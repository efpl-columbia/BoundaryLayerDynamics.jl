# NOTE: This file is supposed to be independent such that it can be included in
# an interactive session to examine convergence rates for these problems.

"""
Analytical solution for transient Poiseuille flow. The solution is normalized
as u = uτ Reτ poiseuille(y/δ, tν/δ²), where Reτ is uτδ/ν and uτ is the friction
velocity. The function works for both scalar and array inputs.
"""
function poiseuille(y, t; tol = eps() / 8, kmax = 100_000)
    t == 0 && return zero(y) # t=0 would require many modes, but u(0) = 0
    u = y .- 0.5 .* y .^ 2 # initialize with steady-state solution (converts range to array)
    t == Inf && return u
    for k in 1:2:kmax # add modes until the contribution is negligible
        α = 16 / (k^3 * π^3) * exp(-0.25 * k^2 * π^2 * t)
        α < tol && return u
        if u isa Number
            u -= α * sin(0.5 * k * π * y)
        else
            @. u -= α * sin(0.5 * k * π * y)
        end
    end
    @warn "Reached maximum number of modes (k=" * string(kmax) * ")"
    u
end

"""
Analytical solution for transient Couette flow. The solution is normalized
as u = uτ Reτ couette(y/δ, tν/δ²), where Reτ is uτδ/ν and uτ is the friction
velocity. The function works for both scalar and array inputs.
"""
function couette(y, t; tol = eps() / 8, kmax = 100_000)
    T = eltype(y)
    t == 0 && return ((y .== 2) .- (y .== 0)) * one(T) # t=0 would require many modes, but u(0) = 0
    u = zero(y) .+ y .- 1 # initialize with steady-state solution (converts range to array)
    t == Inf && return u
    for k in 1:kmax # add modes until their contribution is negligible
        α = 2 / (k * π) * exp(-k^2 * π^2 * t)
        α < tol && return u
        if u isa Number
            u += α * sin(y * k * π)
        else
            @. u += α * sin(y * k * π)
        end
    end
    @warn "Reached maximum number of modes (k=" * string(kmax) * ")"
    u
end

"""
Compute the number of time steps necessary for viscous stability given a
maximum Courant number. The default Cmax of 1/2 appears to be stable for the
laminar problems in this file and the default SSP-RK33 time stepping.
"""
Nt_viscous(T, Nz; δ = one(T), ν = one(T), t = one(T), Cmax = one(T) / 2, η = nothing) =
    ceil(Int, t * ν / (Cmax * dx3_min(δ, Nz, η)^2)) # dtmax = Cmax dz²/ν

dx3_min(δ, N3, η::Nothing) = 2 * δ / N3
function dx3_min(δ::T, N3, η::T) where {T}
    x3, Dx3 = BoundaryLayerDynamics.Domains.instantiate(SinusoidalMapping(η, :symmetric), 2 * δ)
    ζ = LinRange(zero(T), one(T), 2 * N3 + 1) # all nodes
    minimum(Dx3.(ζ)) / N3
end

function laminar_flow_error(T, Nh, Nv, Nt, u_exact; vel_bc = zero(T), f = zero(T), t, ν, δ, dir, η, method)
    dir = dir ./ sqrt(sum(dir .^ 2)) # normalize direction vector
    lbc = CustomBoundary(;
        vel1 = :dirichlet => -vel_bc * dir[1],
        vel2 = :dirichlet => -vel_bc * dir[2],
        vel3 = :dirichlet,
    )
    ubc =
        CustomBoundary(; vel1 = :dirichlet => vel_bc * dir[1], vel2 = :dirichlet => vel_bc * dir[2], vel3 = :dirichlet)
    domain = Domain((1, 1, 2 * δ), lbc, ubc, isnothing(η) ? nothing : SinusoidalMapping(η, :symmetric))
    model = Model((Nh, Nh, Nv), domain, incompressible_flow(ν; constant_forcing = f .* dir))
    evolve!(model, t; dt = t / Nt, method = method, verbose = false)

    uref = T[u_exact(x3, t) for x1 in 1:1, x2 in 1:1, x3 in coordinates(model, :vel1, 3)]
    εu1 = model[:vel1] .- uref * dir[1]
    εu2 = model[:vel2] .- uref * dir[2]
    εu3 = BoundaryLayerDynamics.State.getterm(
        model.state,
        :vel3,
        model.domain,
        model.grid,
        model.physical_spaces,
        BoundaryLayerDynamics.NodeSet(:C),
    )

    # maximum relative error, based on global velocity to avoid division by zero
    sqrt(global_maximum(abs2.(εu1) .+ abs2.(εu2) .+ abs2.(εu3)) / global_maximum(abs2.(uref)))
end

poiseuille_error(T, Nh, Nv, Nt; t, ν, δ, uτ, dir, η, method) = laminar_flow_error(
    T,
    Nh,
    Nv,
    Nt,
    (y, t) -> (uτ^2 * δ / ν) * poiseuille(y / δ, t * ν / δ^2);
    t = t,
    ν = ν,
    δ = δ,
    dir = dir,
    f = (uτ^2 / δ),
    η = η,
    method = method,
)

couette_error(T, Nh, Nv, Nt; t, ν, δ, uτ, dir, η, method) = laminar_flow_error(
    T,
    Nh,
    Nv,
    Nt,
    (y, t) -> (uτ^2 * δ / ν) * couette(y / δ, t * ν / δ^2);
    t = t,
    ν = ν,
    δ = δ,
    dir = dir,
    vel_bc = (uτ * uτ * δ / ν),
    η = η,
    method = method,
)

"""
Return the error for a Taylor-Green vortex with a given set of parameters and a
given discretization. If the direction is set to a non-zero value, the problem
is solved in a vertical plane with the given orientation. If the direction is
set to zero, the problem is solved in a horizontal plane instead. In horizontal
direction we cannot set an arbitrary orientation since we have to be able to
find a domain size for which the problem is periodic.
"""
function taylor_green_vortex_error(
    T,
    Nh,
    Nv,
    Nt;
    t = one(T),
    ν = one(T),
    α = one(T),
    β = one(T),
    γ = one(T),
    U = one(T),
    W = nothing,
    η = nothing,
    method = SSPRK33(),
)

    # define reference solution based on type of problem that is selected
    tg1(x, y, t) = sin(x) * cos(y) * exp(-t)
    tg2(x, y, t) = -cos(x) * sin(y) * exp(-t)
    u1ref, u2ref, u3ref, bc = if isnothing(W) # W is only used for horizontal case
        # vortex in vertical plane
        (
            (x, y, z, t) -> U * α * γ * tg1(α * x + β * y, γ * z, (α^2 + β^2 + γ^2) * ν * t),
            (x, y, z, t) -> U * β * γ * tg1(α * x + β * y, γ * z, (α^2 + β^2 + γ^2) * ν * t),
            (x, y, z, t) -> U * (α^2 + β^2) * tg2(α * x + β * y, γ * z, (α^2 + β^2 + γ^2) * ν * t),
            FreeSlipBoundary(),
        )
    else
        # vortex in horizontal plane
        iszero(W) || error("TODO: implement non-zero u₃")
        (
            (x, y, z, t) -> U * β * tg1(α * x, β * y, (α^2 + β^2) * ν * t),
            (x, y, z, t) -> U * α * tg2(α * x, β * y, (α^2 + β^2) * ν * t),
            (x, y, z, t) -> W,
            CustomBoundary(; vel1 = :neumann, vel2 = :neumann, vel3 = :dirichlet => W),
        )
    end

    ds = convert.(T, (2 * π / α, 2 * π / β, 2 * π / γ))
    mapping = isnothing(η) ? nothing : SinusoidalMapping(η, :symmetric)
    domain = Domain(ds, bc, bc, mapping)
    processes = incompressible_flow(ν; constant_forcing = 0)
    model = Model((Nh, Nh, Nv), domain, processes)

    ic = NamedTuple(
        vel => (x, y, z) -> uref(x, y, z, zero(T)) for (vel, uref) in zip((:vel1, :vel2, :vel3), (u1ref, u2ref, u3ref))
    )
    initialize!(model; ic...)
    evolve!(model, t; dt = t / Nt, method = method, verbose = false)

    xh = coordinates(model, :vel1) # same for vel2
    xv = coordinates(model, :vel3)

    ε1 = model[:vel1] .- T[u1ref(x..., t) for x in xh]
    ε2 = model[:vel2] .- T[u2ref(x..., t) for x in xh]
    # compute error before interpolation (bc3 works for error as well)
    ε3 = BoundaryLayerDynamics.State.interpolate(
        model[:vel3] .- T[u3ref(x..., t) for x in xv],
        :vel3,
        model.domain,
        model.grid,
    )

    # maximum relative error, based on global velocity to avoid divison by zero
    uref_max = exp(-ν * t * (α^2 + β^2 + (isnothing(W) ? γ^2 : 0)))
    sqrt(global_maximum(abs2.(ε1) .+ abs2.(ε2) .+ abs2.(ε3)) / uref_max)
end
