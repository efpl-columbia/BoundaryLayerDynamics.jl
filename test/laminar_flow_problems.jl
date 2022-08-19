# NOTE: This file is supposed to be independent such that it can be included in
# an interactive session to examine convergence rates for these problems.

"""
Analytical solution for transient Poiseuille flow. The solution is normalized
as u = uτ Reτ poiseuille(y/δ, tν/δ²), where Reτ is uτδ/ν and uτ is the friction
velocity. The function works for both scalar and array inputs.
"""
function poiseuille(y, t; tol=eps()/8, kmax=100_000)
    t == 0 && return zero(y) # t=0 would require many modes, but u(0) = 0
    u = y .- 0.5 .* y.^2 # initialize with steady-state solution (converts range to array)
    t == Inf && return u
    for k=1:2:kmax # add modes until the contribution is negligible
        α = 16/(k^3*π^3) * exp(-0.25*k^2*π^2*t)
        α < tol && return u
        if u isa Number
            u -= α * sin(0.5*k*π*y)
        else
            @. u -= α * sin(0.5*k*π*y)
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
function couette(y, t; tol=eps()/8, kmax=100_000)
    T = eltype(y)
    t == 0 && return ((y .== 2) .- (y .== 0)) * one(T) # t=0 would require many modes, but u(0) = 0
    u = zero(y) .+ y .- 1 # initialize with steady-state solution (converts range to array)
    t == Inf && return u
    for k=1:kmax # add modes until their contribution is negligible
        α = 2 / (k * π) * exp(-k^2*π^2*t)
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
Analytical solution for the first component of the velocity in a
Taylor-Green vortex. The solution is normalized as

u₁ = A taylor_green_vortex_u1(x₁β, x₂β, tνβ², α/β)
u₂ = B taylor_green_vortex_u1(x₁β, x₂β, tνβ², α/β)

where A, B, α, and β are the parameters of the initial conditions

u₁(t₀) = A cos(x₁α) sin(x₂β)
u₂(t₀) = B sin(x₁α) cos(x₂β)

under the restriction that B = - Aα/β from the continuity equation.
The function works for both scalar and array inputs.
"""
taylor_green_vortex_u1(x, y, t, λ=one(eltype(x))) = @. cos(λ*x) * sin(y) * exp(-(1+λ^2)*t)

"""
Analytical solution for the second component of the velocity in a
Taylor-Green vortex. The solution is normalized as

u₁ = A taylor_green_vortex_u1(x₁β, x₂β, tνβ², α/β)
u₂ = B taylor_green_vortex_u1(x₁β, x₂β, tνβ², α/β)

where A, B, α, and β are the parameters of the initial conditions

u₁(t₀) = A cos(x₁α) sin(x₂β)
u₂(t₀) = B sin(x₁α) cos(x₂β)

under the restriction that B = - Aα/β from the continuity equation.
The function works for both scalar and array inputs.
"""
taylor_green_vortex_u2(x1, x2, t, λ=one(eltype(x1))) = @. sin(λ*x1) * cos(x2) * exp(-(1+λ^2)*t)

"""
Compute the number of time steps necessary for viscous stability given a
maximum Courant number. The default Cmax of 1/2 appears to be stable for the
laminar problems in this file and the default SSP-RK33 time stepping.
"""
Nt_viscous(T, Nz; δ = one(T), ν = one(T), t = one(T), Cmax = one(T)/2, η = nothing) =
        ceil(Int, t * ν / (Cmax * dx3_min(δ, Nz, η)^2)) # dtmax = Cmax dz²/ν

dx3_min(δ, N3, η::Nothing) = 2 * δ / N3
function dx3_min(δ::T, N3, η::T) where T
    x3, Dx3 = ABL.Domains.instantiate(SinusoidalMapping(η, :symmetric), 2*δ)
    ζ = LinRange(zero(T), one(T), 2*N3+1) # all nodes
    minimum(Dx3.(ζ)) / N3
end

function laminar_flow_error(T, Nh, Nv, Nt, u_exact;
        t = one(T), ν = one(T), δ = one(T), vel_bc = zero(T), f = zero(T),
        dir = (one(T), zero(T)), η = nothing, method = SSPRK33())

    dir = dir ./ sqrt(sum(dir.^2)) # normalize direction vector
    lbc = CustomBoundary(vel1 = :dirichlet => -vel_bc * dir[1],
                         vel2 = :dirichlet => -vel_bc * dir[2],
                         vel3 = :dirichlet)
    ubc = CustomBoundary(vel1 = :dirichlet => vel_bc * dir[1],
                         vel2 = :dirichlet => vel_bc * dir[2],
                         vel3 = :dirichlet)
    domain = Domain((1, 1, 2*δ), lbc, ubc, isnothing(η) ? nothing : SinusoidalMapping(η, :symmetric))
    abl = DiscretizedABL((Nh, Nh, Nv), domain, incompressible_flow(ν, constant_forcing = f .* dir))
    evolve!(abl, t, dt = t / Nt, method = method, verbose = false)

    uref = T[u_exact(x3, t) for x1=1:1, x2=1:1, x3=coordinates(abl, :vel1, 3)]
    εu1 = abl[:vel1] .- uref * dir[1]
    εu2 = abl[:vel2] .- uref * dir[2]
    εu3 = ABL.State.getterm(abl.state, :vel3, abl.domain, abl.grid, abl.physical_spaces, ABL.NodeSet(:C))

    # maximum relative error, based on global velocity to avoid division by zero
    sqrt(global_maximum(abs2.(εu1) .+ abs2.(εu2) .+ abs2.(εu3)) / global_maximum(abs2.(uref)))
end

poiseuille_error(T, Nh, Nv, Nt; t = one(T) / 4, ν = one(T), δ = one(T), uτ = one(T),
                 dir = (one(T), zero(T)), η = nothing) =
    laminar_flow_error(T, Nh, Nv, Nt, (y,t) -> (uτ^2*δ/ν) * poiseuille(y / δ, t * ν / δ^2),
    t=t, ν=ν, δ=δ, dir=dir, f=(uτ^2/δ), η=η)

couette_error(T, Nh, Nv, Nt; t = one(T) / 16, ν = one(T), δ = one(T), uτ = one(T),
              dir = (one(T), zero(T)), η = nothing) =
    laminar_flow_error(T, Nh, Nv, Nt, (y,t) -> (uτ^2*δ/ν) * couette(y/δ, t*ν/δ^2),
    t=t, ν=ν, δ=δ, dir=dir, vel_bc=(uτ*uτ*δ/ν), η=η)

"""
Return the error for a Taylor-Green vortex with a given set of parameters and a
given discretization. If the direction is set to a non-zero value, the problem
is solved in a vertical plane with the given orientation. If the direction is
set to zero, the problem is solved in a horizontal plane instead. In horizontal
direction we cannot set an arbitrary orientation since we have to be able to
find a domain size for which the problem is periodic.
"""
function taylor_green_vortex_error(T, Nh, Nv, Nt;
        t = one(T), ν = one(T), α = one(T), β = one(T), A = one(T),
        dir = (zero(T), zero(T)), η = nothing, method = SSPRK33())

    λ = α / β
    B = - A * λ

    ds, mapping, u1ref, u2ref, u3ref = if dir[1] == dir[2] == zero(T)
        # vortex in horizontal plane
        ((2*π/α, 2*π/β, one(T)),
         isnothing(η) ? nothing : SinusoidalMapping(η, :symmetric),
         (x,y,z,t) -> A * taylor_green_vortex_u1(β*x, β*y, β^2*ν*t, λ),
         (x,y,z,t) -> B * taylor_green_vortex_u2(β*x, β*y, β^2*ν*t, λ),
         (x,y,z,t) -> zero(T))
    else
        # vortex in vertical plane
        dir = dir ./ sqrt(sum(dir.^2)) # normalize direction vector
        ((iszero(dir[1]) ? one(T) : 2*π/α/dir[1], iszero(dir[2]) ? one(T) : 2*π/α/dir[2], π/β),
         isnothing(η) ? nothing : SinusoidalMapping(η, :symmetric),
         (x,y,z,t) -> A * taylor_green_vortex_u1(β*(x*dir[1]+y*dir[2]), π/2+β*z, β^2*ν*t, λ) * dir[1],
         (x,y,z,t) -> A * taylor_green_vortex_u1(β*(x*dir[1]+y*dir[2]), π/2+β*z, β^2*ν*t, λ) * dir[2],
         (x,y,z,t) -> B * taylor_green_vortex_u2(β*(x*dir[1]+y*dir[2]), π/2+β*z, β^2*ν*t, λ))
    end

    ic = NamedTuple(vel => (x,y,z) -> uref(x,y,z,zero(T)) for (vel, uref) in
                    zip((:vel1, :vel2, :vel3), (u1ref, u2ref, u3ref)))
    abl = DiscretizedABL((Nh, Nh, Nv), Domain(ds, FreeSlipBoundary(), FreeSlipBoundary()),
                        incompressible_flow(ν, constant_forcing = (0,0)))
    initialize!(abl; ic...)
    evolve!(abl, t, dt = t / Nt, method = method, verbose = false)

    xh = coordinates(abl, :vel1) # same for vel2
    xv = coordinates(abl, :vel3)

    ε1 = abl[:vel1] .- T[u1ref(x...,t) for x=xh]
    ε2 = abl[:vel2] .- T[u2ref(x...,t) for x=xh]
    # compute error before interpolation (bc3 works for error as well)
    # TODO: figure out why the error is not similar when first interpolating
    # and comparing values on I-nodes
    ε3 = ABL.State.interpolate(abl[:vel3] .- T[u3ref(x...,t) for x=xv], :vel3, abl.domain, abl.grid)

    # maximum relative error, based on global velocity to avoid divison by zero
    sqrt(global_maximum(abs2.(ε1) .+ abs2.(ε2) .+ abs2.(ε3)) / global_maximum(
        T[abs2(u1ref(x...,t)) + abs2(u2ref(x...,t)) + abs2(u3ref(x...,t)) for x=xh]))
end
