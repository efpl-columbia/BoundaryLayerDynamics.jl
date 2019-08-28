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
Nt_viscous(T, Nz; δ = one(T), ν = one(T), t = one(T), Cmax = one(T)/2) =
        ceil(Int, t * ν / (Cmax * (2*δ/Nz)^2)) # dtmax = Cmax dz²/ν

function laminar_flow_error(T, Nh, Nv, Nt, u_exact;
        t = one(T), ν = one(T), δ = one(T), vel_bc = zero(T), f = zero(T), dir = (one(T), zero(T)))

    dir = dir ./ sqrt(sum(dir.^2)) # normalize direction vector
    cf = CF.ChannelFlowProblem(
        (Nh, Nh, Nv), # grid resolution
        (1.0, 1.0, 2*δ), # domain size
        CF.zero_ics(T), # initial conditions
        ("Dirichlet" => -vel_bc * dir[1], "Dirichlet" => -vel_bc * dir[2], "Dirichlet"), # lower BC
        ("Dirichlet" =>  vel_bc * dir[1], "Dirichlet" =>  vel_bc * dir[2], "Dirichlet"), # upper BC
        ν, # kinematic viscosity
        f .* dir, # pressure forcing
        false, # constant flux forcing
    )
    integrate!(cf, t / Nt, Nt, verbose=false)
    vel = CF.get_velocity(cf)

    # need to get local vector of z-values here, for H-nodes
    # coords(
    #uref = T[u_exact(z, t) for x=0:0, y=0:0, z=LinRange(0, 2*δ, 2*Nv+1)[2:2:end-1]]
    uref = T[u_exact(x[3], t) for x=CF.coords(cf, CF.NodeSet(:H))]
    εu1 = vel[1] .- uref * dir[1]
    εu2 = vel[2] .- uref * dir[2]
    # TODO: use interpolation across MPI boundaries for this
    #εu3 = cat(vel[3][:,:,1]/2, (vel[3][:,:,1:end-1]+vel[3][:,:,2:end])/2, vel[3][:,:,end]/2, dims=3)
    εu3 = CF.interpolate(vel[3], cf.lower_bcs[3], cf.upper_bcs[3])

    # maximum relative error, with a safety factor for when the norm of the velocity is zero
    sqrt(CF.global_maximum((abs2.(εu1) .+ abs2.(εu2) .+ abs2.(εu3)) ./ (abs2.(uref) .+ 64*eps())))
end

poiseuille_error(T, Nh, Nv, Nt; t = one(T) / 4, ν = one(T), δ = one(T), uτ = one(T), dir = (one(T), zero(T))) =
    laminar_flow_error(T, Nh, Nv, Nt, (y,t) -> (uτ^2*δ/ν) * poiseuille(y / δ, t * ν / δ^2),
    t=t, ν=ν, δ=δ, dir=dir, f=(uτ^2/δ))

couette_error(T, Nh, Nv, Nt; t = one(T) / 16, ν = one(T), δ = one(T), uτ = one(T), dir = (one(T), zero(T))) =
    laminar_flow_error(T, Nh, Nv, Nt, (y,t) -> (uτ^2*δ/ν) * couette(y/δ, t*ν/δ^2),
    t=t, ν=ν, δ=δ, dir=dir, vel_bc=(uτ*uτ*δ/ν))

"""
Return the error for a Taylor-Green vortex with a given set of parameters and a
given discretization. If the direction is set to a non-zero value, the problem
is solved in a vertical plane with the given orientation. If the direction is
set to zero, the problem is solved in a horizontal plane instead. In horizontal
direction we cannot set an arbitrary orientation since we have to be able to
find a domain size for which the problem is periodic.
"""
function taylor_green_vortex_error(T, Nh, Nv, Nt;
        t = one(T), ν = one(T), α = one(T), β = one(T), A = one(T), dir = (zero(T), zero(T)))

    λ = α / β
    B = - A * λ

    ds, u1ref, u2ref, u3ref = if dir[1] == dir[2] == zero(T)
        # vortex in horizontal plane
        ((2*π/α, 2*π/β, one(T)),
        (x,y,z,t) -> A * taylor_green_vortex_u1(β*x, β*y, β^2*ν*t, λ),
        (x,y,z,t) -> B * taylor_green_vortex_u2(β*x, β*y, β^2*ν*t, λ),
        (x,y,z,t) -> zero(T))
    else
        # vortex in vertical plane
        dir = dir ./ sqrt(sum(dir.^2)) # normalize direction vector
        ((iszero(dir[1]) ? one(T) : 2*π/α/dir[1], iszero(dir[2]) ? one(T) : 2*π/α/dir[2], π/β),
        (x,y,z,t) -> A * taylor_green_vortex_u1(β*(x*dir[1]+y*dir[2]), π/2+β*z, β^2*ν*t, λ) * dir[1],
        (x,y,z,t) -> A * taylor_green_vortex_u1(β*(x*dir[1]+y*dir[2]), π/2+β*z, β^2*ν*t, λ) * dir[2],
        (x,y,z,t) -> B * taylor_green_vortex_u2(β*(x*dir[1]+y*dir[2]), π/2+β*z, β^2*ν*t, λ))
    end

    ic = Tuple((x,y,z) -> uref(x,y,z,zero(T)) for uref=(u1ref, u2ref, u3ref))
    cf = CF.ChannelFlowProblem((Nh, Nh, Nv), ds, ic, CF.bc_freeslip(), CF.bc_freeslip(), ν, (zero(T), zero(T)), false)
    integrate!(cf, t / Nt, Nt, verbose=false)
    vel = CF.get_velocity(cf)

    #nx1, nx2, nx3 = size(vel[1])
    #x1 = LinRange(0, ds[1], nx1+1)[1:end-1]
    #x2 = LinRange(0, ds[2], nx2+1)[1:end-1]
    #x3h, x3v = (x3 = LinRange(0, ds[3], 2*nx3+1); (x3[2:2:end-1], x3[3:2:end-2]))
    xh = CF.coords(cf, CF.NodeSet(:H))
    xv = CF.coords(cf, CF.NodeSet(:V))

    #ε1 = vel[1] .- [u1ref(x1,x2,x3,t) for x1=x1, x2=x2, x3=x3h]
    #ε2 = vel[2] .- [u2ref(x1,x2,x3,t) for x1=x1, x2=x2, x3=x3h]
    #ε3 = (ε = vel[3] .- [u3ref(x1,x2,x3,t) for x1=x1, x2=x2, x3=x3v];
    ε1 = vel[1] .- T[u1ref(x...,t) for x=xh]
    ε2 = vel[2] .- T[u2ref(x...,t) for x=xh]
    ε3 = (ε = vel[3] .- T[u3ref(x...,t) for x=xv];
          CF.interpolate(ε, cf.lower_bcs[3], cf.upper_bcs[3]))
            #cat(ε[:,:,1]/2, (ε[:,:,2:end].+ε[:,:,1:end-1])/2, ε[:,:,end]/2, dims=3))

    # maximum relative error, with a safety factor for when the norm of the velocity is zero
    sqrt(CF.global_maximum((abs2.(ε1) .+ abs2.(ε2) .+ abs2.(ε3)) ./ T[abs2(u1ref(x...,t)) +
        abs2(u2ref(x...,t)) + abs2(u3ref(x...,t)) + 64*eps() for x=xh]))
end
