"""
Test that a standard channel flow with output can be run, including output,
without producing an error.
"""
function test_channel(Nv; Nh = 4, Re = 1.0, CFL = 0.1, T = 1/Re, Nt = 100)
    cf = prepare_closed_channel(Re, (Nh,Nh,Nv), constant_flux = true)
    dt = (2/Nv)^2 * Re * CFL

    mktempdir_parallel() do dir
        integrate!(cf, dt, Nt, output_io = devnull,
            profiles_dir = joinpath(dir, "profiles"), profiles_frequency = 10,
            snapshot_steps = [div(1*Nt,5), div(2*Nt,5), div(3*Nt,5), div(4*Nt,5)],
            snapshot_dir = joinpath(dir, "snapshots"), verbose = true)

        # attempt setting the velocity from the latest snapshot
        last_snapshot = readdir(joinpath(dir, "snapshots"))[end]
        CF.load_snapshot!(cf, joinpath(dir, "snapshots", last_snapshot))
    end
end

include("laminar_flow_problems.jl")

function test_laminar_flow_convergence(T, Nz_min = 3; grid_stretching = false)

    # use a progression of small integers that are close to equidistant in log-space
    N = filter(n -> n>=Nz_min, [11, 14, 18, 24, 32])
    Random.seed!(363613674) # same seed for each process

    # parameter for grid stretching, η ∈ [0.5, 0.75]
    η = grid_stretching ? 0.5 + rand() / 4 : nothing

    # parameters for Poiseuille & Couette flow
    ν  = one(T) / 4 * 3 + rand(T) / 2
    ex = (θ = rand(T) * 2 * π; (cos(θ), sin(θ)))
    δ  = one(T) / 4 * 3 + rand(T) / 2
    uτ = one(T) / 4 * 3 + rand(T) / 2
    t  = (δ^2 / ν) / 6
    Nt = Nt_viscous(T, N[end], t=t, δ=δ, ν=ν, η=η)

    εp = poiseuille_error.(T, 3, N, Nt, t=t, ν=ν, δ=δ, uτ=uτ, dir=ex, η=η)
    εc =    couette_error.(T, 3, N, Nt, t=t, ν=ν, δ=δ, uτ=uτ, dir=ex, η=η)

    # parameters for Taylor-Green vortex
    A  = one(T) / 4 * 3 + rand(T) / 2
    α  = one(T) / 4 * 3 + rand(T) / 2
    β  = one(T) / 4 * 3 + rand(T) / 2
    t = 1 / ((α^2 + β^2) * ν)
    Nt = Nt_viscous(T, N[end], δ=T(π)/(2*β), ν=ν, t=t, Cmax=1/16)

    εtgv = taylor_green_vortex_error.(T, 3, N, Nt, t=t, ν=ν, α=α, β=β, A=A, dir=ex, η=η)
    εtgh = taylor_green_vortex_error.(T, 3, Nz_min, N,  t=t, ν=ν, α=α, β=β, A=A, η=η)

    @test εp[end] < 1e-3
    @test εc[end] < 1e-3
    @test εtgv[end] < 2e-3
    @test εtgh[end] < 1e-5

    test_convergence(N, εp, order=2, threshold_slope=0.9)
    test_convergence(N, εc, order=2, threshold_slope=0.925)
    test_convergence(N, εtgv, order=2)
    test_convergence(N, εtgh, order=3)

    N, εp, εc, εtgv, εtgh # return for plotting in interactive usage
end

function test_constant_flux_poiseuille(nz)

    z = LinRange(0, 2, 2*nz+1)[2:2:end]
    u_exact = poiseuille(z, Inf)
    mf = sum(u_exact) / nz

    # total integration to t=1 in `nit` iteration steps
    CFL, ν, tend, nit = (1/4, 1.0, 1.0, 8)
    dtmax = (2 / nz)^2 / ν * CFL
    nt = ceil(Int, tend / dtmax / nit) # steps per iteration
    dt = tend / (nit * nt)

    ε_constant_force = zeros(nit)
    ε_constant_flux  = zeros(nit)
    ε(cf) = CF.global_sum(abs.(cf.velocity[1][1,1,:] .- u_exact[cf.grid.iz_min:cf.grid.iz_max])) / nz

    # compute errors for constant forcing first
    cf1 = prepare_closed_channel(1.0, (4, 4, nz), constant_flux = false)

    for i=1:nit
        integrate!(cf1, dt, nt, verbose=false)
        ε_constant_force[i] = ε(cf1)
    end

    # compute errors for constant flux next
    cf2 = ChannelFlowProblem((4, 4, nz), (4π, 2π, 2.0),
        CF.bc_noslip(), CF.bc_noslip(), 1.0, (mf, 0.0), true)
    for i=1:nit
        integrate!(cf2, dt, nt, verbose=false)
        ε_constant_flux[i] = ε(cf2)
    end

    # test that difference to steady-state solution is smaller than
    # for constant forcing and that it is monotonically decreasing
    @test all(ε_constant_flux .< ε_constant_force)
    @test all(diff(ε_constant_flux) .< 0)
    @test ε_constant_flux[end] < 1e-6 # should be the case for nz≥4
end

# test that a channel flow with output runs without error
test_channel(16)
MPI.Initialized() && test_channel(MPI.Comm_size(MPI.COMM_WORLD))

# test that laminar flow solutions converge at the right order
test_laminar_flow_convergence(Float64, MPI.Initialized() ? MPI.Comm_size(MPI.COMM_WORLD) : 3)
test_laminar_flow_convergence(Float64, MPI.Initialized() ? MPI.Comm_size(MPI.COMM_WORLD) : 3,
                              grid_stretching = true)

# test that a poiseuille flow driven by a constant flux converges faster
test_constant_flux_poiseuille(16)
MPI.Initialized() && test_constant_flux_poiseuille(MPI.Comm_size(MPI.COMM_WORLD))
