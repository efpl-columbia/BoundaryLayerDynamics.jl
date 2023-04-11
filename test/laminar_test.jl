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

    # TODO: check if these bounds can be made tighter
    test_convergence(N, εp, order=2, threshold_slope=0.875)
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

    # set up identical simulations for constant force & constant flux
    domain = Domain((4π, 2π, 2), SmoothWall(), SmoothWall())
    model1 = Model((4, 4, nz), domain, incompressible_flow(1.0, constant_forcing = 1))
    model2 = Model((4, 4, nz), domain, incompressible_flow(1.0, constant_flux = mf))

    ε_constant_force = zeros(nit)
    ε_constant_flux  = zeros(nit)
    ε(model) = global_sum(abs.(model.state[:vel1][1,1,:] .- u_exact[model.grid.i3min:model.grid.i3max])) / nz

    # run several segments of time integration, measuring errors inbetween
    for i=1:nit
        evolve!(model1, dt * nt, dt = dt, verbose = false)
        ε_constant_force[i] = ε(model1)
        evolve!(model2, dt * nt, dt = dt, verbose = false)
        ε_constant_flux[i] = ε(model2)
    end

    # test that difference to steady-state solution is smaller than
    # for constant forcing and that it is monotonically decreasing
    @test all(ε_constant_flux .< ε_constant_force)
    @test all(diff(ε_constant_flux) .< 0)
    @test ε_constant_flux[end] < 1e-6 # should be the case for nz≥4
end


@timeit "Laminar Flows" @testset "Laminar Flow Solutions" begin

    # test that laminar flow solutions converge at the right order
    test_laminar_flow_convergence(Float64, MPI.Initialized() ? MPI.Comm_size(MPI.COMM_WORLD) : 3)
    test_laminar_flow_convergence(Float64, MPI.Initialized() ? MPI.Comm_size(MPI.COMM_WORLD) : 3,
                                  grid_stretching = true)

    # test that a poiseuille flow driven by a constant flux converges faster
    test_constant_flux_poiseuille(16)
    MPI.Initialized() && test_constant_flux_poiseuille(MPI.Comm_size(MPI.COMM_WORLD))
end
