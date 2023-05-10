include("laminar_flow_problems.jl")

function test_laminar_flow_convergence(T, Nz_min = 3; grid_stretching = false, seed = 62275)
    Random.seed!(seed) # same seed for each process
    method = SSPRK22() # used for tests of spatial convergence

    # use a progression of small integers that are close to equidistant in log-space
    N = filter(n -> n>=Nz_min, [11, 14, 18, 24])

    # parameter for grid stretching, η ∈ [0.4, 0.6]
    η = grid_stretching ? 0.4 + rand() / 5 : nothing

    # parameters for Poiseuille & Couette flow
    ν  = one(T) / 4 * 3 + rand(T) / 2
    ex = (θ = rand(T) * 2 * π; (cos(θ), sin(θ)))
    δ  = one(T) / 4 * 3 + rand(T) / 2
    uτ = one(T) / 4 * 3 + rand(T) / 2
    t  = (δ^2 / ν) / 6
    Nt = Nt_viscous(T, N[end], t=t, δ=δ, ν=ν, η=η, Cmax=one(T)/4)

    params = (t=t, ν=ν, δ=δ, uτ=uτ, dir=ex, η=η, method=method)
    εp = poiseuille_error.(T, 3, N, Nt; params...)
    εc =    couette_error.(T, 3, N, Nt; params...)

    # parameters for Taylor-Green vortex
    α  = one(T) / 4 * 3 + rand(T) / 2
    β  = one(T) / 4 * 3 + rand(T) / 2
    γ  = one(T) / 4 * 3 + rand(T) / 2
    U  = one(T) / 4 * 3 + rand(T) / 2
    t = 1 / ((α^2 + β^2 + γ^2) * ν)
    Nt = Nt_viscous(T, N[end], δ=T(π)/(2*β), ν=ν, t=t, Cmax=one(T)/4)

    # vortex in vertical plane measures convergence in space
    params = (t=t, ν=ν, α=α, β=β, γ=γ, U=U, η=η)
    εtgv = taylor_green_vortex_error.(T, 3, N, Nt; method=method, params...)

    # vortex in horizontal plane measures convergence in time
    params = (params..., W=zero(T))
    εtgeu = taylor_green_vortex_error.(T, 3, Nz_min, N; method=Euler(), params...)
    εtgab = taylor_green_vortex_error.(T, 3, Nz_min, N; method=AB2(), params...)
    εtgr2 = taylor_green_vortex_error.(T, 3, Nz_min, N; method=SSPRK22(), params...)
    εtgr3 = taylor_green_vortex_error.(T, 3, Nz_min, N; method=SSPRK33(), params...)

    # these threshold values have been checked with a number of different seeds
    # and should be fairly robust even if some small details of the
    # implementation are changed and the results are no longer numerically
    # identical
    @test εp[end] < (grid_stretching ? 1e-3 : 2e-3)
    @test εc[end] < (grid_stretching ? 1e-3 : 5e-4)
    @test εtgv[end] < (grid_stretching ? 7e-3 : 5e-3)
    test_convergence(N, εp, order=2, threshold_slope=0.95)
    test_convergence(N, εc, order=2, threshold_slope=0.9)
    test_convergence(N, εtgv, order=2, threshold_slope=0.94)
    @test εtgeu[end] < 2e-2
    @test εtgab[end] < 4e-4
    @test εtgr2[end] < 2e-4
    @test εtgr3[end] < 2e-6
    test_convergence(N, εtgeu, order=1, threshold_slope=0.99)
    test_convergence(N, εtgab, order=2, threshold_slope=0.98)
    test_convergence(N, εtgr2, order=2, threshold_slope=0.99)
    test_convergence(N, εtgr3, order=3, threshold_slope=0.99)

    N, εp, εc, εtgv, εtgeu, εtgab, εtgr2, εtgr3 # return for plotting in interactive usage
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
