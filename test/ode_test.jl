module ODEProblems

    const oscillation = (
        rate! = (du, u, t) -> du .= 1im .* u,
        T = (0.0, 2π),
        uref = t -> [exp(1im*t)],
    )

    const timedependent = (
        rate! = (du, u, t) -> du .= sin(t),
        T = (0.0, 1.0),
        uref = t -> [2 - cos(t)],
    )

    const lane_emden5 = (
        rate! = (du, u, t) -> du .= [u[2], -2 / t * u[2] - u[1]^5],
        T = (eps(), 1.0),
        uref = t -> [1 / sqrt(1 + t^2 / 3), - sqrt(3) * t * (t^2 + 3)^(-3/2)],
    )

    const all = (oscillation, timedependent, lane_emden5)
end

function test_constant_growth(alg)
    # constant growth should be integrated exactly by all methods
    rate!(du, u, t) = du .= 1
    u0 = [0.0]
    t = (0.0, 1.0)
    prob = BLD.ODEProblem(rate!, u0, t)
    dt = 1e-2
    BLD.solve!(prob, alg(), dt)
    @test BLD.ODEMethods.time(prob) ≈ 1.0
    @test BLD.state(prob) ≈ [1.0]
end

function test_projection(alg)
    # a quick check that the projection is applied: every evaluation of f(u)
    # has an imaginary component that is removed by the projection, and the
    # real growth rate is “polluted” by any imaginary parts of u so calling
    # rate! with state that hasn’t been projected should result in a test
    # failure
    rate! = (du, u, t) -> du .= 1 .+ imag.(u) .+ 1im
    projection! = u -> u .= real.(u)
    u0 = [0.0im]
    t = (0.0, 1.0)
    prob = BLD.ODEProblem(rate!, projection!, u0, t)
    dt = 1/8
    BLD.solve!(prob, alg(), dt)
    @test BLD.state(prob) ≈ [1.0]
end

function ode_error(ode, alg, nt)
    dt = ode.T[end]/nt
    u0 = ode.uref(ode.T[1])
    prob = BLD.ODEProblem(ode.rate!, u0, ode.T)
    BLD.solve!(prob, alg(), dt)
    sqrt(sum(abs2.(ode.uref(ode.T[end]) .- BLD.state(prob)))) # error
end

function test_ode_convergence(alg, order)
    N = [8, 16, 32, 64, 128]
    for ode in ODEProblems.all
        ε = [ode_error(ode, alg, nt) for nt=N]
        test_convergence(N, ε; order=order)
    end
end

function test_checkpoints(alg)
    times = []
    rate!(du, u, t; checkpoint = false) = (checkpoint && push!(times, t); du .= 1)
    t = (0.0, 1.0)
    nt = 10
    dt = t[end] / nt

    for checkpoints in (nothing, dt:dt:t[end], 5*dt:5*dt:t[end], 3*dt:3*dt:2*t[end])
        empty!(times)
        prob = BLD.ODEProblem(rate!, [0.0], t)
        if checkpoints == nothing
            BLD.solve!(prob, alg(), dt, checkpoints = checkpoints)
            @test times == []
        elseif maximum(checkpoints) <= t[end]
            BLD.solve!(prob, alg(), dt, checkpoints = checkpoints)
            @test times ≈ collect(checkpoints)
        else
            @test_throws ErrorException BLD.solve!(prob, alg(), dt, checkpoints = checkpoints)
        end
    end
end

@timeit "ODE" @testset "ODE Solution Methods" begin
    for (alg, order) in ((Euler, 1), (AB2, 2), (SSPRK22, 2), (SSPRK33, 3))
        test_constant_growth(alg)
        test_projection(alg)
        test_ode_convergence(alg, order)
        test_checkpoints(alg)
    end
end
