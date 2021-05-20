abstract type TimeIntegrationAlgorithm end
abstract type OrdinaryDiffEqAlgorithm <: TimeIntegrationAlgorithm end

"""
    Euler()

First-order forward Euler time integration.
"""
struct Euler <: TimeIntegrationAlgorithm end

"""
    AB2()

Second-order Adams-Bashforth time integration.
"""
struct AB2 <: TimeIntegrationAlgorithm end

"""
    SSPRK22()

Two-stage second-order strong-stability-preserving Runge-Kutta time
integration.
"""
struct SSPRK22 <: OrdinaryDiffEqAlgorithm end

"""
    SSPRK33()

Three-stage third-order strong-stability-preserving Runge-Kutta time
integration.
"""
struct SSPRK33 <: OrdinaryDiffEqAlgorithm end

abstract type TimeIntegrationCache{T} end

struct EulerCache{T} <: TimeIntegrationCache{T}
    EulerCache(prob) = new{typeof(prob.u)}()
end

struct AB2Cache{T} <: TimeIntegrationCache{T}
    uprev::T
    duprev::T
    initialized::Ref{Bool}
    AB2Cache(prob) = new{typeof(prob.u)}(zero(prob.u), zero(prob.du) * NaN, Ref(false))
end

init_cache(prob, ::Euler) = EulerCache(prob)
init_cache(prob, ::AB2) = AB2Cache(prob)
init_alg(prob, ::SSPRK22) = OrdinaryDiffEq.SSPRK22((u, f, p, t) -> prob.projection!(u))
init_alg(prob, ::SSPRK33) = OrdinaryDiffEq.SSPRK33((u, f, p, t) -> prob.projection!(u))

function perform_step!(problem, dt, ::EulerCache)
    problem.u .+= dt * problem.du
    problem.projection!(problem.u)
    problem.t[] += dt
    problem.rate!(problem.du, problem.u, problem.t[])
end

function perform_step!(problem, dt, c::AB2Cache)
    c.uprev .= problem.u
    if !c.initialized[]
        # first stage of RK2
        @. problem.u += dt * problem.du
        problem.projection!(problem.u)
        # save complete du for multistep method
        @. c.duprev = problem.du + (problem.u - c.uprev) / dt
        # second stage of RK2
        problem.rate!(problem.du, problem.u, problem.t[]+dt)
        @. problem.u = (c.uprev + problem.u + dt * problem.du) / 2
        problem.projection!(problem.u)
        c.initialized[] = true
    else
        @. problem.u += dt/2 * (3 * problem.du - c.duprev)
        problem.projection!(problem.u)
        @. c.duprev = problem.du + (problem.u - c.uprev) * 2 / (3*dt)
    end
    problem.t[] += dt
    problem.rate!(problem.du, problem.u, problem.t[])
end

struct TimeIntegrationProblem{T,R,P}
    rate!::R
    projection!::P
    u::T
    du::T
    t::Ref{Float64}
    tmax::Float64
end

function TimeIntegrationProblem(rate!, projection!, u0, (t1, t2))
    du = zero(u0)
    projection!(u0)
    rate!(du, u0, t1)
    TimeIntegrationProblem(rate!, projection!, u0, du, Ref(t1), t2)
end

function solve(prob::TimeIntegrationProblem,
               alg::TimeIntegrationAlgorithm,
               dt::Float64,
               log = (u, tstep, t) -> ())
    cache = init_cache(prob, alg)
    nt = Int((prob.tmax - prob.t[]) / dt) # may raise error if not divisible
    for tstep in 1:nt
        perform_step!(prob, dt, cache)
        log(prob.u, tstep, prob.t[])
    end
    prob.u
end

function solve(prob::TimeIntegrationProblem,
               alg::OrdinaryDiffEqAlgorithm,
               dt::Float64,
               log = (u, tstep, t) -> ())
    step!(du, u, p, t) = prob.rate!(du, u, t)
    TimerOutputs.@timeit "setup" begin
    odeprob = OrdinaryDiffEq.ODEProblem(step!, prob.u, (prob.t[], prob.tmax))
    odealg = init_alg(prob, alg)
    integrator = OrdinaryDiffEq.init(odeprob, odealg, dt = dt,
                                     save_start = false, save_everystep = false)
    end
    tstep = 0
    for (state, t) in OrdinaryDiffEq.tuples(integrator)
        # this part is run after every step (not before/during)
        tstep += 1
        log(state, tstep, t)
    end
    integrator.sol[end]
end
