abstract type TimeIntegrationAlgorithm end

"""
    Euler(; dt)

First-order forward Euler time integration with constant time step `dt`.
"""
struct Euler <: TimeIntegrationAlgorithm
    constant_dt::Real
    Euler(; dt = nothing) = isnothing(dt) ?
        error("Time integration method requires specifying a fixed time step size") :
        new(dt)
end

"""
    AB2(; dt)

Second-order Adams-Bashforth time integration with constant time step `dt`.
"""
struct AB2 <: TimeIntegrationAlgorithm
    constant_dt::Real
    AB2(; dt = nothing) = isnothing(dt) ?
        error("Time integration method requires specifying a fixed time step size") :
        new(dt)
end

"""
    SSPRK22(; dt)

Two-stage second-order strong-stability-preserving Runge-Kutta time
integration with constant time step `dt`.
"""
struct SSPRK22 <: TimeIntegrationAlgorithm
    constant_dt::Real
    SSPRK22(; dt = nothing) = isnothing(dt) ?
        error("Time integration method requires specifying a fixed time step size") :
        new(dt)
end

"""
    SSPRK33(; dt)

Three-stage third-order strong-stability-preserving Runge-Kutta time
integration with constant time step `dt`.
"""
struct SSPRK33 <: TimeIntegrationAlgorithm
    constant_dt::Real
    SSPRK33(; dt = nothing) = isnothing(dt) ?
        error("Time integration method requires specifying a fixed time step size") :
        new(dt)
end

timestep(alg::TimeIntegrationAlgorithm) = alg.constant_dt

abstract type TimeIntegrationCache{T} end

struct EulerCache{T} <: TimeIntegrationCache{T}
    EulerCache(prob) = new{typeof(prob.u)}()
end

struct AB2Cache{T} <: TimeIntegrationCache{T}
    duprev::T
    last_dt::Ref{Float64}
    AB2Cache(prob) = new{typeof(prob.u)}(zero(prob.du) * NaN, Ref(NaN))
end

struct SSPRK22Cache{T} <: TimeIntegrationCache{T}
    utmp::T
    SSPRK22Cache(prob) = new{typeof(prob.u)}(zero(prob.u) * NaN)
end

struct SSPRK33Cache{T} <: TimeIntegrationCache{T}
    utmp::T
    SSPRK33Cache(prob) = new{typeof(prob.u)}(zero(prob.u) * NaN)
end

init_cache(prob, ::Euler) = EulerCache(prob)
init_cache(prob, ::AB2) = AB2Cache(prob)
init_cache(prob, ::SSPRK22) = SSPRK22Cache(prob)
init_cache(prob, ::SSPRK33) = SSPRK33Cache(prob)

"""
    perform_step!(problem, dt, cache)

Advance the solution `problem.u` by a time step of `dt`. The type of `cache`
determines which integration scheme is used and stores intermediate values
required by the scheme.

The method should assume that `problem.dt` is up-to-date and is allowed to
overwrite it. It is not necessary to update `problem.dt` at the end of the
step, as this is done by the caller.
"""
function perform_step! end

function perform_step!(problem, dt, ::EulerCache)
    problem.u .+= dt * problem.du
    problem.projection!(problem.u)
end

function perform_step!(problem, dt, c::AB2Cache)
    if isnan(c.last_dt[])
        # first step: f(u,t) of previous step is not available yet
        # -> run a single Runge-Kutta step

        # compute u1 = u0 + dt f(u0, t0)
        @. problem.u += dt * problem.du
        problem.projection!(problem.u)

        # store f(u0, t0) for later and compute f(u1, t1)
        @. c.duprev = problem.du
        problem.rate!(problem.du, problem.u, problem.t[]+dt)

        # compute u2 = u1 + ½ dt (f(u1, t1) - f(u0, t0))
        # (u2 = ½ u0 + ½ u1 + ½ Δt f(u1, t1) reformulated to eliminate u0)
        @. problem.u += dt * (problem.du - c.duprev) / 2
        problem.projection!(problem.u)

        c.last_dt[] = dt
        # c.duprev contains f(u0, t0), as required for next step
    else
        c.last_dt[] == dt || error("Cannot change time step for multi-step methods")
        @. problem.u += dt/2 * (3 * problem.du - c.duprev)
        problem.projection!(problem.u)
        c.duprev .= problem.du
    end
end

function perform_step!(problem, dt, c::SSPRK22Cache)

    # u1 = u0 + dt f(u0, t0)
    @. c.utmp = problem.u + dt * problem.du
    problem.projection!(c.utmp)

    # compute f(u1, t1)
    problem.rate!(problem.du, c.utmp, problem.t[] + dt)

    # u2 = ½ u0 + ½ u1 + ½ dt f(u1, t1)
    @. problem.u = (problem.u + c.utmp + dt * problem.du) / 2
    problem.projection!(problem.u)
end

function perform_step!(problem, dt, c::SSPRK33Cache)

    # u1 = u0 + dt f(u0, t0)
    @. c.utmp = problem.u + dt * problem.du
    problem.projection!(c.utmp)

    # compute f(u1, t1)
    problem.rate!(problem.du, c.utmp, problem.t[] + dt)

    # u2 = 3/4 u0 + 1/4 u1 + 1/4 dt f(u1, t1)
    @. c.utmp = (3 * problem.u + c.utmp + dt * problem.du) / 4
    problem.projection!(c.utmp)

    # compute f(u2, t2)
    problem.rate!(problem.du, c.utmp, problem.t[] + dt / 2)

    # u(n+1) = 1/3 u0 + 2/3 u2 + 2/3 dt f(u2, t2)
    @. problem.u = (problem.u + 2 * c.utmp + 2 * dt * problem.du) / 3
    problem.projection!(problem.u)
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

time(prob::TimeIntegrationProblem) = prob.t[]
state(prob::TimeIntegrationProblem) = prob.u

function solve(prob::TimeIntegrationProblem,
               alg::TimeIntegrationAlgorithm;
               checkpoints::Union{Int,Nothing} = nothing)
    cache = init_cache(prob, alg)
    dt = timestep(alg)
    nt_float = (prob.tmax - prob.t[]) / dt
    nt = round(Int, nt_float)
    nt ≈ nt_float || error("Integration time not divisible by (constant) time step.")
    next_checkpoint = checkpoints
    checkpoints == nothing || nt % checkpoints == 0 ||
        error("Time steps not divisible by checkpoint frequency")

    for tstep in 1:nt
        perform_step!(prob, dt, cache)
        prob.t[] += dt
        kwargs = if tstep == next_checkpoint
            next_checkpoint += checkpoints
            (checkpoint = true,)
        else
            ()
        end
        prob.rate!(prob.du, prob.u, prob.t[]; kwargs...)
    end
    prob.u
end
