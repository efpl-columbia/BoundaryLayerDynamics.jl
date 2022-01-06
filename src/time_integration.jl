"""
The `TimeIntegrationAlgorithm` structs are used as arguments to the time
integration methods to specify the algorithm used for integration. They are
only used to select the correct methods in multiple-dispatch functions.
"""
abstract type TimeIntegrationAlgorithm end

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
struct SSPRK22 <: TimeIntegrationAlgorithm end

"""
    SSPRK33()

Three-stage third-order strong-stability-preserving Runge-Kutta time
integration.
"""
struct SSPRK33 <: TimeIntegrationAlgorithm end

"""
The `TimeIntegrationBuffer` structs hold the variables each method requires to
compute the next time step without allocating new memory.

Each algorithm should implement a method `init_buffer(prob, alg)` that takes
the `TimeIntegrationAlgorithm` as an argument and creates the corresponding
buffer, which will then be passed every time [`perform_step!`](@ref) is called.
"""
abstract type TimeIntegrationBuffer{T} end

struct EulerBuffer{T} <: TimeIntegrationBuffer{T}
    EulerBuffer(prob) = new{typeof(prob.u)}()
end

struct AB2Buffer{T} <: TimeIntegrationBuffer{T}
    duprev::T
    last_dt::Ref{Float64}
    AB2Buffer(prob) = new{typeof(prob.u)}(zero(prob.du) * NaN, Ref(NaN))
end

struct SSPRK22Buffer{T} <: TimeIntegrationBuffer{T}
    utmp::T
    SSPRK22Buffer(prob) = new{typeof(prob.u)}(zero(prob.u) * NaN)
end

struct SSPRK33Buffer{T} <: TimeIntegrationBuffer{T}
    utmp::T
    SSPRK33Buffer(prob) = new{typeof(prob.u)}(zero(prob.u) * NaN)
end

init_buffer(prob, ::Euler) = EulerBuffer(prob)
init_buffer(prob, ::AB2) = AB2Buffer(prob)
init_buffer(prob, ::SSPRK22) = SSPRK22Buffer(prob)
init_buffer(prob, ::SSPRK33) = SSPRK33Buffer(prob)

"""
    perform_step!(problem, dt, buffer)

Advance the solution `problem.u` by a time step of `dt`. The type of `buffer`
determines which integration scheme is used and stores intermediate values
required by the scheme.

The method should assume that `problem.dt` is up-to-date and is allowed to
overwrite it. It is not necessary to update `problem.dt` at the end of the
step, as this is done by the caller.
"""
function perform_step! end

function perform_step!(problem, dt, ::EulerBuffer)
    problem.u .+= dt * problem.du
    problem.projection!(problem.u)
end

function perform_step!(problem, dt, c::AB2Buffer)
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

function perform_step!(problem, dt, c::SSPRK22Buffer)

    # u1 = u0 + dt f(u0, t0)
    @. c.utmp = problem.u + dt * problem.du
    problem.projection!(c.utmp)

    # compute f(u1, t1)
    problem.rate!(problem.du, c.utmp, problem.t[] + dt)

    # u2 = ½ u0 + ½ u1 + ½ dt f(u1, t1)
    @. problem.u = (problem.u + c.utmp + dt * problem.du) / 2
    problem.projection!(problem.u)
end

function perform_step!(problem, dt, c::SSPRK33Buffer)

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

struct TimeIntegrationProblem{Tt,Tu,R,P}
    rate!::R
    projection!::P
    u::Tu
    du::Tu
    t::Ref{Tt}
    tmax::Tt
end

TimeIntegrationProblem(rate!, u0, tspan; kwargs...) =
    TimeIntegrationProblem(rate!, u -> u, u0, tspan; kwargs...)

function TimeIntegrationProblem(rate!, projection!, u0, (t1, t2); checkpoint = false)
    t1 <= t2 || error("Integration start time is larger than end time")
    du = zero(u0)
    projection!(u0)
    rate!(du, u0, t1; (checkpoint ? (checkpoint = true,) : ())...)
    TimeIntegrationProblem(rate!, projection!, u0, du, Ref(t1), t2)
end

time(prob::TimeIntegrationProblem) = prob.t[]
state(prob::TimeIntegrationProblem) = prob.u

"""
    solve!(problem, algorithm, dt, checkpoints = nothing)

Step the time integration problem `prob` forward in time with the specified
algorithm and time step, hitting all the checkpoints exactly during the
integration and signaling them to the `rate!` function of the problem.
"""
function solve!(prob::TimeIntegrationProblem{Tt},
        alg::TimeIntegrationAlgorithm,
        dt::Tt;
        checkpoints = nothing) where {Tt}

    buffer = init_buffer(prob, alg)

    # make sure that checkpoints are valid and in the right format
    checkpoints = if isnothing(checkpoints)
        (Inf, )
    elseif minimum(checkpoints) <= prob.t[] || maximum(checkpoints) > prob.tmax
        error("Checkpoints outside of time range")
    else
        sort(checkpoints)
    end

    for t in checkpoints
        # compute number of steps to next checkpoint or end of simulation
        nt = approxdiv(min(prob.tmax, t) - prob.t[], dt)

        # perform steps (the order of these commands is important for correctness!)
        for it in 1:nt
            perform_step!(prob, dt, buffer)
            # update time: make sure time matches exactly at checkpoints
            prob.t[] = (prob.t[] + dt ≈ t) ? t : prob.t[] + dt
            # notify `prob.rate!` if the last evaluation is at a checkpoint
            kwargs = (it == nt && prob.t[] == t) ? (checkpoint = true,) : ()
            prob.rate!(prob.du, prob.u, prob.t[]; kwargs...)
        end
    end
    prob.u
end
