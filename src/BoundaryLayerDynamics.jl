module BoundaryLayerDynamics
__precompile__(false)

# detail-oriented interface
export Model, closedchannelflow, openchannelflow, incompressible_flow,
    initialize!, reset!, evolve!, coordinates

# domain and boundary conditions
export Domain, SmoothWall, RoughWall, FreeSlipBoundary, CustomBoundary, SinusoidalMapping

# physical processes
export MolecularDiffusion, MomentumAdvection, Pressure, ConstantSource, ConstantMean,
    StaticSmagorinskyModel

# ODE methods
export Euler, AB2, SSPRK22, SSPRK33

# logging/output
export MeanProfiles, ProgressMonitor, Snapshots, StepTimer

include("util.jl")
include("fileio.jl")
include("Domains.jl")
include("Grids.jl")
include("physical_space.jl")
include("boundary_conditions.jl")
include("logging.jl")
include("State.jl")
include("derivatives.jl")
include("Processes.jl")
include("ODEMethods.jl")

using .Helpers: Helpers
using .Grids: StaggeredFourierGrid as Grid, NodeSet # nodes exported for convenience in tests
using .PhysicalSpace: init_physical_spaces
using .Processes
using .Domains
using .State: State, init_state
using .ODEMethods
using .Logging: Logging, MeanProfiles, ProgressMonitor, Snapshots, StepTimer,
    Log, flush!

using MPI: Initialized as mpi_initialized, COMM_WORLD as MPI_COMM_WORLD
using TimerOutputs: @timeit
using RecursiveArrayTools: ArrayPartition

"""
    Model(resolution, domain, processes)

A `Model` provides a discretized representation of the dynamics of the
specified `processes` within the specified `domain`.

It contains discretized state variables such as the velocity field that are
initially set to zero, as well as all the information needed to efficiently
compute the rate of change of these state variables. The required state
variables are automatically determined from the specified processes.

# Arguments
- `resolution::NTuple{3, Integer}`: The number of modes along the ``x_1`` and
  ``x_2`` direction, as well as the number of grid points along the ``x_3``
  direction. The first two values should be odd and will be reduced by one if
  they are even to ensure that the resolved wavenumbers are symmetric around
  zero.
- `domain::Domain`: The simulated universum, represented with the
  [`Domain`](@ref) type.
- `processes`: A list of the physical processes that govern the dynamical
  behavior of the state variables. For convenience, the list can be any
  iterable collection and can also be nested.

# Keywords
- `comm::MPI.Comm`: The MPI communicator that the model will make use of. If
  not specified, `MPI.COMM_WORLD` is used. Note that specifying a different
  communicator is poorly tested currently and may lead to unexpected behavior.
"""
struct Model{T,P}
    domain
    grid
    state
    processes::P
    physical_spaces

    function Model(resolution::NTuple{3,Integer}, domain::Domain{T}, processes;
            comm = mpi_initialized() ? MPI_COMM_WORLD : nothing) where T
        grid = Grid(resolution, comm = comm)
        processes = foldl(processes; init = []) do acc, p
            # the list of processes is allowed to contain lists/tuples/generators
            # so we wrap “naked” processes so they are also in the form of an iterable
            p = p isa Union{Tuple,Array,Base.Generator} ? p : (p, )
            append!(acc, init_process(p, domain, grid) for p in p)
        end
        state = init_state(T, grid, state_fields(processes))
        physical_spaces = init_physical_spaces(transformed_fields(processes), domain, grid)

        new{T,typeof(processes)}(domain, grid, state, processes, physical_spaces)
    end

end

function Base.show(io::IO, ::MIME"text/plain", model::Model)
    print(io, "BoundaryLayerDynamics.Model:\n")
    print(io, "→ κ₁ ∈ [−$(model.grid.k1max),$(model.grid.k1max)]")
    print(io, ", κ₂ ∈ [−$(model.grid.k2max),$(model.grid.k2max)]")
    print(io, ", i₃ ∈ [1,$(model.grid.n3global)]")
end

# convenience functions to interact with state through Model struct
initialize!(model::Model; kwargs...) =
    State.initialize!(model.state, model.domain, model.grid, model.physical_spaces; kwargs...)
initialize!(model::Model, path; kwargs...) =
    State.initialize!(model.state, path, model.domain, model.grid, model.physical_spaces)
reset!(model::Model) = State.reset!(model.state)
Base.getindex(model::Model, field::Symbol) =
    State.getterm(model.state, field, model.domain, model.grid, model.physical_spaces)
coordinates(model::Model, opts...) = State.coordinates(model.domain, model.grid, opts...)

function momentum_source(; constant_flux = nothing,
        constant_forcing = isnothing(constant_flux) ? 1 : nothing)
    isnothing(constant_flux) || isnothing(constant_forcing) ||
        error("Momentum source set up for both constant forcing and constant flux")
    vel = (:vel1, :vel2)
    isnothing(constant_forcing) ||
        return (ConstantSource(vel[i], f) for (i, f) in enumerate(constant_forcing))
    isnothing(constant_flux) ||
        return (ConstantMean(vel[i], mean) for (i, mean) in enumerate(constant_flux))
    error("No momentum source defined")
end

incompressible_flow(viscosity; sgs_model = nothing, kwargs...) = [
    MomentumAdvection(),
    (MolecularDiffusion(vel, viscosity) for vel = (:vel1, :vel2, :vel3))...,
    Pressure(),
    momentum_source(; kwargs...)...,
    (isnothing(sgs_model) ? () : (sgs_model,))...,
]

function closedchannelflow(Re, dims; kwargs...)
    domain = Domain((4π, 2π, 2), SmoothWall(), SmoothWall())
    processes = incompressible_flow(1/Re; kwargs...)
    Model(dims, domain, processes)
end

function openchannelflow(Re, dims; roughness_length = nothing, kwargs...)
    wall = isnothing(roughness_length) ? SmoothWall() : RoughWall(roughness_length)
    domain = Domain((4π, 2π, 1), wall, FreeSlipBoundary())
    processes = incompressible_flow(1/Re; kwargs...)
    Model(dims, domain, processes)
end

# generate a function that performs the update of the rate
# based on the current state
rate!(model::Model, log = nothing) = (r, s, t; checkpoint = false) -> begin
    fields = keys(model.state)
    rates = NamedTuple{fields}(r.x)
    state = NamedTuple{fields}(s.x)
    compute_rates!(rates, state, t, model.processes, model.physical_spaces, log, sample = checkpoint)
    for (k, rate) in pairs(rates)
        all(isfinite(val) for val in rate) || error("The simulation has diverged.")
    end
end

# generate a function that performs the projection step of the current state
projection!(model::Model, log = nothing) = (s) -> begin
    fields = keys(model.state)
    state = NamedTuple{fields}(s.x)
    apply_projections!(state, model.processes, log)
end

"""
    evolve!(model, tspan; dt, method, output)

Simulate a `model`’s evolution over time, optionally collecting data along
the way.

# Arguments
- `model::Model`: The [`Model`](@ref) containing the current state as well
  as the discretized processes that describe the rate of change of the
  state.
- `tspan`: The time span over which the evolution of the model should be
  simulated. Can be a single `Number` with the total duration or a `Tuple`
  of `Number`s with the start and end times.

# Keywords
- `dt`: The (constant) size of the time step (required). Note that `tspan`
  has to be divisible by `dt`.
- `method = SSPRK33()`: The [time-integration method](@ref
  Time-Integration-Methods) used to solve the semi-discretized
  initial-value problem as an ordinary differential equation.
- `output = []`: A list of [output modules](@ref Output-Modules) that
  collect data during the simulation.
"""
function evolve!(model::Model{T}, tspan;
        dt = nothing, method = SSPRK33(), output = (),
        verbose = false) where T

    # validate/normalize time arguments
    isnothing(dt) && ArgumentError("The keyword argument `dt` is mandatory")
    t1 = length(tspan) == 1 ? zero(first(tspan)) : first(tspan)
    t2 = last(tspan)
    t1, t2, dt = convert.(T, (t1, t2, dt))

    # set up logging
    log = Log(output, model.domain, model.grid, (t1, t2), dt = dt, verbose = verbose)

    # initialize integrator and perform one step to compile functions
    @timeit log.timer "Initialization" begin
        u0 = ArrayPartition(values(model.state)...)
        prob = ODEProblem(rate!(model, log), projection!(model, log),
                                      u0, (t1, t2), checkpoint = true)
    end

    # perform the full integration
    @timeit log.timer "Time integration" begin
        # allow skipping integration by setting tspan = 0,
        # e.g. to compute RHS terms only
        if t2 > t1
            nt = Helpers.approxdiv(t2-t1, dt)
            solve!(prob, method, dt, checkpoints=range(t1+dt, t2, nt))
        end
    end

    # write remaining output data
    flush!(log)
end

end # module BoundaryLayerDynamics
