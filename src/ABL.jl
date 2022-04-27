module ABL
__precompile__(false)

# detail-oriented interface
export DiscretizedABL, closedchannelflow, openchannelflow, incompressible_flow,
    initialize!, reset!, evolve!, coordinates

# domain and boundary conditions
export Domain, SmoothWall, RoughWall, FreeSlipBoundary, CustomBoundary, SinusoidalMapping

# physical processes
export MolecularDiffusion, MomentumAdvection, Pressure, ConstantSource, ConstantMean,
    StaticSmagorinskyModel

# ODE methods
export Euler, AB2, SSPRK22, SSPRK33

# logging/output
export MeanProfiles, Snapshots

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
using .Grids: StaggeredFourierGrid as Grid, NodeSet # nodes exported for convenience in tessts
using .PhysicalSpace: init_physical_spaces
using .Processes
using .Domains
const Domain = ABLDomain
using .State: State, init_state
using .ODEMethods
using .Logging: Logging, MeanProfiles, Snapshots, Log, process_samples!

using MPI: Initialized as mpi_initialized, COMM_WORLD as MPI_COMM_WORLD
using TimerOutputs: @timeit
using RecursiveArrayTools: ArrayPartition

struct DiscretizedABL{T,P}
    # TODO: decide on name
    # e.g. DiscretizedABL, DiscretizedFlow, DiscretizedFlowSystem, DiscretizedSystem
    domain
    grid
    state
    processes::P
    physical_spaces

    function DiscretizedABL(modes, domain::Domain{T}, processes;
            comm = mpi_initialized() ? MPI_COMM_WORLD : nothing) where T
        grid = Grid(modes, comm = comm)
        processes = [init_process(p, domain, grid) for p in processes]
        state = init_state(T, grid, state_fields(processes))
        physical_spaces = init_physical_spaces(transformed_fields(processes), domain, grid)

        new{T,typeof(processes)}(domain, grid, state, processes, physical_spaces)
    end

end

function Base.show(io::IO, ::MIME"text/plain", abl::DiscretizedABL)
    print(io, "Discretized Atmospheric Boundary Layer:\n")
    print(io, "→ κ₁ ∈ [−$(abl.grid.k1max),$(abl.grid.k1max)]")
    print(io, ", κ₂ ∈ [−$(abl.grid.k2max),$(abl.grid.k2max)]")
    print(io, ", i₃ ∈ [1,$(abl.grid.n3global)]")
end

# convenience functions to interact with state through ABL struct
initialize!(abl::DiscretizedABL; kwargs...) =
    State.initialize!(abl.state, abl.domain, abl.grid, abl.physical_spaces; kwargs...)
initialize!(abl::DiscretizedABL, path; kwargs...) =
    State.initialize!(abl.state, path, abl.domain, abl.grid, abl.physical_spaces)
reset!(abl::DiscretizedABL) = State.reset!(abl.state)
Base.getindex(abl::DiscretizedABL, field::Symbol) =
    State.getterm(abl.state, field, abl.domain, abl.grid, abl.physical_spaces)
coordinates(abl::DiscretizedABL, opts...) = State.coordinates(abl.domain, abl.grid, opts...)

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
    DiscretizedABL(dims, domain, processes)
end

function openchannelflow(Re, dims; kwargs...)
    domain = Domain((4π, 2π, 1), SmoothWall(), FreeSlipBoundary())
    processes = incompressible_flow(1/Re; kwargs...)
    DiscretizedABL(dims, domain, processes)
end

# generate a function that performs the update of the rate
# based on the current state
rate!(abl::DiscretizedABL, log = nothing) = (r, s, t; checkpoint = false) -> begin
    fields = keys(abl.state)
    rates = NamedTuple{fields}(r.x)
    state = NamedTuple{fields}(s.x)
    compute_rates!(rates, state, t, abl.processes, abl.physical_spaces, checkpoint ? log : nothing)
end

# generate a function that performs the projection step of the current state
projection!(abl::DiscretizedABL) = (s) -> begin
    fields = keys(abl.state)
    state = NamedTuple{fields}(s.x)
    apply_projections!(state, abl.processes)
end

function evolve!(abl::DiscretizedABL{T}, tspan;
        dt = nothing, method = SSPRK33(), output = (),
        verbose = true) where T

    # validate/normalize time arguments
    isnothing(dt) && ArgumentError("The keyword argument `dt` is mandatory")
    t1 = length(tspan) == 1 ? zero(first(tspan)) : first(tspan)
    t2 = last(tspan)
    t1, t2, dt = convert.(T, (t1, t2, dt))

    # set up logging
    log = Log(output, abl.domain, abl.grid, t1)

    # initialize integrator and perform one step to compile functions
    @timeit log.timer "Initialization" begin
        u0 = ArrayPartition(values(abl.state)...)
        prob = ODEProblem(rate!(abl, log), projection!(abl),
                                      u0, (t1, t2), checkpoint = true)
    end

    # perform the full integration
    @timeit log.timer "Time integration" begin
        solve!(prob, method, dt, checkpoints=t1+dt:dt:t2)
    end
end


end # module ABL
