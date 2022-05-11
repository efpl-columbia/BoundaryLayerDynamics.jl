module Processes

export init_process, state_fields, transformed_fields, islinear, compute_rates!, apply_projections!

abstract type ProcessDefinition end
abstract type DiscretizedProcess end

using TimerOutputs: TimerOutputs, @timeit
using ..Helpers
using ..Grids: NodeSet, nodes, neighbors, wavenumbers, vrange
using ..Domains: AbstractDomain as Domain, scalefactor, SmoothWall, RoughWall, FreeSlipBoundary,
                 x1range, x2range, x3range, dx1factors, dx2factors
using ..BoundaryConditions: BoundaryCondition, ConstantValue, ConstantGradient, DynamicValues,
                            init_bcs, internal_bc,
                            layers, layer_below, layer_above, layers_c2i, layers_i2c, layers_expand_full
using ..Derivatives: second_derivatives, dx3factors, dx3_c2i!, dx3_i2c!
using ..PhysicalSpace: physical_domain!, pdsize
using ..Logging: prepare_samples!, log_sample!, log_state!, process_log!

function compute_rates!(rates, state, t, processes, transforms, log = nothing; sample = !isnothing(log))

    timer = isnothing(log) ? TimerOutputs.get_defaulttimer() : log.timer
    log = sample ? log : nothing

    # set rate back to zero before adding terms
    reset!(rates)
    prepare_samples!(log, t)
    log_state!(log, state, t)

    # add linear terms in frequency domain
    @timeit timer "Linear Processes" for process in filter(islinear, processes)
        @timeit timer nameof(process) add_rates!(rates, process, state, t, log)
    end

    # add nonlinear terms in physical domain
    @timeit timer "Nonlinear Processes" physical_domain!(rates, state, transforms) do rates, state
        log_state!(log, state, t)
        for process in filter(p -> !(islinear(p) || isprojection(p)), processes)
            @timeit timer nameof(process) add_rates!(rates, process, state, t, log)
        end
    end

    process_log!(log, rates, t)
end

# allow computing rates without specifying log argument
add_rates!(rate, process, state, t) = add_rates!(rate, process, state, t, nothing)

function apply_projections!(state, processes, log = nothing)

    timer = isnothing(log) ? TimerOutputs.get_defaulttimer() : log.timer

    # TODO: allow logging data from projection step
    @timeit timer "Projections" for process in filter(isprojection, processes)
        @timeit timer nameof(process) apply_projection!(state, process)
    end
    state
end

function state_fields(processes)
    fields = []
    for f in state_fields.(processes)
        f isa Tuple ? push!(fields, f...) : push!(fields, f)
    end

    # underlines are used to specify derivatives elsewhere in the code
    any('_' in string(f) for f in fields) &&
        error("The names of state fields are not allowed to contain underlines.")

    unique!(fields)
end

function transformed_fields(processes)
    terms = []
    for f in physical_domain_terms.(processes)
        f isa Tuple ? push!(terms, f...) : push!(terms, f)
    end
    rates = []
    for f in physical_domain_rates.(processes)
        f isa Tuple ? push!(rates, f...) : push!(rates, f)
    end
    (unique!(terms), unique!(rates))
end

# nothing by default
physical_domain_terms(process) = ()
physical_domain_rates(process) = ()

# linear processes are defined by not having physical domain requirements
islinear(process) = isempty(physical_domain_terms(process)) && !isprojection(process)

# by default, processes are not assumed to be projections
isprojection(process) = false

# derive name from type if not set
Base.nameof(p::DiscretizedProcess) = string(nameof(typeof(p)))

include("processes/momentum_advection.jl")
include("processes/molecular_diffusion.jl")
include("processes/pressure.jl")
include("processes/sources.jl")
include("processes/static_smagorinsky.jl")

end
