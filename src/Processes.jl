module Processes

export init_process, state_fields, transformed_fields, islinear, compute_rates!, apply_projections!

abstract type ProcessDefinition end
abstract type DiscretizedProcess end

using ..Helpers
using ..Grids: NodeSet, nodes, neighbors, wavenumbers, vrange
using ..Domains: AbstractDomain as Domain, scalefactor, SmoothWall, RoughWall, FreeSlipBoundary,
                 x1range, x2range, x3range, dx1factors, dx2factors
using ..BoundaryConditions: BoundaryCondition, ConstantValue, ConstantGradient, DynamicValues,
                            init_bcs, internal_bc,
                            layers, layer_below, layer_above, layers_c2i, layers_i2c, layers_expand_full
using ..Derivatives: second_derivatives, dx3factors, dx3_c2i!, dx3_i2c!
using ..PhysicalSpace: physical_domain!, pdsize
using ..Logging: prepare_samples!, log_sample!, process_samples!

function compute_rates!(rates, state, t, processes, transforms, log = nothing)

    # set rate back to zero before adding terms
    reset!(rates)
    prepare_samples!(log, t)

    # add linear terms in frequency domain
    for process in filter(islinear, processes)
        add_rates!(rates, process, state, t, log)
    end

    # add nonlinear terms in physical domain
    physical_domain!(rates, state, transforms) do prates, pstate
        for process in filter(p -> !(islinear(p) || isprojection(p)), processes)
            add_rates!(prates, process, pstate, t, log)
        end

        process_samples!(log, t, state, pstate)
    end
end

# allow computing rates without specifying log argument
add_rates!(rate, process, state, t) = add_rates!(rate, process, state, t, nothing)

function apply_projections!(state, processes)
    # TODO: allow logging data from projection step
    for process in filter(isprojection, processes)
        apply_projection!(state, process)
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

include("processes/momentum_advection.jl")
include("processes/molecular_diffusion.jl")
include("processes/pressure.jl")
include("processes/sources.jl")
include("processes/static_smagorinsky.jl")

end
