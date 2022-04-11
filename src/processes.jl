module Processes

export init_process, state_fields, islinear, add_rate!

abstract type ProcessDefinition end
abstract type DiscretizedProcess end

using ..Helpers
using ..Grid: NodeSet, nodes
using ..BoundaryConditions: ConstantValue, ConstantGradient,
                            init_bcs, layers, layers_expand_full
using ..Derivatives: second_derivatives
using ..Transform: physical_domain!

function rate!(rate, state, t, processes, transforms, log = nothing)

    # set rate back to zero before adding terms
    reset!(rate)

    # add linear terms in frequency domain
    for process in filter(islinear, processes)
        add_rate!(rate, process, state, t, log)
    end

    # add nonlinear terms in physical domain
    physical_domain!(rate, state, transforms) do rate, state
        for process in filter(p -> !islinear(p), processes)
            # TODO: select correct size for each term
            add_rate!(rate, process, state, t, log)
        end
    end
end

add_rate!(rate, process, state, t) = add_rate!(rate, process, state, t, nothing)

state_fields(processes) = unique(vcat((f isa Tuple ? collect(f) : [f] for f in state_fields.(processes))...))

include("momentum_advection.jl")
include("molecular_diffusion.jl")
include("pressure.jl")
include("sources.jl")

end
