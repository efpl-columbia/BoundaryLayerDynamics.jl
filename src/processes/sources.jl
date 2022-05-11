export ConstantSource, ConstantMean

struct ConstantSource <: ProcessDefinition
    field::Symbol
    strength
    ConstantSource(field, strength = 1) = new(field, strength)
end

struct DiscretizedConstantSource{T} <: DiscretizedProcess
    field::Symbol
    strength::T
end

Base.nameof(::DiscretizedConstantSource) = "Constant Source"

function init_process(src::ConstantSource, domain::Domain{T}, grid) where T
    DiscretizedConstantSource(src.field, convert(T, src.strength))
end

state_fields(src::DiscretizedConstantSource) = src.field

function add_rates!(rate, term::DiscretizedConstantSource, state, t, log)
    rate[term.field][1,1,:] .+= term.strength
    rate
end


struct ConstantMean <: ProcessDefinition
    field::Symbol
    mean_value
    ConstantMean(field, mean_value = 1) = new(field, mean_value)
end

struct DiscretizedConstantMean{T,W,C} <: DiscretizedProcess
    field::Symbol
    mean_value::T
    weight::W
    comm::C
end

Base.nameof(::DiscretizedConstantMean) = "Constant Mean"

function init_process(src::ConstantMean, domain::Domain{T}, grid) where T
    weight = 1 ./ dx3factors(domain, grid, nodes(src.field)) / size(domain, 3)
    DiscretizedConstantMean(src.field, convert(T, src.mean_value), weight, grid.comm)
end

state_fields(src::DiscretizedConstantMean) = src.field
isprojection(press::DiscretizedConstantMean) = true

function apply_projection!(state, term::DiscretizedConstantMean)
    field = state[term.field]
    local_sum = sum(real(field[1,1,i]) * term.weight[i] for i=1:size(field, 3))
    current_mean = MPI.Initialized() ? MPI.Allreduce(local_sum, +, term.comm) : local_sum
    state[term.field][1,1,:] .+= term.mean_value - current_mean
    state
end
