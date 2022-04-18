export ConstantSource

struct ConstantSource <: ProcessDefinition
    field::Symbol
    strength

    function ConstantSource(field, strength = 1)
        new(field, strength)
    end
end

struct DiscretizedConstantSource <: DiscretizedProcess
    field::Symbol
end

function init_process(src::ConstantSource, domain, grid)
    DiscretizedConstantSource(src.field)
end

state_fields(src::DiscretizedConstantSource) = src.field
