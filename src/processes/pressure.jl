export Pressure

struct Pressure <: ProcessDefinition
end

struct DiscretizedPressure <: DiscretizedProcess
end

function init_process(T, press::Pressure, grid, domain)
    DiscretizedPressure()
end

state_fields(::DiscretizedPressure) = (:vel1, :vel2, :vel3)
