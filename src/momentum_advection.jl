export MomentumAdvection

struct MomentumAdvection <: ProcessDefinition
end

struct DiscretizedMomentumAdvection <: DiscretizedProcess
end

function init_process(T, adv::MomentumAdvection, grid, domain)
    DiscretizedMomentumAdvection()
end

islinear(::DiscretizedMomentumAdvection) = false
state_fields(::DiscretizedMomentumAdvection) = (:vel1, :vel2, :vel3)
