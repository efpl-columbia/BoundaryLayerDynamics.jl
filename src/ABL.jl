module ABL
__precompile__(false)

include("util.jl")
include("semiperiodic_domain.jl")
include("distributed_grid.jl")
include("boundary_conditions.jl")
include("horizontal_transform.jl")
include("derivatives.jl")
include("processes.jl")


# detail-oriented interface
export DiscretizedABL, incompressible_flow, initialize!, reset!, coordinates

# domain and boundary conditions
export SemiperiodicDomain, SmoothWall, RoughWall, FreeSlipBoundary, CustomBoundary

# physical processes
export MolecularDiffusion

using .Helpers: Helpers
using .Grid: DistributedGrid, NodeSet, nodes, vrange
using .Transform: init_transforms, get_field, set_field!, default_size
using .Domain: SemiperiodicDomain, SmoothWall, RoughWall, FreeSlipBoundary, CustomBoundary,
               x1range, x2range, x3range
using .Processes

using MPI: Initialized as mpi_initialized, COMM_WORLD as MPI_COMM_WORLD

struct DiscretizedABL{T,P}
    # TODO: decide on name
    # e.g. DiscretizedABL, DiscretizedFlow, DiscretizedFlowSystem, DiscretizedSystem
    state
    grid
    processes::P
    transforms
    domain

    function DiscretizedABL(modes, domain::SemiperiodicDomain{T}, processes;
            comm = mpi_initialized() ? MPI_COMM_WORLD : nothing) where T
        grid = DistributedGrid(modes, comm = comm)
        processes = [init_process(T, p, grid, domain) for p in processes]
        state = NamedTuple((f, zeros(T, grid, nodes(f))) for f in state_fields(processes))
        transforms = init_transforms(T, grid, processes)
        new{T,typeof(processes)}(state, grid, processes, transforms, domain)
    end

end

function Base.show(io::IO, ::MIME"text/plain", abl::DiscretizedABL)
    print(io, "Discretized Atmospheric Boundary Layer:\n")
    print(io, "→ κ₁ ∈ [−$(abl.grid.k1max),$(abl.grid.k1max)]")
    print(io, ", κ₂ ∈ [−$(abl.grid.k2max),$(abl.grid.k2max)]")
    print(io, ", i₃ ∈ [1,$(abl.grid.n3global)]")
end

function initialize!(abl::DiscretizedABL; initial_conditions...)
    for (field, ic) in initial_conditions
        set_field!(ic, abl.state[field], abl.transforms[default_size(abl.grid)],
                   abl.domain, abl.grid, nodes(field))
    end
end

reset!(abl::DiscretizedABL) = (Helpers.reset!(abl.state); abl)

Base.getindex(abl::DiscretizedABL, field::Symbol) =
    get_field(abl.transforms[default_size(abl.grid)], abl.state[field])

coordinates(abl::DiscretizedABL) = coordinates(abl, :vel1)
coordinates(abl::DiscretizedABL, dim::Int) = coordinates(abl, :vel1, Val(dim))
coordinates(abl::DiscretizedABL, field, dim::Int) = coordinates(abl, field, Val(dim))
coordinates(abl::DiscretizedABL, field, ::Val{1}) = x1range(abl.domain, h1range(abl.grid, default_size(abl.grid)))
coordinates(abl::DiscretizedABL, field, ::Val{2}) = x2range(abl.domain, h2range(abl.grid, default_size(abl.grid)))
coordinates(abl::DiscretizedABL, field, ::Val{3}) = x3range(abl.domain, vrange(abl.grid, nodes(field)))
function coordinates(abl::DiscretizedABL, field::Symbol)
    ((x1, x2, x3) for x1=coordinates(abl, field, Val(1)),
                      x2=coordinates(abl, field, Val(2)),
                      x3=coordinates(abl, field, Val(3)))
end


incompressible_flow(Re, constant_flux = false) = [
    MomentumAdvection(),
    MolecularDiffusion(:vel1, 1/Re),
    MolecularDiffusion(:vel2, 1/Re),
    MolecularDiffusion(:vel3, 1/Re),
    Pressure(),
    constant_flux ? ConstantMean(:vel1) : ConstantSource(:vel1),
]

# generate a function that performs the update of the rate
# based on the current state
rate!(abl::DiscretizedABL) = (r, s, t; checkpoint = false) -> begin
    rate!(r, s, t, abl.processes, abl.transforms, abl.log)
    checkpoint && process_logs!(log, t) # perform logging activities
end


end
