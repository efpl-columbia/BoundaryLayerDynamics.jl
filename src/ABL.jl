module ABL
__precompile__(false)

# detail-oriented interface
export DiscretizedABL, incompressible_flow, initialize!, reset!, coordinates

# domain and boundary conditions
export Domain, SmoothWall, RoughWall, FreeSlipBoundary, CustomBoundary

# physical processes
export MolecularDiffusion, MomentumAdvection, Pressure

# ODE methods
export Euler, AB2, SSPRK22, SSPRK33

include("util.jl")
include("Domains.jl")
include("Grids.jl")
include("physical_space.jl")
include("boundary_conditions.jl")
include("derivatives.jl")
include("Processes.jl")
include("ODEMethods.jl")

using .Helpers: Helpers
using .Grids: StaggeredFourierGrid as Grid, NodeSet, nodes, vrange
using .PhysicalSpace: init_physical_spaces, get_field, set_field!, default_size
using .Processes
using .Domains
const Domain = ABLDomain
using .ODEMethods


using MPI: Initialized as mpi_initialized, COMM_WORLD as MPI_COMM_WORLD

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
        state = NamedTuple(f => zeros(T, grid, nodes(f)) for f in state_fields(processes))
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

function initialize!(abl::DiscretizedABL; initial_conditions...)
    # TODO: consider setting all other fields to zero
    for (field, ic) in initial_conditions
        set_field!(ic, abl.state[field], abl.physical_spaces[default_size(abl.grid)].transform,
                   abl.domain, abl.grid, nodes(field))
    end
end

reset!(abl::DiscretizedABL) = (Helpers.reset!(abl.state); abl)

Base.getindex(abl::DiscretizedABL, field::Symbol) =
    get_field(abl.physical_spaces[default_size(abl.grid)].transform, abl.state[field])

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
    rate!(r, s, t, abl.processes, abl.physical_spaces, abl.log)
    checkpoint && process_logs!(log, t) # perform logging activities
end


end
