module Grid

using MPI: MPI

export DistributedGrid

struct DistributedGrid{C}
    k1max::Int
    k2max::Int
    n3c::Int
    n3i::Int
    n3global::Int
    i3min::Int
    i3max::Int
    comm::C

    function DistributedGrid((n1, n2, n3)::Tuple{Int,Int,Int};
            comm = MPI.Initialized() ? MPI.COMM_WORLD : nothing)

        # determine largest wavenumber for horizontal directions
        k1max, k2max = div.((n1, n2) .- 1, 2)

        # determine local range of vertical indices
        comm, proc_id, proc_count = init_processes(comm)
        n3 >= proc_count || error("There should not be more processes than vertical layers")
        n3_per_proc, n3_rem = divrem(n3, proc_count)
        i3min = 1 + n3_per_proc * (proc_id - 1) + min(n3_rem, proc_id - 1)
        i3max = min(n3_per_proc * proc_id + min(n3_rem, proc_id), n3)
        n3c = i3max - i3min + 1
        n3i = (proc_id == proc_count ? n3c - 1 : n3c)

        new{typeof(comm)}(k1max, k2max, n3c, n3i, n3, i3min, i3max, comm)
    end
end

DistributedGrid(n::Int; kwargs...) = DistributedGrid((n, n, n); kwargs...)
DistributedGrid((nh, nv)::Tuple{Int,Int}; kwargs...) = DistributedGrid((nh, nh, nv); kwargs...)

init_processes(comm::Nothing) = (comm, 1, 1)
function init_processes(comm)
    count = MPI.Comm_size(comm)
    comm = MPI.Cart_create(comm, [count], [false], true)
    rank = MPI.Comm_rank(comm)
    comm, 1 + rank, np
end

function neighbors(grid, displacement = 1)
    isnothing(grid.comm) && return (nothing, nothing)
    neighbors = MPI.Cart_shift(grid.comm, 0, displacement)
    Tuple(n == MPI.Consts.MPI_PROC_NULL[] ? nothing : n for n in neighbors)
end

wavenumbers(gd, dim::Int) = wavenumbers(gd, Val(dim))
wavenumbers(gd, ::Val{1}) = [0:gd.k1max; ]
wavenumbers(gd, ::Val{2}) = [0:gd.k2max; -gd.k2max:-1]
wavenumbers(gd) = wavenumbers.((gd,), (1,2))

struct NodeSet{NS}
    NodeSet(ns::Symbol) = ns in (:C, :I, :Iext) ? new{ns}() :
        error("Invalid NodeSet: $(ns) (only :C, :I, and :Iext are allowed)")
end

# convenience functions to get array sizes
fdsize(grid) = (1 + grid.k1max, 1 + 2 * grid.k2max)
fdsize(grid, ::NodeSet{:C}) = (fdsize(grid)..., grid.n3c)
fdsize(grid, ::NodeSet{:I}) = (fdsize(grid)..., grid.n3i)

# convenience function to get node set
nodes(field::Symbol) = nodes(Val(field))
nodes(::Val{:vel1}) = NodeSet(:C)
nodes(::Val{:vel2}) = NodeSet(:C)
nodes(::Val{:vel3}) = NodeSet(:I)
nodes(::Val{F}) where F = error("Nodes of field `$F` are unknown. Define `nodes(::Val{:$F})` to resolve this error.")

Base.zeros(T, grid, ::NodeSet{:C}) = zeros(Complex{T}, fdsize(grid)..., grid.n3c)
Base.zeros(T, grid, ::NodeSet{:I}) = zeros(Complex{T}, fdsize(grid)..., grid.n3i)

# returns a range of rational ζ-values between 0 and 1
function vrange(gd, ::NodeSet{:C}; neighbors=false) where T
    ζ = LinRange(0//1, 1//1, 2*gd.n3global+1) # all ζ-values
    imin = 2*gd.i3min - (neighbors ? 1 : 0)
    imax = 2*gd.i3max + (neighbors ? 1 : 0)
    ζ[imin:2:imax]
end
function vrange(gd, ::NodeSet{:I}; neighbors=false) where T
    ζ = LinRange(0//1, 1//1, 2*gd.n3global+1) # all ζ-values
    imin = 2*gd.i3min+1 - (neighbors ? 1 : 0)
    imax = 2*(gd.i3min+gd.n3i-1)+1 + (neighbors ? 1 : 0)
    ζ[imin:2:imax]
end


end
