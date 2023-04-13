module Grids

export StaggeredFourierGrid

using MPI: MPI

abstract type AbstractGrid end
abstract type DistributedGrid{C} <: AbstractGrid end

struct StaggeredFourierGrid{C} <: DistributedGrid{C}
    k1max::Int
    k2max::Int
    n3c::Int
    n3i::Int
    n3global::Int
    i3min::Int
    i3max::Int
    comm::C

    function StaggeredFourierGrid((n1, n2, n3)::Tuple{Int,Int,Int};
            comm = MPI.Initialized() ? MPI.COMM_WORLD : nothing)

        # determine largest wavenumber for horizontal directions
        k1max, k2max = div.((n1, n2) .- 1, 2)

        # determine local range of vertical indices
        comm, proc_id, proc_count = init_processes(comm)
        n3 >= proc_count || error("There should not be more processes than vertical layers")
        i3min, i3max = extrema(i3range(proc_id, proc_count, n3))
        n3c = i3max - i3min + 1
        n3i = (proc_id == proc_count ? n3c - 1 : n3c)

        new{typeof(comm)}(k1max, k2max, n3c, n3i, n3, i3min, i3max, comm)
    end
end

StaggeredFourierGrid(n::Int; kwargs...) = StaggeredFourierGrid((n, n, n); kwargs...)
StaggeredFourierGrid((nh, nv)::Tuple{Int,Int}; kwargs...) = StaggeredFourierGrid((nh, nh, nv); kwargs...)

# ground truth for how layers are distributed amongst processes
function i3range(proc_id, proc_count, n3)
    n3_per_proc, n3_rem = divrem(n3, proc_count)
    i3min = 1 + n3_per_proc * (proc_id - 1) + min(n3_rem, proc_id - 1)
    i3max = min(n3_per_proc * proc_id + min(n3_rem, proc_id), n3)
    i3min:i3max
end

function proc_for_layer(grid::StaggeredFourierGrid, ind)
    isnothing(grid.comm) && return 0
    i3 = ind > 0 ? ind : grid.n3global - abs(ind) + 1
    proc_count = MPI.Comm_size(grid.comm)
    for proc_id in 1:proc_count
        i3 in i3range(proc_id, proc_count, grid.n3global) &&
            return MPI.Cart_rank(grid.comm, proc_id-1)
    end
    error("Layer $i3 does not belong to any process")
end

init_processes(comm::Nothing) = (comm, 1, 1)
function init_processes(comm)
    count = MPI.Comm_size(comm)
    # TODO: change to new interface when updating MPI.jl
    comm = MPI.Cart_create(comm, [count], [0], true)
    coord = MPI.Cart_coords(comm)[] + 1 # convert to one-based coordinate
    comm, coord, count
end

neighbors(grid::StaggeredFourierGrid, displacement = 1) =
    neighbors(grid.comm, displacement)
function neighbors(comm, displacement = 1)
    isnothing(comm) && return (nothing, nothing)
    neighbors = MPI.Cart_shift(comm, 0, displacement)
    Tuple(n == MPI.PROC_NULL ? nothing : n for n in neighbors)
end

wavenumbers(gd) = wavenumbers.((gd,), (1,2))
wavenumbers(gd, dim::Int) = begin
    if dim == 1
        [0:gd.k1max; ]
    elseif dim == 2
        [0:gd.k2max; -gd.k2max:-1]
    else
        error("Invalid dimension `$dim` for wavenumbers")
    end
end

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
nodes(::Val{:vort1}) = NodeSet(:I)
nodes(::Val{:vort2}) = NodeSet(:I)
nodes(::Val{:vort3}) = NodeSet(:C)
nodes(::Val{:vel1_1}) = NodeSet(:C)
nodes(::Val{:vel1_2}) = NodeSet(:C)
nodes(::Val{:vel1_3}) = NodeSet(:I)
nodes(::Val{:vel2_1}) = NodeSet(:C)
nodes(::Val{:vel2_2}) = NodeSet(:C)
nodes(::Val{:vel2_3}) = NodeSet(:I)
nodes(::Val{:vel3_1}) = NodeSet(:I)
nodes(::Val{:vel3_2}) = NodeSet(:I)
nodes(::Val{:vel3_3}) = NodeSet(:C)
nodes(::Val{:strain12}) = NodeSet(:C)
nodes(::Val{:strain13}) = NodeSet(:I)
nodes(::Val{:strain23}) = NodeSet(:I)
nodes(::Val{:adv1}) = NodeSet(:C)
nodes(::Val{:adv2}) = NodeSet(:C)
nodes(::Val{:adv3}) = NodeSet(:I)
nodes(::Val{:sgs11}) = NodeSet(:C)
nodes(::Val{:sgs12}) = NodeSet(:C)
nodes(::Val{:sgs13}) = NodeSet(:I)
nodes(::Val{:sgs22}) = NodeSet(:C)
nodes(::Val{:sgs23}) = NodeSet(:I)
nodes(::Val{:sgs33}) = NodeSet(:C)
nodes(::Val{F}) where F = error("Nodes of field `$F` are unknown. Define `nodes(::Val{:$F})` to resolve this error.")

Base.zeros(T, grid, ::NodeSet{:C}) = zeros(Complex{T}, fdsize(grid)..., grid.n3c)
Base.zeros(T, grid, ::NodeSet{:I}) = zeros(Complex{T}, fdsize(grid)..., grid.n3i)

# returns a range of rational ζ-values between 0 and 1
function vrange(gd, ::NodeSet{:C}; neighbors=false)
    ζ = LinRange(0//1, 1//1, 2*gd.n3global+1) # all ζ-values
    imin = 2*gd.i3min - (neighbors ? 1 : 0)
    imax = 2*gd.i3max + (neighbors ? 1 : 0)
    ζ[imin:2:imax]
end
function vrange(gd, ::NodeSet{:I}; neighbors=false)
    ζ = LinRange(0//1, 1//1, 2*gd.n3global+1) # all ζ-values
    imin = 2*gd.i3min+1 - (neighbors ? 1 : 0)
    imax = 2*(gd.i3min+gd.n3i-1)+1 + (neighbors ? 1 : 0)
    ζ[imin:2:imax]
end


end
