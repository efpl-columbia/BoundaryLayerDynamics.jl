abstract type ProcType end
struct SingleProc <: ProcType end
struct MinProc    <: ProcType end
struct InnerProc  <: ProcType end
struct MaxProc    <: ProcType end
LowestProc  = Union{SingleProc, MinProc}
HighestProc = Union{SingleProc, MaxProc}

proc_below() = (MPI.Initialized() ? MPI.Comm_rank(MPI.COMM_WORLD) : 0) - 1
proc_above() = (MPI.Initialized() ? MPI.Comm_rank(MPI.COMM_WORLD) : 0) + 1

proc(i, n) = (i,n) == (1,1) ? SingleProc() : i == 1 ? MinProc() :
             i == n ? MaxProc() : InnerProc()
proc_type() = typeof(proc(proc_info()...))

abstract type BoundaryCondition{P<:ProcType,T} end
abstract type SolidWallBC{P,T} <: BoundaryCondition{P,T} end
struct DirichletBC{P,T} <: SolidWallBC{P,T}
    value::T
    buffer_fd::Array{Complex{T},2}
    buffer_pd::Array{T,2}
    neighbor_below::Int
    neighbor_above::Int
    DirichletBC(value::T, nh_fd, nh_pd) where T =
        new{proc_type(),T}(value, zeros(Complex{T}, nh_fd...),
        zeros(T, nh_pd...), proc_below(), proc_above())
end
DirichletBC(value::T, gd::DistributedGrid) where T =
    DirichletBC(value, (gd.nx_fd, gd.ny_fd), (gd.nx_pd, gd.ny_pd))

struct NeumannBC{P,T} <: BoundaryCondition{P,T}
    gradient::T
    buffer_fd::Array{Complex{T},2}
    buffer_pd::Array{T,2}
    neighbor_below::Int
    neighbor_above::Int
    NeumannBC(value::T, nh_fd, nh_pd) where T =
        new{proc_type(),T}(value, zeros(Complex{T}, nh_fd...),
        zeros(T, nh_pd...), proc_below(), proc_above())
end
NeumannBC(gradient::T, gd::DistributedGrid) where T =
    NeumannBC(gradient, (gd.nx_fd, gd.ny_fd), (gd.nx_pd, gd.ny_pd))

"""
The UnspecifiedBC can be used as boundary condition for terms that do not have
a boundary condition defined, such as pressure and the horizontal derivatives
of u₁ and u₂. These might still require a boundary condition to pass data
between layers, but they do not have a value specified for the actual boundary
and will cause an error if the values is attempted to be accessed.
"""
struct UnspecifiedBC{P,T} <: BoundaryCondition{P,T}
    buffer_fd::Array{Complex{T},2}
    buffer_pd::Array{T,2}
    neighbor_below::Int
    neighbor_above::Int
    UnspecifiedBC(T, gd::DistributedGrid) =
        new{proc_type(),T}(zeros(Complex{T}, gd.nx_fd, gd.ny_fd),
        zeros(T, gd.nx_pd, gd.ny_pd), proc_below(), proc_above())
end

@inline buffer_for_field(bc, ::AbstractArray{T}) where {T<:SupportedReals} = bc.buffer_pd
@inline buffer_for_field(bc, ::AbstractArray{Complex{T}}) where {T<:SupportedReals} = bc.buffer_fd
bc_as_layer(bc::DirichletBC, ::AbstractArray{T}) where {T<:SupportedReals} =
    bc.buffer_pd .= bc.value
bc_as_layer(bc::DirichletBC, ::AbstractArray{Complex{T}}) where {T<:SupportedReals} =
    (b = bc.buffer_fd; b .= 0; b[1,1] = bc.value; b)

const MTAG_UP = 8
const MTAG_DN = 9

# NOTE: the views passed to these helper functions should have a range of indices
# since zero-dimensional subarrays are not considered contiguous in julia 1.0
send_to_proc_above(x) = MPI.Send(x, MPI.Comm_rank(MPI.COMM_WORLD) + 1, MTAG_UP, MPI.COMM_WORLD)
send_to_proc_below(x) = MPI.Send(x, MPI.Comm_rank(MPI.COMM_WORLD) - 1, MTAG_DN, MPI.COMM_WORLD)
get_from_proc_above!(x) = MPI.Recv!(x, MPI.Comm_rank(MPI.COMM_WORLD) + 1, MTAG_DN, MPI.COMM_WORLD)
get_from_proc_below!(x) = MPI.Recv!(x, MPI.Comm_rank(MPI.COMM_WORLD) - 1, MTAG_UP, MPI.COMM_WORLD)

layers(field::Array{T,3}) where T =
        Tuple(view(field, :, :, iz) for iz=1:size(field,3))
first_layer(A) = view(A, :, :, 1)
last_layer(A) = view(A, :, :, size(A, 3))

@inline function get_layer_below(layers::Tuple, lower_bc::BoundaryCondition{SingleProc})
    lower_bc
end
@inline function get_layer_below(layers::Tuple, lower_bc::BoundaryCondition{MinProc})
    MPI.Send(layers[end], lower_bc.neighbor_above, 1, MPI.COMM_WORLD)
    lower_bc
end
@inline function get_layer_below(layers::NTuple{N,AbstractArray{Complex{T}}},
        lower_bc::BoundaryCondition{MaxProc,T}) where {N,T}
    MPI.Recv!(lower_bc.buffer_fd, lower_bc.neighbor_below, 1, MPI.COMM_WORLD)
    lower_bc.buffer_fd
end
@inline function get_layer_below(layers::NTuple{N,AbstractArray{Complex{T}}},
        lower_bc::BoundaryCondition{InnerProc,T}) where {N,T}
    r = MPI.Irecv!(lower_bc.buffer_fd, lower_bc.neighbor_below, 1, MPI.COMM_WORLD)
    MPI.Send(layers[end], lower_bc.neighbor_above, 1, MPI.COMM_WORLD)
    MPI.Wait!(r)
    lower_bc.buffer_fd
end

# TODO: consider using a different way of handling pd & fd
@inline function get_layer_below_pd(layers::Tuple, lower_bc::BoundaryCondition{SingleProc})
    lower_bc
end
@inline function get_layer_below_pd(layers::Tuple, lower_bc::BoundaryCondition{MinProc})
    MPI.Send(layers[end], lower_bc.neighbor_above, 1, MPI.COMM_WORLD)
    lower_bc
end
@inline function get_layer_below_pd(layers::NTuple{N,AbstractArray{T}},
        lower_bc::BoundaryCondition{MaxProc,T}) where {N,T}
    # NOTE: it needs to be explicit that the pd version of this method is wanted,
    # since layers can be empty, in which case the type information is lost
    MPI.Recv!(lower_bc.buffer_pd, lower_bc.neighbor_below, 1, MPI.COMM_WORLD)
    lower_bc.buffer_pd
end
@inline function get_layer_below_pd(layers::NTuple{N,AbstractArray{T}},
        lower_bc::BoundaryCondition{InnerProc,T}) where {N,T}
    r = MPI.Irecv!(lower_bc.buffer_pd, lower_bc.neighbor_below, 1, MPI.COMM_WORLD)
    MPI.Send(layers[end], lower_bc.neighbor_above, 1, MPI.COMM_WORLD)
    MPI.Wait!(r)
    lower_bc.buffer_pd
end

@inline function get_layer_above(layers::Tuple, upper_bc::BoundaryCondition{SingleProc})
    upper_bc
end
@inline function get_layer_above(layers::Tuple, upper_bc::BoundaryCondition{MaxProc})
    MPI.Send(layers[1], upper_bc.neighbor_below, 2, MPI.COMM_WORLD)
    upper_bc
end
@inline function get_layer_above(layers::Tuple{}, upper_bc::DirichletBC{MaxProc,T}) where T
    # this is a special case for when the top process does not have any layers,
    # which is the case if there is only one layer per process. in this case, we
    # fill the BC buffer with the boundary condition and pass that down to the
    # process below
    fill!(upper_bc.buffer_fd, zero(T))
    upper_bc.buffer_fd[1,1] = upper_bc.value
    MPI.Send(upper_bc.buffer_fd, upper_bc.neighbor_below, 2, MPI.COMM_WORLD)
    nothing # prevent the caller from trying to use the return value
end
@inline function get_layer_above(layers::NTuple{N,AbstractArray{Complex{T}}},
        upper_bc::BoundaryCondition{MinProc,T}) where {N,T}
    MPI.Recv!(upper_bc.buffer_fd, upper_bc.neighbor_above, 2, MPI.COMM_WORLD)
    upper_bc.buffer_fd
end
@inline function get_layer_above(layers::NTuple{N,AbstractArray{T}},
        upper_bc::BoundaryCondition{MinProc,T}) where {N,T}
    MPI.Recv!(upper_bc.buffer_pd, upper_bc.neighbor_above, 2, MPI.COMM_WORLD)
    upper_bc.buffer_pd
end
@inline function get_layer_above(layers::NTuple{N,AbstractArray{Complex{T}}},
        upper_bc::BoundaryCondition{InnerProc,T}) where {N,T}
    r = MPI.Irecv!(upper_bc.buffer_fd, upper_bc.neighbor_above, 2, MPI.COMM_WORLD)
    MPI.Send(layers[1], upper_bc.neighbor_below, 2, MPI.COMM_WORLD)
    MPI.Wait!(r)
    upper_bc.buffer_fd
end
@inline function get_layer_above(layers::NTuple{N,AbstractArray{T}},
        upper_bc::BoundaryCondition{InnerProc,T}) where {N,T}
    r = MPI.Irecv!(upper_bc.buffer_pd, upper_bc.neighbor_above, 2, MPI.COMM_WORLD)
    MPI.Send(layers[1], upper_bc.neighbor_below, 2, MPI.COMM_WORLD)
    MPI.Wait!(r)
    upper_bc.buffer_pd
end

"""
This function takes a field defined on C-nodes, converts it into layers, and expands
them through communication such that the list includes the layers just above and
below all I-nodes. This means passing data down throughout the domain.
The boundary condition is only used for passing data, the actual values are unused.
"""
function layers_expand_c_to_i(field::AbstractArray{T}, bc_above::BoundaryCondition{P}) where {T,P}
    #TODO: find a better name for this function

    l = layers(field)

    if P == SingleProc
        l

    elseif P == MinProc
        buffer_above = buffer_for_field(bc_above, field)
        rq = (MPI.Irecv!(buffer_above, bc_above.neighbor_above, MTAG_DN, MPI.COMM_WORLD), )
        map(MPI.Wait!, rq) # not using Waitall! to avoid allocation of vector
        l..., buffer_above

    elseif P == InnerProc
        buffer_above = buffer_for_field(bc_above, field)
        rq = (MPI.Isend(l[1], bc_above.neighbor_below, MTAG_DN, MPI.COMM_WORLD),
              MPI.Irecv!(buffer_above, bc_above.neighbor_above, MTAG_DN, MPI.COMM_WORLD))
        map(MPI.Wait!, rq) # not using Waitall! to avoid allocation of vector
        l..., buffer_above

    # there is a special case for when the top process does not have any V-nodes,
    # which is the case if there is only one layer per process. in this case, we
    # only have to pass data down, but we still return the (single) H-layer for
    # consistency so we always return Nv+1 layers
    elseif P == MaxProc
        rq = (MPI.Isend(l[1], bc_above.neighbor_below, MTAG_DN, MPI.COMM_WORLD), )
        map(MPI.Wait!, rq) # not using Waitall! to avoid allocation of vector
        l

    else
        @error "Invalid process type" P
    end
end

"""
This function takes a field defined on I-nodes, converts it into layers, and expands
them through communication such that the list includes the layers just above and
below all C-nodes. This means passing data up throughout the domain.
"""
function layers_expand_i_to_c(field::AbstractArray{T}, bc_below::BoundaryCondition{P},
                        bc_above::BoundaryCondition{P}) where {T,P}
    #TODO: find a better name for this function

    l = layers(field)
    neighbor_below = equivalently(bc_below.neighbor_below, bc_above.neighbor_below)
    neighbor_above = equivalently(bc_below.neighbor_above, bc_above.neighbor_above)

    if P == SingleProc
        bc_below, l..., bc_above

    elseif P == MinProc
        rq = (MPI.Isend(l[end], neighbor_above, MTAG_UP, MPI.COMM_WORLD), )
        map(MPI.Wait!, rq) # not using Waitall! to avoid allocation of vector
        bc_below, l...

    elseif P == InnerProc
        buffer_below = buffer_for_field(bc_below, field)
        rq = (MPI.Isend(l[end], neighbor_above, MTAG_UP, MPI.COMM_WORLD),
              MPI.Irecv!(buffer_below, neighbor_below, MTAG_UP, MPI.COMM_WORLD))
        map(MPI.Wait!, rq) # not using Waitall! to avoid allocation of vector
        buffer_below, l...

    # there is a special case for when the top process does not have any V-nodes,
    # which is the case if there is only one layer per process. in this case, we
    # only have to pass data down, but we still return the (single) H-layer for
    # consistency so we always return Nv+1 layers
    elseif P == MaxProc
        buffer_below = buffer_for_field(bc_below, field)
        rq = (MPI.Irecv!(buffer_below, neighbor_below, MTAG_UP, MPI.COMM_WORLD), )
        map(MPI.Wait!, rq) # not using Waitall! to avoid allocation of vector
        buffer_below, l..., bc_above

    else
        @error "Invalid process type" P
    end
end

layers_expand_half(field, _, bc_above, ::NodeSet{:H}) = layers_expand_c_to_i(field, bc_above)
layers_expand_half(field, bc_below, bc_above, ::NodeSet{:V}) = layers_expand_i_to_c(field, bc_below, bc_above)

function layers_expand_full(field::AbstractArray{T}, bc_below::BoundaryCondition{P},
                         bc_above::BoundaryCondition{P}) where {T,P}

    l = layers(field)
    neighbor_below = equivalently(bc_below.neighbor_below, bc_above.neighbor_below)
    neighbor_above = equivalently(bc_below.neighbor_above, bc_above.neighbor_above)

    if P == SingleProc
        bc_below, l..., bc_above

    elseif P == MinProc
        buffer_above = buffer_for_field(bc_above, field)
        rq = (MPI.Isend(l[end], neighbor_above, MTAG_UP, MPI.COMM_WORLD),
              MPI.Irecv!(buffer_above, neighbor_above, MTAG_DN, MPI.COMM_WORLD))
        map(MPI.Wait!, rq) # not using Waitall! to avoid allocation of vector
        bc_below, l..., buffer_above

    elseif P == InnerProc
        buffer_below = buffer_for_field(bc_below, field)
        buffer_above = buffer_for_field(bc_above, field)
        rq = (MPI.Isend(l[end], neighbor_above, MTAG_UP, MPI.COMM_WORLD),
              MPI.Isend(l[1], neighbor_below, MTAG_DN, MPI.COMM_WORLD),
              MPI.Irecv!(buffer_below, neighbor_below, MTAG_UP, MPI.COMM_WORLD),
              MPI.Irecv!(buffer_above, neighbor_above, MTAG_DN, MPI.COMM_WORLD))
        map(MPI.Wait!, rq) # not using Waitall! to avoid allocation of vector
        buffer_below, l..., buffer_above

    # there is a special case for when the top process does not have any layers,
    # which is the case if there is only one layer per process. in this case, we
    # fill the BC buffer with the boundary condition and pass that down to the
    # process below
    elseif P == MaxProc
        buffer_below = buffer_for_field(bc_below, field)
        lowest_layer = length(l) == 0 ? bc_as_layer(bc_above, field) : l[1]
        rq = (MPI.Isend(lowest_layer, neighbor_below, MTAG_DN, MPI.COMM_WORLD),
              MPI.Irecv!(buffer_below, neighbor_below, MTAG_UP, MPI.COMM_WORLD))
        map(MPI.Wait!, rq) # not using Waitall! to avoid allocation of vector
        buffer_below, l..., bc_above

    else
        @error "Invalid process type: $(P)"
    end
end

# interpolation v-nodes to h-nodes
add_interpolated_layer!(layer, layer¯, layer⁺) =
    (@. layer += (layer¯ + layer⁺) / 2; layer)
add_interpolated_layer!(layer, lbc::DirichletBC, layer⁺) =
        (@. layer += (lbc.value + layer⁺) / 2; layer)
add_interpolated_layer!(layer, layer¯, ubc::DirichletBC) =
        (@. layer += (layer¯ + ubc.value) / 2; layer)

function add_interpolation!(layers_h::NTuple{NZH}, layers_v::NTuple{NZV}, lower_bc, upper_bc) where {NZH, NZV}
    layer_below = get_layer_below_pd(layers_v, lower_bc)
    for i=1:NZH
        add_interpolated_layer!(layers_h[i], i==1 ? layer_below : layers_v[i-1],
                                i==NZH==NZV+1 ? upper_bc : layers_v[i])
    end
    layers_h
end

function interpolate(field_v, lower_bc::BoundaryCondition{P}, upper_bc::BoundaryCondition{P}) where {P}
    field_h = (P <: HighestProc) ? zeros(eltype(field_v), size(field_v, 1), size(field_v, 2), size(field_v, 3) + 1) : zero(field_v)
    add_interpolation!(layers(field_h), layers(field_v), lower_bc, upper_bc)
    field_h
end

global_minimum(val::T) where {T<:Real} =
        MPI.Initialized() ? MPI.Allreduce(val, MPI.MIN, MPI.COMM_WORLD) : val

function global_minimum(field::Array{T}) where {T<:SupportedReals}
    # specifying T avoids accidentially taking the maximum in Fourier space
    global_minimum(mapreduce(abs, min, field))
end

global_maximum(val::T) where {T<:Real} =
        MPI.Initialized() ? MPI.Allreduce(val, MPI.MAX, MPI.COMM_WORLD) : val

function global_maximum(field::Array{T}) where {T<:SupportedReals}
    # specifying T avoids accidentially taking the maximum in Fourier space
    global_maximum(mapreduce(abs, max, field))
end

global_sum(Ns) = MPI.Initialized() ? MPI.Allreduce(sum(Ns), MPI.SUM, MPI.COMM_WORLD) : sum(Ns)
