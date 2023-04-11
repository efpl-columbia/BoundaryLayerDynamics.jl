module BoundaryConditions

using MPI: MPI
using ..Domains: Domain, SmoothWall, RoughWall, CustomBoundary, FreeSlipBoundary
using ..Grids: AbstractGrid as Grid, fdsize, neighbors
using ..PhysicalSpace: pdsize

struct BoundaryCondition{BC,Nb,Na,C,A}
    type::BC
    buffer::A
    comm::C
    BoundaryCondition(type::BC, buffer::A, comm::C, (Nb, Na)) where {BC,A,C} =
        new{BC,Nb,Na,C,A}(type, buffer, comm)
end

# set up frequency domain boundary condition
function BoundaryCondition(::Type{T}, type, grid::Grid) where T
    type = init_bctype(T, type)
    buffer = zeros(Complex{T}, fdsize(grid))
    BoundaryCondition(type, buffer, grid.comm, neighbors(grid))
end

# set up physical domain boundary condition
function BoundaryCondition(::Type{T}, type, grid::Grid, dealiasing) where T
    type = init_bctype(T, type)
    buffer = zeros(T, pdsize(grid, dealiasing))
    BoundaryCondition(type, buffer, grid.comm, neighbors(grid))
end

struct ConstantValue{T}
    value::T
end

struct ConstantGradient{T}
    gradient::T
end

const LowerBoundary{BC} = BoundaryCondition{BC,nothing,Na} where {Na}
const UpperBoundary{BC} = BoundaryCondition{BC,Nb,nothing} where {Nb}

struct DynamicValues{T}
    values::Array{T,2}
end

# convert boundary conditions into concrete types
init_bctype(::Type{T}, type::Symbol) where T = init_bctype(T, Val(type), zero(T))
init_bctype(::Type{T}, type::Pair) where T = init_bctype(T, Val(first(type)), last(type))
init_bctype(::Type{T}, ::Val{:dirichlet}, value) where T = ConstantValue(convert(T, value))
init_bctype(::Type{T}, ::Val{:neumann}, gradient) where T = ConstantGradient(convert(T, gradient))
init_bctype(::Type{T}, ::Nothing) where T = nothing
init_bctype(::Type{T}, ::Val{:dynamic}, dims::Tuple{Int,Int}) where T = DynamicValues(zeros(T, dims))


function init_bcs(field, domain::Domain{T}, grid::Grid, opts...) where T
    # options passed on to boundary condition allow for initializing both
    # frequency-space and physical-space boundary conditions
    lbc = BoundaryCondition(T, bctype(domain.lower_boundary, field), grid, opts...)
    ubc = BoundaryCondition(T, bctype(domain.upper_boundary, field), grid, opts...)
    (lbc, ubc)
end

internal_bc(domain::Domain{T}, grid) where T =
    BoundaryCondition(T, nothing, grid)
internal_bc(domain::Domain{T}, grid, dims) where T =
    BoundaryCondition(T, nothing, grid, dims)

# create boundary conditions from boundary definitions
bctype(boundary, field::Symbol) = bctype(boundary, Val(field))
bctype(boundary::CustomBoundary, ::Val{F}) where F = boundary.behaviors[F]
bctype(::SmoothWall, ::Val{:vel1}) = :dirichlet
bctype(::SmoothWall, ::Val{:vel2}) = :dirichlet
bctype(::SmoothWall, ::Val{:vel3}) = :dirichlet
bctype(::RoughWall, ::Val{:vel1}) = :dirichlet # TODO: free slip boundary for resolved velocity
bctype(::RoughWall, ::Val{:vel2}) = :dirichlet
bctype(::RoughWall, ::Val{:vel3}) = :dirichlet
bctype(::FreeSlipBoundary, ::Val{:vel1}) = :neumann
bctype(::FreeSlipBoundary, ::Val{:vel2}) = :neumann
bctype(::FreeSlipBoundary, ::Val{:vel3}) = :dirichlet

const MTAG_UP = 8
const MTAG_DN = 9

layers(field::AbstractArray{T,3}) where T =
        Tuple(view(field, :, :, i3) for i3=1:size(field,3))

# single process
function layer_below(layers, lower_bc::BoundaryCondition{BC,nothing,nothing}) where {BC}
    lower_bc.type
end

# lowest process
function layer_below(layers, lower_bc::BoundaryCondition{BC,nothing,Na}) where {BC,Na}
    MPI.Send(layers[end], Na, MTAG_UP, lower_bc.comm)
    lower_bc.type
end

# highest process
function layer_below(layers, lower_bc::BoundaryCondition{BC,Nb,nothing}) where {BC,Nb}
    MPI.Recv!(lower_bc.buffer, Nb, MTAG_UP, lower_bc.comm)
    lower_bc.buffer
end

# inner process
function layer_below(layers, lower_bc::BoundaryCondition{BC,Nb,Na}) where {BC,Nb,Na}
    r = MPI.Irecv!(lower_bc.buffer, Nb, MTAG_UP, lower_bc.comm)
    MPI.Send(layers[end], Na, MTAG_UP, lower_bc.comm)
    MPI.Wait!(r)
    lower_bc.buffer
end

# single process
function layer_above(layers, upper_bc::BoundaryCondition{BC,nothing,nothing}) where {BC}
    upper_bc.type
end

# lowest process
function layer_above(layers, upper_bc::BoundaryCondition{BC,nothing,Na}) where {BC,Na}
    MPI.Recv!(upper_bc.buffer, Na, MTAG_DN, upper_bc.comm)
    upper_bc.buffer
end

# highest process
function layer_above(layers, upper_bc::BoundaryCondition{BC,Nb,nothing}) where {BC,Nb}
    MPI.Send(layers[1], Nb, MTAG_DN, upper_bc.comm)
    upper_bc.type
end
function layer_above(layers::Tuple{}, upper_bc::BoundaryCondition{ConstantValue{T},Nb,nothing}) where {T,Nb}
    # this is a special case for when the top process does not have any layers,
    # which is the case if there is only one layer per process. in this case, we
    # fill the BC buffer with the boundary condition and pass that down to the
    # process below
    if eltype(upper_bc.buffer) <: Real
        fill!(upper_bc.buffer, upper_bc.type.value)
    else
        fill!(upper_bc.buffer, 0)
        upper_bc.buffer[1,1] = upper_bc.type.value
    end
    MPI.Send(upper_bc.buffer, Nb, MTAG_DN, upper_bc.comm)
    nothing # prevent the caller from trying to use the return value
end

# inner process
function layer_above(layers, upper_bc::BoundaryCondition{BC,Nb,Na}) where {BC,Nb,Na}
    r = MPI.Irecv!(upper_bc.buffer, Na, MTAG_DN, upper_bc.comm)
    MPI.Send(layers[1], Nb, MTAG_DN, upper_bc.comm)
    MPI.Wait!(r)
    upper_bc.buffer
end

"""
This function takes a field defined on C-nodes, converts it into layers, and expands
them through communication such that the list includes the layers just above and
below all I-nodes. This means passing data down throughout the domain.
The boundary condition is only used for passing data, the actual values are unused.
"""
function layers_c2i(field::AbstractArray, bc_above::BoundaryCondition{BC,Nb,Na}) where {BC,Nb,Na}
    field = layers(field)
    field_above = layer_above(field, bc_above)
    isnothing(Na) ? field : (field..., field_above)
end

"""
This function takes a field defined on I-nodes, converts it into layers, and expands
them through communication such that the list includes the layers just above and
below all C-nodes. This means passing data up throughout the domain.
"""
function layers_i2c(field::AbstractArray{T}, bc_below::BoundaryCondition{BCb,Nb,Na},
                        bc_above::BoundaryCondition{BCa,Nb,Na}) where {T,BCb,BCa,Nb,Na}
    field = layers(field)
    field_below = layer_below(field, bc_below)
    isnothing(Na) ? (field_below, field..., bc_above.type) : (field_below, field...)
end

function layers_expand_full(field::AbstractArray,
        bc_below::BoundaryCondition{BCb,Nb,Na},
        bc_above::BoundaryCondition{BCa,Nb,Na}) where {BCb,BCa,Nb,Na}

    field = layers(field)
    layer_below(field, bc_below), field..., layer_above(field, bc_above)
end

end # module BoundaryConditions
