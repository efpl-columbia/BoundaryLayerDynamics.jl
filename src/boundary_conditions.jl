module BoundaryConditions

using MPI: MPI
using ..Domain: SmoothWall, RoughWall, CustomBoundary, FreeSlipBoundary
using ..Grid: fdsize, neighbors

struct BoundaryCondition{BC,Nb,Na,C,A}
    type::BC
    buffer::A
    comm::C
    BoundaryCondition(type::BC, buffer::A, comm::C, (Nb, Na)) where {BC,A,C} =
        new{BC,Nb,Na,C,A}(type, buffer, comm)
end

struct ConstantValue{T}
    value::T
end

struct ConstantGradient{T}
    gradient::T
end

const DirichletBC = BoundaryCondition{ConstantValue}
const NeumannBC = BoundaryCondition{ConstantGradient}


# set up frequency domain boundary condition
function BoundaryCondition(::Type{T}, type, grid) where T
    type = init_bctype(T, type)
    buffer = zeros(Complex{T}, fdsize(grid))
    BoundaryCondition(type, buffer, grid.comm, neighbors(grid))
end

# set up physical domain boundary condition
function BoundaryCondition(::Type{T}, type, grid, dealiasing) where T
    type = init_bctype(T, type)
    buffer = zeros(T, pdsize(grid, dealiasing))
    BoundaryCondition(type, buffer, neighbors(grid), grid.comm)
end

# convert boundary conditions into concrete types
init_bctype(::Type{T}, type::Symbol) where T = init_bctype(T, Val(type), zero(T))
init_bctype(::Type{T}, type::Pair) where T = init_bctype(T, Val(first(type)), convert(T, last(type)))
init_bctype(::Type{T}, ::Val{:dirichlet}, value::T) where T = ConstantValue(value)
init_bctype(::Type{T}, ::Val{:neumann}, gradient::T) where T = ConstantGradient(gradient)

function init_bcs(::Type{T}, domain, grid, field) where T
    lbc = BoundaryCondition(T, bctype(domain.lower_boundary, field), grid)
    ubc = BoundaryCondition(T, bctype(domain.upper_boundary, field), grid)
    (lbc, ubc)
end

# create boundary conditions from boundary definitions
bctype(boundary, field::Symbol) = bctype(boundary, Val(field))
bctype(boundary::CustomBoundary, ::Val{F}) where F = boundary.behaviors[F]
bctype(::SmoothWall, ::Val{:vel1}) = :dirichlet
bctype(::SmoothWall, ::Val{:vel2}) = :dirichlet
bctype(::SmoothWall, ::Val{:vel3}) = :dirichlet
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

function layers_expand_full(field::AbstractArray,
        bc_below::BoundaryCondition{BCb,Nb,Na},
        bc_above::BoundaryCondition{BCa,Nb,Na}) where {BCb,BCa,Nb,Na}

    field = layers(field)
    layer_below(field, bc_below), field..., layer_above(field, bc_above)
end

end
