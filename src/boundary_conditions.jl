module BoundaryConditions

using ..Domain: SmoothWall, RoughWall, CustomBoundary, FreeSlipBoundary
using ..Grid: fdsize, neighbors

struct BoundaryCondition{BC,Nb,Na,C,A}
    type::BC
    buffer::A
    neighbors::Tuple{Nb,Na}
    comm::C
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
    BoundaryCondition(type, buffer, neighbors(grid), grid.comm)
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
function layer_below(layers, lower_bc::BoundaryCondition{BC,Nothing,Nothing}) where {BC}
    lower_bc.type
end

# lowest process
function layer_below(layers, lower_bc::BoundaryCondition{BC,Nothing,Na}) where {BC,Na}
    MPI.Send(layers[end], Na, MTAG_UP, lower_bc.comm)
    lower_bc.type
end

# highest process
function layer_below(layers, lower_bc::BoundaryCondition{BC,Nb,Nothing}) where {BC,Nb}
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
function layer_above(layers, upper_bc::BoundaryCondition{BC,Nothing,Nothing}) where {BC}
    upper_bc.type
end

# lowest process
function layer_above(layers, upper_bc::BoundaryCondition{BC,Nothing,Na}) where {BC,Na}
    MPI.Recv!(upper_bc.buffer, Na, MTAG_DOWN, upper_bc.comm)
    upper_bc.buffer
end

# highest process
function layer_above(layers, upper_bc::BoundaryCondition{BC,Nb,Nothing}) where {BC,Nb}
    MPI.Send(layers[1], Nb, MTAG_DOWN, upper_bc.comm)
    upper_bc.type
end

# inner process
function layer_above(layers, upper_bc::BoundaryCondition{BC,Nb,Nothing}) where {BC,Nb}
    r = MPI.Irecv!(upper_bc.buffer, Na, MTAG_DOWN, upper_bc.comm)
    MPI.Send(layers[1], Nb, MTAG_DOWN, upper_bc.comm)
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
