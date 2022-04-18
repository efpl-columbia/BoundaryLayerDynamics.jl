module Derivatives

using ..Helpers
using ..Grids: NodeSet, wavenumbers, fdsize, vrange
using ..Domains: AbstractDomain as Domain
using ..PhysicalSpace: get_field!
import ..PhysicalSpace: default_term, init_term, compute_term! # imported to add new methods
using ..BoundaryConditions: ConstantValue, internal_bc, layers, layers_c2i

dx1factors(domain, grid) = begin
    d = reshape(1im * wavenumbers(grid, 1) * (2π/domain.hsize[1]), (:, 1))
    d
end
dx2factors(domain, grid) = begin
    d = reshape(1im * wavenumbers(grid, 2) * (2π/domain.hsize[2]), (1, :))
    d
end

function dx3factors(domain::Domain{T}, grid, nodes; neighbors=false) where T
    α(ζ) = grid.n3global / domain.Dvmap(convert(T, ζ))
    ζ = vrange(grid, nodes)
    if neighbors
        ζnb = vrange(grid, nodes, neighbors=true)
        [(α(ζnb[i]), α(ζ[i]), α(ζnb[i+1])) for i=1:equivalently(length(ζ), length(ζnb)-1)]
    else
        collect(α.(ζ))
    end
end

function second_derivatives(domain::Domain{T}, grid, nodes) where T
    k1, k2 = wavenumbers(grid)
    DD1 = reshape( - k1.^2 * (2π/convert(T, domain.hsize[1]))^2, (:, 1))
    DD2 = reshape( - k2.^2 * (2π/convert(T, domain.hsize[2]))^2, (1, :))
    DD3 = dx3factors(domain, grid, nodes, neighbors=true)
    (DD1=DD1, DD2=DD2, DD3=DD3)
end


# TERMS FOR SIMPLE DERIVATIVES -----------------------------

# default behavior: derivatives if name contains underline, direct transform otherwise
function default_term(field, domain, grid, dims)
    if '_' in string(field) # set up a derivative
        field, dim = split(string(field), '_')
        if dim == "1"
            (default=true, D1 = dx1factors(domain, grid),)
        elseif dim == "2"
            (default=true, D2 = dx2factors(domain, grid),)
        elseif dim == "3"
            field = Symbol(field)
            ns = nodes(field)
            # note: derivatives are computed on opposite grid points
            if ns isa NodeSet{:C}
                (default=true, dependencies = (field,),
                 D3i = dx3factors(domain, grid, NodeSet(:I)),
                 bc = internal_bc(domain, grid, dims))
            elseif ns isa NodeSet{:I}
                (default=true, dependencies = (field,),
                 D3c = dx3factors(domain, grid, NodeSet(:C)),
                 bcs = init_bcs(field, domain, grid, dims))
            end
        end
    else # set up a direct transform
        (default=true,)
    end
end

# default behavior: transform term directly from FD state
compute_term!(term::NamedTuple{(:values,:default),T},
              ::Val{F}, _, state, transform) where {F,T} =
    get_field!(term.values, transform, state[F])

# default behavior: compute derivative along first dimension
compute_term!(term::NamedTuple{(:values,:default,:D1),T},
              ::Val{F}, _, state, transform) where {F,T} =
    get_field!(term.values, transform, state[F], term.D1)

# default behavior: compute derivative along second dimension
compute_term!(term::NamedTuple{(:values,:default,:D2),T},
              ::Val{F}, _, state, transform) where {F,T} =
    get_field!(term.values, transform, state[F], term.D2)

# default behavior: compute derivative along third dimension (C→I)
compute_term!(term::NamedTuple{(:values,:default,:dependencies,:D3i,:bc),T},
              ::Val{F}, pdfields, _, _) where {F,T} =
    dx3_c2i!(term.values, pdfields[F], term.bc, term.D3i)

# default behavior: compute derivative along third dimension (I→C)
compute_term!(term::NamedTuple{(:values,:default,:dependencies,:D3c,:bcs),T},
              ::Val{F}, pdfields, _, _) where {F,T} =
    dx3_i2c!(term.values, pdfields[F], term.bcs, term.D3c)

# all layers of vertical derivatives in physical & frequency domain
function dx3_c2i!(dx3, term, bc, D3i)
    dx3 = layers(dx3)
    term = layers_c2i(term, bc)
    for i=1:equivalently(length(dx3), length(term)-1, length(D3i))
        dx3!(dx3[i], term[i:i+1], D3i)
    end
end
function dx3_i2c!(dx3, term, bcs, D3)
    dx3 = layers(dx3)
    term = layers_i2c(term, bcs...)
    for i=1:equivalently(length(dx3), length(term)-1, length(D3c))
        dx3!(dx3[i], term[i:i+1], D3c)
    end
end

# single layer of vertical derivatives in physical & frequency domain
# → boundary conditions: only Dirichlet on I-nodes supported
dx3_c2i!(dx3, (term¯, term⁺), D3i) = @. dx3 = - D3i * term¯ + D3i * term⁺
dx3_i2c!(dx3, (term¯, term⁺), D3c) = @. dx3 = - D3c * term¯ + D3c * term⁺
function dx3_i2c!(dx3, (lbc, term⁺)::Tuple{ConstantValue,A}, D3) where A
    if eltype(dx3) <: Real # physical domain
        @. dx3 = - D3 * lbc.value + D3 * term⁺
    else
        @. dx3 = D3 * term⁺
        dx3[1,1] -= D3 * lbc.value
    end
end
function dx3_i2c!(dx3, (lbc, term⁺)::Tuple{ConstantValue,A}, D3) where A
    if eltype(dx3) <: Real # physical domain
        @. dx3 = - D3 * lbc.value + D3 * term⁺
    else
        @. dx3 = D3 * term⁺
        dx3[1,1] -= D3 * lbc.value
    end
end


# TERMS FOR VORTICITY --------------------------------------

# if we know that the velocity gradients will be computed in physical domain
# anyway, we can use those to compute the vorticity; otherwise it is cheaper to
# compute the vorticity in frequency domain and only transform its three
# components
init_term(::Val{:vort1}, domain, grid, _, fields) =
    :S23 in fields || (:vel2 in fields && :vel3_2 in fields) ?
        (dependencies = (:vel2, :vel3_2),) :
        (bc = internal_bc(domain, grid),
         buffer = init_buffer(domain, grid),
         D2 = dx2factors(domain, grid),
         D3i = dx3factors(domain, grid, NodeSet(:I)))
init_term(::Val{:vort2}, domain, grid, _, fields) =
    :S13 in fields || (:vel1 in fields && :vel3_1 in fields) ?
        (dependencies = (:vel1, :vel3_1),) :
        (bc = internal_bc(domain, grid),
         buffer = init_buffer(domain, grid),
         D1 = dx1factors(domain, grid),
         D3i = dx3factors(domain, grid, NodeSet(:I)))
init_term(::Val{:vort3}, domain, grid, _, fields) =
    :S12 in fields || (:vel1_2 in fields && :vel2_1 in fields) ?
        (dependencies = (:vel1_2, :vel2_1),) :
        (buffer = init_buffer(domain, grid),
         D1 = dx1factors(domain, grid),
         D2 = dx2factors(domain, grid))

# convenience function to generate a single layer of the right size
init_buffer(domain::Domain{T}, grid) where T = zeros(Complex{T}, fdsize(grid))
init_buffer(domain::Domain{T}, dims::Tuple) where T = zeros(T, dims)

# we check the dependencies to determine whether the vorticity is computed in
# physical or frequency domain
compute_term!(term, ::Val{:vort1}, pdfields, state, transform) =
    if haskey(term, :dependencies)
        @. term.values = pdfields.vel3_2 - pdfields.vel2_3
    else
        vort1!(term.values, state.vel2, state.vel3, transform, term.D2, term.D3i, term.bc, term.buffer)
        term.values
    end
compute_term!(term, ::Val{:vort2}, pdfields, state, transform) =
    if haskey(term, :dependencies)
        @. term.values = pdfields.vel1_3 - pdfields.vel3_1
    else
        vort2!(term.values, state.vel1, state.vel3, transform, term.D1, term.D3i, term.bc, term.buffer)
        term.values
    end
compute_term!(term, ::Val{:vort3}, pdfields, state, transform) =
    if haskey(term, :dependencies)
        @. term.values = pdfields.vel2_1 - pdfields.vel1_2
    else
        vort3!(term.values, state.vel1, state.vel2, transform, term.D1, term.D2, term.buffer)
        term.values
    end

# compute vorticity in frequency domain
function vort1!(vort1, vel2, vel3, transform, D2, D3i, ubc2, buffer)
    vort1 = layers(vort1)
    vel2exp = layers_c2i(vel2, ubc2)
    vel3 = layers(vel3)
    for i=1:equivalently(length(vort1), length(vel2exp)-1, length(vel3), length(D3i))
        set_vort1!(buffer, vel2exp[i:i+1], vel3[i], D2, D3i[i])
        get_field!(vort1[i], transform, buffer)
    end
end
function vort2!(vort2, vel1, vel3, transform, D1, D3i, ubc1, buffer)
    vort2 = layers(vort2)
    vel1exp = layers_c2i(vel1, ubc1)
    vel3 = layers(vel3)
    for i=1:equivalently(length(vort2), length(vel1exp)-1, length(vel3), length(D3i))
        set_vort2!(buffer, vel1exp[i:i+1], vel3[i], D1, D3i[i])
        get_field!(vort2[i], transform, buffer)
    end
end
function vort3!(vort3, vel1, vel2, transform, D1, D2, buffer)
    vort3 = layers(vort3)
    vel1 = layers(vel1)
    vel2 = layers(vel2)
    for i=1:equivalently(length(vort3), length(vel1), length(vel2))
        set_vort3!(buffer, vel1[i], vel2[i], D1, D2)
        get_field!(vort3[i], transform, buffer)
    end
end

# compute single layer of vorticity in frequency domain
set_vort1!(vort1, (vel2¯, vel2⁺), vel3, D2, D3) = @. vort1 = D2 * vel3 - (-D3 * vel2¯ + D3 * vel2⁺)
set_vort2!(vort2, (vel1¯, vel1⁺), vel3, D1, D3) = @. vort2 = (-D3 * vel1¯ + D3 * vel1⁺) - D1 * vel3
set_vort3!(vort3, vel1, vel2, D1, D2) = @. vort3 = D1 * vel2 - D2 * vel1


# TERMS FOR RATE OF STRAIN ---------------------------------

# strain rates can be computed in frequency or physical domain
init_term(::Val{:strain11}) = error("Not yet implemented :(")
init_term(::Val{:strain12}) = error("Not yet implemented :(")
init_term(::Val{:strain13}) = error("Not yet implemented :(")
init_term(::Val{:strain22}) = error("Not yet implemented :(")
init_term(::Val{:strain33}) = error("Not yet implemented :(")
init_term(::Val{:strain33}) = error("Not yet implemented :(")

# total strain is computed in physical domain
init_term(::Val{:strain}) = (dependencies = (:strain11, :strain12, :strain13,
                                             :strain22, :strain23, :strain33))
end # module Derivatives
