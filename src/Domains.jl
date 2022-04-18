module Domains

export ABLDomain, SinusoidalMapping, SmoothWall, RoughWall, FreeSlipBoundary, CustomBoundary, x1range, x2range, x3range

abstract type AbstractDomain{T} end

struct ABLDomain{T,F1,F2} <: AbstractDomain{T}
    hsize::Tuple{T,T}
    vmap::F1 # [0,1] → physical domain
    Dvmap::F2
    lower_boundary
    upper_boundary

    function ABLDomain(::Type{T}, size, lower_boundary, upper_boundary, mapping = nothing) where T

        one_sided = any(isa.((lower_boundary, upper_boundary), FreeSlipBoundary))

        hsize(L::T) = (L, L)
        hsize(L::Tuple{T,T}) = (L[1], L[1])
        hsize(L::Tuple{T,T,T}) = (L[1], L[2])
        hsize(L) = hsize(convert.(T, size))
        x3, Dx3 = instantiate(mapping, last(size), one_sided)

        new{T,typeof(x3),typeof(Dx3)}(hsize(size), x3, Dx3, lower_boundary, upper_boundary)
    end
end

# use double precision by default
ABLDomain(size, args...) = ABLDomain(Float64, size, args...)

"""
    SinusoidalMapping(η)

Define a transformed vertical coordinate with the mapping

``x_3/δ = 1 + \\frac{sin(η (ζ-1) π/2)}{sin(η π/2)}``

for a half-channel, where ``0≤ζ≤1``. For a full channel, the transformed grid
is mirrored in the upper half.

The parameter ``0<η<1`` controls the strength of the grid stretching, where
values close to ``0`` result in a more equidistant spacing and values close to
``1`` result in a higher density of grid points close to the wall(s). The value
``η=1`` is not allowed since it produces a vanishing derivative of the mapping
function at the wall.
"""
struct SinusoidalMapping
    parameter
    SinusoidalMapping(η) = (@assert 0 < η < 1; new(η))
end

function instantiate(m::SinusoidalMapping, (x3min, x3max)::Tuple{T,T}, one_sided) where T

    @assert x3min < x3max
    η = convert(T, m.parameter)
    l = x3max - x3min

    if one_sided
        x3 = ζ -> convert(T, x3min + l * (1 + sin(η*(ζ-1)*π/2) / sin(η*π/2)))
        Dx3 = ζ -> convert(T, l * (η*π/2) * (cos(η*(ζ-1)*π/2) / sin(η*π/2)))
        x3, Dx3
    else
        m = (x3min + x3max) / 2
        x3 = ζ -> convert(T, m + l/2 * (sin(η*(2*ζ-1)*π/2) / sin(η*π/2)))
        Dx3 = ζ -> convert(T, l/2 * (2*η*π/2) * (cos(η*(-1 + 2*ζ)*π/2) / sin(η*π/2)))
        x3, Dx3
    end
end

instantiate(::Nothing, (x3min, x3max)::Tuple{T,T}, one_sided) where T =
    ζ -> x3min + ζ * (x3max - x3min), x -> x3max - x3min

instantiate(mapping, L3, one_sided) = instantiate(mapping, (zero(L3), L3), one_sided)

# get physical coordinates from normalized domain positions ∈ [0,1]
x1range(domain::ABLDomain{T}, ξ) where T = (convert(T, domain.hsize[1] * ξ) for ξ in ξ)
x2range(domain::ABLDomain{T}, η) where T = (convert(T, domain.hsize[2] * η) for η in η)
x3range(domain::ABLDomain{T}, ζ) where T = (convert(T, domain.vmap(ζ)) for ζ in ζ)

# struct holding the information necessary to compute the jacobian of the
# mapping from the simulation space [0,1]×[0,1]×[0,1] to the physical domain
#struct DomainJacobian
    # TODO
#end

# Base.getindex(j::DomainJacobian, i::Int)

"""
Computes the constant coordinate scaling along dimension `dim`, i.e. 1/L
where L is the domain size. This can only be computed for coordinates that are
at most linearly transformed and will return an error otherwise.
"""
function scalefactor(domain::ABLDomain, dim::Int)
    # TODO: support this for vertical directions if there is no transform in use
    dim == 1 ? 1/domain.hsize[1] : dim == 2 ? 1/domain.hsize[2] :
        error("Coordinates along dimension 3 might be transformed")
end

function scalefactor(domain::ABLDomain{T}, dim::Int, pos::Rational) where T
    # factors are constant along horizontal dimensions
    dim in (1, 2) && return scalefactor(domain, dim)
    dim == 3 && return convert(T, 1/domain.Dvmap(pos))
    error("Invalid dimension `$pos`")
end

struct RoughWall
    roughness
end

struct SmoothWall
end

struct FreeSlipBoundary
end

struct CustomBoundary
    behaviors
    CustomBoundary(; behaviors...) = new(NamedTuple(behaviors))
end

end
