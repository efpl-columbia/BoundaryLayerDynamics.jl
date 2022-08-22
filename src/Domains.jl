module Domains

export ABLDomain, SinusoidalMapping, SmoothWall, RoughWall, FreeSlipBoundary, CustomBoundary, x1range, x2range, x3range

abstract type AbstractDomain{T} end

struct ABLDomain{T,F1,F2} <: AbstractDomain{T}
    hsize::Tuple{T,T}
    vmap::F1  # ζ ∈ [0,1] → x₃ ∈ physical domain
    Dvmap::F2 # dx₃/dζ
    lower_boundary
    upper_boundary

    function ABLDomain(::Type{T}, dims::Union{Tuple,AbstractArray},
            lower_boundary, upper_boundary, mapping = nothing) where T

        l1, l2 = convert.(T, dims[1:2])
        l3 = length(dims) == 2 ? one(T) : dims[3]

        x3, Dx3 = instantiate(mapping, l3, (lower_boundary, upper_boundary))

        new{T,typeof(x3),typeof(Dx3)}((l1, l2), x3, Dx3, lower_boundary, upper_boundary)
    end
end

Base.size(domain::ABLDomain) = Tuple(size(domain, i) for i=1:3)
Base.size(domain::ABLDomain{T}, dim) where T = begin
    dim in (1,2) && return domain.hsize[dim]
    dim == 3 && return convert(T, domain.vmap(one(T)) - domain.vmap(zero(T)))
    error("Invalid dimension `$dim`")
end

Base.extrema(domain::ABLDomain) = begin
    x1min, x1max = extrema(domain, 1)
    x2min, x2max = extrema(domain, 2)
    x3min, x3max = extrema(domain, 3)
    ((x1min, x2min, x3min), (x1max, x2max, x3max))
end
Base.extrema(domain::ABLDomain{T}, dim::Int) where T = begin
    dim in (1,2) && return (zero(T), domain.hsize[dim])
    dim == 3 && return convert.(T, domain.vmap.((zero(T), one(T))))
    error("Invalid dimension `$dim`")
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
    variant
    SinusoidalMapping(η, variant=:auto) = (@assert 0 < η < 1; new(η, variant))
end

function instantiate(m::SinusoidalMapping, (x3min, x3max)::Tuple{T,T}, bcs = nothing) where T

    @assert x3min < x3max
    η = convert(T, m.parameter)
    l = x3max - x3min

    variant = m.variant
    if variant == :auto
        bls = hasboundarylayer.(bcs)
        if bls == (true, true)
            variant = :symmetric
        elseif bls == (true, false)
            variant = :below
        elseif bls == (false, true)
            variant = :above
        else
            error("Could not determine variant of SinusoidalMapping")
        end
    end

    if variant == :symmetric
        x3 = ζ -> convert(T, x3min + l/2 * (1 + sin(η*(2*ζ-1)*π/2) / sin(η*π/2)))
        Dx3 = ζ -> convert(T, l * η*π/2 / sin(η*π/2) * cos(η*(2*ζ-1)*π/2))
    elseif variant == :below
        x3 = ζ -> convert(T, x3min + l * (1 + sin(η*(ζ-1)*π/2) / sin(η*π/2)))
        Dx3 = ζ -> convert(T, l * η*π/2 / sin(η*π/2) * cos(η*(ζ-1)*π/2))
    elseif variant == :above
        x3 = ζ -> convert(T, x3min + l * (sin(η*(ζ)*π/2) / sin(η*π/2)))
        Dx3 = ζ -> convert(T, l * η*π/2 / sin(η*π/2) * cos(η*(ζ)*π/2))
    else
        error("Unknown variant of SinusoidalMapping")
    end

    x3, Dx3
end

instantiate(::Nothing, (x3min, x3max)::Tuple{T,T}, bcs = nothing) where T =
    ζ -> x3min + ζ * (x3max - x3min), x -> x3max - x3min
instantiate(m::Tuple{Function,Function}, (x3min, x3max)::Tuple{T,T}, bcs = nothing) where T = m
instantiate(mapping, L3::Real, bcs = nothing) = instantiate(mapping, (zero(L3), L3), bcs)

# get physical coordinates from normalized domain positions ∈ [0,1]
x1range(domain::ABLDomain{T}, ξ) where T = (convert(T, domain.hsize[1] * ξ) for ξ in ξ)
x2range(domain::ABLDomain{T}, η) where T = (convert(T, domain.hsize[2] * η) for η in η)
x3range(domain::ABLDomain{T}, ζ) where T = (convert(T, domain.vmap(ζ)) for ζ in ζ)

dx1factors(domain, wavenumbers::Tuple) = dx1factors(domain, wavenumbers[1])
dx1factors(domain, wavenumbers) = reshape(1im * wavenumbers * (2π/domain.hsize[1]), (:, 1))
dx2factors(domain, wavenumbers::Tuple) = dx2factors(domain, wavenumbers[2])
dx2factors(domain, wavenumbers) = reshape(1im * wavenumbers * (2π/domain.hsize[2]), (1, :))

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
    von_karman_constant
    RoughWall(z0, kappa = 0.4) = new(z0, kappa)
end

struct SmoothWall
end

struct FreeSlipBoundary
end

struct CustomBoundary
    behaviors
    CustomBoundary(; behaviors...) = new(NamedTuple(behaviors))
end

hasboundarylayer(::RoughWall) = true
hasboundarylayer(::SmoothWall) = true
hasboundarylayer(::FreeSlipBoundary) = false

end
