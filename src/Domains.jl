module Domains

export Domain, SinusoidalMapping, SmoothWall, RoughWall, FreeSlipBoundary, CustomBoundary, x1range, x2range, x3range

abstract type AbstractDomain{T} end

"""
    Domain([T=Float64,] dimensions, lower_boundary, upper_boundary, [mapping])

Three-dimensional Cartesian domain of size `dimensions`, periodic along the
first two coordinate directions and delimited by `lower_boundary` and
`upper_boundary` along the third coordinate direction.

# Arguments

- `T::DataType = Float64`: Type that is used for coordinates and fields inside domain.
- `dimensions::Tuple`: Size of the domain. The third dimension can be specified
  as a single value, in which case the domain is assumed to start at ``x_3=0``
  or as a tuple with the minimum and maximum ``x_3`` values. If it is omitted,
  the default of ``x_3 ∈ [0,1]`` is assumed, unless a custom `mapping` is
  provided.
- `lower_boundary`, `upper_boundary`: Boundary definitions.
- `mapping`: A non-linear mapping from the interval ``[0,1]`` to the range of
  ``x_3`` values, instead of the default linear mapping. The mapping can either
  be specified as a tuple of two functions representing the mapping and its
  derivative, or using the predefined [`SinusoidalMapping`](@ref). In the
  former case, the third element of `dimensions` is ignored if specified; in
  the latter case the mapping is adjusted to the domain size.
"""
struct Domain{T,F1,F2} <: AbstractDomain{T}
    hsize::Tuple{T,T}
    vmap::F1  # ζ ∈ [0,1] → x₃ ∈ physical domain
    Dvmap::F2 # dx₃/dζ
    lower_boundary
    upper_boundary

    function Domain(::Type{T}, dims::Tuple,
            lower_boundary, upper_boundary, mapping = nothing) where T

        l1, l2 = convert.(T, dims[1:2])
        # l3 could be omitted, one value, or two values
        l3 = length(dims) == 2 ? one(T) : convert.(T, dims[3])

        # type of `l3` defines type of mapping
        x3, Dx3 = instantiate(mapping, l3, (lower_boundary, upper_boundary))

        new{T,typeof(x3),typeof(Dx3)}((l1, l2), x3, Dx3, lower_boundary, upper_boundary)
    end
end

Base.size(domain::Domain) = Tuple(size(domain, i) for i=1:3)
Base.size(domain::Domain{T}, dim) where T = begin
    dim in (1,2) && return domain.hsize[dim]
    dim == 3 && return convert(T, domain.vmap(one(T)) - domain.vmap(zero(T)))
    error("Invalid dimension `$dim`")
end

Base.extrema(domain::Domain) = begin
    x1min, x1max = extrema(domain, 1)
    x2min, x2max = extrema(domain, 2)
    x3min, x3max = extrema(domain, 3)
    ((x1min, x2min, x3min), (x1max, x2max, x3max))
end
Base.extrema(domain::Domain{T}, dim::Int) where T = begin
    dim in (1,2) && return (zero(T), domain.hsize[dim])
    dim == 3 && return convert.(T, domain.vmap.((zero(T), one(T))))
    error("Invalid dimension `$dim`")
end

# use double precision by default
Domain(size::Tuple, args...) = Domain(Float64, size, args...)

"""
    SinusoidalMapping(η, variant = :auto)

Define a transformed vertical coordinate with a mapping in the form of

``x_3 = \\frac{\\sin(ζ η π/2)}{\\sin(η π/2)}``

appropriately rescaled such that it maps the range ``0≤ζ≤1`` to the
``x_3``-range of the [`Domain`](@ref).

The parameter ``0<η<1`` controls the strength of the grid stretching, where
values close to ``0`` result in a more equidistant spacing and values close to
``1`` result in a higher density of grid points close to the wall(s). The value
``η=1`` is not allowed since it produces a vanishing derivative of the mapping
function at the wall.

The `variant` defines at which of the boundaries the coordinates are refined
and can be set to `:below`, `:above`, `:symmetric` (both boundaries refined),
or `:auto` (refined for boundaries that produce a boundary-layer).
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
x1range(domain::Domain{T}, ξ) where T = (convert(T, domain.hsize[1] * ξ) for ξ in ξ)
x2range(domain::Domain{T}, η) where T = (convert(T, domain.hsize[2] * η) for η in η)
x3range(domain::Domain{T}, ζ) where T = (convert(T, domain.vmap(ζ)) for ζ in ζ)

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
function scalefactor(domain::Domain, dim::Int)
    # TODO: support this for vertical directions if there is no transform in use
    dim == 1 ? 1/domain.hsize[1] : dim == 2 ? 1/domain.hsize[2] :
        error("Coordinates along dimension 3 might be transformed")
end

function scalefactor(domain::Domain{T}, dim::Int, pos::Rational) where T
    # factors are constant along horizontal dimensions
    dim in (1, 2) && return scalefactor(domain, dim)
    dim == 3 && return convert(T, 1/domain.Dvmap(pos))
    error("Invalid dimension `$pos`")
end

"""
    SmoothWall()

A wall that is aerodynamically smooth and has no-slip, no-penetration boundary
conditions for the velocity field.
"""
struct SmoothWall
end

"""
    RoughWall(roughness, [von_karman_constant = 0.4])

A wall that is aerodynamically rough with a `roughness` length scale ``z₀``.
The mean streamwise velocity near the wall can be assumed to follow a
logarithmic profile with the specified von Kármán constant.
"""
struct RoughWall
    roughness
    von_karman_constant
    RoughWall(z0, kappa = 0.4) = new(z0, kappa)
end

"""
    FreeSlipBoundary()

A no-penetration boundary with vanishing gradients for tangential velocity
components.
"""
struct FreeSlipBoundary
end

"""
    CustomBoundary(; boundary_behaviors...)

A boundary that explicitly specifies the mathematical boundary conditions for
all state variables. Boundary conditions can be specified as keyword arguments,
where the key is the name of the field and the value is either one of
`:dirichlet` or `:neumann` for homogeneous boundary conditions, or a `Pair`
that includes the non-zero value, such as `:dirichlet => 1`.

# Example

```
CustomBoundary(vel1 = :dirichlet => 1, vel2 = :dirichlet, vel3 = :neumann)
```
"""
struct CustomBoundary
    behaviors
    CustomBoundary(; behaviors...) = new(NamedTuple(behaviors))
end

hasboundarylayer(::RoughWall) = true
hasboundarylayer(::SmoothWall) = true
hasboundarylayer(::FreeSlipBoundary) = false

end
