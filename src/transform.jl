SupportedReals = Union{Float32,Float64}

struct NodeSet{NS}
    NodeSet(ns::Symbol) = ns == :H ? new{:H}() : ns == :V ? new{:V}() :
        error("Invalid NodeSet: $(NS) (only :H and :V are allowed)")
end

staggered_nodes() = (NodeSet(:H), NodeSet(:H), NodeSet(:V))
inverted_nodes() = (NodeSet(:V), NodeSet(:V), NodeSet(:H))

function vertical_range(nz, proc_id, proc_count)
    nz >= proc_count || error("There should not be more processes than vertical layers")
    nz_per_proc, nz_rem = divrem(nz, proc_count)
    iz_min = 1 + nz_per_proc * (proc_id - 1) + min(nz_rem, proc_id - 1)
    iz_max = min(nz_per_proc * proc_id + min(nz_rem, proc_id), nz)
    nz_local_h = iz_max - iz_min + 1
    nz_local_v = (proc_id == proc_count ? nz_local_h - 1 : nz_local_h)
    iz_min, iz_max, nz_local_h, nz_local_v
end

# these more specific methods of vertical_range give the distribution of the
# vertical indices for a specific set of nodes. note that the variant for the
# V-nodes takes the global number of V-layers as argument, which is reduced by 1.
proc_info() = MPI.Initialized() ? (MPI.Comm_rank(MPI.COMM_WORLD) + 1,
        MPI.Comm_size(MPI.COMM_WORLD)) : (1,1)
vertical_range(nz_global_h, ::NodeSet{:H}) =
    (vr = vertical_range(nz_global_h, proc_info()...); (vr[1], vr[2]))
vertical_range(nz_global_v, ::NodeSet{:V}) =
    (vr = vertical_range(nz_global_v+1, proc_info()...); (vr[1], vr[1]+vr[4]-1))

struct DistributedGrid
    nx_fd::Int
    nx_pd::Int
    ny_fd::Int
    ny_pd::Int
    nz_h::Int
    nz_v::Int
    nz_global::Int
    iz_min::Int
    iz_max::Int
end

function DistributedGrid(nx, ny, nz, proc_id, proc_count)

    # NOTE: IDs are one-based, unlike MPI ranks!

    # horizontal sizes: nx & ny inform number of frequencies, but for even
    # values we remove the Nyquist frequency such that there is always an odd
    # number of frequencies going from -k_max to 0 to k_max. in the physical
    # domain the number of grid points is 3/2 times the number of frequencies
    # plus one (in order to have an even number of frequencies such that
    # multiplication with 3/2 results in an integer value)
    kx_max = div(nx - 1, 2)
    ky_max = div(ny - 1, 2)
    nx_fd = 1 + kx_max
    ny_fd = 1 + ky_max * 2
    nx_pd = 3 * (1 + kx_max)
    ny_pd = 3 * (1 + ky_max)

    # vertical sizes
    iz_min, iz_max, nz_local_h, nz_local_v = vertical_range(nz, proc_id, proc_count)

    DistributedGrid(nx_fd, nx_pd, ny_fd, ny_pd,
            nz_local_h, nz_local_v, nz, iz_min, iz_max)
end

# convenience functions
DistributedGrid(nx, ny, nz) = DistributedGrid(nx, ny, nz, proc_info()...)
DistributedGrid(n) = DistributedGrid(n, n, n)
Broadcast.broadcastable(gd::DistributedGrid) = Ref(gd) # to use as argument of elementwise functions

function Base.show(io::IO, gd::DistributedGrid)

    ip = MPI.Initialized() ? MPI.Comm_rank(MPI.COMM_WORLD) + 1 : 1
    np = MPI.Initialized() ? MPI.Comm_size(MPI.COMM_WORLD) : 1

    println(io, "DistributedGrid(", gd.nx_fd * 2,  ", ", gd.ny_fd + 1, ", ",
            gd.nz_global, ")", " part ", ip, "/", np)

    println(io, "  Physical Domain:  ",
            "ix=1…", gd.nx_pd, ", ",
            "iy=1…", gd.ny_pd, ", ",
            "iz=", gd.iz_min, "…", gd.iz_max, ", ",
            "array size: ", gd.nx_pd, "×", gd.ny_pd, "×", gd.nz_h, " (local) ",
            "or ", gd.nx_pd, "×", gd.ny_pd, "×", gd.nz_global, " (global)")

    println(io, "  Frequency Domain: ",
            "kx=-", gd.nx_fd - 1, "…0…", gd.nx_fd - 1, ", ",
            "ky=-", div(gd.ny_fd, 2), "…0…", div(gd.ny_fd, 2), ", ",
            "iz=", gd.iz_min, "…", gd.iz_max, ", ",
            "array size: ", gd.nx_fd, "×", gd.ny_fd, "×", gd.nz_h, " (local) ",
            "or ", gd.nx_fd, "×", gd.ny_fd, "×", gd.nz_global, " (global)")
end

# create new fields initialized to zero
get_nz(gd::DistributedGrid, ns::NodeSet{:H}) = gd.nz_h
get_nz(gd::DistributedGrid, ns::NodeSet{:V}) = gd.nz_v
zeros_fd(T, gd::DistributedGrid, ns::NodeSet) = zeros(Complex{T}, gd.nx_fd, gd.ny_fd, get_nz(gd, ns))
zeros_pd(T, gd::DistributedGrid, ns::NodeSet) = zeros(T, gd.nx_pd, gd.ny_pd, get_nz(gd, ns))
zeros_fd(T, gd, ns::Symbol) = zeros_fd(T, gd, NodeSet(ns))
zeros_pd(T, gd, ns::Symbol) = zeros_pd(T, gd, NodeSet(ns))


"""
GridMapping defines a transformed vertical coordinate system. It is defined as
a mapping ζ: [0,1] → [0,L₃], such that the coordinate in a Cartesian coordinate
system can be obtained as x₃ = ζ(X₃), where x₃ is the Cartesian
coordinate and X₃ is the coordinate in the transformed system, i.e. ζ is
defined as the function going from normalized transformed coordinates to the
normalized "original" Cartesian coordinate system.  To compute both first and
second derivatives, we also need the derivative of this transform as well as
the first and second derivative of its inverse. We currently require these to
be supplied explicitly and do not check whether they are consistent.
"""
# GridMapping describes how the computational grid relates to the physical domain.
# In horizontal direction, this contains the length of periodicity
struct GridMapping{T}
    hsize1::T
    hsize2::T
    vmap::Function # [0,1] → physical domain
    Dvmap::Function
end

GridMapping(L1::T, L2::T, L3::T) where T =
    GridMapping(L1, L2, ζ -> ζ*L3, ζ -> L3)
GridMapping(L1::T, L2::T, (L3¯, L3⁺)::Tuple{T,T}) where T =
    GridMapping(L1, L2, ζ -> L3¯ + ζ * (L3⁺ - L3¯), x -> L3⁺ - L3¯)
GridMapping(L1::T, L2::T, (x3, dx3dζ)::Tuple{Function,Function}) where T =
    GridMapping(L1, L2, x3, dx3dζ)
Broadcast.broadcastable(gm::GridMapping) = Ref(gm) # to use as argument of elementwise functions

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

instantiate(mapping, L3, one_sided) = instantiate(mapping, (zero(L3), L3), one_sided)

function vrange(T, gd, ::NodeSet{:H}; neighbors=false)
    ζ = LinRange(zero(T), one(T), 2*gd.nz_global+1) # all ζ-values
    imin = 2*gd.iz_min - (neighbors ? 1 : 0)
    imax = 2*gd.iz_max + (neighbors ? 1 : 0)
    ζ[imin:2:imax]
end

function vrange(T, gd, ::NodeSet{:V}; neighbors=false)
    ζ = LinRange(zero(T), one(T), 2*gd.nz_global+1) # all ζ-values
    imin = 2*gd.iz_min+1 - (neighbors ? 1 : 0)
    imax = 2*(gd.iz_min+gd.nz_v-1)+1 + (neighbors ? 1 : 0)
    ζ[imin:2:imax]
end

k1(gd) = [0:gd.nx_fd-1; ]
k2(gd) = [0:div(gd.ny_fd, 2); -div(gd.ny_fd, 2):-1]
wavenumbers(gd) = (k1(gd), k2(gd))
x1(gd, gm::GridMapping{T}) where T = LinRange(zero(T), gm.hsize1, gd.nx_pd+1)[1:end-1]
x2(gd, gm::GridMapping{T}) where T = LinRange(zero(T), gm.hsize2, gd.ny_pd+1)[1:end-1]
x3(gd, gm::GridMapping{T}, ns) where T = (gm.vmap(x) for x=vrange(T, gd, ns))
coords(gd::DistributedGrid, gm::GridMapping, ns::NodeSet) =
    ((x1, x2, x3) for x1=x1(gd, gm), x2=x2(gd, gm), x3=x3(gd, gm, ns))

struct HorizontalTransform{T<:SupportedReals}

    plan_fwd_h::FFTW.rFFTWPlan{T,FFTW.FORWARD,false,3}
    plan_fwd_v::FFTW.rFFTWPlan{T,FFTW.FORWARD,false,3}
    plan_bwd_h::FFTW.rFFTWPlan{Complex{T},FFTW.BACKWARD,false,3}
    plan_bwd_v::FFTW.rFFTWPlan{Complex{T},FFTW.BACKWARD,false,3}

    buffer_pd_h::Array{T,3}
    buffer_pd_v::Array{T,3}
    buffer_fd_h::Array{Complex{T},3}
    buffer_fd_v::Array{Complex{T},3}

    HorizontalTransform(T, nx, ny, nz_h, nz_v) = begin

        buffer_pd_h = zeros(T, nx, ny, nz_h)
        buffer_pd_v = zeros(T, nx, ny, nz_v)
        buffer_fd_h = zeros(Complex{T}, div(nx, 2) + 1, ny, nz_h)
        buffer_fd_v = zeros(Complex{T}, div(nx, 2) + 1, ny, nz_v)

        new{T}(FFTW.plan_rfft(buffer_pd_h, (1,2)),
               FFTW.plan_rfft(buffer_pd_v, (1,2)),
               FFTW.plan_brfft(buffer_fd_h, nx, (1,2)),
               FFTW.plan_brfft(buffer_fd_v, nx, (1,2)),
               buffer_pd_h, buffer_pd_v, buffer_fd_h, buffer_fd_v)
    end
end

HorizontalTransform(T, gd::DistributedGrid; expand=true) = HorizontalTransform(T,
        expand ? gd.nx_pd : 2*gd.nx_fd,
        expand ? gd.ny_pd : gd.ny_fd + 1,
        gd.nz_h, gd.nz_v)
Broadcast.broadcastable(ht::HorizontalTransform) = Ref(ht) # to use as argument of elementwise functions

# utility functions to select the right plans & buffers at compile time
@inline get_plan_fwd(ht, ::NodeSet{:H}) = ht.plan_fwd_h
@inline get_plan_fwd(ht, ::NodeSet{:V}) = ht.plan_fwd_v
@inline get_plan_bwd(ht, ::NodeSet{:H}) = ht.plan_bwd_h
@inline get_plan_bwd(ht, ::NodeSet{:V}) = ht.plan_bwd_v
@inline get_buffer_pd(ht, ::NodeSet{:H}) = ht.buffer_pd_h
@inline get_buffer_pd(ht, ::NodeSet{:V}) = ht.buffer_pd_v
@inline get_buffer_fd(ht, ::NodeSet{:H}) = ht.buffer_fd_h
@inline get_buffer_fd(ht, ::NodeSet{:V}) = ht.buffer_fd_v

"""
Transform a field from the frequency domain to an extended set of nodes in the
physical domain by adding extra frequencies set to zero.
The function optionally takes an array of prefactors which will be multiplied
with the values before the inverse transform. These prefactors don’t have to be
specified along all dimensions since singleton dimensions are broadcast, but
they should be provided as three-dimensional array.
"""
function get_field!(field_pd, ht::HorizontalTransform, field_fd, prefactors, ns::NodeSet)

    # highest frequencies in non-expanded array
    # this assumes that there are no Nyquist frequencies in field_fd
    # while this could be checked in y-direction, the x-direction is ambiguous
    # so we just assume that there is no Nyquist frequency since that’s how the
    # whole code is set up
    kx_max = size(field_fd, 1) - 1 # [0, 1…K]
    ky_max = div(size(field_fd, 2), 2) # [0, 1…K, -K…1]

    # copy frequencies and set the extra frequencies to zero
    buffer = get_buffer_fd(ht, ns)
    @views buffer[1:kx_max+1, 1:ky_max+1, :] .=
        field_fd[:, 1:ky_max+1, :] .*
        prefactors[:, 1:(size(prefactors, 2) == 1 ? 1 : ky_max+1), :]
    @views buffer[1:kx_max+1, end-ky_max+1:end, :] .=
        field_fd[:, ky_max+2:end, :] .*
        prefactors[:, (size(prefactors, 2) == 1 ? 1 : ky_max+2):end, :]
    buffer[1:kx_max+1, ky_max+2:end-ky_max, :] .= 0
    buffer[kx_max+2:end, :, :] .= 0

    # perform ifft (note that this overwrites buffer_big_fd due to the way fftw
    # works, and also note that the factor 1/(nx*ny) is always multiplied during
    # the forward transform and not here)
    LinearAlgebra.mul!(field_pd, get_plan_bwd(ht, ns), buffer)
    field_pd
end

# default when prefactors are missing
get_field!(field_pd, ht::HorizontalTransform, field_fd, ns::NodeSet) =
    get_field!(field_pd, ht, field_fd, ones(eltype(field_pd), (1, 1, 1)), ns::NodeSet)

# convenience function allocating a new array
get_field(field_fd, gd::DistributedGrid, ht::HorizontalTransform{T}, ns::NodeSet) where T =
    get_field!(zeros_pd(T, gd, ns), ht, field_fd, ns)

"""
Transform a field from an extended set of nodes in physical domain back to the
frequency domain and remove extra frequencies.
"""
function set_field!(field_fd, ht::HorizontalTransform, field_pd::Array, ns::NodeSet)

    nx, ny, nz = size(field_fd)
    nx_pd, ny_pd, nz_pd = size(field_pd)
    buffer = get_buffer_fd(ht, ns)

    # safety check to avoid hidden issues later on
    nx_pd >= 2 * nx && ny_pd > ny ||
            error("Physical domain does not contain enough information for FFT.")
    isodd(ny) || error("Target field of FFT includes Nyquist frequency.")
    nz == nz_pd == size(buffer,3) ||
            error("Arrays for FFT do not have the same number of vertical levels")

    LinearAlgebra.mul!(buffer, get_plan_fwd(ht, ns), field_pd)

    # fft normalization factor 1/(nx*ny) is applied whenever forward transform
    # is performed, such that the first frequency corresponds to the average
    # value
    fft_factor = 1 / (nx_pd * ny_pd)
    ky_max = div(ny,2) # [0, 1…kmax, -kmax…1]

    for i in CartesianIndices(field_fd)
        field_fd[i] = i[2] <= 1+ky_max ? buffer[i] * fft_factor :
                buffer[i[1], i[2] + (ny_pd - ny), i[3]] * fft_factor
    end
    field_fd
end

function set_field!(field_fd, field_pd::Function, gd, gm, ht::HorizontalTransform, ns::NodeSet)
    buffer = get_buffer_pd(ht, ns)
    for (i, x) in zip(CartesianIndices(buffer), coords(gd, gm, ns))
        buffer[i] = field_pd(x...)
    end
    set_field!(field_fd, ht, buffer, ns)
end

# convenience function allocating a new array
set_field(field_pd::Function, gd, gm::GridMapping{T}, ht::HorizontalTransform, ns::NodeSet) where T =
    set_field!(zeros_fd(T, gd, ns), field_pd, gd, gm, ht, ns)
