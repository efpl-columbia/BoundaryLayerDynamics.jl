SupportedReals = Union{Float32,Float64}

struct NodeSet{NS}
    NodeSet(ns::Symbol) = ns == :H ? new{:H}() : ns == :V ? new{:V}() :
        error("Invalid NodeSet: $(NS) (only :H and :V are allowed)")
end

abstract type ProcType end
struct SingleProc <: ProcType end
struct MinProc    <: ProcType end
struct InnerProc  <: ProcType end
struct MaxProc    <: ProcType end
LowestProc  = Union{SingleProc, MinProc}
HighestProc = Union{SingleProc, MaxProc}

proc_below() = (MPI.Initialized() ? MPI.Comm_rank(MPI.COMM_WORLD) : 0) - 1
proc_above() = (MPI.Initialized() ? MPI.Comm_rank(MPI.COMM_WORLD) : 0) + 1
proc_id() = MPI.Initialized() ? (MPI.Comm_rank(MPI.COMM_WORLD) + 1,
        MPI.Comm_size(MPI.COMM_WORLD)) : (1,1)
proc(i, n) = (i,n) == (1,1) ? SingleProc() : i == 1 ? MinProc() :
             i == n ? MaxProc() : InnerProc()
proc_type(p::P) where {P<:ProcType} = P
proc_type() = proc_type(proc(proc_id()...))

global_maximum(val::T) where {T<:Real} =
        MPI.Initialized() ? MPI.Allreduce(val, MPI.MAX, MPI.COMM_WORLD) : val

function global_maximum(field::Array{T}) where {T<:SupportedReals}
    # specifying T avoids accidentially taking the maximum in Fourier space
    global_maximum(mapreduce(abs, max, field))
end

global_sum(Ns) = MPI.Initialized() ? MPI.Allreduce(sum(Ns), MPI.SUM, MPI.COMM_WORLD) : sum(Ns)

struct DistributedGrid{P<:ProcType}
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
vertical_range(nz_global_h, ns::NodeSet{:H}) =
    (vr = vertical_range(nz_global_h, proc_id()...); (vr[1], vr[2]))
vertical_range(nz_global_v, ns::NodeSet{:V}) =
    (vr = vertical_range(nz_global_v+1, proc_id()...); (vr[1], vr[1]+vr[4]-1))

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

    DistributedGrid{proc_type()}(nx_fd, nx_pd, ny_fd, ny_pd,
            nz_local_h, nz_local_v, nz, iz_min, iz_max)
end

function DistributedGrid(nx, ny, nz)
    if MPI.Initialized()
        comm = MPI.COMM_WORLD
        mpi_size = MPI.Comm_size(comm)
        mpi_rank = MPI.Comm_rank(comm)
        DistributedGrid(nx, ny, nz, mpi_rank + 1, mpi_size)
    else
        DistributedGrid(nx, ny, nz, 1, 1)
    end
end

DistributedGrid(n) = DistributedGrid(n, n, n)

function Base.show(io::IO, gd::DistributedGrid{P}) where {P}

    ip = MPI.Initialized() ? MPI.Comm_rank(MPI.COMM_WORLD) + 1 : 1
    np = MPI.Initialized() ? MPI.Comm_size(MPI.COMM_WORLD) : 1

    println(io, "DistributedGrid(", gd.nx_fd * 2,  ", ", gd.ny_fd + 1, ", ",
            gd.nz_global, ")", " part ", ip, "/", np, " (", P, ")")

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
@inline zeros_fd(T, gd::DistributedGrid, ns::NodeSet) =
        zeros(Complex{T}, gd.nx_fd, gd.ny_fd, get_nz(gd, ns))
@inline zeros_pd(T, gd::DistributedGrid, ns::NodeSet) =
        zeros(T, gd.nx_pd, gd.ny_pd, get_nz(gd, ns))

abstract type BoundaryCondition{P<:ProcType,T} end
struct DirichletBC{P,T} <: BoundaryCondition{P,T}
    value::T
    buffer_fd::Array{Complex{T},2}
    buffer_pd::Array{T,2}
    neighbor_below::Int
    neighbor_above::Int
    DirichletBC(gd::DistributedGrid, value::T) where T =
        new{proc_type(),T}(value, zeros(Complex{T}, gd.nx_fd, gd.ny_fd),
        zeros(T, gd.nx_pd, gd.ny_pd), proc_below(), proc_above())
end
struct NeumannBC{P,T} <: BoundaryCondition{P,T}
    value::T
    buffer_fd::Array{Complex{T},2}
    buffer_pd::Array{T,2}
    neighbor_below::Int
    neighbor_above::Int
    NeumannBC(gd::DistributedGrid, value::T) where T =
        new{proc_type(),T}(value, zeros(Complex{T}, gd.nx_fd, gd.ny_fd),
        zeros(T, gd.nx_pd, gd.ny_pd), proc_below(), proc_above())
end

bc_noslip(T, gd) = (DirichletBC(gd, zero(T)),
                    DirichletBC(gd, zero(T)),
                    DirichletBC(gd, zero(T)))
bc_freeslip(T, gd) = (NeumannBC(gd, zero(T)),
                      NeumannBC(gd, zero(T)),
                    DirichletBC(gd, zero(T)))

layers(field::Array{T,3}) where T =
        Tuple(view(field, :, :, iz) for iz=1:size(field,3))

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
        expand ? gd.nx_pd : 2*gd.nx_fd-1,
        expand ? gd.ny_pd : gd.ny_fd,
        gd.nz_h, gd.nz_v)

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
"""
function get_field!(field_pd, ht::HorizontalTransform, field_fd, ns::NodeSet)

    # highest frequencies in non-expanded array
    # this assumes that there are no Nyquist frequencies in field_hat
    ny, ny_pd = size(field_fd,2), size(field_pd,2)
    kx_max = size(field_fd,1) - 1 # [0, 1…kmax]
    ky_max = div(ny,2) # [0, 1…kmax, -kmax…1]

    # copy frequencies such that the extra frequencies are zero
    buffer = get_buffer_fd(ht, ns)
    for i in CartesianIndices(buffer)
        if 1+kx_max < i[1] || 1+ky_max < i[2] <= ny_pd - ky_max
            buffer[i] = 0
        else
            buffer[i] = i[2] <= 1+ky_max ? field_fd[i] :
                field_fd[i[1], i[2] - (ny_pd - ny), i[3]]
        end
    end

    # perform ifft (note that this overwrites buffer_big_fd due to the way fftw
    # works, and also note that the factor 1/(nx*ny) is always multiplied during
    # the forward transform and not here)
    LinearAlgebra.mul!(field_pd, get_plan_bwd(ht, ns), buffer)
    field_pd
end

get_field(gd::DistributedGrid, ht::HorizontalTransform{T}, field_fd, ns::NodeSet) where T=
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

# nodes generally start at zero, vertical direction is centered for H-nodes
@inline coord(i, δ, ::NodeSet{:H}) = (δ[1] * (i[1]-1),
                                      δ[2] * (i[2]-1),
                                      δ[3] * (2*i[3]-1)/2)
@inline coord(i, δ, ::NodeSet{:V}) = (δ[1] * (i[1]-1),
                                      δ[2] * (i[2]-1),
                                      δ[3] * (i[3]))

function set_field!(field_fd, ht::HorizontalTransform, field_pd::Function,
        grid_spacing, iz_min, ns::NodeSet)
    buffer = get_buffer_pd(ht, ns)
    for i in CartesianIndices(buffer)
        buffer[i] = field_pd(coord((i[1], i[2], i[3] + iz_min - 1), grid_spacing, ns)...)
    end
    set_field!(field_fd, ht, buffer, ns)
end

set_field(gd::DistributedGrid, ht::HorizontalTransform{T}, field_pd::Function,
        grid_spacing, ns::NodeSet) where {T} =
        set_field!(zeros_fd(T, gd, ns), ht, field_pd, grid_spacing, gd.iz_min, ns)

struct DerivativeFactors{T<:SupportedReals}
    dx1::Array{Complex{T},2}
    dy1::Array{Complex{T},2}
    dz1::T
    dx2::Array{T,2}
    dy2::Array{T,2}
    dz2::T
    DerivativeFactors(gd::DistributedGrid, domain_size::Tuple{T,T,T}) where T = begin
        kx = [0:gd.nx_fd-1; ]
        ky = [0:div(gd.ny_fd, 2); -div(gd.ny_fd, 2):-1]
        new{T}(
            reshape(1im * kx * (2π/domain_size[1]), (gd.nx_fd, 1)),
            reshape(1im * ky * (2π/domain_size[2]), (1, gd.ny_fd)),
            gd.nz_global / domain_size[3], # 1/δz
            reshape( - kx.^2 * (2π/domain_size[1])^2, (gd.nx_fd, 1)),
            reshape( - ky.^2 * (2π/domain_size[2])^2, (1, gd.ny_fd)),
            gd.nz_global^2 / domain_size[3]^2, # 1/δz²
    )
    end
end
