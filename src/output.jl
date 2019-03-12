max_dt_diffusion(coeff, dx, dy, dz) = min(dx, dy, dz)^2 / coeff

function global_maximum(field::Array{T}) where {T<:SupportedReals}
    # specifying T avoids accidentially taking the maximum in Fourier space
    local_max = mapreduce(abs, max, field)
    MPI.Initialized() ? MPI.Allreduce(local_max, MPI.MAX, MPI.COMM_WORLD) : local_max
end

global_sum(Ns) = MPI.Initialized() ? MPI.Allreduce(sum(Ns), MPI.SUM, MPI.COMM_WORLD) : sum(Ns)

function max_dt_advection(vel, grid_spacing)
    max_vel = map(global_maximum, vel)
    maximum(grid_spacing ./ max_vel)
end

function sequential_open(f, filename, mode)
    id, N = proc_id()
    r = nothing
    for i=1:N
        if i == id
            r = open(f, filename, mode)
        end
        MPI.Initialized() && MPI.Barrier(MPI.COMM_WORLD)
    end
    r
end

function write_field(filename, field, x, y, z, domain_size::NTuple{3}; output_type=eltype(field))

    # build first UInt64 value, describing the version of the file format that is used
    magic_number = 0xFFFFFFFF00000000 # TODO: define a good magic number
    precision = output_type == Float64 ? 64 : output_type == Float32 ? 32 :
            error("Only 64-bit and 32-bit precision is supported.")

    # make sure the dimensions are compatible
    Nx, Ny, Nz = size(field)
    Nz = global_sum(Nz)
    length(x) == Nx || error("Length of x-values incompatible with dimensions of field.")
    length(y) == Ny || error("Length of y-values incompatible with dimensions of field.")
    length(z) == Nz || error("Length of z-values incompatible with dimensions of field.")

    sequential_open(filename, "a") do f
        if proc_id()[1] == 1
            write(f, magic_number | precision)
            write(f, collect(UInt64, (Nx, Ny, Nz)))
            write(f, collect(Float64, domain_size))
            write(f, convert.(Float64, x))
            write(f, convert.(Float64, y))
            write(f, convert.(Float64, z))
        end
        write(f, convert.(output_type, field))
    end
end

function read_field(filename, nodeset::NodeSet)

    magic_number = 0xFFFFFFFF00000000 # TODO: define a good magic number
    grid_points = zeros(UInt64, 3)
    domain_size = zeros(Float64, 3)

    sequential_open(filename, "r") do f

        # parse first values specifying data format
        identifier = read(f, UInt64)
        identifier & 0xFFFFFFFF00000000 == magic_number || error("Invalid file format")
        variant = identifier & 0x00000000FFFFFFFF
        T = variant == 64 ? Float64 : variant == 32 ? Float32 : error("Invalid data format")

        read!(f, grid_points)
        read!(f, domain_size)

        x = zeros(Float64, grid_points[1])
        y = zeros(Float64, grid_points[2])
        z = zeros(Float64, grid_points[3])
        read!(f, x)
        read!(f, y)
        read!(f, z)

        iz_min, iz_max = vertical_range(grid_points[3], nodeset)
        data = zeros(T, grid_points[1], grid_points[2], 1+iz_max-iz_min)
        bytes_per_value = T == Float64 ? 8 : T == Float32 ? 4 :
            error("Bytes per value not known for type ", T)
        skip(f, grid_points[1] * grid_points[2] * (iz_min-1) * bytes_per_value)
        read!(f, data)

        Tuple(domain_size), x, y, z, data
    end
end

function write_field(filename, field::Array{Complex{T}}, ds::NTuple{3},
        ht::HorizontalTransform{T}, sf::NTuple{2}, ns::NodeSet;
        shift=true, output_type=T) where {T}

    buffer_fd = get_buffer_fd(ht, ns)
    buffer_pd = get_buffer_pd(ht, ns)

    if shift
        broadcast!(*, buffer_fd, field, sf[1], sf[2])
    else
        buffer_fd .= field
    end

    shifth, shiftv = (shift, ns isa NodeSet{:H})
    gs = (size(buffer_pd, 1), size(buffer_pd, 2), global_sum(size(buffer_pd, 3)) + (shiftv ? 0 : 1))

    # when the indices are not shifted, the H-nodes start at zero
    # but the V-nodes start at Δz
    x = LinRange(0, ds[1], 2*gs[1]+1)[(shifth ? 2 : 1):2:end-1]
    y = LinRange(0, ds[2], 2*gs[2]+1)[(shifth ? 2 : 1):2:end-1]
    z = LinRange(0, ds[3], 2*gs[3]+1)[(shiftv ? 2 : 3):2:end-1]

    LinearAlgebra.mul!(buffer_pd, get_plan_bwd(ht, ns), buffer_fd)
    write_field(filename, buffer_pd, x, y, z, ds, output_type=output_type)
end

function shift_factors(T, nx_pd, ny_pd)
    nkx = div(nx_pd, 2)
    nky = div(ny_pd, 2)
    kx = 0:nkx
    ky = vcat(0:nky, (isodd(ny_pd) ? -nky : -nky + 1):-1)
    shiftx = Complex{T}[exp(1im * kx * π / nx_pd) for kx=kx, ky=1:1, z=1:1]
    shifty = Complex{T}[exp(1im * ky * π / ny_pd) for kx=1:1, ky=ky, z=1:1]
    shiftx[nkx+1] *= (iseven(nx_pd) ? 0 : 1)
    shifty[nky+1] *= (iseven(ny_pd) ? 0 : 1)
    shiftx, shifty
end

"""
Return a tuple with the number of digits before and after the decimal point
needed to describe `x` in a meaningful way. This algorithm is an attempt at a
heuristic to format decimal numbers in a useful way.
"""
function relevant_digits(x)

    s1, s2 = split(strip(Printf.@sprintf("%.15f", x)[1:end-1], '0'), '.')
    periodic_part = ""
    leading_zeros = 0
    i = 1
    repetitions = 1
    for c=s2
        if isempty(periodic_part)
            if c == '0'
                leading_zeros += 1
            else
                periodic_part *= c
            end
        elseif periodic_part[i] == c
            if i == length(periodic_part)
                repetitions += 1
                i = 1
            else
                i += 1
            end
        else # character is not matching the periodic part
            periodic_part ^= repetitions
            periodic_part *= periodic_part[1:i-1] * c
            repetitions = 1
            i = 1
        end
    end

    Np = length(periodic_part)
    N = leading_zeros + (repetitions == 1 ? (Np < 5 ? Np : 3) : (Np == 1 ? 2 : Np == 2 ? 4 : 3))
    max(1, length(s1)), N
end

relevant_digits(xs::Array) = isempty(xs) ? (0,0) :
        mapreduce(relevant_digits, (x,y) -> max.(x,y), vcat(xs, abs.(diff(xs))))

function format_timestamp(timestamp, tsdigits)
    tint = floor(Int, timestamp)
    tdec = timestamp - tint
    "t" * string(tint, pad=tsdigits[1]) * (tsdigits[2] == 0 ? "" : Printf.@sprintf("%.15f", tdec)[2:end-15+tsdigits[2]])
end

function format_dirname(basedir, i, timestamps, tsdigits)
    prefix = "snapshot"
    id = string(i-1, pad=ndigits(length(timestamps)-1))
    timestamp = format_timestamp(timestamps[i], tsdigits)
    dirname = join((prefix, id, timestamp), '-')
    joinpath(basedir, dirname)
end

struct OutputCache{T}

    domain_size::NTuple{3,T}
    transform::HorizontalTransform{T}
    shift_factors::NTuple{2,Array{Complex{T},3}}
    snapshot_dir::String
    snapshot_counter::Array{Int,1}
    snapshot_timestamps::Array{T,1}
    snapshot_tsdigits::NTuple{2,Int}

    function OutputCache(gd::DistributedGrid, ds::NTuple{3,T}, dt::T,
                         snapshot_steps::Array{Int,1}, snapshot_dir) where {T}
        timestamps = snapshot_steps .* dt
        new{T}(ds, HorizontalTransform(T, gd, expand=false),
                shift_factors(T, 2*gd.nx_fd-1, gd.ny_fd), snapshot_dir, [0],
                timestamps, relevant_digits(timestamps))
    end
end

function next_snapshot_time(output::OutputCache)
    ts = output.snapshot_timestamps
    i = output.snapshot_counter[1] + 1
    i > length(ts) ? Inf : ts[i]
end

function write_snapshot(output::OutputCache, state, t)
    output.snapshot_counter[1] += 1
    output.snapshot_timestamps[output.snapshot_counter[1]] ≈ t || error("Snapshot output out of sync")
    dir = format_dirname(output.snapshot_dir, output.snapshot_counter[1],
                         output.snapshot_timestamps, output.snapshot_tsdigits)
    mkpath(dir)
    write_field(joinpath(dir, "u.cbd"), state[1], output.domain_size, output.transform, output.shift_factors, NodeSet(:H))
    write_field(joinpath(dir, "v.cbd"), state[2], output.domain_size, output.transform, output.shift_factors, NodeSet(:H))
    write_field(joinpath(dir, "w.cbd"), state[3], output.domain_size, output.transform, output.shift_factors, NodeSet(:V))
end

function log_state!(output::OutputCache, state, t)
    next_snapshot_time(output) ≈ t && write_snapshot(output, state, t)
end
