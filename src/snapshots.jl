const CBD_MAGIC_NUMBER = hex2bytes("CBDF")
const CBD_VERSION = 1

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

write_field(filename, field::Array{T}, x, y, z, domain_min, domain_max) where T =
    write_field(eltype(field), filename, field, x, y, z, domain_min, domain_max)

function write_field(T, filename, field, x, y, z, domain_min, domain_max)

    # build first value, describing the version of the file format that is used
    identifier = zeros(UInt8, 8)
    identifier[1:2] = CBD_MAGIC_NUMBER
    identifier[3]   = CBD_VERSION
    identifier[end] = T == Float64 ? 0x8 : T == Float32 ? 0x4 :
            error("Only 64-bit and 32-bit precision is supported.")

    # make sure the dimensions are compatible
    Nx, Ny, Nz = size(field)
    Nz = global_sum(Nz)
    length(x) == Nx || error("Length of x-values incompatible with dimensions of field.")
    length(y) == Ny || error("Length of y-values incompatible with dimensions of field.")
    length(z) == Nz || error("Length of z-values incompatible with dimensions of field.")

    sequential_open(filename, "a") do f
        if proc_id()[1] == 1
            write(f, identifier)
            write(f, collect(UInt64, (Nx, Ny, Nz)))
            write(f, collect(Float64, domain_min))
            write(f, collect(Float64, domain_max))
            write(f, convert(Array{Float64}, x))
            write(f, convert(Array{Float64}, y))
            write(f, convert(Array{Float64}, z))
        end
        write(f, convert(Array{T}, field))
    end
end

function read_field(filename, nodeset::NodeSet)

    grid_points = zeros(UInt64, 3)
    domain_min = zeros(Float64, 3)
    domain_max = zeros(Float64, 3)

    sequential_open(filename, "r") do f

        # parse identifier and select precision
        identifier = read!(f, zeros(UInt8, 8))
        identifier[1:2] == CBD_MAGIC_NUMBER ||
            @error "Not a CBD file (magic number not matching)." identifier[1:2]
        identifier[3] == CBD_VERSION ||
            @error "Unsupported version of CBD format." identifier[3]
        T = identifier[end] == 0x4 ? Float32 : identifier[end] == 0x8 ? Float64 :
            @error "Unsupported data type: Only 32-bit and 64-bit precision is supported."

        read!(f, grid_points)
        read!(f, domain_min)
        read!(f, domain_max)

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

        Tuple(domain_min), Tuple(domain_max), x, y, z, data
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
    write_field(output_type, filename, buffer_pd, x, y, z,
                (zero(T), zero(T), zero(T)), ds)
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

function next_snapshot_time(output::OutputCache)
    ts = output.snapshot_timestamps
    i = output.snapshot_counter[] + 1
    i > length(ts) ? Inf : ts[i]
end

function write_state(dir, state, domain_size, transform, shift_factors)
    mkpath(dir)
    write_field(joinpath(dir, "u.cbd"), state[1], domain_size, transform, shift_factors, NodeSet(:H))
    write_field(joinpath(dir, "v.cbd"), state[2], domain_size, transform, shift_factors, NodeSet(:H))
    write_field(joinpath(dir, "w.cbd"), state[3], domain_size, transform, shift_factors, NodeSet(:V))
end

function write_snapshot(output::OutputCache, state, t)
    output.snapshot_counter[] += 1
    output.snapshot_timestamps[output.snapshot_counter[]] ≈ t || error("Snapshot output out of sync")
    dir = format_dirname(output.snapshot_dir, output.snapshot_counter[],
                         output.snapshot_timestamps, output.snapshot_tsdigits)
    write_state(dir, state, output.domain_size, output.transform, output.shift_factors)
end
