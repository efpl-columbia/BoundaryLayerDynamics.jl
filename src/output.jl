max_dt_diffusion(coeff, dx, dy, dz) = min(dx, dy, dz)^2 / coeff

function global_maximum(field::Array{T}) where {T<:SupportedReals}
    # specifying T avoids accidentially taking the maximum in Fourier space
    local_max = mapreduce(abs, max, field)
    MPI.Initialized() ? MPI.Allreduce(local_max, MPI.MAX, MPI.COMM_WORLD) : local_max
end

global_sum(N) = MPI.Initialized() ? MPI.Allreduce(N, MPI.SUM, MPI.COMM_WORLD) : N

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
