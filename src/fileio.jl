module CBD

const MAGIC_NUMBER = hex2bytes("CBDF")
const CBD_VERSION = 1

using MPI: MPI

function sequential_open(f, filename, mode, comm)
    N, id = MPI.Initialized() ? (MPI.Comm_size(comm), MPI.Cart_coords(comm)[] + 1) : (1, 1)

    if id == 1 && !(mode in ("r", "r+"))
        mkpath(dirname(filename))
    end
    MPI.Initialized() && MPI.Barrier(comm)

    r = nothing
    for i in 1:N
        if i == id
            r = open(f, filename, mode)
        end
        MPI.Initialized() && MPI.Barrier(comm)
    end
    r
end

function writecbd(::Type{T}, filename, field, x1, x2, x3, xmin, xmax, comm) where {T}

    # build first value, describing the version of the file format that is used
    identifier = zeros(UInt8, 8)
    identifier[1:2] = MAGIC_NUMBER
    identifier[3] = CBD_VERSION
    identifier[end] = T == Float64 ? 0x8 : T == Float32 ? 0x4 : error("Only 64-bit and 32-bit precision is supported.")

    # make sure the dimensions are compatible
    n1, n2, n3 = size(field)
    length(x1) == n1 || error("Length of x1-values incompatible with dimensions of field.")
    length(x2) == n2 || error("Length of x2-values incompatible with dimensions of field.")
    length(x3) == n3 || error("Length of x3-values incompatible with dimensions of field.")

    # write file header at root process only
    n3global = MPI.Initialized() ? MPI.Allreduce(n3, +, MPI.COMM_WORLD) : n3
    if !MPI.Initialized() || MPI.Cart_coords(comm)[] == 0
        mkpath(dirname(filename))
        open(filename, "w") do f
            write(f, identifier)
            write(f, collect(UInt64, (n1, n2, n3global)))
            write(f, collect(Float64, xmin))
            write(f, collect(Float64, xmax))
            write(f, collect(Float64, x1))
            write(f, collect(Float64, x2))
        end
    end

    # write x3-values sequentially
    sequential_open(filename, "a", comm) do f
        write(f, collect(Float64, x3))
    end

    # write main data sequentially
    sequential_open(filename, "a", comm) do f
        write(f, convert(Array{T}, field))
    end
end

function readcbd(filename, x3range, comm; tol = 1e-9)
    grid_points = zeros(UInt64, 3)
    xmin = zeros(Float64, 3)
    xmax = zeros(Float64, 3)

    sequential_open(filename, "r", comm) do f

        # parse identifier and select precision
        identifier = read!(f, zeros(UInt8, 8))
        identifier[1:2] == MAGIC_NUMBER || @error "Not a CBD file (magic number not matching)." identifier[1:2]
        identifier[3] == CBD_VERSION || @error "Unsupported version of CBD format." identifier[3]
        T =
            identifier[end] == 0x4 ? Float32 :
            identifier[end] == 0x8 ? Float64 :
            @error "Unsupported data type: Only 32-bit and 64-bit precision is supported."
        bytes_per_value = T == Float64 ? 8 : T == Float32 ? 4 : error("Bytes per value not known for type ", T)

        read!(f, grid_points)
        read!(f, xmin)
        read!(f, xmax)

        x1 = zeros(Float64, grid_points[1])
        x2 = zeros(Float64, grid_points[2])
        x3 = zeros(Float64, grid_points[3])
        read!(f, x1)
        read!(f, x2)
        read!(f, x3)

        # determine index range to read based on requested interval
        i3min, i3max = if isempty(x3range)
            (1, 0) # process might request no data at all
        else
            x3min = max(first(x3), first(x3range))
            x3max = min(last(x3), last(x3range))
            (findlast(x -> x < x3min + tol, x3), findfirst(x -> x > x3max - tol, x3))
        end

        data = zeros(T, grid_points[1], grid_points[2], 1 + i3max - i3min)
        skip(f, grid_points[1] * grid_points[2] * (i3min - 1) * bytes_per_value)
        read!(f, data)

        Tuple(xmin), Tuple(xmax), x1, x2, x3[i3min:i3max], data
    end
end

# for serial i/o, the communicator (and the local x3-range) can be omitted
writecbd(T, filename, field, x1, x2, x3, xmin, xmax) = writecbd(T, filename, field, x1, x2, x3, xmin, xmax, nothing)
readcbd(filename; kwargs...) = readcbd(filename, (-Inf, Inf), nothing; kwargs...)

end # module CBD
