function test_file_io(; n1=6, n2=8, n3=12)

    Random.seed!(73330459108) # same seed for each process

    mktempdir_parallel() do p

        for T=(Float32, Float64), ns=(:C, :I)

            ds = (rand(T), rand(T), rand(T))
            vel0(x, y, z) = x / ds[1] + y / ds[2] + z / ds[3]
            xmin, xmax = (zero(T), zero(T), zero(T)), ds

            domain = Domain(ds, FreeSlipBoundary(), FreeSlipBoundary())
            grid = BLD.Grid((n1, n2, n3))
            pdims = BLD.PhysicalSpace.default_size(grid)
            transform = BLD.PhysicalSpace.Transform2D(T, pdims)

            x1 = collect(LinRange(0, ds[1], pdims[1]+1)[1:pdims[1]])
            x2 = collect(LinRange(0, ds[2], pdims[2]+1)[1:pdims[2]])
            x3 = collect(LinRange(0, ds[3], 2*n3+1)[(ns == :C ? 2 : 3):2:end-1])
            x3local = x3[grid.i3min:min(grid.i3max, length(x3))]

            fvel = BLD.PhysicalSpace.set_field!(vel0, zeros(T, grid, NS(ns)),
                                                transform, domain, grid, NS(ns))
            vel = BLD.PhysicalSpace.get_field(transform, fvel)

            for T_files=(Float32, Float64)

                fn = joinpath(p, string("vel-", T_files, "-from-", T, "-", ns, ".cbd"))
                BLD.CBD.writecbd(T_files, fn, vel, x1, x2, x3local, xmin, xmax, grid.comm)

                xmin_after, xmax_after, x1_after, x2_after, x3_after, vel_after =
                    BLD.CBD.readcbd(fn, x3local, grid.comm)

                @test xmin_after == xmin
                @test xmax_after == xmax
                @test x1_after == x1
                @test x2_after == x2
                @test global_vector(x3_after) == x3

                if T == T_files
                    @test vel_after == vel
                else
                    @test vel_after ≈ vel
                end
            end
        end
    end
end

function test_shifted_file_output(T=Float64; n1=6, n2=8, n3=12)

    Random.seed!(839905175168) # same seed for each process
    k1max, k2max = div(n1 - 1, 2), div(n2 - 1, 2)
    C0, Cx, Cy = rand(T), rand(T, 2, k1max), rand(T, 2, k2max)
    vel0(x, y, z) = ( sum(Cx[1,i] * sin(i * x + Cx[2,i]) for i=1:k1max)
                    + sum(Cy[1,i] * sin(i * y + Cy[2,i]) for i=1:k1max)
                    + C0 ) * z * (1-z)

    ds = convert.(T, (2π, 2π, 1))
    xmin, xmax = (zero(T), zero(T), zero(T)), ds
    domain = Domain(ds, FreeSlipBoundary(), FreeSlipBoundary())

    grid = BLD.Grid((n1, n2, n3))
    pdims = BLD.PhysicalSpace.default_size(grid)
    x1 = collect(LinRange(0, ds[1], 2*pdims[1]+1)[2:2:end-1])
    x2 = collect(LinRange(0, ds[2], 2*pdims[2]+1)[2:2:end-1])
    x3 = collect(LinRange(0, ds[3], 2*n3+1)[2:end-1]) # both sets of nodes
    #x3local = x3[grid.i3min:min(grid.i3max, length(x3))]

    transform = BLD.PhysicalSpace.Transform2D(T, pdims)

    for ns = (:C, :I)

        fvel = BLD.PhysicalSpace.set_field!(vel0, zeros(T, grid, NS(ns)),
                                            transform, domain, grid, NS(ns))
        vel = BLD.PhysicalSpace.get_field(transform, fvel, centered = true)

        x3ns = x3[(ns == :C ? 1 : 2):2:end]
        x3local = x3ns[grid.i3min:min(grid.i3max, length(x3ns))]

        xmin_file, xmax_file, x1_file, x2_file, x3_file, data_file = mktempdir_parallel() do p
            fn = joinpath(p, "vel-$ns.cbd")
            BLD.CBD.writecbd(T, fn, vel, x1, x2, x3local, xmin, xmax, grid.comm)
            BLD.CBD.readcbd(fn, x3local, grid.comm)
        end

        @test xmin_file == xmin
        @test xmax_file == xmax
        @test x1_file == x1
        @test x2_file == x2
        @test global_vector(x3_file) == x3ns
        @test data_file ≈ [vel0(x1, x2, x3) for x1=x1_file, x2=x2_file, x3=x3local]
    end
end

@timeit "Output" @testset "File Output" begin
    test_file_io()
    test_shifted_file_output()

    # also test the parallel version with one layer per process
    MPI.Initialized() && test_file_io(n3=MPI.Comm_size(MPI.COMM_WORLD))
    MPI.Initialized() && test_shifted_file_output(n3=MPI.Comm_size(MPI.COMM_WORLD))
end
