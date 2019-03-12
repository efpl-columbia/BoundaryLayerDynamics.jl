function test_file_io(; Nx=6, Ny=8, Nz=12)

    Random.seed!(73330459108) # same seed for each process

    mktempdir_parallel() do p

        for T=(Float32, Float64), ns=(:H, :V)

            ds = (rand(T), rand(T), rand(T))
            vel0(x, y, z) = x / ds[1] + y / ds[2] + z / ds[3]
            dmin, dmax = (zero(T), zero(T), zero(T)), ds

            gd = CF.DistributedGrid(Nx, Ny, Nz)
            gs = ds ./ (gd.nx_pd, gd.ny_pd, gd.nz_global)
            df = CF.DerivativeFactors(gd, ds)
            ht = CF.HorizontalTransform(T, gd)
            lbcs = CF.bc_noslip(T, gd)
            ubcs = CF.bc_noslip(T, gd)

            x = LinRange(0, ds[1], gd.nx_pd+1)[1:gd.nx_pd]
            y = LinRange(0, ds[2], gd.ny_pd+1)[1:gd.ny_pd]
            z = LinRange(0, ds[3], 2*Nz+1)[(ns == :H ? 2 : 3):2:end-1]

            vel_fd = CF.set_field(gd, ht, vel0, gs, CF.NodeSet(ns))
            vel_pd = CF.get_field(gd, ht, vel_fd, CF.NodeSet(ns))

            for T_files=(Float32, Float64)

                fn = joinpath(p, string("vel-", T_files, "-from-", T, "-", ns, ".bin"))
                CF.write_field(T_files, fn, vel_pd, x, y, z, dmin, dmax)
                dmin_after, dmax_after, x_after, y_after, z_after, vel_after = CF.read_field(fn, CF.NodeSet(ns))

                @test dmin_after == dmin
                @test dmax_after == dmax
                @test x_after == x
                @test y_after == y
                @test z_after == z

                if T == T_files
                    @test vel_after == vel_pd
                else
                    @test vel_after ≈ vel_pd
                end
            end
        end
    end
end

function test_shifted_file_output(T=Float64; Nx=6, Ny=8, Nz=12)

    # reduced number of frequencies (number of frequencies stored in file)
    Nx¯ = div(Nx - 1, 2)
    Ny¯ = div(Ny - 1, 2)

    # extended number of frequencies (that can be represented during initialization)
    Nx⁺ = div(floor(Int, Nx / 2 * 3), 2)
    Ny⁺ = div(floor(Int, Ny / 2 * 3), 2)

    Random.seed!(839905175168) # same seed for each process
    C0, Cx, Cy = rand(T), rand(T, 2, Nx⁺), rand(T, 2, Ny⁺)
    vel0(x, y, z) = ( sum(Cx[1,i] * sin(i * x + Cx[2,i]) for i=1:Nx⁺)
                    + sum(Cy[1,i] * sin(i * y + Cy[2,i]) for i=1:Ny⁺)
                    + C0 ) * z * (1-z)
    vel0_file(x, y, z) = ( sum(Cx[1,i] * sin(i * x + Cx[2,i]) for i=1:Nx¯)
                         + sum(Cy[1,i] * sin(i * y + Cy[2,i]) for i=1:Ny¯)
                         + C0 ) * z * (1-z)

    ds = convert.(T, (2π, 2π, 1))
    dmin, dmax = (zero(T), zero(T), zero(T)), ds
    gd = CF.DistributedGrid(Nx, Ny, Nz)
    gs = ds ./ (gd.nx_pd, gd.ny_pd, gd.nz_global)
    df = CF.DerivativeFactors(gd, ds)
    ht = CF.HorizontalTransform(T, gd)
    lbcs = CF.bc_noslip(T, gd)
    ubcs = CF.bc_noslip(T, gd)

    nx_file = 2*gd.nx_fd-1
    ny_file = gd.ny_fd
    sf_file = CF.shift_factors(T, nx_file, ny_file)
    ht_file = CF.HorizontalTransform(T, gd, expand=false)

    for NS = (:H, :V)

        ns = CF.NodeSet(NS)
        vel_fd = CF.set_field(gd, ht, vel0, gs, ns)

        dmin_file, dmax_file, x_file, y_file, z_file, data_file = mktempdir_parallel() do p
            fn = joinpath(p, "vel.cbd")
            CF.write_field(fn, vel_fd, ds, ht_file, sf_file, ns)
            CF.read_field(fn, ns)
        end

        @test collect(dmin_file) ≈ collect(dmin)
        @test collect(dmax_file) ≈ collect(dmax)
        @test x_file ≈ LinRange(0, ds[1], 2*nx_file+1)[2:2:end]
        @test y_file ≈ LinRange(0, ds[2], 2*ny_file+1)[2:2:end]
        @test z_file ≈ LinRange(0, ds[3], 2*Nz+1)[(NS == :H ? 2 : 3):2:end-1]
        @test data_file ≈ [vel0_file(x, y, z) for x=x_file, y=y_file,
                z=z_file[gd.iz_min:(gd.iz_min + size(vel_fd,3) - 1)]]
    end
end

test_file_io()
test_shifted_file_output()

# also test the parallel version with one layer per process
MPI.Initialized() && test_file_io(Nz=MPI.Comm_size(MPI.COMM_WORLD))
MPI.Initialized() && test_shifted_file_output(Nz=MPI.Comm_size(MPI.COMM_WORLD))
