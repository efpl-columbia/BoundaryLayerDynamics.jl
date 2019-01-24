function mktempdir_parallel(f)

    MPI.Initialized() || return mktempdir(f)

    mktempdir_once(f0) = MPI.Comm_rank(MPI.COMM_WORLD) == 0 ? mktempdir(f0) : f0("")

    mktempdir_once() do p
        p = MPI.bcast(p, 0, MPI.COMM_WORLD)
        f(p)
        MPI.Barrier(MPI.COMM_WORLD)
    end
end

function test_file_io(; Nx=6, Ny=8, Nz=12)

    Random.seed!(73330459108) # same seed for each process

    mktempdir_parallel() do p

        for T=(Float32, Float64), ns=(:H, :V)

            ds = (rand(T), rand(T), rand(T))
            vel0(x, y, z) = x / ds[1] + y / ds[2] + z / ds[3]

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
                CF.write_field(fn, vel_pd, x, y, z, ds, output_type = T_files)
                domain_after, x_after, y_after, z_after, vel_after = CF.read_field(fn, CF.NodeSet(ns))

                @test domain_after == ds
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

test_file_io()

# also test the parallel version with one layer per process
MPI.Initialized() && test_file_io(Nz=MPI.Comm_size(MPI.COMM_WORLD))
