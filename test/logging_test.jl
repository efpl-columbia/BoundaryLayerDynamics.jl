function test_sgs_logging(; Nx=6, Ny=8, Nz=8)

    # set up test problem
    z0 = 5e-3
    u1, u2, dudz = 0.64917, 0.18572, 0.40595
    gs = (Nx, Ny, Nz)
    ds = (2*π, 4/3*π, 2.0)
    ic = ((x,y,z)->u1 + dudz * (z-1), (x,y,z)->u2, (x,y,z)->0.0)
    lbcs = RoughWallBoundary(z0)
    ubcs = RoughWallBoundary(z0)
    Re = 1e5
    f = (1.0, 0.0)

    # compute exact sgs-stresses
    utot = sqrt(u1^2+u2^2)
    τw = 0.4^2 * utot^2 / log(ds[3]/Nz/2/z0)^2
    # linear stress
    S13 = S31 = 1/2 * (dudz + 0) # dw/dx == 0
    Stot = sqrt(2 * (S13^2 + S31^2)) # other entries of Sij are zero
    Δ = cbrt(prod(ds ./ gs))
    νT = (Δ * 0.1)^2 * Stot
    #uw = (1-ds[3]/Nz/2)*dudz
    #τw = 0.4^2 * (-uw*uw) / log(ds[3]/Nz/2/z0)^2
    #advw = (2 * νT * S13 - τw) / (ds[3]/Nz)
    u1below = u1 - (1-ds[3]/Nz/2)*dudz
    u1above = u1 + (1-ds[3]/Nz/2)*dudz
    ubelow = sqrt(u1below^2 + u2^2)
    uabove = sqrt(u1above^2 + u2^2)
    τwbelow = 0.4^2 * ubelow^2 / log(ds[3]/Nz/2/z0)^2
    τwabove = 0.4^2 * uabove^2 / log(ds[3]/Nz/2/z0)^2
    τ13below = τwbelow * u1below / ubelow
    τ23below = τwbelow * u2 / ubelow
    τ13above = -τwabove * u1above / uabove
    τ23above = -τwabove * u2 / uabove

    cfp = ChannelFlowProblem(gs, ds, lbcs, ubcs, 1/Re, f, false,
                             sgs_model = StaticSmagorinskyModel())
    set_velocity!(cfp, ic)

    # set up logging of SGS terms
    Random.seed!(901666522) # same seed for each process

    mktempdir_parallel() do dir

        # set up log for all sgs terms
        log = CF.FlowLog(Float64, cfp.grid, [0.0], dir,
                     [:sgs11, :sgs12, :sgs13, :sgs22, :sgs23, :sgs33])

        # compute advection term for constant velocity
        CF.set_advection!(cfp.rhs, cfp.velocity, cfp.derivatives, cfp.transform,
                          cfp.lower_bcs, cfp.upper_bcs, log, cfp.advection_buffers)

        # trigger write of log files
        CF.process_logs!(log, 0.0)
        MPI.Initialized() && MPI.Barrier(MPI.COMM_WORLD)

        HDF5.h5open(joinpath(dir, "profiles-000.h5")) do h5
            @test read(h5, "sgs13") ≈ [τ13below; 2 * νT * S13 * ones(Nz-1); τ13above]
            @test read(h5, "sgs23") ≈ [τ23below; zeros(Nz-1); τ23above]
            @test read(h5, "sgs12") ≈ zeros(Nz)
            @test read(h5, "sgs11") ≈ zeros(Nz)
            @test read(h5, "sgs22") ≈ zeros(Nz)
            @test read(h5, "sgs33") ≈ zeros(Nz)
        end
    end
end

test_sgs_logging()
MPI.Initialized() && test_sgs_logging(Nz = MPI.Comm_size(MPI.COMM_WORLD))
