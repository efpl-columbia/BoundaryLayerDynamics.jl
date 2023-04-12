function test_abl_setup()

    Re = 1e6
    N = 64
    L = (4π, 2π, 1)

    domain = Domain(L, RoughWall(1e-4), FreeSlipBoundary())
    processes = incompressible_flow(1/Re)
    abl = Model((N, N, N), domain, incompressible_flow(Re))

    io = IOBuffer()
    show(io, MIME("text/plain"), abl)
    @test String(take!(io)) == """
        BoundaryLayerDynamics.Model:
        → κ₁ ∈ [−31,31], κ₂ ∈ [−31,31], i₃ ∈ [1,64]"""
end

"""
Test that a standard channel flow with output can be run, including output,
without producing an error.
"""
function test_channel(Nv; Nh = 4, Re = 1.0, CFL = 0.1, T = 1/Re, Nt = 100)
    cf = closedchannelflow(Re, (Nh,Nh,Nv), constant_flux = true)
    dt = (2/Nv)^2 * Re * CFL

    mktempdir_parallel() do dir
        redirect_stdout(devnull) do # keep output verbose to check for errors in output routines
            output = [MeanProfiles(path=joinpath(dir, "profiles"), output_frequency=10*dt),
                      Snapshots(path=joinpath(dir, "snapshots"), frequency=div(Nt, 5)*dt)]
            evolve!(cf, dt * Nt, dt = dt, output = output)
        end

        # attempt setting the velocity from the latest snapshot
        last_snapshot = readdir(joinpath(dir, "snapshots"))[end]
        initialize!(cf, joinpath(dir, "snapshots", last_snapshot))
    end
end

@timeit "ABL Simulation" @testset "Detailed ABL Flow Setup" begin
    test_abl_setup()

    # test that a channel flow with output runs without error
    test_channel(16)
    MPI.Initialized() && test_channel(MPI.Comm_size(MPI.COMM_WORLD))
end
