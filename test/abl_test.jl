function test_abl_setup()

    Re = 1e6
    N = 64
    L = (4π, 2π, 1)

    domain = Domain(L, RoughWall(1e-4), FreeSlipBoundary())
    processes = incompressible_flow(1/Re)
    abl = DiscretizedABL(N, domain, incompressible_flow(Re))

    io = IOBuffer()
    show(io, MIME("text/plain"), abl)
    @test String(take!(io)) == """
        Discretized Atmospheric Boundary Layer:
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
            integrate!(cf, dt * Nt, dt = dt,
                profiles_dir = joinpath(dir, "profiles"), profiles_frequency = 10,
                snapshot_steps = [div(1*Nt,5), div(2*Nt,5), div(3*Nt,5), div(4*Nt,5)],
                snapshot_dir = joinpath(dir, "snapshots"), verbose = false)
        end

        # attempt setting the velocity from the latest snapshot
        last_snapshot = readdir(joinpath(dir, "snapshots"))[end]
        CF.load_snapshot!(cf, joinpath(dir, "snapshots", last_snapshot))
    end
end

@timeit "ABL Simulation" @testset "Detailed ABL Flow Setup" begin
    test_abl_setup()
end
