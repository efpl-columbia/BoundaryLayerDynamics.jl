function test_velocity_log(; n=8)

    Random.seed!(92719753017) # same seed for each process
    domain = Domain(rand(3), FreeSlipBoundary(), FreeSlipBoundary())
    u1, u2, f1, f2 = rand(4)
    processes = [ConstantSource(:vel1, f1), ConstantSource(:vel2, f2)]
    abl = DiscretizedABL((n, n, n), domain, processes)
    initialize!(abl, vel1=(x,y,z)->u1, vel2=(x,y,z)->u2)
    tspan = rand() .* [1, 2] # integrate in [T, 2T], with random T

    mktempdir_parallel() do dir

        profiles = MeanProfiles((:vel1, :vel2),
                                output_frequency = diff(tspan)[]/2,
                                path = joinpath(dir, "profiles"))
        evolve!(abl, tspan, dt = diff(tspan)[] / 10, output = [profiles])
        @test length(readdir(dir)) == 2

        # first profiles: mean from t=0 to t=tmax/2
        HDF5.h5open(joinpath(dir, "profiles-01.h5")) do h5
            @test HDF5.read_attribute(h5, "timespan") ≈ [tspan[1], tspan[1] + diff(tspan)[]/2]
            @test HDF5.read_attribute(h5, "intervals") == 5
            tmean = 1/4 * diff(tspan)[]
            @test read(h5, "vel1") ≈ ones(n) * (u1 + tmean * f1)
            @test read(h5, "vel2") ≈ ones(n) * (u2 + tmean * f2)
        end

        # second profiles: mean from t=tmax/2 to t=tmax
        HDF5.h5open(joinpath(dir, "profiles-02.h5")) do h5
            @test HDF5.read_attribute(h5, "timespan") ≈ [tspan[1] + diff(tspan)[]/2, tspan[2]]
            @test HDF5.read_attribute(h5, "intervals") == 5
            tmean = 3/4 * diff(tspan)[]
            @test read(h5, "vel1") ≈ ones(n) * (u1 + tmean * f1)
            @test read(h5, "vel2") ≈ ones(n) * (u2 + tmean * f2)
        end
    end
end

@timeit "Logging" @testset "Logging Mean Statistics" begin
    test_velocity_log()
end
