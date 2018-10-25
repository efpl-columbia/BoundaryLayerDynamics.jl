function test_advection(NZ)

    gd = CF.DistributedGrid(12, 14, NZ)
    ht = CF.HorizontalTransform(Float64, gd)
    ds = (2*π, 2*π, 1.0)
    df = CF.DerivativeFactors(gd, ds)

    lbcs = CF.bc_noslip(Float64, gd)
    ubcs = CF.bc_noslip(Float64, gd)

    vel = (CF.zeros_fd(Float64, gd, CF.NodeSet(:H)),
           CF.zeros_fd(Float64, gd, CF.NodeSet(:H)),
           CF.zeros_fd(Float64, gd, CF.NodeSet(:V)))
    rhs = (CF.zeros_fd(Float64, gd, CF.NodeSet(:H)),
           CF.zeros_fd(Float64, gd, CF.NodeSet(:H)),
           CF.zeros_fd(Float64, gd, CF.NodeSet(:V)))
    b = CF.AdvectionBuffers(Float64, gd)

    CF.set_advection!(rhs, vel, df, ht, lbcs, ubcs, b)
    @test true
end

test_advection(16)
# also test the parallel version with one layer per process
MPI.Initialized() && test_advection(MPI.Comm_size(MPI.COMM_WORLD))
