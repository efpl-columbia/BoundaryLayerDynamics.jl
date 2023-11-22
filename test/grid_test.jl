i3_range_from_neighbors(gd) =
    if MPI.Initialized()
        c = MPI.COMM_WORLD
        r, s = MPI.Comm_rank(c), MPI.Comm_size(c)
        r < s - 1 && MPI.Send(gd.i3max, r + 1, 1, c)
        i3min = (r == 0 ? 1 : MPI.Recv(Int, r - 1, 1, c)[1] + 1)
        r > 0 && MPI.Send(gd.i3min, r - 1, 2, c)
        i3max = (r == s - 1 ? gd.n3global : MPI.Recv(Int, r + 1, 2, c)[1] - 1)
        i3min, i3max
    else
        1, gd.n3global
    end

function test_grid_setup(N3)
    gd = BLD.Grid((64, 32, N3))

    # even number of frequencies should be rounded down as Nyquist is removed
    gd2 = BLD.Grid((63, 31, N3))
    @test all([
        getproperty(gd, p) == getproperty(gd2, p) for p in (:k1max, :k2max, :n3c, :n3i, :n3global, :i3min, :i3max)
    ])

    # check all properties
    @test [getproperty(gd, i) for i in (:k1max, :k2max, :n3global)] == [31, 15, 16]
    @test global_sum(gd.n3c) == 16
    @test global_sum(gd.n3i) == 15
    @test (gd.i3min, gd.i3max) == i3_range_from_neighbors(gd)

    # make sure the correct wavenumbers are generated
    BLD.Grids.wavenumbers(gd, 1) == [0:31;]
    BLD.Grids.wavenumbers(gd, 2) == [0:31; -31:-1]
end

# helper to test grid distribution without actual MPI setup
BLD.Grids.init_processes(comm::Tuple) = (comm, comm...)

function test_grid_distribution()
    @test BLD.Grid(64; comm = (1, 32)).n3c == 2
    @test [BLD.Grid(64; comm = (i, 8)).n3i for i in 1:8] == [8, 8, 8, 8, 8, 8, 8, 7]
    @test [BLD.Grid(60; comm = (i, 8)).n3c for i in 1:8] == [8, 8, 8, 8, 7, 7, 7, 7]
end

@timeit "Grid" @testset "Staggered Fourier Grid" begin
    test_grid_setup(16)
    test_grid_distribution()
end
