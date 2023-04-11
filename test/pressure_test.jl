function test_batch_ldlt(N)

    T = Float64
    gd = BLD.Grid((4, 4, N)) # size in FD: (2, 3, N)
    batch_size = 4 # not evenly divisible by 2x3
    n3c = BLD.Grids.fdsize(gd, NS(:C))[3]
    n3i = BLD.Grids.fdsize(gd, NS(:I))[3]

    # initialize rhs vectors with random values
    b_dist = zeros(T, gd, NS(:C))
    map!(x -> rand(eltype(b_dist)), b_dist, b_dist)
    nx, ny = size(b_dist)[1:2]

    # solve system with identity matrix
    dv_dist = ones(T, n3c)
    ev_dist = zeros(T, n3i)
    dvs = (dv_dist for kx=1:nx, ky=1:ny)
    evs = (ev_dist for kx=1:nx, ky=1:ny)
    A = BLD.Processes.DistributedBatchLDLt(T, dvs, evs, b_dist, batch_size, gd.comm)
    @test A.γ ≈ ones(T, size(A.γ))
    @test A.β ≈ zeros(T, size(A.β))
    x_dist = A \ b_dist
    @test x_dist ≈ b_dist

    # solve system with random matrix, with different prefactor for each (kx,ky) pair
    dv_dist = rand(T, n3c)
    ev_dist = rand(T, n3i)
    dvs = ((kx + (ky-1) * nx) * dv_dist for kx=1:nx, ky=1:ny)
    evs = ((kx + (ky-1) * nx) * ev_dist for kx=1:nx, ky=1:ny)
    A = BLD.Processes.DistributedBatchLDLt(T, dvs, evs, b_dist, batch_size, gd.comm)
    x_dist = A \ b_dist
    T = LinearAlgebra.SymTridiagonal(global_vector(dv_dist), global_vector(ev_dist))
    @test 1 * T * global_vector(x_dist[1,1,:]) ≈ global_vector(b_dist[1,1,:])
    @test 2 * T * global_vector(x_dist[2,1,:]) ≈ global_vector(b_dist[2,1,:])
    @test 3 * T * global_vector(x_dist[1,2,:]) ≈ global_vector(b_dist[1,2,:])
    @test 4 * T * global_vector(x_dist[2,2,:]) ≈ global_vector(b_dist[2,2,:])
    @test 5 * T * global_vector(x_dist[1,3,:]) ≈ global_vector(b_dist[1,3,:])
    @test 6 * T * global_vector(x_dist[2,3,:]) ≈ global_vector(b_dist[2,3,:])
end

function test_pressure_solver(N)

    T = Float64
    dims = (4, 4, N) # size in FD: (2, 3, N)
    gd = BLD.Grid(dims)
    batch_size = 4 # not evenly divisible by 2x3
    domain_size = (2*(π*one(T)), 2*(π*one(T)), one(T))
    bc3 = 0.5700899056030746 # random but same value everywhere
    lbc = CustomBoundary(vel1 = :dirchlet, vel2 = :dirichlet, vel3 = :dirichlet => bc3)
    ubc = deepcopy(lbc)
    domain = Domain(domain_size, lbc, ubc)
    abl = DiscretizedABL(dims, domain, [Pressure()])
    rhs = deepcopy(abl.state)

    for u in values(abl.state)
        map!(x->rand(eltype(u)), u, u)
    end

    # check pressure in u-direction
    initialize!(abl, vel1 = (x,y,z) -> 1 + sin(x), vel2 = (x,y,z) -> 0, vel3 = (x,y,z) -> 0)
    BLD.Processes.apply_projections!(abl.state, abl.processes)
    u_pd = abl[:vel1]
    @test u_pd ≈ ones(T, size(u_pd))

    # check pressure in v-direction
    initialize!(abl, vel1 = (x,y,z) -> 0, vel2 = (x,y,z) -> 1 + sin(y), vel3 = (x,y,z) -> 0)
    BLD.Processes.apply_projections!(abl.state, abl.processes)
    v_pd = abl[:vel2]
    @test v_pd ≈ ones(T, size(v_pd))

    # check pressure in w-direction
    initialize!(abl, vel1 = (x,y,z) -> 0, vel2 = (x,y,z) -> 0, vel3 = (x,y,z) -> 0)
    BLD.Processes.apply_projections!(abl.state, abl.processes)
    w_pd = abl[:vel3]
    @test w_pd ≈ bc3 * ones(T, size(w_pd))
end

@timeit "Pressure" @testset "Pressure Solver" begin
    test_batch_ldlt(16)
    test_pressure_solver(16)
    MPI.Initialized() && test_batch_ldlt(MPI.Comm_size(MPI.COMM_WORLD))
    MPI.Initialized() && test_pressure_solver(MPI.Comm_size(MPI.COMM_WORLD))
end
