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
    model = Model(dims, domain, [Pressure()])
    rhs = deepcopy(model.state)

    for u in values(model.state)
        map!(x->rand(eltype(u)), u, u)
    end

    # check pressure in u-direction
    initialize!(model, vel1 = (x,y,z) -> 1 + sin(x), vel2 = (x,y,z) -> 0, vel3 = (x,y,z) -> 0)
    BLD.Processes.apply_projections!(model.state, model.processes)
    u_pd = model[:vel1]
    @test u_pd ≈ ones(T, size(u_pd))

    # check pressure in v-direction
    initialize!(model, vel1 = (x,y,z) -> 0, vel2 = (x,y,z) -> 1 + sin(y), vel3 = (x,y,z) -> 0)
    BLD.Processes.apply_projections!(model.state, model.processes)
    v_pd = model[:vel2]
    @test v_pd ≈ ones(T, size(v_pd))

    # check pressure in w-direction
    initialize!(model, vel1 = (x,y,z) -> 0, vel2 = (x,y,z) -> 0, vel3 = (x,y,z) -> 0)
    BLD.Processes.apply_projections!(model.state, model.processes)
    w_pd = model[:vel3]
    @test w_pd ≈ bc3 * ones(T, size(w_pd))
end

function ke_pressure(n, bc, η = nothing)
    # set up random velocity field
    mapping = isnothing(η) ? () : (SinusoidalMapping(η, :symmetric),)
    domain = Domain((1, 1, 1), bc, bc, mapping...)
    model = Model((n, n, n), domain, [Pressure()])

    # initialize to random divergence-free field and store copy
    vel0(x, y, z) = rand()
    initialize!(model, vel1 = vel0, vel2 = vel0, vel3 = vel0)
    BLD.apply_projections!(model.state, model.processes)
    state_before = deepcopy(model.state)

    # reset to (different) random field, representing an update in the form
    # ui = ui_before + dt rhs_i or any linear combination of such updates
    BLD.initialize!(model, vel1 = vel0, vel2 = vel0, vel3 = vel0)
    @assert state_before.vel1 != model.state.vel1 # sanity check

    # compute contribution of pressure solver dt dp/dxi as the difference
    # of the state before and after the pressure projection
    rates = deepcopy(model.state) # initialize
    BLD.apply_projections!(model.state, model.processes)
    for vel in keys(rates)
        rates[vel] .-= model.state[vel]
    end

    # compute energy contribution in physical domain
    pd = BLD.PhysicalSpace.pdsize(model.grid, nothing)
    tf = model.physical_spaces[pd].transform
    get(fd) = BLD.PhysicalSpace.get_field(tf, fd)
    e1 = sum(get(state_before.vel1) .* get(rates.vel1), dims=(1,2))[:]
    e2 = sum(get(state_before.vel2) .* get(rates.vel2), dims=(1,2))[:]
    e3 = sum(get(state_before.vel3) .* get(rates.vel3), dims=(1,2))[:]
    wc, wi = map((:C, :I)) do ns
      1 ./ BLD.Derivatives.dx3factors(model.domain, model.grid, NS(ns))
    end
    global_sum(wc .* e1) + global_sum(wc .* e2) + global_sum(wi .* e3)
end

function test_pressure_energy()
    # have at least one layer per process
    n = MPI.Initialized() ? max(MPI.Comm_size(MPI.COMM_WORLD), 8) : 8

    Random.seed!(871286) # seed RNG for deterministic results

    # check that total kinetic energy is conserved, with & without grid stretching
    @test ke_pressure(n, SmoothWall()) + 1 ≈ 1
    @test ke_pressure(n, FreeSlipBoundary()) + 1 ≈ 1
    @test ke_pressure(n, SmoothWall(), 0.97) + 1 ≈ 1
    @test ke_pressure(n, FreeSlipBoundary(), 0.97) + 1 ≈ 1
end

@timeit "Pressure" @testset "Pressure Solver" begin
    test_batch_ldlt(16)
    test_pressure_solver(16)
    test_pressure_energy()
    MPI.Initialized() && test_batch_ldlt(max(MPI.Comm_size(MPI.COMM_WORLD), 3))
    MPI.Initialized() && test_pressure_solver(max(MPI.Comm_size(MPI.COMM_WORLD), 3))
end
