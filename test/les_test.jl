function test_closed_channel_les(n)

    z0 = 5e-3
    u1, u2 = 0.64917, 0.18572
    gs = (n, n, n)
    ds = (2*π, 4/3*π, 2.0)
    ic = ((x,y,z)->u1, (x,y,z)->u2)

    domain = Domain(ds, RoughWall(z0), RoughWall(z0))
    processes = [StaticSmagorinskyModel(wall_damping = false)]
    model = Model(gs, domain, processes)
    rhs = deepcopy(model.state)

    # compute advection term for constant velocity and check that advective
    # stress is equal to wall stress from wall model
    initialize!(model, vel1=ic[1], vel2=ic[2])
    BLD.Processes.compute_rates!(rhs, model.state, 0.0, model.processes, model.physical_spaces)

    utot = sqrt(u1^2+u2^2)
    τw = 0.4^2 * utot^2 / log(ds[3]/n/2/z0)^2
    @test global_vector(rhs[1][1,1,:]) ≈ [1; zeros(n-2); 1] * τw * u1 / utot / (-ds[3]/n)
    @test global_vector(rhs[2][1,1,:]) ≈ [1; zeros(n-2); 1] * τw * u2 / utot / (-ds[3]/n)

    # compute advection term for linear velocity field
    dudz = 0.40595 # pseudo-random value
    initialize!(model, vel1 = (x,y,z) -> (z-1) * dudz)
    BLD.Processes.compute_rates!(rhs, model.state, 0.0, model.processes, model.physical_spaces)

    # check that eddy viscosity is correct
    S13 = S31 = 1/2 * (dudz + 0) # dw/dx == 0
    Stot = sqrt(2 * (S13^2 + S31^2)) # other entries of Sij are zero
    Δ = cbrt(prod(ds ./ (gs .+ isodd(n) .* (1, 1, 0)))) # physical-domain size uses even numbers
    νT = (Δ * 0.1)^2 * Stot
    @test global_vector(model.processes[].eddyviscosity_c[1,1,:])[2:end-1] ≈ νT * ones(n-2) # top & bottom rely on boundary condition
    @test global_vector(model.processes[].eddyviscosity_i[1,1,:]) ≈ νT * ones(n-1)

    # check that the advection term is correct
    uw = (1-ds[3]/n/2)*dudz
    τw = 0.4^2 * (-uw*uw) / log(ds[3]/n/2/z0)^2
    advw = (2 * νT * S13 - τw) / (ds[3]/n)
    @test global_vector(rhs[1][1,1,:]) ≈ [advw; zeros(n-2); -advw]
    @test global_vector(rhs[2][1,1,:]) .+ 1 ≈ ones(n)
    @test global_vector(rhs[3][1,1,:]) .+ 1 ≈ ones(n-1)

    # check that integration runs without error
    dt = 1e-3
    nt = 10
    evolve!(model, dt * nt, dt = dt)
end

function test_open_channel_les(n)

    z0 = 5e-3
    ds = (2*π, 4/3*π, 1.0)
    u1, u2 = 0.64917, 0.18572
    ic = ((x,y,z)->u1, (x,y,z)->u2)
    Re = 1e5

    domain = Domain(ds, RoughWall(z0), FreeSlipBoundary())
    processes = incompressible_flow(1/Re, constant_forcing = (1, 0),
                                    sgs_model = StaticSmagorinskyModel())
    model = Model((n, n, n), domain, processes)
    initialize!(model, vel1 = ic[1], vel2 = ic[2])

    # check that integration runs without error
    dt = 1e-3
    nt = 10
    evolve!(model, dt * nt, dt = dt)
end

@timeit "LES" @testset "Large-Eddy Simulation" begin
    test_closed_channel_les(16)
    MPI.Initialized() && test_closed_channel_les(max(MPI.Comm_size(MPI.COMM_WORLD), 3))
    test_open_channel_les(16)
end
