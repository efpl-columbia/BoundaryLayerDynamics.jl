function test_closed_channel_les(n)

    z0 = 5e-3
    u1, u2 = 0.64917, 0.18572
    gs = (n, n, n)
    ds = (2*π, 4/3*π, 2.0)
    ic = ((x,y,z)->u1, (x,y,z)->u2, (x,y,z)->0.0)
    lbcs = RoughWallBoundary(z0)
    ubcs = RoughWallBoundary(z0)
    Re = 1e5
    f = (1.0, 0.0)

    cfp = ChannelFlowProblem(gs, ds, lbcs, ubcs, 1/Re, f, false,
                             sgs_model = StaticSmagorinskyModel())
    set_velocity!(cfp, ic)

    # compute advection term for constant velocity
    CF.set_advection!(cfp.rhs, cfp.velocity, cfp.derivatives, cfp.transform,
                      cfp.lower_bcs, cfp.upper_bcs, nothing, cfp.advection_buffers)

    # check that advective stress is equal to wall stress from wall model
    utot = sqrt(u1^2+u2^2)
    τw = 0.4^2 * utot^2 / log(ds[3]/n/2/z0)^2
    @test global_vector(cfp.rhs[1][1,1,:]) ≈ [1; zeros(n-2); 1] * τw * u1 / utot / (-ds[3]/n)
    @test global_vector(cfp.rhs[2][1,1,:]) ≈ [1; zeros(n-2); 1] * τw * u2 / utot / (-ds[3]/n)

    # compute advection term for linear velocity field
    dudz = 0.40595
    CF.set_field!(cfp.velocity[1], (x,y,z) -> (z - 1) * dudz,
                  cfp.grid, cfp.mapping, cfp.transform, CF.NodeSet(:H))
    cfp.velocity[2] .= 0
    cfp.velocity[3] .= 0
    CF.set_advection!(cfp.rhs, cfp.velocity, cfp.derivatives, cfp.transform,
                      cfp.lower_bcs, cfp.upper_bcs, nothing, cfp.advection_buffers)

    # check that eddy viscosity is correct
    S13 = S31 = 1/2 * (dudz + 0) # dw/dx == 0
    Stot = sqrt(2 * (S13^2 + S31^2)) # other entries of Sij are zero
    Δ = cbrt(prod(ds ./ gs))
    νT = (Δ * 0.1)^2 * Stot
    @test global_vector(cfp.advection_buffers.eddy_viscosity_h[1,1,:])[2:end-1] ≈ νT * ones(n-2) # top & bottom rely on boundary condition
    @test global_vector(cfp.advection_buffers.eddy_viscosity_v[1,1,:]) ≈ νT * ones(n-1)

    # check that the advection term is correct
    uw = (1-ds[3]/n/2)*dudz
    τw = 0.4^2 * (-uw*uw) / log(ds[3]/n/2/z0)^2
    advw = (2 * νT * S13 - τw) / (ds[3]/n)
    @test global_vector(cfp.rhs[1][1,1,:]) ≈ [advw; zeros(n-2); -advw]
    @test global_vector(cfp.rhs[2][1,1,:]) .+ 1 ≈ ones(n)
    @test global_vector(cfp.rhs[3][1,1,:]) ≈ dudz * (LinRange(-1,1,n+1)[2:end-1] * dudz) # −(ω1u2 − ω2u1) = (∂₃u₁ − ∂₁u₃) u₁ = (∂₃u₁) u₁

    # check that integration runs without error
    dt = 1e-3
    nt = 10
    integrate!(cfp, dt * nt, dt = dt, verbose = false)

end

function test_open_channel_les(n)

    z0 = 5e-3
    u1, u2 = 0.64917, 0.18572
    gs = (n, n, n)
    ds = (2*π, 4/3*π, 1.0)
    ic = ((x,y,z)->u1, (x,y,z)->u2, (x,y,z)->0.0)
    lbcs = RoughWallBoundary(z0)
    ubcs = FreeSlipBoundary()
    Re = 1e5
    f = (1.0, 0.0)

    cfp = ChannelFlowProblem(gs, ds, lbcs, ubcs, 1/Re, f, false,
                             sgs_model = StaticSmagorinskyModel())
    set_velocity!(cfp, ic)

    # check that integration runs without error
    dt = 1e-3
    nt = 10
    integrate!(cfp, dt * nt, dt = dt, verbose = false)

end

test_closed_channel_les(16)
MPI.Initialized() && test_closed_channel_les(MPI.Comm_size(MPI.COMM_WORLD))
test_open_channel_les(16)
