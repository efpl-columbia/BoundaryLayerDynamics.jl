reset!(arrays...) = map(a -> fill!(a, zero(eltype(a))), arrays)
randomize!(arrays...) = map(a -> fill!(a, rand(eltype(a))), arrays)

function test_diffusion(NZ)

    gd, ht, uh, uv = setup_random_fields(Float64, 12, 14, NZ)
    ds = (2*π, 2*π, 1.0)
    gm = CF.GridMapping(ds...)
    df = CF.DerivativeFactors(gd, gm)

    # some functions to test dirichlet & neumann boundary conditions
    rval = 0.9134954544827887
    u0d = (x,y,z) -> sin(x) * cos(y) * (z*(1-z)) + rval
    u0n = (x,y,z) -> cos(x) + sin(y) + (z*(1-z)) + rval * z
    Lu0d = (x,y,z) -> - (2 * z*(1-z) + 2) * sin(x) * cos(y)
    Lu0n = (x,y,z) -> - (cos(x) + sin(y) + 2)

    # set up boundary conditions with random value to test non-zero bc
    bcd1 = CF.DirichletBC(rval, gd)
    bcd2 = CF.DirichletBC(rval, gd)
    bcn1 = CF.NeumannBC(1.0 + rval, gd)
    bcn2 = CF.NeumannBC(-1.0 + rval, gd)

    rhsh_pd = CF.zeros_pd(Float64, gd, CF.NodeSet(:H))
    rhsv_pd = CF.zeros_pd(Float64, gd, CF.NodeSet(:V))

    ν = 1e-6
    z_h = 1/(2*NZ):1/NZ:1-1/(2*NZ)
    z_v = 1/NZ:1/NZ:1-1/NZ

    # check 2nd derivative for H-nodes with Dirichlet BCs
    randomize!(uh, rhsh_pd)
    CF.set_field!(uh, u0d, gd, gm, ht, CF.NodeSet(:H))
    rhs = CF.zeros_fd(Float64, gd, CF.NodeSet(:H))
    CF.add_laplacian!(rhs, uh, bcd1, bcd2, df, CF.NodeSet(:H), ν)
    CF.get_field!(rhsh_pd, ht, rhs, CF.NodeSet(:H))
    @test global_vector(rhsh_pd[5,8,:]) ≈ ν * [Lu0d(2*π*4/18, 2*π*7/21, z) for z=z_h]

    # check 2nd derivative for H-nodes with Neumann BCs
    randomize!(uh, rhsh_pd)
    CF.set_field!(uh, u0n, gd, gm, ht, CF.NodeSet(:H))
    rhs = CF.zeros_fd(Float64, gd, CF.NodeSet(:H))
    CF.add_laplacian!(rhs, uh, bcn1, bcn2, df, CF.NodeSet(:H), ν)
    CF.get_field!(rhsh_pd, ht, rhs, CF.NodeSet(:H))
    @test global_vector(rhsh_pd[5,8,:]) ≈ ν * [Lu0n(2*π*4/18, 2*π*7/21, z) for z=z_h]

    # check 2nd derivative for V-nodes with Dirichlet BCs
    randomize!(uv, rhsv_pd)
    CF.set_field!(uv, u0d, gd, gm, ht, CF.NodeSet(:V))
    rhs = CF.zeros_fd(Float64, gd, CF.NodeSet(:V))
    CF.add_laplacian!(rhs, uv, bcd1, bcd2, df, CF.NodeSet(:V), ν)
    CF.get_field!(rhsv_pd, ht, rhs, CF.NodeSet(:V))
    @test global_vector(rhsv_pd[5,8,:]) ≈ ν * [Lu0d(2*π*4/18, 2*π*7/21, z) for z=z_v]

end

test_diffusion(16)

# also test the parallel version with one layer per process
MPI.Initialized() && test_diffusion(MPI.Comm_size(MPI.COMM_WORLD))
