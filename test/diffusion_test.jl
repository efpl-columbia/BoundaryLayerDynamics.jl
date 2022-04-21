function test_diffusion(NZ)

    # some functions to test dirichlet & neumann boundary conditions
    rval = 0.9134954544827887
    u0d = (x,y,z) -> sin(x) * cos(y) * (z*(1-z)) + rval
    u0n = (x,y,z) -> cos(x) + sin(y) + (z*(1-z)) + rval * z
    Lu0d = (x,y,z) -> - (2 * z*(1-z) + 2) * sin(x) * cos(y)
    Lu0n = (x,y,z) -> - (cos(x) + sin(y) + 2)
    x3c = 1/(2*NZ):1/NZ:1-1/(2*NZ)
    x3i = 1/NZ:1/NZ:1-1/NZ

    # set up diffusion problems
    ν = 1e-6
    diff = [MolecularDiffusion(:vel1, ν), MolecularDiffusion(:vel2, ν), MolecularDiffusion(:vel3, ν)]

    # set up domain
    ds = (2*π, 2*π, 1.0)
    lbc = CustomBoundary(vel1 = (:neumann => 1.0 + rval),
                         vel2 = (:dirichlet => rval),
                         vel3 = (:dirichlet => rval))
    ubc = CustomBoundary(vel1 = (:neumann => - 1.0 + rval),
                         vel2 = (:dirichlet => rval),
                         vel3 = (:dirichlet => rval))
    domain = Domain(ds, lbc, ubc)

    # set up grid
    dims = (12, 14, NZ)

    # set up state & rhs
    abl = DiscretizedABL(dims, domain, diff)
    rhs = deepcopy(abl.state)

    # initialize values after resetting arrays with some arbitrary value
    #map(a -> fill!(a, rand(eltype(a))), [values(abl.state)..., values(rhs)...])
    initialize!(abl, vel1 = u0n, vel2 = u0d, vel3 = u0d)

    # compute diffusion
    ABL.Processes.compute_rates!(rhs, abl.state, 0.0, abl.processes, abl.physical_spaces)

    isample = (5, 7)
    xsample = (isample .- 1) ./ dims[1:2] .* ds[1:2]
    x3c = global_vector(coordinates(abl, :vel1, 3))
    x3i = global_vector(coordinates(abl, :vel3, 3))

    # check 2nd derivative for C-nodes with Neumann BCs
    #@test global_vector(abl[:vel1][5,8,:]) ≈ ν * [Lu0n(2*π*4/18, 2*π*7/21, x3) for x3=coordinates(abl, :vel1, 3)]
    result = ABL.PhysicalSpace.get_field(abl.physical_spaces[dims[1:2]].transform, rhs[:vel1])
    @test global_vector(result[isample...,:]) ≈ ν * [Lu0n(xsample..., x3) for x3=x3c]

    # check 2nd derivative for C-nodes with Dirichlet BCs
    result = ABL.PhysicalSpace.get_field(abl.physical_spaces[dims[1:2]].transform, rhs[:vel2])
    @test global_vector(result[isample...,:]) ≈ ν * [Lu0d(xsample..., x3) for x3=x3c]

    # check 2nd derivative for I-nodes with Dirichlet BCs
    result = ABL.PhysicalSpace.get_field(abl.physical_spaces[dims[1:2]].transform, rhs[:vel3])
    @test global_vector(result[isample...,:]) ≈ ν * [Lu0d(xsample..., x3) for x3=x3i]
end

@timeit "Diffusion" @testset "Molecular Diffusion" begin
    test_diffusion(16)
    MPI.Initialized() && test_diffusion(MPI.Comm_size(MPI.COMM_WORLD))
end
