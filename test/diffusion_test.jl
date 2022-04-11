
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
    domain = SemiperiodicDomain(ds, lbc, ubc)

    # set up grid
    size = (12, 14, NZ)

    # set up state & rhs
    abl = DiscretizedABL(size, domain, diff)
    rhs = deepcopy(abl.state)

    # initialize values after resetting arrays with some arbitrary value
    #map(a -> fill!(a, rand(eltype(a))), [values(abl.state)..., values(rhs)...])
    initialize!(abl, vel1 = u0n, vel2 = u0d, vel3 = u0d)

    # compute diffusion
    ABL.Processes.rate!(rhs, abl.state, 0.0, abl.processes, abl.transforms)

    isample = (5, 7)
    xsample = (isample .- 1) ./ size[1:2] .* ds[1:2]

    # check 2nd derivative for C-nodes with Neumann BCs
    #@test global_vector(abl[:vel1][5,8,:]) ≈ ν * [Lu0n(2*π*4/18, 2*π*7/21, x3) for x3=coordinates(abl, :vel1, 3)]
    result = ABL.Transform.get_field(abl.transforms[size[1:2]], rhs[:vel1])
    @test global_vector(result[isample...,:]) ≈ ν * [Lu0n(xsample..., x3) for x3=coordinates(abl, :vel1, 3)]

    # check 2nd derivative for C-nodes with Dirichlet BCs
    result = ABL.Transform.get_field(abl.transforms[size[1:2]], rhs[:vel2])
    @test global_vector(result[isample...,:]) ≈ ν * [Lu0d(xsample..., x3) for x3=coordinates(abl, :vel2, 3)]

    # check 2nd derivative for I-nodes with Dirichlet BCs
    result = ABL.Transform.get_field(abl.transforms[size[1:2]], rhs[:vel3])
    @test global_vector(result[isample...,:]) ≈ ν * [Lu0d(xsample..., x3) for x3=coordinates(abl, :vel3, 3)]
end
