function test_batch_ldlt(N)

    T = Float64
    gd = CF.DistributedGrid(4, 4, N) # size in FD: (2, 3, N)
    batch_size = 4 # not evenly divisible by 2x3

    # initialize rhs vectors with random values
    b_dist = CF.zeros_fd(T, gd, CF.NodeSet(:H))
    map!(x -> rand(eltype(b_dist)), b_dist, b_dist)
    nx, ny = size(b_dist)[1:2]

    # solve system with identity matrix
    dv_dist = ones(T, CF.get_nz(gd, CF.NodeSet(:H)))
    ev_dist = zeros(T, CF.get_nz(gd, CF.NodeSet(:V)))
    dvs = (dv_dist for kx=1:nx, ky=1:ny)
    evs = (ev_dist for kx=1:nx, ky=1:ny)
    A = CF.DistributedBatchLDLt(T, dvs, evs, b_dist, batch_size)
    @test A.γ ≈ ones(T, size(A.γ))
    @test A.β ≈ zeros(T, size(A.β))
    x_dist = A \ b_dist
    @test x_dist ≈ b_dist

    # solve system with random matrix, with different prefactor for each (kx,ky) pair
    dv_dist = rand(T, CF.get_nz(gd, CF.NodeSet(:H)))
    ev_dist = rand(T, CF.get_nz(gd, CF.NodeSet(:V)))
    dvs = ((kx + (ky-1) * nx) * dv_dist for kx=1:nx, ky=1:ny)
    evs = ((kx + (ky-1) * nx) * ev_dist for kx=1:nx, ky=1:ny)
    A = CF.DistributedBatchLDLt(T, dvs, evs, b_dist, batch_size)
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
    gd = CF.DistributedGrid(4, 4, N) # size in FD: (2, 3, N)
    batch_size = 4 # not evenly divisible by 2x3
    domain_size = (2*(π*one(T)), 2*(π*one(T)), one(T))
    gm = CF.GridMapping(domain_size...)

    u = CF.zeros_fd(T, gd, CF.NodeSet(:H))
    v = CF.zeros_fd(T, gd, CF.NodeSet(:H))
    w = CF.zeros_fd(T, gd, CF.NodeSet(:V))
    p = CF.zeros_fd(T, gd, CF.NodeSet(:H))
    u_pd = CF.zeros_pd(T, gd, CF.NodeSet(:H))
    v_pd = CF.zeros_pd(T, gd, CF.NodeSet(:H))
    w_pd = CF.zeros_pd(T, gd, CF.NodeSet(:V))

    map!(x->rand(eltype(u)), u, u)
    map!(x->rand(eltype(v)), v, v)
    map!(x->rand(eltype(w)), w, w)
    map!(x->rand(eltype(p)), p, p)

    lbc = CF.bc_noslip(T, gd)
    ubc = CF.bc_freeslip(T, gd)
    pbc = CF.UnspecifiedBC(T, gd)

    df = CF.DerivativeFactors(gd, gm)
    ht = CF.HorizontalTransform(T, gd)
    ps = CF.prepare_pressure_solver(gd, gm, batch_size)

    # check pressure in u-direction
    CF.set_field!(u, (x,y,z) -> 1.0 + sin(x), gd, gm, ht, CF.NodeSet(:H))
    CF.set_field!(v, (x,y,z) -> 0.0, gd, gm, ht, CF.NodeSet(:H))
    CF.set_field!(w, (x,y,z) -> 0.0, gd, gm, ht, CF.NodeSet(:V))
    CF.solve_pressure!(p, (u,v,w), lbc, ubc, gd, df, ps)
    CF.add_gradient!((u,v,w), p, pbc, df, -1)
    CF.get_field!(u_pd, ht, u, CF.NodeSet(:H))
    @test u_pd[1,1,:] ≈ ones(T, size(u_pd, 3))
    @test u_pd ≈ ones(T, size(u_pd))

    # check pressure in v-direction
    CF.set_field!(u, (x,y,z) -> 0.0, gd, gm, ht, CF.NodeSet(:H))
    CF.set_field!(v, (x,y,z) -> 1.0 + sin(y), gd, gm, ht, CF.NodeSet(:H))
    CF.set_field!(w, (x,y,z) -> 0.0, gd, gm, ht, CF.NodeSet(:V))
    CF.solve_pressure!(p, (u,v,w), lbc, ubc, gd, df, ps)
    CF.add_gradient!((u,v,w), p, pbc, df, -1)
    CF.get_field!(v_pd, ht, v, CF.NodeSet(:H))
    @test v_pd[1,1,:] ≈ ones(T, size(v_pd, 3))
    @test v_pd ≈ ones(T, size(v_pd))

    # check pressure in w-direction
    CF.set_field!(u, (x,y,z) -> 0.0, gd, gm, ht, CF.NodeSet(:H))
    CF.set_field!(v, (x,y,z) -> 0.0, gd, gm, ht, CF.NodeSet(:H))
    CF.set_field!(w, (x,y,z) -> 0.0, gd, gm, ht, CF.NodeSet(:V))
    w_bc = convert(T, 0.5700899056030746) # random but same value everywhere
    CF.solve_pressure!(p, (u,v,w), (lbc[1], lbc[2], CF.DirichletBC(w_bc, gd)),
            (ubc[1], ubc[2], CF.DirichletBC(w_bc, gd)), gd, df, ps)
    CF.add_gradient!((u,v,w), p, pbc, df, -1)
    CF.get_field!(w_pd, ht, w, CF.NodeSet(:V))
    @test w_pd[1,1,:] ≈ w_bc * ones(T, size(w_pd, 3))
    @test w_pd ≈ w_bc * ones(T, size(w_pd))
end

test_batch_ldlt(16)
test_pressure_solver(16)
MPI.Initialized() && test_batch_ldlt(MPI.Comm_size(MPI.COMM_WORLD))
MPI.Initialized() && test_pressure_solver(MPI.Comm_size(MPI.COMM_WORLD))
