function iz_range_from_neighbors(gd)
    if MPI.Initialized()
        c = MPI.COMM_WORLD
        r, s = MPI.Comm_rank(c), MPI.Comm_size(c)
        r < s-1 && MPI.Send(gd.iz_max, r+1, 1, c)
        iz_min = (r == 0 ? 1 : MPI.Recv(Int, r-1, 1, c)[1] + 1)
        r > 0 && MPI.Send(gd.iz_min, r-1, 2, c)
        iz_max = (r == s-1 ? gd.nz_global : MPI.Recv(Int, r+1, 2, c)[1] - 1)
        iz_min, iz_max
    else
        1, gd.nz_global
    end
end

function test_grid_setup()
    # even number of frequencies should be rounded down as Nyquist is removed
    for dims in ((64, 32, 16), (63, 31, 16))
        gd = ChannelFlow.DistributedGrid(dims...)
        @test gd.nx_fd == 32
        @test gd.nx_pd == 96
        @test gd.ny_fd == 31
        @test gd.ny_pd == 48
        @test gd.nz_global == 16
        @test global_sum(gd.nz_h) == 16
        @test global_sum(gd.nz_v) == 15
        @test (gd.iz_min, gd.iz_max) == iz_range_from_neighbors(gd)
    end
end

function setup_random_fields(T, nx, ny, nz)
    gd = CF.DistributedGrid(nx, ny, nz)
    uh = CF.zeros_fd(T, gd, CF.NodeSet(:H))
    uv = CF.zeros_fd(T, gd, CF.NodeSet(:V))
    fill!(uh, rand(Complex{T}))
    fill!(uv, rand(Complex{T}))
    ht = CF.HorizontalTransform(T, gd)
    gd, ht, uh, uv
end

function test_transform_constant()
    gd, ht, uh, uv = setup_random_fields(Float64, 64, 32, 16)
    ds = (3*0.5176366701850579, 2*0.9360371626346338, 0.10477115957769456)
    gs = ds ./ (gd.nx_pd, gd.ny_pd, gd.nz_global)
    CF.set_field!(uh, ht, (x,y,z)->1, gs, gd.iz_min, CF.NodeSet(:H))
    CF.set_field!(uv, ht, (x,y,z)->2, gs, gd.iz_min, CF.NodeSet(:V))
    @test uh[1,1,:] ≈ 1 * ones(gd.nz_h)
    @test uv[1,1,:] ≈ 2 * ones(gd.nz_v)
    @test uh[2:end,1,:] ≈ zeros(31,gd.nz_h)
    @test uv[2:end,1,:] ≈ zeros(31,gd.nz_v)
    @test uh[:,2:end,:] ≈ zeros(32,30,gd.nz_h)
    @test uv[:,2:end,:] ≈ zeros(32,30,gd.nz_v)
    uh_pd = CF.zeros_pd(Float64, gd, CF.NodeSet(:H))
    uv_pd = CF.zeros_pd(Float64, gd, CF.NodeSet(:V))
    CF.get_field!(uh_pd, ht, uh, CF.NodeSet(:H))
    CF.get_field!(uv_pd, ht, uv, CF.NodeSet(:V))
    @test uh_pd ≈ 1 * ones(gd.nx_pd, gd.ny_pd, gd.nz_h)
    @test uv_pd ≈ 2 * ones(gd.nx_pd, gd.ny_pd, gd.nz_v)
end

function test_transform_horizontally_varying()
    gd, ht, uh, uv = setup_random_fields(Float64, 16, 16, 16)
    ds = (3*0.5176366701850579, 2*0.9360371626346338, 0.10477115957769456)
    gs = ds ./ (gd.nx_pd, gd.ny_pd, gd.nz_global)
    k = -7:8
    ax = [rand() for k=k]
    ay = [rand() for k=k]
    u0(x,y,z) = sum(ax .* [cos(2*π*k*x/ds[1]) for k=k]) +
                sum(ay .* [cos(2*π*k*y/ds[2]) for k=k])
    u0n(x,y,z) = sum((ax .* [cos(2*π*k*x/ds[1]) for k=k])[1:end-1]) +
                 sum((ay .* [cos(2*π*k*y/ds[2]) for k=k])[1:end-1])
    CF.set_field!(uh, ht, u0, gs, gd.iz_min, CF.NodeSet(:H))
    @test uh[1,1,:] ≈ (ax[8] + ay[8]) .* ones(gd.nz_h)
    @test uh[2:end,1,:] ≈ (ax[9:end-1] + ax[7:-1:1]) / 2 .* ones(1,gd.nz_h)
    @test uh[1,2:8,:] ≈ (ay[9:end-1] + ay[7:-1:1]) / 2 .* ones(1,gd.nz_h)
    @test uh[1,end:-1:9,:] ≈ (ay[9:end-1] + ay[7:-1:1]) / 2 .* ones(1,gd.nz_h)
    # do not compare almost-zero values directly with zero, since the algorithm
    # of “≈” checks for relative errors, which makes it too hard to pass “≈ 0”
    @test uh[2:end,2:end,:] .+ 1 ≈ ones(gd.nx_fd-1, gd.ny_fd-1, gd.nz_h)
    uh_pd = CF.zeros_pd(Float64, gd, CF.NodeSet(:H))
    CF.get_field!(uh_pd, ht, uh, CF.NodeSet(:H))
    @test uh_pd ≈ [u0n(x, y, z) for x=(0:size(uh_pd,1)-1)/size(uh_pd,1)*ds[1],
                                    y=(0:size(uh_pd,2)-1)/size(uh_pd,2)*ds[2],
                                    z=zeros(size(uh_pd,3))]
end

function test_transform_vertically_varying()
    gd, ht, uh, uv = setup_random_fields(Float64, 64, 32, 16)
    ds = (3*0.5176366701850579, 2*0.9360371626346338, 0.10477115957769456)
    gs = ds ./ (gd.nx_pd, gd.ny_pd, gd.nz_global)
    global_values_h = LinRange(0, ds[3], 2*gd.nz_global+1)[2:2:end-1]
    global_values_v = LinRange(0, ds[3], 2*gd.nz_global+1)[3:2:end-2]
    CF.set_field!(uh, ht, (x,y,z)->z, gs, gd.iz_min, CF.NodeSet(:H))
    CF.set_field!(uv, ht, (x,y,z)->z, gs, gd.iz_min, CF.NodeSet(:V))
    @test global_vector(uh[1,1,:]) ≈ global_values_h
    @test global_vector(uv[1,1,:]) ≈ global_values_v
    @test uh[2:end,1,:] ≈ zeros(31,gd.nz_h)
    @test uv[2:end,1,:] ≈ zeros(31,gd.nz_v)
    @test uh[:,2:end,:] ≈ zeros(32,30,gd.nz_h)
    @test uv[:,2:end,:] ≈ zeros(32,30,gd.nz_v)
    uh_pd = CF.zeros_pd(Float64, gd, CF.NodeSet(:H))
    uv_pd = CF.zeros_pd(Float64, gd, CF.NodeSet(:V))
    CF.get_field!(uh_pd, ht, uh, CF.NodeSet(:H))
    CF.get_field!(uv_pd, ht, uv, CF.NodeSet(:V))
    @test global_vector(uh_pd[1,1,:]) ≈ global_values_h
    @test global_vector(uv_pd[1,1,:]) ≈ global_values_v
    @test uh_pd ≈ uh_pd[1:1,1:1,:] .* ones(gd.nx_pd, gd.ny_pd, gd.nz_h)
    @test uv_pd ≈ uv_pd[1:1,1:1,:] .* ones(gd.nx_pd, gd.ny_pd, gd.nz_v)
end

test_grid_setup()
test_transform_constant()
test_transform_horizontally_varying()
test_transform_vertically_varying()
