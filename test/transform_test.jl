function test_transform_constant()
    gd, ht, uh, uv = setup_random_fields(Float64, 64, 32, 16)
    ds = (3*0.5176366701850579, 2*0.9360371626346338, 0.10477115957769456)
    domain = ABL.SemiperiodicDomain(ds, nothing, nothing)
    ABL.set_field!((x,y,z)->1, uh, ht, domain, gd, ABL.NodeSet(:C))
    ABL.set_field!((x,y,z)->2, uv, ht, domain, gd, ABL.NodeSet(:I))
    @test uh[1,1,:] ≈ 1 * ones(gd.n3c)
    @test uv[1,1,:] ≈ 2 * ones(gd.n3i)
    @test uh[2:end,1,:] ≈ zeros(31,gd.n3c)
    @test uv[2:end,1,:] ≈ zeros(31,gd.n3i)
    @test uh[:,2:end,:] ≈ zeros(32,30,gd.n3c)
    @test uv[:,2:end,:] ≈ zeros(32,30,gd.n3i)
    uh_pd = ABL.get_field(ht, uh)
    uv_pd = ABL.get_field(ht, uv)
    npd = ABL.default_size(gd)
    @test uh_pd ≈ 1 * ones(npd..., gd.n3c)
    @test uv_pd ≈ 2 * ones(npd..., gd.n3i)
end

function test_transform_horizontally_varying()
    gd, ht, uh, uv = setup_random_fields(Float64, 16, 16, 16)
    ds = (3*0.5176366701850579, 2*0.9360371626346338, 0.10477115957769456)
    domain = ABL.SemiperiodicDomain(ds, nothing, nothing)
    k = -7:8
    ax = [rand() for k=k]
    ay = [rand() for k=k]
    u0(x,y,z) = sum(ax .* [cos(2*π*k*x/domain.hsize[1]) for k=k]) +
                sum(ay .* [cos(2*π*k*y/domain.hsize[2]) for k=k])
    u0n(x,y,z) = sum((ax .* [cos(2*π*k*x/domain.hsize[1]) for k=k])[1:end-1]) +
                 sum((ay .* [cos(2*π*k*y/domain.hsize[2]) for k=k])[1:end-1])
    ABL.set_field!(u0, uh, ht, domain, gd, ABL.NodeSet(:C))
    @test uh[1,1,:] ≈ (ax[8] + ay[8]) .* ones(gd.n3c)
    @test uh[2:end,1,:] ≈ (ax[9:end-1] + ax[7:-1:1]) / 2 .* ones(1,gd.n3c)
    @test uh[1,2:8,:] ≈ (ay[9:end-1] + ay[7:-1:1]) / 2 .* ones(1,gd.n3c)
    @test uh[1,end:-1:9,:] ≈ (ay[9:end-1] + ay[7:-1:1]) / 2 .* ones(1,gd.n3c)
    # do not compare almost-zero values directly with zero, since the algorithm
    # of “≈” checks for relative errors, which makes it too hard to pass “≈ 0”
    @test uh[2:end,2:end,:] .+ 1 ≈ ones(gd.k1max, 2*gd.k2max, gd.n3c)
    uh_pd = ABL.get_field(ht, uh)
    @test uh_pd ≈ [u0n(x, y, z) for x=(0:size(uh_pd,1)-1)/size(uh_pd,1)*domain.hsize[1],
                                    y=(0:size(uh_pd,2)-1)/size(uh_pd,2)*domain.hsize[2],
                                    z=zeros(size(uh_pd,3))]
end

function test_transform_vertically_varying()
    gd, ht, uh, uv = setup_random_fields(Float64, 64, 32, 16)
    ds = (3*0.5176366701850579, 2*0.9360371626346338, 0.10477115957769456)
    domain = ABL.SemiperiodicDomain(ds, nothing, nothing)
    global_values_h = LinRange(0, ds[3], 2*gd.n3global+1)[2:2:end-1]
    global_values_v = LinRange(0, ds[3], 2*gd.n3global+1)[3:2:end-2]
    ABL.set_field!((x,y,z)->z, uh, ht, domain, gd, ABL.NodeSet(:C))
    ABL.set_field!((x,y,z)->z, uv, ht, domain, gd, ABL.NodeSet(:I))
    @test global_vector(uh[1,1,:]) ≈ global_values_h
    @test global_vector(uv[1,1,:]) ≈ global_values_v
    @test uh[2:end,1,:] ≈ zeros(31,gd.n3c)
    @test uv[2:end,1,:] ≈ zeros(31,gd.n3i)
    @test uh[:,2:end,:] ≈ zeros(32,30,gd.n3c)
    @test uv[:,2:end,:] ≈ zeros(32,30,gd.n3i)
    uh_pd = ABL.get_field(ht, uh)
    uv_pd = ABL.get_field(ht, uv)
    n1pd, n2pd = ABL.default_size(gd)
    @test global_vector(uh_pd[1,1,:]) ≈ global_values_h
    @test global_vector(uv_pd[1,1,:]) ≈ global_values_v
    @test uh_pd ≈ uh_pd[1:1,1:1,:] .* ones(ABL.default_size(gd)..., gd.n3c)
    @test uv_pd ≈ uv_pd[1:1,1:1,:] .* ones(ABL.default_size(gd)..., gd.n3i)
end

@timeit "Transforms" @testset "Horizontal Transforms" begin
    test_transform_constant()
    test_transform_horizontally_varying()
    test_transform_vertically_varying()
end
