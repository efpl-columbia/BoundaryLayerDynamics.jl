# set up new process that just requests the vorticity as input and checks if it
# matches some reference
struct Vorticity
    dealiasing
    reference
    Vorticity(dealiasing = :quadratic; kwargs...) = new(dealiasing, kwargs)
end
struct VorticityTest
    dims
    reference
    coords
end

ABL.Processes.state_fields(::VorticityTest) = (:vel1, :vel2, :vel3)
ABL.Processes.physical_domain_terms(adv::VorticityTest) =
    Tuple(f => adv.dims for f in (:vort1, :vort2, :vort3))
ABL.Processes.physical_domain_rates(::VorticityTest) = ()
ABL.Processes.init_process(vort::Vorticity, domain::Domain{T}, grid) where T = begin
    dims = ABL.PhysicalSpace.pdsize(grid, vort.dealiasing)
    x1 = ABL.x1range(domain, ABL.PhysicalSpace.h1range(grid, dims))
    x2 = ABL.x2range(domain, ABL.PhysicalSpace.h2range(grid, dims))
    x3c = ABL.x3range(domain, ABL.Grids.vrange(grid, NS(:C)))
    x3i = ABL.x3range(domain, ABL.Grids.vrange(grid, NS(:I)))
    VorticityTest(dims, vort.reference, (x1, x2, x3c, x3i))
end
ABL.Processes.add_rate!(_rates, term::VorticityTest, state, _t, _log) = begin
    state = state[term.dims]
    for (f, ref) in pairs(term.reference)
        ns = ABL.nodes(f)
        x1, x2 = term.coords[1:2]
        x3 = term.coords[ns isa NS{:C} ? 3 : ns isa NS{:I} ? 4 : error("Invalid field")]

        @test state[f] ≈ [ref(x1, x2, x3) for x1=x1, x2=x2, x3=x3]
    end
end


# compute vorticity in FS, transform, and check if it is correct
function test_vorticity_fs()

    k = 2
    u0(x,y,z) = 0.856 * sin(k*x)
    vort3(x,y,z) = 0.856 * k * cos(k*x) # d/x1 vel2 - dx/2 vel1

    dims = (12, 14, 4)
    domain = Domain((2*π, 2*π, 1.0), SmoothWall(), SmoothWall())
    abl = DiscretizedABL(dims, domain, [Vorticity(vort3 = vort3)])

    initialize!(abl, vel2 = u0)
    ABL.Processes.compute_rates!(deepcopy(abl.state), abl.state, 0.0, abl.processes, abl.physical_spaces)

end

@timeit "Derivatives" @testset "Spatial Derivatives" begin
    test_vorticity_fs()
end
