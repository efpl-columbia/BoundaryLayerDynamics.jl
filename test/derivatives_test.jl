# set up new process that just requests the vorticity as input and checks if it
# matches some reference
struct Vorticity <: BLD.Processes.DiscretizedProcess
    dealiasing
    reference
    Vorticity(dealiasing = :quadratic; kwargs...) = new(dealiasing, kwargs)
end
struct VorticityTest <: BLD.Processes.DiscretizedProcess
    dims
    reference
    coords
end

BLD.Processes.state_fields(::VorticityTest) = (:vel1, :vel2, :vel3)
BLD.Processes.physical_domain_terms(adv::VorticityTest) =
    Tuple(f => adv.dims for f in (:vort1, :vort2, :vort3))
BLD.Processes.physical_domain_rates(::VorticityTest) = ()
BLD.Processes.init_process(vort::Vorticity, domain::Domain{T}, grid) where T = begin
    dims = BLD.PhysicalSpace.pdsize(grid, vort.dealiasing)
    x1 = BLD.x1range(domain, BLD.PhysicalSpace.h1range(grid, dims))
    x2 = BLD.x2range(domain, BLD.PhysicalSpace.h2range(grid, dims))
    x3c = BLD.x3range(domain, BLD.Grids.vrange(grid, NS(:C)))
    x3i = BLD.x3range(domain, BLD.Grids.vrange(grid, NS(:I)))
    VorticityTest(dims, vort.reference, (x1, x2, x3c, x3i))
end
BLD.Processes.add_rates!(_rates, term::VorticityTest, state, _t, _log) = begin
    state = state[term.dims]
    for (f, ref) in pairs(term.reference)
        ns = BLD.Grids.nodes(f)
        x1, x2 = term.coords[1:2]
        x3 = term.coords[ns isa NS{:C} ? 3 : ns isa NS{:I} ? 4 : error("Invalid field")]

        @test state[f] ≈ [ref(x1, x2, x3) for x1=x1, x2=x2, x3=x3]
    end
end


# compute vorticity in FS, transform, and check if it is correct
function test_vorticity_fs(n3 = 4)

    k = 2
    u0(x,y,z) = 0.856 * sin(k*x)
    vort3(x,y,z) = 0.856 * k * cos(k*x) # d/x1 vel2 - dx/2 vel1

    dims = (12, 14, n3)
    domain = Domain((2*π, 2*π, 1.0), SmoothWall(), SmoothWall())
    model = Model(dims, domain, [Vorticity(vort3 = vort3)])

    initialize!(model, vel2 = u0)
    BLD.Processes.compute_rates!(deepcopy(model.state), model.state, 0.0, model.processes, model.physical_spaces)

end

@timeit "Derivatives" @testset "Spatial Derivatives" begin
    # have at least one layer per process
    n = MPI.Initialized() ? max(MPI.Comm_size(MPI.COMM_WORLD), 3) : 3
    test_vorticity_fs(n)
end
