# set up minimal load path (remove system-wide packages)
deleteat!(LOAD_PATH, 1:length(LOAD_PATH))
push!(LOAD_PATH, "@", "@stdlib")

using BenchmarkTools, BoundaryLayerDynamics, MPI
MPI.Init()

# set up parameters for benchmarks
# IMPORTANT: the time limit should not be reached, otherwise it is possible
# that the MPI ranks execute the code a different number of times and get stuck
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 3600*24

# set up different combinations of horizontal/vertical resolutions
resolutions = if length(ARGS) > 2
    nh, nv = parse.(Int, ARGS[1:2])
    [(nh, nv)]
else
    # NOTE: `union` avoids including the same resolution twice
    union([(Nh, 64) for Nh = (32, 48, 64, 96, 128, 192, 256)],
          [(64, Nv) for Nv = (32, 48, 64, 96, 128)])
end

# set up & run test suite for each resolution
const suite = BenchmarkGroup()
for N in resolutions
    redirect_stdout(devnull) do # suppress output from BoundaryLayerDynamics package

        # setup problem
        # compute_rates!(rates, state, t, abl.processes, abl.physical_spaces, log, sample = checkpoint)
        abl = openchannelflow(1.0, (N[1], N[1], N[2]),
                              sgs_model = StaticSmagorinskyModel(),
                              roughness_length = 1e-6) # should be less than Δx₃
        pdiff = filter(p -> p isa BoundaryLayerDynamics.Processes.DiscretizedMolecularDiffusion, abl.processes)
        pcont = filter(p -> p isa BoundaryLayerDynamics.Processes.DiscretizedPressure, abl.processes)
        padv  = filter(p -> p isa BoundaryLayerDynamics.Processes.DiscretizedMomentumAdvection, abl.processes)
        psgs  = filter(p -> p isa BoundaryLayerDynamics.Processes.DiscretizedStaticSmagorinskyModel, abl.processes)
        pdims = [(0,0), sort(collect(keys(abl.physical_spaces)))...]
        ps0, ps1, ps2 = [filter(x -> first(x) == d, abl.physical_spaces) for d in pdims]
        rates = deepcopy(abl.state)

        # define tests for individual parts of the code
        bm = BenchmarkGroup()
        bm["diffusion"] = @benchmarkable BoundaryLayerDynamics.compute_rates!($rates, $abl.state, 0.0, $pdiff, $ps0)
        bm["sgs_stress"] = @benchmarkable BoundaryLayerDynamics.compute_rates!($rates, $abl.state, 0.0, $psgs, $ps1)
        bm["advection"] = @benchmarkable BoundaryLayerDynamics.compute_rates!($rates, $abl.state, 0.0, $padv, $ps2)
        bm["continuity"] = @benchmarkable BoundaryLayerDynamics.apply_projections!($abl.state, $pcont)

        # run tests
        warmup(bm, verbose=false) # avoid including compilation in samples
        suite[N] = run(bm)
    end
end

const mpir = MPI.Comm_rank(MPI.COMM_WORLD)
const mpis = MPI.Comm_size(MPI.COMM_WORLD)

function normalize(suite)
    if length(suite) == 1
        # drop key with size if only one entry (i.e. if size is specified from command line)
        first(values(suite))
    else
        suite
    end
end

# save serialized results to file or print to standard output
if length(ARGS) >= 1
    mpir == 1 && BenchmarkTools.save(ARGS[end], normalize(suite))
else
    mpir == 1 && BenchmarkTools.save(Base.stdout, normalize(suite))
end
