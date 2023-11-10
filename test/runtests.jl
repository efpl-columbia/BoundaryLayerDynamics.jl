using BoundaryLayerDynamics, Test
const BLD = BoundaryLayerDynamics
const NS = BLD.Grids.NodeSet # for convenience

using MPI: MPI
using HDF5: HDF5
using Random: Random
using LinearAlgebra: LinearAlgebra
using TimerOutputs: @timeit, print_timer
include("test_utils.jl")

# allow selecting individual tests through command-line arguments
# (starting Julia 1.3, these can be passed using Pkg.test(..., test_args=``))
const tests = map(fn -> first(split(fn, '_')), filter(endswith("_test.jl"), readdir(@__DIR__)))
selection = filter(a -> !startswith(a, '-'), ARGS)
selection == ["all"] && pop!(selection)
isempty(selection) && append!(selection, tests)
let unknown = filter(t -> !(t in tests), selection)
    isempty(unknown) || error("Unknown test(s): ", join(unknown, ", "))
end

# use command-line argument to launch in MPI mode or not
const nolaunch = "--no-mpi" in ARGS # do not launch parallel run after this one
const parallel = "--mpi" in ARGS # current run is in parallel mode
let i = findfirst(startswith("--mpi="), ARGS)
    if !isnothing(i)
        run_mpi_test("runtests.jl", parse(Int, split(ARGS[i], '=')[2]), selection)
        exit(0) # skip single-process tests when nproc is given explicitly
    end
end

# run tests for current (single- or multiple-process) run
parallel && MPI.Init()
const show_output = parallel ? MPI.Comm_rank(MPI.COMM_WORLD) == 0 : true
show_output && println("Testing BoundaryLayerDynamics.jl... ($(parallel ? "$(MPI.Comm_size(MPI.COMM_WORLD)) processes" : "single process"))")
@timeit "BoundaryLayerDynamics.jl Tests" @testset MPITestSet "Atmospheric Boundary Layer Simulations" begin
    for test in selection
        include("$(test)_test.jl")
    end
end
show_output && print_timer()
parallel && MPI.Finalize()

# launch parallel tests after single-process run
parallel || nolaunch || run_mpi_test("runtests.jl", 4, selection)
