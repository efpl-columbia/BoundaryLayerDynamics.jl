using BoundaryLayerDynamics, Test
const BLD = BoundaryLayerDynamics
const NS = BLD.Grids.NodeSet # for convenience

using MPI: MPI
using HDF5: HDF5
using Random: Random
using LinearAlgebra: LinearAlgebra
using TimerOutputs: @timeit, print_timer
include("test_utils.jl")

# use command-line argument to launch in MPI mode or not
const parallel = "--mpi" in ARGS
const serial = "--no-mpi" in ARGS
parallel && MPI.Init()
const nproc = parallel ? MPI.Comm_size(MPI.COMM_WORLD) : 1
const show_output = parallel ? MPI.Comm_rank(MPI.COMM_WORLD) == 0 : true
show_output && println("Testing BoundaryLayerDynamics.jl... ($(nproc == 1 ? "single process" : "$nproc processes"))")

# allow selecting individual tests through command-line arguments
# (starting Julia 1.3, these can be passed using Pkg.test(..., test_args=``))
const tests = ["grid", "transform", "derivatives", "diffusion", "advection",
               "pressure", "ode", "laminar", "output", "les", "logging", "abl"]
selection = filter(a -> !startswith(a, '-'), ARGS)
selection == ["all"] && pop!(selection)
isempty(selection) && append!(selection, tests)
let unknown = filter(t -> !(t in tests), selection)
    isempty(unknown) || error("Unknown test(s): ", join(unknown, ", "))
end

@timeit "BoundaryLayerDynamics.jl Tests" @testset MPITestSet "Atmospheric Boundary Layer Simulations" begin
    for test in selection
        include("$(test)_test.jl")
    end
end

show_output && print_timer()
parallel && MPI.Finalize()
parallel || serial || run_mpi_test("runtests.jl", 4, selection)
