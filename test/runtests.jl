using ABL, Test

using MPI: MPI
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
show_output && println("Testing ABL.jl... ($(nproc == 1 ? "single process" : "$nproc processes"))")

# allow selecting individual tests through command-line arguments
# (starting Julia 1.3, these can be passed using Pkg.test(..., test_args=``))
tests = ["grid", "transform", "diffusion", "advection", "pressure", "ode", "abl"]
selection = filter(a -> !startswith(a, '-'), ARGS)
if !isempty(selection) && selection != ["all"]
    filter!(t -> t in selection, tests)
end

@timeit "ABL.jl Tests" @testset MPITestSet "Atmospheric Boundary Layer Simulations" begin
    for test in tests
        include("$(test)_test.jl")
    end
end

show_output && print_timer()
parallel && MPI.Finalize()
parallel || serial || run_mpi_test("runtests.jl", 4, tests)
