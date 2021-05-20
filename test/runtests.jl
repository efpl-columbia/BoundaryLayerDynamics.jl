using ChannelFlow, Test
import LinearAlgebra, Random, MPI, TimerOutputs
include("test_utils.jl")

const CF = ChannelFlow # shorthand for convenience, since we use a lot of unexported functions in tests
const to = TimerOutputs.TimerOutput()

# TODO: use ARGS to decide which test sets should be added & run
# (starting Julia 1.3, these can be passed using Pkg.test(..., test_args=``))
println("Testing ChannelFlow.jl... (serial)")
@testset "Direct Numerical Simulation" begin
    TimerOutputs.@timeit to "transform test" @testset "Basics & Transform" begin include("transform_test.jl") end
    TimerOutputs.@timeit to "advection test" @testset "Advection Term" begin include("advection_test.jl") end
    TimerOutputs.@timeit to "diffusion test" @testset "Diffusion Term" begin include("diffusion_test.jl") end
    TimerOutputs.@timeit to "pressure test" @testset "Pressure Solver" begin include("pressure_solver_test.jl") end
    TimerOutputs.@timeit to "output test" @testset "File I/O" begin include("output_test.jl") end
    TimerOutputs.@timeit to "integration test" @testset "Laminar 2D Flows" begin include("integration_test.jl") end
    TimerOutputs.@timeit to "les test" @testset "Large-Eddy Simulation" begin include("les_test.jl") end
    show(to); println() # TimerOutput has no newline at the end
end

run_mpi_test("runtests_mpi.jl", 4)
