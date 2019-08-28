using ChannelFlow, Test
import LinearAlgebra, Random, MPI, TimerOutputs
include("test_utils.jl")

const CF = ChannelFlow
const to = TimerOutputs.TimerOutput()

MPI.Init()
const show_output = MPI.Comm_rank(MPI.COMM_WORLD) == 1

show_output && println("Testing ChannelFlow.jl... (parallel, n=", MPI.Comm_size(MPI.COMM_WORLD), ")")
@testset MPITestSet "Direct Numerical Simulation" begin
    TimerOutputs.@timeit to "transform test" include("transform_test.jl")
    TimerOutputs.@timeit to "advection test" include("advection_test.jl")
    TimerOutputs.@timeit to "diffusion test" include("diffusion_test.jl")
    TimerOutputs.@timeit to "pressure test" include("pressure_solver_test.jl")
    TimerOutputs.@timeit to "output test" include("output_test.jl")
    TimerOutputs.@timeit to "integration test" include("integration_test.jl")
    show_output && (show(to); println())
end

MPI.Finalize()
