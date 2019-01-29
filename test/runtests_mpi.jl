using ChannelFlow, Test
import LinearAlgebra, Random, MPI

include("test_utils.jl")

CF = ChannelFlow
MPI.Init()

@testset MPITestSet "Basic MPI Tests" begin
    include("transform_test.jl")
    include("advection_test.jl")
    include("diffusion_test.jl")
    include("pressure_solver_test.jl")
    include("integration_test.jl")
    include("output_test.jl")
end

MPI.Finalize()
