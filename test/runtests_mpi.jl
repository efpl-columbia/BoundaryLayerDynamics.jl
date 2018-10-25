using ChannelFlow, Test
import LinearAlgebra, MPI

include("mpi_test_tools.jl")

CF = ChannelFlow
MPI.Init()

@testset MPITestSet "Basic MPI Tests" begin
    include("test_transform.jl")
    include("test_advection.jl")
    include("test_diffusion.jl")
    include("test_pressure_solver.jl")
end

MPI.Finalize()
