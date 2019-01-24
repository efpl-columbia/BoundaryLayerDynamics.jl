using ChannelFlow, Test

import LinearAlgebra, Random, MPI
include("mpi_test_tools.jl")

# shorthand for convenience, since we use a lot of unexported functions in tests
CF = ChannelFlow

println("Testing ChannelFlow.jl... (serial)")
@testset "Direct Numerical Simulation" begin
    @testset "Basics & Transform" begin include("test_transform.jl") end
    @testset "Advection Term" begin include("test_advection.jl") end
    @testset "Diffusion Term" begin include("test_diffusion.jl") end
    @testset "Pressure Solver" begin include("test_pressure_solver.jl") end
    @testset "Poiseuille Flow" begin include("test_integration.jl") end
    @testset "File I/O" begin include("test_output.jl") end
end

println("Testing ChannelFlow.jl... (parallel, n=4)")
begin run_mpi_test("runtests_mpi.jl", 4) end
