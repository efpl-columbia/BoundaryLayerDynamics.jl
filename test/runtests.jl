using ChannelFlow, Test

import LinearAlgebra, Random, MPI
include("test_utils.jl")

# shorthand for convenience, since we use a lot of unexported functions in tests
CF = ChannelFlow

println("Testing ChannelFlow.jl... (serial)")
@testset "Direct Numerical Simulation" begin
    @testset "Basics & Transform" begin include("transform_test.jl") end
    @testset "Advection Term" begin include("advection_test.jl") end
    @testset "Diffusion Term" begin include("diffusion_test.jl") end
    @testset "Pressure Solver" begin include("pressure_solver_test.jl") end
    @testset "Poiseuille Flow" begin include("integration_test.jl") end
    @testset "File I/O" begin include("output_test.jl") end
end

println("Testing ChannelFlow.jl... (parallel, n=4)")
begin run_mpi_test("runtests_mpi.jl", 4) end
