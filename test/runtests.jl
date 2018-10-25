using ChannelFlow, Test

import LinearAlgebra, MPI
include("mpi_test_tools.jl")

# shorthand for convenience, since we use a lot of unexported functions in tests
CF = ChannelFlow

println("Testing ChannelFlow.jl... (serial)")
@testset "Direc Numerical Simulation" begin
    @testset "Basics & Transform" begin include("test_transform.jl") end
    @testset "Advection Term" begin include("test_advection.jl") end
    @testset "Diffusion Term" begin include("test_diffusion.jl") end
    @testset "Pressure Solver" begin include("test_pressure_solver.jl") end
end

println("Testing ChannelFlow.jl... (parallel, n=4)")
begin run_mpi_test("runtests_mpi.jl", 4) end
