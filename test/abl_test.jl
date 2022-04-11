using ABL, Test, MPI

# helper to test grid distribution without actual MPI setup
ABL.Grid.init_processes(comm::Tuple) = (comm, comm...)

function test_grid()
    @test ABL.DistributedGrid(64, comm=(1, 32)).n3c == 2
    @test [ABL.DistributedGrid(64, comm=(i, 8)).n3i for i=1:8] == [8,8,8,8,8,8,8,7]
    @test [ABL.DistributedGrid(60, comm=(i, 8)).n3c for i=1:8] == [8,8,8,8,7,7,7,7]
    @test [getproperty(ABL.DistributedGrid((64, 32, 16)), i) for i in
           (:k1max, :k2max, :n3c, :n3i, :n3global, :i3min, :i3max)] == [31, 15, 16, 15, 16, 1, 16]
    @test ABL.DistributedGrid((64, 32, 16)) == ABL.DistributedGrid((63, 31, 16))
end

function test_abl_setup()

    Re = 1e6
    N = 64
    L = (4π, 2π, 1)

    domain = SemiperiodicDomain(L, RoughWall(1e-4), FreeSlipBoundary())
    processes = incompressible_flow(Re)
    abl = DiscretizedABL(N, domain, incompressible_flow(Re))

    io = IOBuffer()
    show(io, MIME("text/plain"), abl)
    @test String(take!(io)) == """
        Discretized Atmospheric Boundary Layer:
        → κ₁ ∈ [−31,31], κ₂ ∈ [−31,31], i₃ ∈ [1,64]"""
end

function setup_random_fields(T, n1, n2, n3)
    gd = ABL.DistributedGrid((n1, n2, n3))
    uh = ABL.zeros(T, gd, ABL.NodeSet(:C))
    uv = ABL.zeros(T, gd, ABL.NodeSet(:I))
    fill!(uh, rand(Complex{T}))
    fill!(uv, rand(Complex{T}))
    ht = ABL.HorizontalTransform(T, gd, ABL.pdsize(gd))
    gd, ht, uh, uv
end

global_sum(x) = MPI.Initialized() ? MPI.Allreduce(x, +, MPI.COMM_WORLD) : x
global_vector(x) = MPI.Initialized() ? MPI.Allgatherv(x, convert(Vector{Cint},
        MPI.Allgather(length(x), MPI.COMM_WORLD)), MPI.COMM_WORLD) : x



@testset "Atmospheric Boundary Layer Simulations" begin

    @testset "Distributed Grid" begin
        test_grid()
    end

    @testset "Molecular Diffusion" begin
        include("diffusion_test.jl")
        test_diffusion(16)
        MPI.Initialized() && test_diffusion(MPI.Comm_size(MPI.COMM_WORLD))
    end

    @testset "Detailed ABL Flow Setup" begin
        test_abl_setup()
    end
end
