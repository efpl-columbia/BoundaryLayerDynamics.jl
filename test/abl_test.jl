function test_abl_setup()

    Re = 1e6
    N = 64
    L = (4π, 2π, 1)

    domain = Domain(L, RoughWall(1e-4), FreeSlipBoundary())
    processes = incompressible_flow(Re)
    abl = DiscretizedABL(N, domain, incompressible_flow(Re))

    io = IOBuffer()
    show(io, MIME("text/plain"), abl)
    @test String(take!(io)) == """
        Discretized Atmospheric Boundary Layer:
        → κ₁ ∈ [−31,31], κ₂ ∈ [−31,31], i₃ ∈ [1,64]"""
end

@timeit "ABL Simulation" @testset "Detailed ABL Flow Setup" begin
    test_abl_setup()
end
