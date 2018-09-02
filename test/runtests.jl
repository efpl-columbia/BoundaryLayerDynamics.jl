using ChannelFlow, Test

import LinearAlgebra, MPI

CF = ChannelFlow

println("Testing ChannelFlow.jl...")

function test_basics()

    @test all(CF.wavenumbers(8) .== (0,1,2,3,0,-3,-2,-1))
    @test all(CF.wavenumbers(9) .== (0,1,2,3,4,-4,-3,-2,-1))

    gd = CF.Grid(64)
    @test all(gd.δ .== (π/32, π/32, 1/64))

    p = CF.ChannelFlowProblem(gd)
    @test p.vel_hat[1][1,2,0] == 0

    p.vel_hat[1][:,:,1:end-1] .= 1
    CF.set_lower_bc!(p.vel_hat[1], p.lower_bc[1])
    CF.set_upper_bc!(p.vel_hat[1], p.upper_bc[1])
    @test p.vel_hat[1][1,2,0] == -1
    @test p.vel_hat[1][1,2,end] == -1

    CF.add_laplacian_fd!(p.rhs_hat[1], p.vel_hat[1], CF.DerivativeFactors(gd))
    @test p.rhs_hat[1][1,1,1] ≈ -2*64^2 # Laplacian at boundary for kx=ky=0

    CF.initialize!(p, (x,y,z) -> cos(x))
    @test all(p.vel_hat[1][2,1,1:end-1] .≈ 0.5)
    @test p.vel_hat[1][2,1,0] ≈ -0.5
    @test p.vel_hat[1][2,1,end] ≈ -0.5

    CF.integrate!(p, 1e-6, 1)
end

@testset "Basics" begin
    test_basics()
end

exit()

@testset "Horizontal Derivatives" begin

    function test_dx(N=64)
        u = zeros(N,N,N)
        dudx = similar(u)
        gd = ChannelFlow.Grid((N,N,N), (1.0,1.0,1.0))

        f = Float64[sin(2*π*i/N)^2 for i=0:N-1]
        dfdx = Float64[4*π*sin(2*π*i/N)*cos(2*π*i/N) for i=0:N-1]
        for j=1:N, k=1:N
            u[:,j,k] = f[:]
        end

        tf = ChannelFlow.prepare_state(gd)[3]
        u_hat = similar(tf.buffer)
        LinearAlgebra.mul!(u_hat, tf.plan_fwd, u)
        ChannelFlow.ddx!(dudx, u_hat, tf)
        err = broadcast(-, dudx, reshape(dfdx, (N,1,1)))
        maximum(abs.(err))
    end

    function test_dy(N=64)
        u = zeros(N,N,N)
        dudy = similar(u)
        gd = ChannelFlow.Grid((N,N,N), (1.0,1.0,1.0))

        f = Float64[sin(2*π*i/N)^2 for i=0:N-1]
        dfdy = Float64[4*π*sin(2*π*i/N)*cos(2*π*i/N) for i=0:N-1]
        for i=1:N, j=1:N, k=1:N
            u[i,j,k] = f[j]
        end

        tf = ChannelFlow.prepare_state(gd)[3]
        u_hat = similar(tf.buffer)
        LinearAlgebra.mul!(u_hat, tf.plan_fwd, u)
        ChannelFlow.ddy!(dudy, u_hat, tf)
        err = broadcast(-, dudy, reshape(dfdy, (1,N,1)))
        maximum(abs.(err))
    end

    @test test_dx() < 1e-12
    @test test_dy() < 1e-12
end


@testset "Thomas Algorithm for Symmetric Tridiagonal System" begin

    N = 1024

    # define random system to be solved
    A = LinearAlgebra.SymTridiagonal(rand(N), rand(N-1))
    b = rand(N)

    # comparison: integrated solver
    Â = LinearAlgebra.ldlt(A) # LDL' factorization
    x = Â \ b
    ε = maximum(abs.(A*x-b))

    # thomas algorithm for symmetric matrix
    T = ChannelFlow.SymThomas(A)
    x_st = T \ b
    ε_st = maximum(abs.(A*x_st-b))

    ε_rel = ε/ε_st
    @test ε_rel < 100

    # run again, testing performance with preallocated output
    t, t_st = 0.0, 0.0
    for i=1:ceil(Int, 1e6/N)
        t    += @elapsed LinearAlgebra.ldiv!(x, Â, b)
        t_st += @elapsed LinearAlgebra.ldiv!(x, T, b)
    end
    t_rel = t/t_st
    @test t_rel > 0.8 # not more than 20% slower
end

@testset "Simple Channel Flow" begin
    MPI.Init()
    (MPI.Comm_size(MPI.COMM_WORLD) > 1 ? ChannelFlow.channelflow_mpi :
        ChannelFlow.channelflow)(
            ChannelFlow.Grid((64, 64, 64), (2π, 2π, 1)),
            (dt=1e-3, nt=3),
            (x,y,z) -> 0.0,#1.0 + sin(z),
        )
    MPI.Finalize()
    @test true # only test that everything returns without error
end
