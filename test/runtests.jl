using ChannelFlow, Test

import LinearAlgebra, MPI

println("Testing ChannelFlow.jl...")

@testset "Horizontal Derivatives" begin

    function test_dx(N=64)
        u = zeros(N,N,N)
        dudx = similar(u)
        gd = (lx=1.0,ly=1.0,lz=1.0,nx=N,ny=N,nz=N)

        f = Float64[sin(2*π*i/N)^2 for i=0:N-1]
        dfdx = Float64[4*π*sin(2*π*i/N)*cos(2*π*i/N) for i=0:N-1]
        for j=1:N, k=1:N
            u[:,j,k] = f[:]
        end

        tf = ChannelFlow.prepare_state(Float64, gd)[3]
        u_hat = similar(tf.buffer)
        LinearAlgebra.mul!(u_hat, tf.plan_fwd, u)
        ChannelFlow.ddx!(dudx, u_hat, tf)
        err = broadcast(-, dudx, reshape(dfdx, (N,1,1)))
        maximum(abs.(err))
    end

    function test_dy(N=64)
        u = zeros(N,N,N)
        dudy = similar(u)
        gd = (lx=1.0,ly=1.0,lz=1.0,nx=N,ny=N,nz=N)

        f = Float64[sin(2*π*i/N)^2 for i=0:N-1]
        dfdy = Float64[4*π*sin(2*π*i/N)*cos(2*π*i/N) for i=0:N-1]
        for i=1:N, j=1:N, k=1:N
            u[i,j,k] = f[j]
        end

        tf = ChannelFlow.prepare_state(Float64, gd)[3]
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
            (lx=2π, ly=2π, lz=1, nx=64, ny=64, nz=64),
            (dt=1e-3, nt=3),
            (x,y,z) -> 0.0,#1.0 + sin(z),
        )
    MPI.Finalize()
    @test true # only test that everything returns without error
end
