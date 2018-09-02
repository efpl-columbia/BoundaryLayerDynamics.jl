SupportedReals = Union{Float32,Float64}

struct Grid{T<:SupportedReals} # only allow Float64 for now
    n::Tuple{Int,Int,Int}
    l::Tuple{T,T,T}
    δ::Tuple{T,T,T}
    k::Tuple{Array{Int,1},Array{Int,1}}
    Grid(n::Tuple{Int,Int,Int}, l::Tuple{T,T,T}) where T<:SupportedReals = new{T}(
            n, l, l./n, (wavenumbers(n[1])[1:div(n[1],2)+1], wavenumbers(n[2])))
end

Grid(n) = Grid(n, (2π, 2π, 1.0))
Grid(n::Integer, l::NTuple{3,Real}) = Grid((n, n, n), l)
Grid(n::NTuple{3,Integer}, l::NTuple{3,Real}) =
        Grid(map(ni -> convert(Int, ni), n), map(li -> convert(Float64, li), l))

abstract type BoundaryCondition{T} end
struct DirichletBC{T} <: BoundaryCondition{T} value::T end
struct NeumannBC{T} <: BoundaryCondition{T}
    value::T # values has δz premultiplied
    NeumannBC(value::T, δz::T) where T = new{T}(value * δz)
end

struct BigTransform{T<:SupportedReals}
    n::Tuple{Int,Int,Int}
    plan_fwd::FFTW.rFFTWPlan{T,FFTW.FORWARD,false,3}
    plan_bwd::FFTW.rFFTWPlan{Complex{T},FFTW.BACKWARD,false,3}
    buffers_pd::NTuple{9,Array{T,3}}
    buffers_fd::NTuple{1,Array{Complex{T},3}}
    buffer_layers_pd::NTuple{5,Array{T,2}}
    BigTransform(gd::Grid{T}) where T = begin
        n = (3*div(gd.n[1],2), 3*div(gd.n[2],2), gd.n[3])
        buffers_pd = Tuple(zeros(T, n) for i=1:9)
        buffers_fd = Tuple(zeros(Complex{T}, div(n[1],2)+1, n[2], n[3]) for i=1:1)
        new{T}(n, plan_rfft(buffers_pd[1], (1,2)), plan_brfft(buffers_fd[1], n[1], (1,2)),
            buffers_pd, buffers_fd, Tuple(zeros(T, n[1], n[2]) for i=1:5))
    end
end

struct DerivativeFactors{T<:SupportedReals}
    dx1::Array{Complex{T},3}
    dy1::Array{Complex{T},3}
    dz1::T
    dx2::Array{T,3}
    dy2::Array{T,3}
    dz2::T
    DerivativeFactors(gd::Grid{T}) where T = new{T}(
            reshape(1im * gd.k[1] * (2π/gd.l[1]), length(gd.k[1]), 1, 1),
            reshape(1im * gd.k[2] * (2π/gd.l[2]), 1, length(gd.k[2]), 1),
            1/gd.δ[3],
            reshape( - gd.k[1].^2 * (2π/gd.l[1])^2, length(gd.k[1]), 1, 1),
            reshape( - gd.k[2].^2 * (2π/gd.l[2])^2, 1, length(gd.k[2]), 1),
            1/gd.δ[3]^2,
    )
end

"""
Structure holding the data to efficiently solve a series of symmetric
tridiagonal linear systems. The input is an iterator that returns a series of
symmetric tridiagonal matrices. For each matrix, the constructor prepares two
arrays to apply the Thomas algorithm with a minimal amount of operations.

The implementatioh of the Thomas algorithm closely follows the “modthomas”
routine in section 3.7.2 in the 2nd ed. of the book “Numerical Mathematics” by
Quarteroni, Sacco & Saleri, modified to take advantage of the symmetrical
structure of the matrix.
"""
struct SymThomasBatch{T}
    γ::Array{T,2}
    β::Array{T,2}
    SymThomasBatch{T}(As, nz) where T = begin
        nk = length(As) # batch size
        γ = zeros(T, nk, nz)
        β = zeros(T, nk, nz-1)
        for (ik,A::LinearAlgebra.SymTridiagonal{T,Array{T,1}}) in enumerate(As)
            size(A) == (nz,nz) || error("Matrix size not compatible")
            γ[ik,1] = 1 / A.dv[1]
            for iz = 2:nz
                γ[ik,iz] = 1 / (A.dv[iz] - A.ev[iz-1].^2 * γ[ik,iz-1])
            end
            β[ik,:] = copy(A.ev)
        end
        new{T}(γ, β)
    end
end

"""
This is a new attempt at how to best implement the channel flow DNS in Julia.
This time, we compute everything in frequency domain, only going back to the
physical domain when necessary. Currently, this is only the case for the
non-linear advection term, for which we need pass through Fourier space anyway
in order to perform the dealiasing.

Currently, there are (almost) complete implementations of the viscous term as
well as the non-linear advection term. We are still missing the initialization,
which could be performed with dealiasing as well, adding the pressure terms to
the RHS, and the pressure solver, as well as the time stepping.
"""
struct ChannelFlowProblem{T<:SupportedReals}

    # problem definition
    grid::Grid{T}
    lower_bc::NTuple{3,BoundaryCondition{T}}
    upper_bc::NTuple{3,BoundaryCondition{T}}
    forcing::NTuple{3,T}

    # state
    vel_hat::NTuple{3,OffsetArray{Complex{T},3,Array{Complex{T},3}}}
    #p_hat::OffsetArray{Complex{T},3,Array{Complex{T},3}}
    p_hat::Array{Complex{T},3}

    # buffered values
    rhs_hat::NTuple{3,Array{Complex{T},3}}
    rot_hat::NTuple{3,Array{Complex{T},3}}
    tf_big::BigTransform{T}
    df::DerivativeFactors{T}
    p_solver::SymThomasBatch{T}

    # inner constructor
    ChannelFlowProblem(grid::Grid{T}, lower_bc::NTuple{3,BoundaryCondition{T}},
            upper_bc::NTuple{3,BoundaryCondition{T}}, forcing::NTuple{3,T},
            ) where T = new{T}(grid, lower_bc, upper_bc, forcing,
            Tuple(make_buffered_array_fd(grid) for i=1:3), # vel_hat
            make_array_fd(grid), # phat
            Tuple(make_array_fd(grid) for i=1:3), # rhs_hat
            Tuple(make_array_fd(grid) for i=1:3), # rot_hat
            BigTransform(grid), DerivativeFactors(grid),
            SymThomasBatch{T}((prepare_laplacian_fd(grid, kx, ky)
                    for kx=grid.k[1], ky=grid.k[2]), grid.n[3]),
        )
end

ChannelFlowProblem(grid::Grid{T}) where T = ChannelFlowProblem(grid,
        noslip(T), noslip(T), (one(T), zero(T), zero(T)))

# small helper functions -------------------------------------------------------

# wavenumbers in fft order, with zero for nyquist frequency
wavenumbers(n) = map(i -> i<n/2 ? i : i>n/2 ? i-n : 0 , 0:n-1)

noslip(T) = (DirichletBC{T}(zero(T)), DirichletBC{T}(zero(T)),
        DirichletBC{T}(zero(T)))
freeslip(T) = (NeumannBC{T}(zero(T), zero(T)), NeumannBC{T}(zero(T), zero(T)),
        DirichletBC{T}(zero(T)))

make_array_fd(gd::Grid{T}) where T = Array(zeros(Complex{T},
        div(gd.n[1],2)+1, gd.n[2], gd.n[3]))

make_buffered_array_fd(gd::Grid{T}) where T = OffsetArray(zeros(Complex{T},
        div(gd.n[1],2)+1, gd.n[2], gd.n[3]+2),
        1:div(gd.n[1],2)+1, 1:gd.n[2], 0:gd.n[3]+1)
