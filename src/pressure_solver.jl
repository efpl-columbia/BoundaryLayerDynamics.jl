"""
The DistributedBatchLDLt{P,Tm,Tv,B} holds the data of a decomposition A=L·D·Lᵀ
for a batch of symmetric tridiagonal matrices A. The purpose is to solve a
system A·x=b efficiently for several A, x, and b. The vectors and matrices are
distributed across MPI processes, with `P` designating the role of the current
MPI process. `Tm` is the element type of the matrices A while `Tv` is the
element type of the vectors x and b.

Since the algorithm for solving the linear system is sequential by nature, there
is a tradeoff between solving many systems at once, maximizing the throughput of
each indvidual solution, and solving smaller batches of systems, minimizing the
time MPI processes have to wait for each other. This tradeoff is represented by
the value of `B`, which is the number of systems solved in batch at a time.
Ideally the optimal value should be found automatically by measuring the
computation time for different values of `B` and choosing the optimal value.

The type is constructed from an iterable type (such as a generator) holding the
diagonal vectors `dvs` and another one holding the off-diagonals `evs`. The
diagonals are distributed amongst MPI processes like variables on H-nodes while
the off-diagonals are distributed like variables on V-nodes. An example batch of
b-vectors (or x-vectors) is also passed to the constructor, since those might
have a different element type than the matrices.

Internally, we store a set of vectors γ and another set of vectors β,
representing the L and D matrices of the L·D·Lᵀ decomposition. D is the diagonal
matrix with γ as its diagnoal vector while L is the matrix with γ¯¹ as the
diagonal and β as the lower off-diagonal. Another vector is allocated as buffer
for the values exchanged between MPI processes when solving the system A·x=b.

The algorithm used for the decomposition and the solution of the linear systems
can be found in section 3.7.2 of the book “Numerical Mathematics” by Quarteroni,
Sacco & Saleri (2007).
"""
struct DistributedBatchLDLt{P,Tm,Tv,B} #<: AbstractMatrix{T}

    γ::Matrix{Tm} # diagonal
    β::Matrix{Tm} # off-diagonal
    buffer::Vector{Tv}

    DistributedBatchLDLt(::Type{Tm}, dvs, evs, rhs, batch_size) where Tm = begin

        P = proc_type()
        nk = length(dvs)
        nz = size(rhs, ndims(rhs)) # last dimension of rhs
        length(dvs) == length(evs) == prod(size(rhs)[1:ndims(rhs)-1]) ||
                error("Batch size of dv (nk=", length(dvs), "), ev (nk=",
                length(evs), "), and rhs (nk=", prod(size(rhs)[1:ndims(rhs)-1]),
                ") not compatible")

        nz_dv = nz
        nz_ev = (P == SingleProc ? nz - 1 : P == InnerProc ? nz + 1 : nz)
        iz_min_ev = (P <: LowestProc) ? 1 : 2

        γ = zeros(Tm, nk, nz_dv)
        β = zeros(Tm, nk, nz_ev)

        # copy elements from iterable inputs to internal arrays
        for (ik, dv, ev) in zip(1:nk, dvs, evs)
            eltype(dv) == eltype(ev) == Tm || error("Incompatible element types")
            ndims(dv) == ndims(ev) == 1 || error("More than one dimensions for diagonal vectors")
            length(dv) == nz_dv || error("Length of diagonal (", length(dv), ") not equal to ", nv_dv)
            length(ev) == length(iz_min_ev:nz_ev) || error("Length of diagonal (", length(dv), ") not equal to ", length(iz_min_ev:nv_dv))
            copyto!(view(γ, ik, :), dv)
            copyto!(view(β, ik, iz_min_ev:nz_ev), ev)
        end

        # add missing ev-values, obtaining them from the neighbor below
        if !(P <: HighestProc)
            send_to_proc_above(view(β, :, nz_ev))
        end
        if !(P <: LowestProc)
            get_from_proc_below!(view(β, :, 1))
        end

        # compute gamma (TODO: transform to batched version if too slow)
        if !(P <: LowestProc)
            γ_below = zeros(Tm, nk)
            get_from_proc_below!(γ_below)
            @views @. γ[:,1] = 1 / (γ[:,1] - γ_below * β[:,1]^2)
        else
            @views @. γ[:,1] = 1 / γ[:,1]
        end
        for i=2:nz
            @views @. γ[:,i] = 1 / (γ[:,i] - γ[:,i-1] * β[:,(iz_min_ev-1)+i-1]^2)
        end
        if !(P <: HighestProc)
            send_to_proc_above(view(γ, :, nz_dv))
        end

        Tv, B = eltype(rhs), batch_size
        new{P,Tm,Tv,B}(γ, β, zeros(Tv, B))
    end
end

function symthomas_batch_fwd!(b::AbstractArray{Tv,2},
    γ::AbstractArray{Tm,2}, β::AbstractArray{Tm,2},
    buffer::AbstractArray{Tv,1}, ::Val{P}) where {P<:ProcType,Tm,Tv}
    nk, nz = size(b)
    if !(P <: LowestProc)
        b_orig = b[1]
        get_from_proc_below!(buffer)
        for ik=1:nk
            b[ik,1] = γ[ik,1] * (b[ik,1] - β[ik,1] * buffer[ik])
        end
    else
        for ik=1:nk
            b[ik,1] = γ[ik,1] * b[ik,1]
        end
    end
    for iz=2:nz
        iz_β = (P <: LowestProc) ? iz-1 : iz
        #@views @. b[:,iz] = γ[:,iz] * (b[:,iz] - β[:,iz_β] * b[:,iz-1])
        for ik=1:nk
            b[ik,iz] = γ[ik,iz] * (b[ik,iz] - β[ik,iz_β] * b[ik,iz-1])
        end
    end
    if !(P <: HighestProc)
        send_to_proc_above(view(b, :, nz))
    end
    b
end

function symthomas_batch_bwd!(b::AbstractArray{Tv,2},
    γ::AbstractArray{Tm,2}, β::AbstractArray{Tm,2},
    buffer::AbstractArray{Tv,1}, ::Val{P}) where {P<:ProcType,Tm,Tv}
    nk, nz = size(b)
    if !(P <: HighestProc) # at the top b[end] = b[end], no change is needed
        b_orig = b[nz]
        get_from_proc_above!(buffer)
        for ik=1:nk
            b[ik,end] = b[ik,end] - γ[ik,end] * β[ik,end] * buffer[ik]
        end
    end
    for iz=nz-1:-1:1
        iz_β = (P <: LowestProc) ? iz : iz+1
        for ik=1:nk
            b[ik,iz] = b[ik,iz] - γ[ik,iz] * β[ik,iz_β] * b[ik,iz+1]
        end
    end
    if !(P <: LowestProc)
        send_to_proc_below(view(b, :, 1))
    end
    b
end

function LinearAlgebra.ldiv!(A_batch::DistributedBatchLDLt{P,Tm,Tv,B},
        b_batch::AbstractArray{Tv}) where {P,Tm,Tv,B}
    nk, nz = size(A_batch.γ)
    b_batch_reshaped = reshape(b_batch, nk, nz)

    # forward pass
    for ik_min=1:B:nk
        ik_max = min(ik_min + B - 1, nk) # batch size might not be divisible by B
        symthomas_batch_fwd!(view(b_batch_reshaped, ik_min:ik_max, :),
            view(A_batch.γ, ik_min:ik_max, :), view(A_batch.β, ik_min:ik_max, :),
            view(A_batch.buffer, 1:1+ik_max-ik_min), Val(P))
    end

    # backward pass
    for ik_min=1:B:nk
        ik_max = min(ik_min + B - 1, nk) # batch size might not be divisible by B
        symthomas_batch_bwd!(view(b_batch_reshaped, ik_min:ik_max, :),
            view(A_batch.γ, ik_min:ik_max, :), view(A_batch.β, ik_min:ik_max, :),
            view(A_batch.buffer, 1:1+ik_max-ik_min), Val(P))
    end

    b_batch
end

LinearAlgebra.ldiv!(x, A::DistributedBatchLDLt, b) = LinearAlgebra.ldiv!(A, copyto!(x, b))
LinearAlgebra.:\(A::DistributedBatchLDLt, b) = LinearAlgebra.ldiv!(similar(b), A, b)

# Compute the diagonal of D₃G₃, excluding the αc-factor: -α(1), -α(1)-α(2), …, -α(N-2)-α(N-1), -α(N-1)
function d3g3_diagonal(::Type{T}, gd, gm) where T
    αi_ext = T[gd.nz_global / gm.Dvmap(ζ) for ζ=vrange(T, gd, NodeSet(:H), neighbors=true)]
    imin = gd.iz_min == 1 ? 2 : 1 # exclude value at lower boundary
    imax = gd.nz_v # exclude value at upper boundary
    dvs = zeros(T, gd.nz_h)
    dvs[imin:end] .-= αi_ext[imin:end-1]
    dvs[1:imax] .-= αi_ext[2:imax+1]
    dvs
end

# Compute the off-diagonal of D₃G₃, excluding the αc-factor: α(0), α(1), …, α(N-1)
function d3g3_offdiagonal(::Type{T}, gd, gm) where T
    T[gd.nz_global / gm.Dvmap(ζ) for ζ=vrange(T, gd, NodeSet(:V))]
end

"""
For k1=k2=0, we want to replace the last equation of the linear system with the
equation 3 p(N₃-1/2) - p(N₃-3/2) = 0, which (approximately) sets the mean
pressure at the top of the domain to zero. To retain the symmetry of the
matrix, this function computes the last value of the off-diagonal and and sets
the last value of the diagonal to minus three times that value. For this to
work correctly, the last entry of the vector for which the system is solved
should also be set to zero. Otherwise, the solver will still work correctly but
the pressure will differ by some additive factor.
"""
function adjust_d3g3_diagonal(dvs, gd, gm)
    gd.iz_max == gd.nz_global || return dvs # only adjust at the very top
    # the last off-diagonal value is evaluated at the last I-node, which might
    # already belong to the process below, so we select it as the lower
    # neighbor of the last C-node
    last_ζi = vrange(eltype(dvs), gd, NodeSet(:H), neighbors=true)[end-1]
    last_ev = gd.nz_global / gm.Dvmap(last_ζi)
    dvs[end] = - 3 * last_ev
    dvs
end

# for kx=ky=0, the top and bottom boundary condition have to be the same,
# which corresponds to having the same integrated mass flux over both the
# top and bottom boundary. otherwise it is impossible to conserve mass.
# in fact, the formulation used for the other wavenumbers results in nz-1
# equations for the pressure plus one equation w_top-w_bottom=0. therefore,
# we need to specify an additional equation, which is equivalent to fixing
# the absolute value of the horizontal mean of pressure somewhere in the
# domain. the absolute value is undetermined because only the gradients
# of pressure appear in the problem. we can replace any of the equations
# with a different one that places a restriction on the value of p. one way
# of doing so is replacing the last line with (p[nz-1]-3p[nz])/δz²=0, which
# keeps the system of equations symmetric tridiagonal and sets the (second
# order FD approximation of) pressure at the top of the domain to zero.
# it could also be done by replacing a different line in order to get a
# higher order approximation at the boundary, but there is little reason to
# do so, since the absolute values don’t matter anyway.
adjust_pressure_rhs!(p, gd) = (gd.iz_max == gd.nz_global) ? (p[1,1,end] = 0; p) : p

function prepare_pressure_solver(gd::DistributedGrid, gm::GridMapping{T}, batch_size) where T

    # compute off-diagonal terms
    ps_offdiagonal = d3g3_offdiagonal(T, gd, gm)

    # set up function to compute diagonal terms of matrix
    d3g3 = d3g3_diagonal(T, gd, gm)
    αc = T[gd.nz_global / gm.Dvmap(ζ) for ζ=vrange(T, gd, NodeSet(:H))]
    ps_diagonal(k1, k2) = (k1 == k2 == 0) ? adjust_d3g3_diagonal(copy(d3g3), gd, gm) :
        T[d3g3[i] - 4 * π^2 * (k1^2 / gm.hsize1^2 + k2^2 / gm.hsize2^2) ./ αc[i] for i=1:gd.nz_h]

    # set up generators that produce diagonals for each wavenumber pair
    # note: the offdiagonal is always the same so the generator simply returns
    # the precomputed array
    k1, k2 = wavenumbers(gd)
    dvs = (ps_diagonal(k1, k2) for k1=k1, k2=k2)
    evs = (ps_offdiagonal for k1=k1, k2=k2)

    DistributedBatchLDLt(T, dvs, evs, zeros_fd(T, gd, NodeSet(:H)), batch_size)
end

# Set the output to the divergence of the input, but using 1/N₃ ∂/∂ζ instead of
# ∂/∂x₃ in the vertical direction. This is required to build the RHS of the
# pressure solver.
function set_divergence_rescaled!(div, vel, (lbc3, ubc3), gd, df)

    u1 = layers(vel[1])
    u2 = layers(vel[2])
    u3_expanded = layers_expand_i_to_c(vel[3], lbc3, ubc3)

    for (i, div) in zip(1:size(div, 3), layers(div))
        hfactor = 1/df.D3_h[i] # TODO: this might be best computed directly
        div!(div, u1[i], u2[i], u3_expanded[i:i+1], df.D1, df.D2, 1, hfactor)
    end

    div
end

"""
The pressure solver computes a pressure field `p` such that the provided
velocity field becomes divergence-free when the pressure gradient is
subtracted, i.e. `∇·(u − ∇p) = 0`. The solver only requires boundary conditions
for the vertical velocity component due to the way the staggered grid is set
up.

Note that the formulation of this solver does not correspond exactly to the way
pressure appears in the Navier-Stokes equations and the problem has to be
rescaled to apply the solver. It is important that the provided boundary
conditions are compatible with the rescaled problem.

In practice, the solver is employed in two different ways. One option is to
apply it to a velocity field to obtain its projection into a divergence-free
space. This corresponds more or less to the formulation given above, but the
pressure does not really have a physical meaning in this case. The other option
is to apply it to one or all terms of the RHS to obtain the pressure field they
induce. In this case, the `u` of the solver is really a `∂u/∂t` and the
boundary conditions provided to the pressure solver should be those of the time
derivative of the velocity, i.e. constant values becomes zero.

The second option is useful for obtaining pressure fields in post-processing,
but for time stepping it is best to rely on the first formulation and apply the
the solver to `u + Δt RHS` rather than directly to `RHS`. Otherwise, errors can
accumulate in `u` and the solution may drift away from a divergence-free flow
field over time.
"""
function solve_pressure!(p, vel, lbcs, ubcs, gd, df, solver)

    # the formulation of the pressure solver matrix implicitly relies on the
    # assumption that the vertical velocity component has Dirichlet boundary
    # conditions
    (lbcs[3] isa DirichletBC && ubcs[3] isa DirichletBC) ||
        error("The pressure solver only supports Dirichlet BCs")

    # since the linear solver works in-place, we save the velocity divergence
    # to the pressure array
    set_divergence_rescaled!(p, vel, (lbcs[3], ubcs[3]), gd, df)

    # when k1=k2=0, one equation is redundant (as long as the mean flow at the
    # top and bottom is the same, which is required for mass conservation) and
    # is overwritten here to set the absolute value of pressure to zero at the
    # top of the domain
    (lbcs[3].value == ubcs[3].value) ||
        error("The mean flow must be equal at both vertical boundaries.")
    adjust_pressure_rhs!(p, gd)

    LinearAlgebra.ldiv!(solver, p)
end

struct PressureSolver{P,T}
    pressure::Array{Complex{T},3}
    solver::DistributedBatchLDLt{P,T,Complex{T}}
    pressure_bc::UnspecifiedBC
    PressureSolver(gd, gm::GridMapping{T}, batch_size) where T = new{proc_type(),T}(
            zeros_fd(T, gd, NodeSet(:H)),
            prepare_pressure_solver(gd, gm, batch_size),
            UnspecifiedBC(T, gd))
end

function enforce_continuity!(vel, lbcs, ubcs, gd, df, ps::PressureSolver)
    solve_pressure!(ps.pressure, vel, lbcs, ubcs, gd, df, ps.solver)
    add_gradient!(vel, ps.pressure, ps.pressure_bc::UnspecifiedBC, df, -1)
    vel
end
