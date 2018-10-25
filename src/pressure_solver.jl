const MTAG_UP = 8
const MTAG_DN = 9

printdirect(s...) = MPI.Initialized() ? println("process ", MPI.Comm_rank(MPI.COMM_WORLD) + 1, " ", s...) : println(s...)

# NOTE: the views passed to these helper functions should have a range of indices
# since zero-dimensional subarrays are not considered contiguous in julia 1.0
send_to_proc_above(x) = MPI.Send(x, MPI.Comm_rank(MPI.COMM_WORLD) + 1, MTAG_UP, MPI.COMM_WORLD)
send_to_proc_below(x) = MPI.Send(x, MPI.Comm_rank(MPI.COMM_WORLD) - 1, MTAG_DN, MPI.COMM_WORLD)
get_from_proc_above!(x) = MPI.Recv!(x, MPI.Comm_rank(MPI.COMM_WORLD) + 1, MTAG_DN, MPI.COMM_WORLD)
get_from_proc_below!(x) = MPI.Recv!(x, MPI.Comm_rank(MPI.COMM_WORLD) - 1, MTAG_UP, MPI.COMM_WORLD)

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

    DistributedBatchLDLt(Tm, dvs, evs, rhs, batch_size) = begin

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

# if kx=ky=0 (or i=j=1), the nz equations only have nz-1 entries that are
# linearly independent, so we drop the last line and use it to set the mean
# pressure at the top of the domain to (approximately) zero with
# (-3p[nz]+p[nz-1])/δz²=0
pressure_solver_diagonal(gd::DistributedGrid{SingleProc}, df::DerivativeFactors{T}, i, j) where T =
        df.dx2[i] .+ df.dy2[j] .- df.dz2 *
        T[one(T); 2 * ones(T, get_nz(gd, NodeSet(:H))-2); one(T) * (i==j==1 ? 3 : 1)]
pressure_solver_diagonal(gd::DistributedGrid{MinProc}, df::DerivativeFactors{T}, i, j) where T =
        df.dx2[i] .+ df.dy2[j] .- df.dz2 *
        T[one(T); 2 * ones(T, get_nz(gd, NodeSet(:H))-1)]
pressure_solver_diagonal(gd::DistributedGrid{InnerProc}, df::DerivativeFactors{T}, i, j) where T =
        df.dx2[i] .+ df.dy2[j] .- df.dz2 *
        T[2 * ones(T, get_nz(gd, NodeSet(:H)));]
pressure_solver_diagonal(gd::DistributedGrid{MaxProc}, df::DerivativeFactors{T}, i, j) where T =
        df.dx2[i] .+ df.dy2[j] .- df.dz2 *
        T[2 * ones(T, get_nz(gd, NodeSet(:H))-1); one(T) * (i==j==1 ? 3 : 1)]
pressure_solver_offdiagonal(gd::DistributedGrid, df::DerivativeFactors{T}) where T =
        df.dz2 * ones(T, get_nz(gd, NodeSet(:V)))

prepare_pressure_solver(gd::DistributedGrid, df::DerivativeFactors{T}, batch_size) where T =
        DistributedBatchLDLt(T, (pressure_solver_diagonal(gd, df, i, j) for i=1:gd.nx_fd, j=1:gd.ny_fd),
        (pressure_solver_offdiagonal(gd, df) for i=1:gd.nx_fd, j=1:gd.ny_fd),
        zeros_fd(T, gd, NodeSet(:H)), batch_size)


@inline div!(div, u, v, w¯, w⁺, dx, dy, dz) =
    (@. div = u * dx + v * dy + (w⁺ - w¯) * dz; div)
@inline div!(div, u, v, w¯::DirichletBC, w⁺, dx, dy, dz) =
    (@. div = u * dx + v * dy + w⁺ * dz; div[1,1] -= w¯.value * dz; div)
@inline div!(div, u, v, w¯, w⁺::DirichletBC, dx, dy, dz) =
    (@. div = u * dx + v * dy - w¯ * dz; div[1,1] += w⁺.value * dz; div)


function set_divergence!(div_layers::NTuple{NZH},
    u_layers::NTuple{NZH}, v_layers::NTuple{NZH},  w_layers::NTuple{NZV},
    lower_bcw::BoundaryCondition, upper_bcw::BoundaryCondition,
    df::DerivativeFactors) where {NZH,NZV}

    w_below = get_layer_below(w_layers, lower_bcw)

    if NZV >= 1
        # as long as there is at least one local w-layer, the first layer is
        # computed with the w below and the first w-layer
        div!(div_layers[1], u_layers[1], v_layers[1], w_below, w_layers[1], df.dx1, df.dy1, df.dz1)
        # add the remaining layers that are using local w-layers
        for i=2:NZV
            div!(div_layers[i], u_layers[i], v_layers[i], w_layers[i-1], w_layers[i], df.dx1, df.dy1, df.dz1)
        end
        # if this is the highest process, there is an extra layer to be computed
        # using the top boundary condition
        if NZH > NZV
            div!(div_layers[end], u_layers[end], v_layers[end], w_layers[end], upper_bcw, df.dx1, df.dy1, df.dz1)
        end
    else
        # if there are no local w-layers (highest process with one layer per
        # process), a single layer is computed with the w below and the upper BC
        div!(div_layers[1], u_layers[1], v_layers[1], w_below, upper_bcw, df.dx1, df.dy1, df.dz1)
    end

    div_layers
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
set_top_pressure(p, bc_pressure::DirichletBC{HighestProc}) =
        (p[1,1,end] = bc_pressure.value; p)
set_top_pressure(p, bc_pressure) = p # other processes do nothing

"""
This computes a pressure field `p`, the gradient of which can be added to the
velocity field vel_hat to make it divergence free. Note that in a time stepping
algorithm, we usually want (vel + dt ∇ p) to be divergence free. In this case,
we simply solve for dt·p with the following algorithm.
"""
function solve_pressure!(p, vel::Tuple, lower_bcs::Tuple, upper_bcs::Tuple,
        bc_pressure, df::DerivativeFactors, p_solver::DistributedBatchLDLt)
    # since the solver works in-place, we set p to the divergence of vel
    set_divergence!(layers(p), layers(vel[1]), layers(vel[2]), layers(vel[3]),
            lower_bcs[3], upper_bcs[3], df)
    set_top_pressure(p, bc_pressure)
    LinearAlgebra.ldiv!(p_solver, p)
end

function subtract_pressure_gradient!(vel::Tuple, p, df::DerivativeFactors, bc_pressure)
    @. vel[1] -= p .* df.dx1
    @. vel[2] -= p .* df.dy1
    p_above = get_layer_above(layers(p), bc_pressure)
    for iz=1:size(p, 3)-1
        #TODO @. vel[3][:,:,iz] = view(vel[3], :, :, iz) - (view(p, :, :, iz+1) - view(p, :, :, iz)) * df.dz1
        vel[3][:,:,iz] .= view(vel[3], :, :, iz) .-
                (view(p, :, :, iz+1) .- view(p, :, :, iz)) .* df.dz1
    end
    if !(p_above isa BoundaryCondition)
        vel[3][:,:,end] .= view(vel[3], :, :, size(vel[3], 3)) .-
                (p_above .- view(p, :, :, size(p,3))) .* df.dz1
    end
    vel
end
