export Pressure
using LinearAlgebra: LinearAlgebra
using MPI: MPI

"""
The DistributedBatchLDLt{P,Tm,Tv,B} holds the data of a decomposition A=L·D·Lᵀ
for a batch of symmetric tridiagonal matrices A. The purpose is to solve a
system A·x=b efficiently for several A, x, and b. The vectors and matrices are
distributed across MPI processes, with `P` designating the role of the current
MPI process. `Tm` is the element type of the matrices A while `Tv` is the
element type of the vectors x and b.

Since the algorithm for solving the linear system is sequential by nature, there
is a tradeoff between solving many systems at once, maximizing the throughput of
each individual solution, and solving smaller batches of systems, minimizing the
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
matrix with γ as its diagonal vector while L is the matrix with γ¯¹ as the
diagonal and β as the lower off-diagonal. Another vector is allocated as buffer
for the values exchanged between MPI processes when solving the system A·x=b.

The algorithm used for the decomposition and the solution of the linear systems
can be found in section 3.7.2 of the book “Numerical Mathematics” by Quarteroni,
Sacco & Saleri (2007).
"""
struct DistributedBatchLDLt{Nb,Na,Tm,Tv,B,C} #<: AbstractMatrix{T}

    γ::Matrix{Tm} # diagonal
    β::Matrix{Tm} # off-diagonal

    # TODO: try to implement solver without this buffer vector
    buffer::Vector{Tv}

    comm::C

    DistributedBatchLDLt(::Type{Tm}, dvs, evs, rhs, batch_size, comm) where Tm = begin

        C = typeof(comm)
        Nb, Na = neighbors(comm)

        nk = length(dvs)
        nz = size(rhs, ndims(rhs)) # last dimension of rhs
        length(dvs) == length(evs) == prod(size(rhs)[1:ndims(rhs)-1]) ||
                error("Batch size of dv (nk=", length(dvs), "), ev (nk=",
                length(evs), "), and rhs (nk=", prod(size(rhs)[1:ndims(rhs)-1]),
                ") not compatible")

        nz_dv = nz
        nz_ev = nz + 1 - sum(isnothing.((Nb, Na)))
        iz_min_ev = isnothing(Nb) ? 1 : 2

        γ = zeros(Tm, nk, nz_dv)
        β = zeros(Tm, nk, nz_ev)

        # copy elements from iterable inputs to internal arrays
        for (ik, dv, ev) in zip(1:nk, dvs, evs)
            eltype(dv) == eltype(ev) == Tm || error("Incompatible element types")
            ndims(dv) == ndims(ev) == 1 || error("More than one dimensions for diagonal vectors")
            length(dv) == nz_dv || error("Length of diagonal (", length(dv), ") not equal to ", nz_dv)
            length(ev) == length(iz_min_ev:nz_ev) || error("Length of off-diagonal (", length(ev), ") not equal to ", length(iz_min_ev:nz_ev))
            copyto!(view(γ, ik, :), dv)
            copyto!(view(β, ik, iz_min_ev:nz_ev), ev)
        end

        # add missing ev-values, obtaining them from the neighbor below
        isnothing(Na) || send_to_proc(view(β, :, nz_ev), Na, comm)
        isnothing(Nb) || get_from_proc!(view(β, :, 1), Nb, comm)

        # compute gamma (TODO: transform to batched version if too slow)
        if !isnothing(Nb)
            γ_below = zeros(Tm, nk)
            get_from_proc!(γ_below, Nb, comm)
            @views @. γ[:,1] = 1 / (γ[:,1] - γ_below * β[:,1]^2)
        else
            @views @. γ[:,1] = 1 / γ[:,1]
        end
        for i=2:nz
            @views @. γ[:,i] = 1 / (γ[:,i] - γ[:,i-1] * β[:,(iz_min_ev-1)+i-1]^2)
        end
        isnothing(Na) || send_to_proc(view(γ, :, nz_dv), Na, comm)

        Tv, B = eltype(rhs), batch_size
        new{Nb,Na,Tm,Tv,B,C}(γ, β, zeros(Tv, B), comm)
    end
end

send_to_proc(data, proc, comm, tag=0) =
    MPI.Send(data, proc, tag, comm)
get_from_proc!(data, proc, comm, tag=MPI.ANY_TAG) =
    MPI.Recv!(data, proc, tag, comm)

function symthomas_batch_fwd!(b::AbstractArray{Tv,2},
    γ::AbstractArray{Tm,2}, β::AbstractArray{Tm,2},
    buffer::AbstractArray{Tv,1}, (Nb, Na), comm) where {Tm,Tv}
    nk, nz = size(b)
    if !isnothing(Nb)
        b_orig = b[1]
        get_from_proc!(buffer, Nb, comm)
        for ik=1:nk
            b[ik,1] = γ[ik,1] * (b[ik,1] - β[ik,1] * buffer[ik])
        end
    else
        for ik=1:nk
            b[ik,1] = γ[ik,1] * b[ik,1]
        end
    end
    for iz=2:nz
        iz_β = isnothing(Nb) ? iz-1 : iz
        #@views @. b[:,iz] = γ[:,iz] * (b[:,iz] - β[:,iz_β] * b[:,iz-1])
        for ik=1:nk
            b[ik,iz] = γ[ik,iz] * (b[ik,iz] - β[ik,iz_β] * b[ik,iz-1])
        end
    end
    if !isnothing(Na)
        send_to_proc(view(b, :, nz), Na, comm)
    end
    b
end

function symthomas_batch_bwd!(b::AbstractArray{Tv,2},
    γ::AbstractArray{Tm,2}, β::AbstractArray{Tm,2},
    buffer::AbstractArray{Tv,1}, (Nb, Na), comm) where {Tm,Tv}
    nk, nz = size(b)
    if !isnothing(Na) # at the top b[end] = b[end], no change is needed
        b_orig = b[nz]
        get_from_proc!(buffer, Na, comm)
        for ik=1:nk
            b[ik,end] = b[ik,end] - γ[ik,end] * β[ik,end] * buffer[ik]
        end
    end
    for iz=nz-1:-1:1
        iz_β = isnothing(Nb) ? iz : iz+1
        for ik=1:nk
            b[ik,iz] = b[ik,iz] - γ[ik,iz] * β[ik,iz_β] * b[ik,iz+1]
        end
    end
    if !isnothing(Nb)
        send_to_proc(view(b, :, 1), Nb, comm)
    end
    b
end

function LinearAlgebra.ldiv!(A_batch::DistributedBatchLDLt{Nb,Na,Tm,Tv,B},
        b_batch::AbstractArray{Tv}) where {Nb,Na,Tm,Tv,B}
    nk, nz = size(A_batch.γ)
    b_batch_reshaped = reshape(b_batch, nk, nz)

    # forward pass
    for ik_min=1:B:nk
        ik_max = min(ik_min + B - 1, nk) # batch size might not be divisible by B
        symthomas_batch_fwd!(view(b_batch_reshaped, ik_min:ik_max, :),
            view(A_batch.γ, ik_min:ik_max, :), view(A_batch.β, ik_min:ik_max, :),
            view(A_batch.buffer, 1:1+ik_max-ik_min), (Nb, Na), A_batch.comm)
    end

    # backward pass
    for ik_min=1:B:nk
        ik_max = min(ik_min + B - 1, nk) # batch size might not be divisible by B
        symthomas_batch_bwd!(view(b_batch_reshaped, ik_min:ik_max, :),
            view(A_batch.γ, ik_min:ik_max, :), view(A_batch.β, ik_min:ik_max, :),
            view(A_batch.buffer, 1:1+ik_max-ik_min), (Nb, Na), A_batch.comm)
    end

    b_batch
end

LinearAlgebra.ldiv!(x, A::DistributedBatchLDLt, b) = LinearAlgebra.ldiv!(A, copyto!(x, b))
LinearAlgebra.:\(A::DistributedBatchLDLt, b) = LinearAlgebra.ldiv!(similar(b), A, b)


"""
    Pressure(batch_size = 64)

Transport of momentum by a pressure-like variable that enforces a divergence-free velocity field.

Arguments

- `batch_size::Int`: The number of wavenumber pairs that are included in each batch of the tri-diagonal solver. The batching serves to stagger the computation such that different MPI ranks can work on different batches at the same time.
"""
struct Pressure <: ProcessDefinition
    batch_size::Int
    Pressure(batch_size = 64) = new(batch_size)
end

struct DiscretizedPressure{T,D,BC,S} <: DiscretizedProcess
    pressure::Array{T,3}
    derivatives::D
    bcs::BC
    solver::S
end

Base.nameof(::DiscretizedPressure) = "Pressure Solver"

state_fields(::DiscretizedPressure) = (:vel1, :vel2, :vel3)

function init_process(press::Pressure, domain::Domain{T}, grid) where T

    # Compute the off-diagonal of D₃G₃, excluding the αc-factor: α(0), α(1), …, α(N-1)
    ps_offdiagonal = dx3factors(domain, grid, NodeSet(:I))

    # set up function to compute diagonal terms of matrix
    d3g3 = d3g3_diagonal(domain, grid)
    αc = dx3factors(domain, grid, NodeSet(:C))
    ps_diagonal(k1, k2) = (k1 == k2 == 0) ? adjust_d3g3_diagonal(copy(d3g3), domain, grid) :
        T[d3g3[i] - 4 * π^2 * (k1^2 * scalefactor(domain, 1)^2 + k2^2 * scalefactor(domain, 2)^2) ./ αc[i] for i=1:length(αc)]

    # set up generators that produce diagonals for each wavenumber pair
    # note: the offdiagonal is always the same so the generator simply returns
    # the precomputed array
    k1, k2 = wavenumbers(grid)
    dvs = (ps_diagonal(k1, k2) for k1=k1, k2=k2)
    evs = (ps_offdiagonal for k1=k1, k2=k2)

    solver = DistributedBatchLDLt(T, dvs, evs, zeros(T, grid, NodeSet(:C)), press.batch_size, grid.comm)

    derivatives = (D1 = dx1factors(domain, wavenumbers(grid)),
                   D2 = dx2factors(domain, wavenumbers(grid)),
                   D3c = dx3factors(domain, grid, NodeSet(:C)),
                   D3i = dx3factors(domain, grid, NodeSet(:I)))
    bcs = (vel3 = init_bcs(:vel3, domain, grid), p = internal_bc(domain, grid))

    DiscretizedPressure(zeros(T, grid, NodeSet(:C)), derivatives, bcs, solver)
end

# Compute the diagonal of D₃G₃, excluding the αc-factor: -α(1), -α(1)-α(2), …, -α(N-2)-α(N-1), -α(N-1)
function d3g3_diagonal(domain::Domain{T}, grid) where T
    αi_ext = T[grid.n3global * scalefactor(domain, 3, ζ)
               for ζ=vrange(grid, NodeSet(:C), neighbors=true)]
    imin = grid.i3min == 1 ? 2 : 1 # exclude value at lower boundary
    imax = grid.n3i # exclude value at upper boundary
    dvs = zeros(T, grid.n3c)
    dvs[imin:end] .-= αi_ext[imin:end-1]
    dvs[1:imax] .-= αi_ext[2:imax+1]
    dvs
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
function adjust_d3g3_diagonal(dvs, domain, grid)
    grid.i3max == grid.n3global || return dvs # only adjust at the very top
    # the last off-diagonal value is evaluated at the last I-node, which might
    # already belong to the process below, so we select it as the lower
    # neighbor of the last C-node
    last_ζi = vrange(grid, NodeSet(:C), neighbors=true)[end-1]
    last_ev = grid.n3global * scalefactor(domain, 3, last_ζi)
    dvs[end] = - 3 * last_ev
    dvs
end

isprojection(press::DiscretizedPressure) = true

function apply_projection!(state, term::DiscretizedPressure)
    vel = (state.vel1, state.vel2, state.vel3)
    solve_pressure!(term.pressure, vel, term.bcs.vel3, term.derivatives, term.solver)
    add_gradient!(vel, term.pressure, term.bcs.p, term.derivatives, -1)
    vel
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
function solve_pressure!(p, vel, (lbc3, ubc3), df, solver)
    # the formulation of the pressure solver matrix implicitly relies on the
    # assumption that the vertical velocity component has Dirichlet boundary
    # conditions (enforced with `ConstantValue` argument definition)
    (lbc3.type isa ConstantValue && ubc3.type isa ConstantValue) ||
        error("The pressure solver only supports Dirichlet BCs")

    # Set the output to the divergence of the input, but using 1/N₃ ∂/∂ζ instead of
    # ∂/∂x₃ in the vertical direction. This is required to build the RHS of the
    # pressure solver.
    # since the linear solver works in-place, we save the velocity divergence
    # to the pressure array
    u1 = layers(vel[1])
    u2 = layers(vel[2])
    u3_expanded = layers_i2c(vel[3], lbc3, ubc3)
    for (i, p) in zip(1:size(p, 3), layers(p))
        div!(p, u1[i], u2[i], u3_expanded[i:i+1], df.D1, df.D2, 1, 1/df.D3c[i])
    end

    # when k1=k2=0, one equation is redundant (as long as the mean flow at the
    # top and bottom is the same, which is required for mass conservation) and
    # is overwritten here to set the absolute value of pressure to zero at the
    # top of the domain
    (lbc3.type.value == ubc3.type.value) ||
        error("The mean flow must be equal at both vertical boundaries.")
    adjust_pressure_rhs!(p, ubc3)

    LinearAlgebra.ldiv!(solver, p)
end

# Divergence in frequency domain
# Note: The divergence functions allow for rescaling the horizontal
# contributions with a (horizontally) constant value. This is used to run the
# pressure solver on the system (α^C¯¹ DG) p = α^C¯¹ (D u + b_c), for which the
# tridiagonal matrix is symmetric.
div!(div, u1, u2, (u3¯, u3⁺), D1, D2, D3, hfactor=1) =
    (@. div = u1 * D1 * hfactor + u2 * D2 * hfactor - D3 * u3¯ + D3 * u3⁺; div)
div!(div::AbstractArray{T}, u1, u2, (u3¯, u3⁺)::Tuple{ConstantValue,A}, D1, D2, D3, hfactor=1) where {T <: Complex, A} =
(@. div = u1 * D1 * hfactor + u2 * D2 * hfactor + D3 * u3⁺; div[1,1] -= D3 * u3¯.value; div)
div!(div::AbstractArray{T}, u1, u2, (u3¯, u3⁺)::Tuple{A,ConstantValue}, D1, D2, D3, hfactor=1) where {T <: Complex, A} =
    (@. div = u1 * D1 * hfactor + u2 * D2 * hfactor - D3 * u3¯; div[1,1] += D3 * u3⁺.value; div)

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
adjust_pressure_rhs!(p, ::BoundaryCondition{BC,Nb,nothing}) where {BC,Nb} = (p[1,1,end] = 0; p)
adjust_pressure_rhs!(_, _) = nothing

"""
Compute the gradient of a scalar field and add it to a vector field. The scalar
field and the horizontal components of the vector field are defined on C-nodes
while the vertical component of the vector field are defined on I-nodes. An
optional prefactor can be used to rescale the gradient before it is added.
"""
function add_gradient!(vector_output, scalar_input, bc::BoundaryCondition{Nothing}, df, prefactor = 1)

    v1 = layers(vector_output[1])
    v2 = layers(vector_output[2])
    v3 = layers(vector_output[3])

    s = layers(scalar_input)
    s_expanded = layers_c2i(scalar_input, bc)

    add_derivative!.(v1, s, (df.D1, ), prefactor)
    add_derivative!.(v2, s, (df.D2, ), prefactor)
    for i = 1:equivalently(length(v3), length(s_expanded)-1)
        add_derivative!(v3[i], s_expanded[i:i+1], df.D3i[i], prefactor)
    end

    vector_output
end

# Horizontal derivatives in frequency domain
add_derivative!(f_out, f_in::AbstractArray, D, prefactor = 1) = @. f_out += prefactor * D * f_in

# Vertical derivatives in frequency or physical domain
add_derivative!(f_out, (f_in¯, f_in⁺)::Tuple, D, prefactor = 1) =
        @. f_out += prefactor * (-D * f_in¯ + D * f_in⁺)
