@inline function checkbounds_symthomas_batch(B, γ, β)
    k, n = size(B)
    n > 0 && size(γ) == (k,n) && size(β) == (k,n-1) ||
            error("Cannot apply forward pass: array sizes do not match")
end

"""
This function applies the forward pass of the symmetric Thomas algorithm, where
the operations are applied elementwise along the first dimension of the arrays
such that several systems can be solved in one go.
"""
@inline function symthomas_batch_fwd!(B, γ, β)
    @boundscheck checkbounds_symthomas_batch(B, γ, β)
    @inbounds @. @views B[:,1] = γ[:,1] * B[:,1]
    for i=2:size(B,2)
        @inbounds @. @views B[:,i] = γ[:,i] * (B[:,i] - β[:,i-1] * B[:,i-1])
    end
end

"""
This function applies the backward pass of the symmetric Thomas algorithm, where
the operations are applied elementwise along the first dimension of the arrays
such that several systems can be solved in one go.
"""
@inline function symthomas_batch_bwd!(B, γ, β)
    @boundscheck checkbounds_symthomas_batch(B, γ, β)
    for i=size(B,2)-1:-1:1
        @inbounds @. @views B[:,i] -= γ[:,i] * β[:,i] * B[:,i+1]
    end
end

function LinearAlgebra.ldiv!(A::SymThomasBatch, B)
    dims = size(B)
    B = reshape(B, size(A.γ)) # treat all dimensions but last as a single one
    checkbounds_symthomas_batch(B, A.γ, A.β) # could be ommited, reshape already confirms size
    @inbounds symthomas_batch_fwd!(B, A.γ, A.β)
    @inbounds symthomas_batch_bwd!(B, A.γ, A.β)
    reshape(B, dims)
end

# matrices for laplacian in Fourier space, for each kx & ky
function prepare_laplacian_fd(gd::Grid{T}, kx, ky) where T

    dx2 = - (kx * 2*π/gd.l[1]).^2
    dy2 = - (ky * 2*π/gd.l[2]).^2
    dz2_diag0 = - [one(T); 2*ones(T, gd.n[3]-2); one(T)] / gd.δ[3]^2
    dz2_diag1 = ones(T, gd.n[3]-1) / gd.δ[3]^2

    # if kx=ky=0, the nz equations only have nz-1 that are linearly independent,
    # so we drop the last line and use it to set the mean pressure at the top
    # of the domain to (approximately) zero with (-3p[nz]+p[nz-1])/δz²=0
    if kx == ky == 0
        dz2_diag0[end] = -3 / gd.δ[3]^2
    end

    LinearAlgebra.SymTridiagonal{T}(dz2_diag0 .+ dx2 .+ dy2, dz2_diag1)
end

"""
Compute the vertical derivative of a function, with the result being on a
different set of nodes. The boundary versions assume that the derivative is
computed in frequency domain.
"""
@inline dvel_dz!(dvel_dz, vel¯, vel⁺, dz) = @. dvel_dz = (vel⁺ - vel¯) * dz
@inline dvel_dz!(dvel_dz, vel¯::DirichletBC, vel⁺, dz) =
        (@. dvel_dz = vel⁺ * dz; dvel_dz[1,1] -= vel¯.value; dvel_dz)
@inline dvel_dz!(dvel_dz, vel¯, vel⁺::DirichletBC, dz) =
        (@. dvel_dz = - vel¯ * dz; dvel_dz[1,1] += vel⁺.value; dvel_dz)

"""
Compute the divergence of a field with three components in frequency domain.
"""
function compute_divergence_fd!(div_hat, field_hat, bc_w, dx, dy, dz)

    # start with the vertical derivatives, overwriting existing values in div_hat
    for k = 2:size(div_hat,3)-1
        @views dvel_dz!(div_hat[:,:,k], field_hat[3][:,:,k], field_hat[3][:,:,k+1], dz)
    end
    @views dvel_dz!(div_hat[:,:,1], bc_w, field_hat[3][:,:,2], dz)
    @views dvel_dz!(div_hat[:,:,end], field_hat[3][:,:,end], bc_w, dz)

    # add horizontal derivatives
    @. div_hat += field_hat[2] * dy
    @. div_hat += field_hat[1] * dx

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
    div_hat[1,1,end] = 0

    div_hat
end

"""
This computes a pressure field `p_hat`, the gradient of which can be added to
the velocity field vel_hat to make it divergence free. Note that in a time
stepping algorithm, we usually want (vel_hat + dt ∇ p_hat) to be divergence
free. In this case, we simply solve for dt·p_hat with the following algorithm.
"""
function solve_pressure_fd!(p_hat, p_solver::SymThomasBatch, vel_hat,
        bc_w::DirichletBC, dx, dy, dz)

    # build divergence of RHS in frequency domain, write into p_hat array
    compute_divergence_fd!(p_hat, vel_hat, bc_w, dx, dy, dz)

    # solve for pressure, overwriting divergence
    LinearAlgebra.ldiv!(p_solver, p_hat)
end

function subtract_pressure_gradient_fd!(field_hat, p_hat, dx, dy, dz)
    @. field_hat[1] -= p_hat .* dx
    @. field_hat[2] -= p_hat .* dy
    for k=2:size(p_hat,3) # do no use size(field_hat) since it can contain buffers
        @. @views field_hat[3][:,:,k] -= (p_hat[:,:,k] - p_hat[:,:,k-1]) * dz
    end
    field_hat
end
