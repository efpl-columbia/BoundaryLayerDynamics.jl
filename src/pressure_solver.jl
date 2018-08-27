# structure holding the data to efficiently
# apply the thomas algorithm for a symmetric
# tridiagonal matrix
struct SymThomas{T} <: LinearAlgebra.Factorization{T}
    γ::Array{T,1}
    β::Array{T,1}
    SymThomas(A::LinearAlgebra.SymTridiagonal{T,Array{T,1}}) where T = begin
        γ, β = similar(A.dv), copy(A.ev)
        γ[1] = 1 / A.dv[1]
        for i=2:length(A.dv)
            γ[i] = 1 / (A.dv[i] - A.ev[i-1].^2 * γ[i-1])
        end
        new{T}(γ, β)
    end
end

Base.size(A::SymThomas, i::Integer) = i > 2 ? 1 : length(A.γ)

function LinearAlgebra.ldiv!(A::SymThomas, B) # A \ B
    n = length(A.γ)
    n == length(B) == length(A.β)+1 ||
        error("Cannot solve system: array sizes do not match")
    B[1] = A.γ[1] * B[1]
    @inbounds for i=2:n # forward pass
        B[i] = A.γ[i] * (B[i] - A.β[i-1] * B[i-1])
    end
    @inbounds for i=n-1:-1:1
        B[i] = B[i] - A.γ[i] * A.β[i] * B[i+1]
    end
    B
end

# matrices for laplacian in Fourier space, for each kx & ky
function prepare_laplacian(T, grid, kx, ky)

    dz = grid.lz / grid.nz

    dx2 = - (kx * 2*π/grid.lx).^2
    dy2 = - (ky * 2*π/grid.ly).^2
    dz2_diag0 = - [one(T); 2*ones(T, grid.nz-2); one(T)] / dz^2
    dz2_diag1 = ones(T, grid.nz-1) / dz^2

    L = LinearAlgebra.SymTridiagonal{T}(dz2_diag0 .+ dx2 .+ dy2, dz2_diag1)

    SymThomas(L) # precompute first step of Thomas algorithm
end

function solve_pressure!(state_fd, rhs, transform)

    #= pressure solver:
    - solve for p: T¯¹ dx² T p + T¯¹ dy² T p + Dz T¯¹ T p = T¯¹ dx T rhs_x + T¯¹ dy T rhs_y + Dz T¯¹ T rhs_z
    - can compute d/dz in Fourier space for all of these, i.e swap order of Dz & T¯¹
    - solve for (Tp): dx²(i,j) Tp(i,j,:) + dy²(i,j) Tp(i,j,:) + Dz(:,:) Tp(i,j,:)
                        = dx(i,j) Trhs_x(i,j,:) + dy(i,j) Trhs_y(i,j,:) + Dz(:,:) Trhs_z(i,j,:)

    steps:
    - need to build divergence of RHS in Fourier space
    - need to find a good way of solving the tridiagonal system (backslash?)
    - need to worry about boundaries, especially for k=0
    =#

    # compute divergence of RHS
    update_rhs_div!(rhs, transform)
    return state_fd

    # solve for remaining pressure in frequency domain, skipping nyquist frequencies
    nx, ny, nz = size(rhs.div_fd) # for matrices in frequency domain, not the same as grid.nx etc.
    dz = transform.dpdz.dz

    for i=1:nx, j=1:ny

        # skip nyquist frequencies
        i == nx && isodd(nx) && continue
        i == div(ny,2)+1 && iseven(ny) && continue

        if i == j == 1 # first wave-number, can simply integrate
            state_fd.p_hat[i,j,1] = 0
            for k=2:nz
                state_fd.p_hat[i,j,k] = (state_fd.p_hat[i,j,k-1] + rhs.div_fd[i,j,k-1]) / dz^2
            end
            for k=2:nz
                state_fd.p_hat[i,j,k] += state_fd.p_hat[i,j,k-1]
            end
        else
            LinearAlgebra.ldiv!(view(state_fd.p_hat, i, j, :), transform.dd2[i,j], view(rhs.div_fd, i, j, :))
            #state_fd.p_hat[i,j,:] = (transform.dx2[i]*I + transform.dy2[j]*I + transform.dz2) \ rhs.div_fd[i,j,:]
        end
    end

    state_fd
end

function update_pressure_gradients!(gradients, state_pd, state_fd, transform)
    idft!(state_pd.p, state_fd.p_hat, transform)
    ddx!(gradients.dpdx, state_fd.p_hat, transform)
    ddy!(gradients.dpdy, state_fd.p_hat, transform)
    ddz!(gradients.dpdz, state_pd.p, transform.dpdz)
end

function add_pressure_gradient!(rhs, gradients)
    rhs.rhs_u .-= gradients.dpdx
    rhs.rhs_v .-= gradients.dpdy
    rhs.rhs_w .-= gradients.dpdz
end

function euler_pressure_correction!(state_pd, state_fd, rhs, gradients, tf, dt)
    solve_pressure!(state_fd, rhs, tf)
    update_pressure_gradients!(gradients, state_pd, state_fd, tf)
    add_pressure_gradient!(rhs, gradients)
end
