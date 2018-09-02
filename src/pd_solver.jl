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
function prepare_laplacian(gd::Grid{T}, kx, ky) where T

    dz = gd.l[3] / gd.n[3]

    dx2 = - (kx * 2*π/gd.l[1]).^2
    dy2 = - (ky * 2*π/gd.l[2]).^2
    dz2_diag0 = - [one(T); 2*ones(T, gd.n[3]-2); one(T)] / dz^2
    dz2_diag1 = ones(T, gd.n[3]-1) / dz^2

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

function prepare_state(gd::Grid{T}) where T

    # shorthand to initialize arrays
    init_field_pd() = zeros(T, gd.n...)
    init_field_fd() = zeros(Complex{T}, div(gd.n[1],2)+1, gd.n[2:3]...)

    # shorthand to initialize enlarged arrays for dealiasing
    n_big = (3*div(gd.n[1],2), 3*div(gd.n[2],2), gd.n[3])
    init_field_big_pd() = zeros(T, n_big)
    init_field_big_fd() = zeros(Complex{T}, div(n_big[1],2)+1, n_big[2:3]...)

    state_pd = (u = init_field_pd(), u_big = init_field_big_pd(),
                v = init_field_pd(), v_big = init_field_big_pd(),
                w = init_field_pd(), w_big = init_field_big_pd(),
                p = init_field_pd())

    # since the fft of real values is symmetric, only half of the
    # complex values are needed in the first direction of the transform,
    # but for even number of values an extra number is needed for the
    # nyquist frequency
    state_fd = (u_hat = init_field_fd(), u_big_hat = init_field_big_fd(),
                v_hat = init_field_fd(), v_big_hat = init_field_big_fd(),
                w_hat = init_field_fd(), w_big_hat = init_field_big_fd(),
                p_hat = init_field_fd())

    # wavenumbers
    kx = wavenumbers(gd.n[1])[1:div(gd.n[1],2)+1]
    ky = wavenumbers(gd.n[2])

    transform = (
        plan_fwd     = plan_rfft(state_pd.u, (1,2)),
        plan_bwd     = plan_brfft(state_fd.u_hat, gd.n[1], (1,2)),
        plan_fwd_big = plan_rfft(state_pd.u_big, (1,2)),
        plan_bwd_big = plan_brfft(state_fd.u_big_hat, n_big[1], (1,2)),
        buffer=init_field_fd(),
        buffer_pd=init_field_pd(),
        buffer_big_pd=init_field_big_pd(),
        buffer_big_fd=init_field_big_fd(),
        dx  = Complex{T}[1im * 2*π/gd.l[1] * k for k=kx],
        dy  = Complex{T}[1im * 2*π/gd.l[2] * k for k=ky],
        dx_premultiplied = Complex{T}[1im * 2*π/gd.l[1] * k / (gd.n[1]*gd.n[2]) for k=kx],
        dy_premultiplied = Complex{T}[1im * 2*π/gd.l[2] * k / (gd.n[1]*gd.n[2]) for k=ky],
        dx2 = T[1/(gd.n[1]*gd.n[2]) * (-4*π*π)/(gd.l[1]*gd.l[1]) * k * k
                for k=wavenumbers(gd.n[1])[1:div(gd.n[1],2)+1]],
        dy2 = T[1/(gd.n[1]*gd.n[2]) * (-4*π*π)/(gd.l[2]*gd.l[2]) * k * k
                for k=wavenumbers(gd.n[2])],
        dd2 = [prepare_laplacian(gd, kx, ky) for kx=kx, ky=ky],
        duvdz = (uvp=true, dirichlet=true, dz=convert(T, gd.l[3]/gd.n[3]),
            ghost=zeros(T, gd.n[1], gd.n[2])),
        dpdz = (uvp=true, dirichlet=false, dz=convert(T, gd.l[3]/gd.n[3]),
            ghost=zeros(T, gd.n[1], gd.n[2])),
        )

    state_pd, state_fd, transform
end

function update_fd!(state_fd, state_pd, transform)
    LinearAlgebra.mul!(state_fd.u_hat, transform.plan_fwd, state_pd.u)
    LinearAlgebra.mul!(state_fd.v_hat, transform.plan_fwd, state_pd.v)
    LinearAlgebra.mul!(state_fd.w_hat, transform.plan_fwd, state_pd.w)
    LinearAlgebra.mul!(state_fd.p_hat, transform.plan_fwd, state_pd.p)
    state_fd
end

function idft!(u, u_hat, tr)
    α = 1 / (size(u,1) * size(u,2))
    nfx = iseven(size(u,1)) ? div(size(u,1),2)+1 : 0
    nfy = iseven(size(u,2)) ? div(size(u,2),2)+1 : 0
    @. tr.buffer = α * u_hat
    @. tr.buffer[nfx,:,:] = 0
    @. tr.buffer[:,nfy,:] = 0
    LinearAlgebra.mul!(u, tr.plan_bwd, tr.buffer)
end

function ddx!(dudx, u_hat, tr)
    for i in CartesianIndices(u_hat)
        tr.buffer[i] = u_hat[i] * tr.dx_premultiplied[i[1]]
    end
    LinearAlgebra.mul!(dudx, tr.plan_bwd, tr.buffer)
end

function ddy!(dudy, u_hat, tr)
    for i in CartesianIndices(u_hat)
        tr.buffer[i] = u_hat[i] * tr.dy_premultiplied[i[2]]
    end
    LinearAlgebra.mul!(dudy, tr.plan_bwd, tr.buffer)
end

function ddz!(dudz, u, vdiff)

    α = 1/vdiff.dz

    # u is un uvp-nodes, dudz on w-nodes, ghost layer below lowest w-layer
    if vdiff.uvp # u is on uvp-nodes, dudz on w-nodes, going down
        if vdiff.dirichlet
            for i in CartesianIndices(vdiff.ghost)
                vdiff.ghost[i] = - u[i[1], i[2], 1] # set u==0 at lowest w-layer
            end
        else # neumann bc
            for i in CartesianIndices(vdiff.ghost)
                vdiff.ghost[i] = u[i[1], i[2], 1] # set dudz==1 at lowest w-layer
            end
        end
        for i in CartesianIndices(dudz)
            dudz[i] = (i[3] > 1) ? α * (u[i] - u[i[1], i[2], i[3]-1]) :
                                   α * (u[i] - vdiff.ghost[i[1], i[2]])
        end

    # u is on w-nodes, dudz on uvp-nodes, ghost layer above highest uvp-layer
    else # u is on w-nodes, dudz on uvp-nodes, going up
        if vdiff.dirichlet
            for i in CartesianIndices(vdiff.ghost)
                vdiff.ghost[i] = zero(eltype(u))
            end
        else # neumann bc
            for i in CartesianIndices(vdiff.ghost)
                vdiff.ghost[i] = (4/3) * u[i[1], i[2], end] - (1/3) * u[i[1], i[2], end-1]
            end
        end
        for i in CartesianIndices(dudz)
            dudz[i] = (i[3] < size(u,3)) ? α * (u[i[1], i[2], i[3]+1]   - u[i]) :
                                           α * (vdiff.ghost[i[1], i[2]] - u[i])
        end
    end

    dudz
end

function laplacian!(lvel, vel_hat, tr, uvp)
    dz2 = tr.duvdz.dz^2
    nz = size(lvel,3)
    if uvp # uvp-nodes, bottom bc: no slip, top bc: free slip
        for i in CartesianIndices(vel_hat)
            tr.buffer[i] =
                (i[3] > 1 ? vel_hat[i[1],i[2],i[3]-1] : - vel_hat[i]) / dz2 + # u[0] = -u[1]
                (tr.dx2[i[1]] + tr.dy2[i[2]] - 2 / dz2) * vel_hat[i] +
                (i[3] < nz ? vel_hat[i[1],i[2],i[3]+1] : vel_hat[i]) / dz2 # u[end+1] = u[end]
        end
    else # w-nodes, bottom & top bc: no penetration
        # iz=1: u[0]=0, u[-1]=-u[1]; u[1] - 2 u[0] + u[-1] = 0
        # iz=2: u[1]=0
        for i in CartesianIndices(vel_hat)
            if i[3] == 1
                tr.buffer[i] = 0
                continue
            end
            tr.buffer[i] =
                (i[3] > 2 ? vel_hat[i[1],i[2],i[3]-1] : 0.0) / dz2 + # u[1] = 0
                (tr.dx2[i[1]] + tr.dy2[i[2]] - 2 / dz2) * vel_hat[i] +
                (i[3] < nz ? vel_hat[i[1],i[2],i[3]+1] : 0.0) / dz2 # u[end+1] = 0
        end
    end
    LinearAlgebra.mul!(lvel, tr.plan_bwd, tr.buffer)
end


function prepare_gradients(gd::Grid{T}) where T
    init_field() = zeros(T, gd.n)
    n_big = (3*div(gd.n[1],2), 3*div(gd.n[2],2), gd.n[3])
    init_field_big_pd() = zeros(T, n_big)
    init_field_big_fd() = zeros(Complex{T}, div(n_big[1],2)+1, n_big[2:3]...)
    (
        dudy = init_field(), dudz = init_field(), lapu = init_field(),
        dvdz = init_field(), dvdx = init_field(), lapv = init_field(),
        dwdx = init_field(), dwdy = init_field(), lapw = init_field(),
        dpdx = init_field(), dpdy = init_field(), dpdz = init_field(),
        rotx = init_field(), roty = init_field(), rotz = init_field(),
        rotx_big = init_field_big_pd(), rotx_big_hat = init_field_big_fd(),
        roty_big = init_field_big_pd(), roty_big_hat = init_field_big_fd(),
        rotz_big = init_field_big_pd(), rotz_big_hat = init_field_big_fd(),
    )
end

function update_velocity_gradients!(gradients, state_pd, state_fd, transform)

    ddy!(gradients.dudy, state_fd.u_hat, transform)
    ddz!(gradients.dudz, state_pd.u, transform.duvdz)
    laplacian!(gradients.lapu, state_fd.u_hat, transform, true)

    ddz!(gradients.dvdz, state_pd.v, transform.duvdz)
    ddx!(gradients.dvdx, state_fd.v_hat, transform)
    laplacian!(gradients.lapv, state_fd.v_hat, transform, true)

    ddx!(gradients.dwdx, state_fd.w_hat, transform)
    ddy!(gradients.dwdy, state_fd.w_hat, transform)
    laplacian!(gradients.lapw, state_fd.w_hat, transform, false)

    # curl [u, v, w] = [dw/dy - dv/dz, du/dz - dw/dx, dv/dx - du/dy]
    @. gradients.rotx = gradients.dwdy - gradients.dvdz # w-nodes
    @. gradients.roty = gradients.dudz - gradients.dwdx # w-nodes
    @. gradients.rotz = gradients.dvdx - gradients.dudy # uvp-nodes

    gradients
end

function add_advection!(rhs, state_pd, gradients)

    #=
    for now, there is no dealiasing and the products are computed directly
    on the regular nodes in physical space. to keep the spectral accuracy,
    the products should be computed with 3/2 of the frequencies, i.e. on
    1.5 times as many nodes in physical space. this will be done as follows:
    - compute dudy, dudz, dvdz, dvdx, dwdx, dwdy
    - expand velocities to 3/2 of nodes
    - expand curl terms (dwdy-dvdz), (dudz-dwdx), (dvdx-dudy) to 3/2 of nodes
    - compute product and reduce to regular nodes again
    =#

    nz = size(rhs.rhs_u,3)
    fill!(rhs.ghost, zero(eltype(rhs.ghost)))

    # -(roty[w]*w[w]-rotz[uvp]*v[uvp]) (interpolate w up to uvp)
    # ghost layer is roty*w at w=top (== zero)
    for i in CartesianIndices(rhs.rhs_u)
        rhs.rhs_u[i] -= (
            0.5 * (gradients.roty[i] * state_pd.w[i] + ((i[3] == nz) ? rhs.ghost[i[1], i[2]] :
            gradients.roty[i[1], i[2], i[3]+1] * state_pd.w[i[1], i[2], i[3]+1]))
            + gradients.rotz[i] * state_pd.v[i])
    end

    # -(rotz[uvp]*u[uvp]-rotx[w]*w[w]) (interpolate w up to uvp)
    # ghost layer is rotx*w at w=top (== zero)
    for i in CartesianIndices(rhs.rhs_v)
        rhs.rhs_v[i] -= (
            gradients.rotz[i] * state_pd.u[i]
            - 0.5 * (gradients.rotx[i] * state_pd.w[i] + ((i[3] == nz) ? rhs.ghost[i[1], i[2]] :
            gradients.rotx[i[1], i[2], i[3]+1] * state_pd.w[i[1], i[2], i[3]+1]))
            )
    end

    # -(rotx[w]*v[uvp]-roty[w]*u[uvp]) (interpolate uvp down to w)
    # ghost layer is rotx*v-roty*u at w=bottom (== zero)
    for i in CartesianIndices(rhs.rhs_v)
        rhs.rhs_w[i] -= (i[3] == 1) ? rhs.ghost[i[1], i[2]] :
            ( gradients.rotx[i] * 0.5 * (state_pd.v[i] + state_pd.v[i[1], i[2], i[3]-1])
            - gradients.roty[i] * 0.5 * (state_pd.u[i] + state_pd.u[i[1], i[2], i[3]-1]) )
    end

    rhs
end

function pad_frequencies!(vel_big_hat, vel_hat)
    # warning: this assumes that nx & ny (before the fft)
    #          are even in the non-expanded array

    # highest frequencies (excluding nyquist) in non-expanded array
    ny, ny_big = size(vel_hat,2), size(vel_big_hat,2)
    kx_max = size(vel_hat,1) - 2 # [0, 1…kmax, nyquist]
    ky_max = div(ny,2) - 1 # [0, 1…kmax, nyquist, -kmax…1]

    for i in CartesianIndices(vel_big_hat)
        if 1+kx_max < i[1] || 1+ky_max < i[2] <= ny_big - ky_max
            vel_big_hat[i] = 0
        else
            vel_big_hat[i] = i[2] <= 1+ky_max ? vel_hat[i] :
                vel_hat[i[1], i[2] - (ny_big - ny), i[3]]
        end
    end
    vel_big_hat
end

function unpad_frequencies!(vel_hat, vel_big_hat)
    # warning: this assumes that nx & ny (before the fft)
    #          are even in the non-expanded array

    # highest frequencies (excluding nyquist) in non-expanded array
    ny, ny_big = size(vel_hat,2), size(vel_big_hat,2)
    kx_max = size(vel_hat,1) - 2 # [0, 1…kmax, nyquist]
    ky_max = div(ny,2) - 1 # [0, 1…kmax, nyquist, -kmax…1]

    for i in CartesianIndices(vel_hat)
        if i[1] == ky_max+2 || i[2] == ky_max+2
            vel_hat[i] = 0
        else
            vel_hat[i] = i[2] <= 1+ky_max ? vel_big_hat[i] :
                vel_big_hat[i[1], i[2] + (ny_big - ny), i[3]]
        end
    end
    vel_hat
end

function add_advection_dealiased!(rhs, state_pd, state_fd, gradients, tf)

    #=
    the products are computed with 3/2 of the frequencies, i.e. on
    1.5 times as many nodes in physical space. this will be done as follows:
    - compute dudy, dudz, dvdz, dvdx, dwdx, dwdy
    - expand velocities to 3/2 of nodes
    - expand curl terms (dwdy-dvdz), (dudz-dwdx), (dvdx-dudy) to 3/2 of nodes
    - compute product and reduce to regular nodes again
    =#

    # pad velocities
    pad_frequencies!(state_fd.u_big_hat, state_fd.u_hat)
    pad_frequencies!(state_fd.v_big_hat, state_fd.v_hat)
    pad_frequencies!(state_fd.w_big_hat, state_fd.w_hat)

    # pad vorticities (TODO: build vorticity directly in Fourier domain)
    LinearAlgebra.mul!(tf.buffer, tf.plan_fwd, gradients.rotx)
    pad_frequencies!(gradients.rotx_big_hat, tf.buffer)
    LinearAlgebra.mul!(tf.buffer, tf.plan_fwd, gradients.roty)
    pad_frequencies!(gradients.roty_big_hat, tf.buffer)
    LinearAlgebra.mul!(tf.buffer, tf.plan_fwd, gradients.rotz)
    pad_frequencies!(gradients.rotz_big_hat, tf.buffer)

    # transform all six values to the physical domain
    LinearAlgebra.mul!(state_pd.u_big, tf.plan_bwd_big, state_fd.u_big_hat)
    LinearAlgebra.mul!(state_pd.v_big, tf.plan_bwd_big, state_fd.v_big_hat)
    LinearAlgebra.mul!(state_pd.w_big, tf.plan_bwd_big, state_fd.w_big_hat)
    LinearAlgebra.mul!(gradients.rotx_big, tf.plan_bwd_big, gradients.rotx_big_hat)
    LinearAlgebra.mul!(gradients.roty_big, tf.plan_bwd_big, gradients.roty_big_hat)
    LinearAlgebra.mul!(gradients.rotz_big, tf.plan_bwd_big, gradients.rotz_big_hat)

    # compute advection term in physical domain, transform back to frequency domain,
    # remove padding, transform to physical domain again, and add to RHS

    nz = size(rhs.rhs_u,3)
    fill!(rhs.ghost, zero(eltype(rhs.ghost)))

    # -(roty[w]*w[w]-rotz[uvp]*v[uvp]) (interpolate w up to uvp)
    # ghost layer is roty*w at w=top (== zero)
    for i in CartesianIndices(tf.buffer_big_pd)
        tf.buffer_big_pd[i] = (
            0.5 * (gradients.roty_big[i] * state_pd.w_big[i] + ((i[3] == nz) ? rhs.ghost_big[i[1], i[2]] :
            gradients.roty_big[i[1], i[2], i[3]+1] * state_pd.w_big[i[1], i[2], i[3]+1]))
            + gradients.rotz_big[i] * state_pd.v_big[i])
    end
    LinearAlgebra.mul!(tf.buffer_big_fd, tf.plan_fwd_big, tf.buffer_big_pd)
    unpad_frequencies!(tf.buffer, tf.buffer_big_fd)
    LinearAlgebra.mul!(tf.buffer_pd, tf.plan_bwd, tf.buffer)
    @. rhs.rhs_u -= tf.buffer_pd

    # -(rotz[uvp]*u[uvp]-rotx[w]*w[w]) (interpolate w up to uvp)
    # ghost layer is rotx*w at w=top (== zero)
    for i in CartesianIndices(tf.buffer_big_pd)
        tf.buffer_big_pd[i] = (
            gradients.rotz_big[i] * state_pd.u_big[i]
            - 0.5 * (gradients.rotx_big[i] * state_pd.w_big[i] + ((i[3] == nz) ? rhs.ghost_big[i[1], i[2]] :
            gradients.rotx_big[i[1], i[2], i[3]+1] * state_pd.w_big[i[1], i[2], i[3]+1]))
            )
    end
    LinearAlgebra.mul!(tf.buffer_big_fd, tf.plan_fwd_big, tf.buffer_big_pd)
    unpad_frequencies!(tf.buffer, tf.buffer_big_fd)
    LinearAlgebra.mul!(tf.buffer_pd, tf.plan_bwd, tf.buffer)
    @. rhs.rhs_v -= tf.buffer_pd

    # -(rotx[w]*v[uvp]-roty[w]*u[uvp]) (interpolate uvp down to w)
    # ghost layer is rotx*v-roty*u at w=bottom (== zero)
    for i in CartesianIndices(tf.buffer_big_pd)
        tf.buffer_big_pd[i] -= (i[3] == 1) ? rhs.ghost_big[i[1], i[2]] :
            ( gradients.rotx_big[i] * 0.5 * (state_pd.v_big[i] + state_pd.v_big[i[1], i[2], i[3]-1])
            - gradients.roty_big[i] * 0.5 * (state_pd.u_big[i] + state_pd.u_big[i[1], i[2], i[3]-1]) )
    end
    LinearAlgebra.mul!(tf.buffer_big_fd, tf.plan_fwd_big, tf.buffer_big_pd)
    unpad_frequencies!(tf.buffer, tf.buffer_big_fd)
    LinearAlgebra.mul!(tf.buffer_pd, tf.plan_bwd, tf.buffer)
    @. rhs.rhs_w -= tf.buffer_pd

    rhs
end

function prepare_rhs(gd::Grid{T}) where T
    n_big = (div(gd.n[1],2)*3, div(gd.n[2],2)*3, gd.n[3])
    (
        rhs_u = zeros(T, gd.n[1], gd.n[2], gd.n[3]),
        rhs_v = zeros(T, gd.n[1], gd.n[2], gd.n[3]),
        rhs_w = zeros(T, gd.n[1], gd.n[2], gd.n[3]),
        ghost = zeros(T, gd.n[1], gd.n[2]),
        ghost_big = zeros(T, n_big[1:2]),
        div_fd = zeros(Complex{T}, div(gd.n[1],2)+1, gd.n[2:3]...)
    )
end

function update_rhs!(rhs, state_pd, state_fd, gradients, transform, forcing, ν)

    # apply pressure forcing first to avoid leftover values in rhs arrays
    fill!(rhs.rhs_u, forcing[1])
    fill!(rhs.rhs_v, forcing[2])
    fill!(rhs.rhs_w, forcing[3])

    # add pressure gradients from last time step
    @. rhs.rhs_u += gradients.dpdx
    @. rhs.rhs_v += gradients.dpdy
    @. rhs.rhs_w += gradients.dpdz

    # add advection terms - (curl u) × u
    #add_advection!(rhs, state_pd, gradients)
    add_advection_dealiased!(rhs, state_pd, state_fd, gradients, transform)

    # add viscous term (molecular diffusion)
    @. rhs.rhs_u += ν * gradients.lapu
    @. rhs.rhs_v += ν * gradients.lapv
    @. rhs.rhs_w += ν * gradients.lapw

    rhs
end

function update_rhs_div!(rhs, transform)

    # initialize with drhs_u/dx
    LinearAlgebra.mul!(transform.buffer, transform.plan_fwd, rhs.rhs_u)
    for i in CartesianIndices(rhs.div_fd)
        rhs.div_fd[i] = transform.buffer[i] * transform.dx[i[1]]
    end

    # add drhs_v/dy
    LinearAlgebra.mul!(transform.buffer, transform.plan_fwd, rhs.rhs_v)
    for i in CartesianIndices(rhs.div_fd)
        rhs.div_fd[i] += transform.buffer[i] * transform.dy[i[2]]
    end

    # add drhs_w/dz (from w-nodes up to uvp-nodes)
    LinearAlgebra.mul!(transform.buffer, transform.plan_fwd, rhs.rhs_w)
    for i in CartesianIndices(rhs.div_fd)
        rhs.div_fd[i] += ((i[3] == size(transform.buffer, 3) ? zero(transform.buffer[1]) :
                transform.buffer[i[1], i[2], i[3]+1]) - transform.buffer[i])
    end

    rhs
end

function euler_step!(state, rhs, dt)
    @. state.u += dt * rhs.rhs_u
    @. state.v += dt * rhs.rhs_v
    @. state.w += dt * rhs.rhs_w
    state
end

function channelflow(gd::Grid{T}, tspan, u0; verbose = false) where T

    ν = 1e-2 # kinematic viscosity, 1.5e-5 for air

    to = TimerOutput()

    @timeit to "allocations" begin
        state_pd, state_fd, tf = prepare_state(gd)
        gradients = prepare_gradients(gd)
        rhs = prepare_rhs(gd)
        forcing = (one(T), zero(T), zero(T))
    end

    # initialize velocity field
    @timeit to "initialization" begin
        for i in CartesianIndices(state_pd.u)
            state_pd.u[i] = u0((i[1]-1)*gd.δ[1], i[2]-1*gd.δ[2], i[3]-1*gd.δ[3])
        end
    end

    tsteps = (0:tspan.nt-1) * tspan.dt
    @timeit to "time stepping" for t = tsteps
        @timeit to "fourier transforms" update_fd!(state_fd, state_pd, tf)
        @timeit to "gradients" update_velocity_gradients!(gradients, state_pd, state_fd, tf)
        @timeit to "build RHS" update_rhs!(rhs, state_pd, state_fd, gradients, tf, forcing, ν)
        @timeit to "time integration" euler_step!(state_pd, rhs, tspan.dt)
        @timeit to "pressure correction" euler_pressure_correction!(state_pd, state_fd,
            rhs, gradients, tf, tspan.dt)
    end

    verbose && show(to)
    state_pd
end
