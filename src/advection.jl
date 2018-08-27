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
