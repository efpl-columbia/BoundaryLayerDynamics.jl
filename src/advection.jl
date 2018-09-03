# compute one layer of vorticity (in frequency domain)
# (dw/dy - dv/dz) and (du/dz - dw/dx) on w-nodes, (dv/dx - du/dy) on uvp-nodes
@inline rot_u!(rot_u, v¯, v⁺, w, dy, dz) = @. rot_u = w * dy - (v⁺ - v¯) * dz
@inline rot_v!(rot_v, u¯, u⁺, w, dx, dz) = @. rot_v = (u⁺ - u¯) * dz - w * dx
@inline rot_w!(rot_w, u, v, dx, dy)      = @. rot_w = v * dx - u * dy

@inline function compute_vorticity_fd!(rot_hat, vel_hat, dx, dy, dz)
    for k=1:size(vel_hat[1],3)-2 # inner z-layers
        @views rot_u!(rot_hat[1][:,:,k],
                      vel_hat[2][:,:,k-1], vel_hat[2][:,:,k], vel_hat[3][:,:,k],
                      dy[:,:,1], dz)
        @views rot_v!(rot_hat[2][:,:,k],
                      vel_hat[1][:,:,k-1], vel_hat[1][:,:,k], vel_hat[3][:,:,k],
                      dy[:,:,1], dz)
        @views rot_w!(rot_hat[3][:,:,k], vel_hat[1][:,:,k], vel_hat[2][:,:,k],
                      dx[:,:,1], dy[:,:,1])
    end
end

"""
Transform a field from the frequency domain to an extended set of nodes in the
physical domain by adding extra frequencies set to zero.
"""
function ifft_dealiased!(field_big, field_hat, plan_big_bwd, buffer_big_fd)

    # highest frequencies (excluding nyquist) in non-expanded array
    ny, ny_big = size(field_hat,2), size(buffer_big_fd,2)
    kx_max = size(field_hat,1) - 2 # [0, 1…kmax, nyquist]
    ky_max = div(ny,2) - 1 # [0, 1…kmax, nyquist, -kmax…1]

    # copy frequencies such that the extra frequencies are zero
    # TODO: multiply with 1/(nx*ny) -> which nx & ny?
    # TODO: test these expansions for new approach where the frequency domain
    #       arrays no longer include the nyquist frequency
    # -> which nx & ny? should be small values to get correct velocities in
    # physical domain, but needs to be big one to get correct frequencies again
    # when going back to the frequency domain (should always multiply when
    # doing forward transform, not for backward transform)
    for i in CartesianIndices(buffer_big_fd)
        if 1+kx_max < i[1] || 1+ky_max < i[2] <= ny_big - ky_max
            buffer_big_fd[i] = 0
        else
            buffer_big_fd[i] = i[2] <= 1+ky_max ? field_hat[i] :
                field_hat[i[1], i[2] - (ny_big - ny), i[3]]
        end
    end

    LinearAlgebra.mul!(field_big, plan_big_bwd, buffer_big_fd)
    field_big
end

"""
Transform a field from an extended set of nodes in physical domain back to the
frequency domain and remove extra frequencies.
"""
function fft_dealiased!(field_hat, field_big, plan_big_fwd, buffer_big_fd)

    LinearAlgebra.mul!(buffer_big_fd, plan_big_fwd, field_big)

    # highest frequencies (excluding nyquist) in non-expanded array
    # warning: this assumes that nx & ny (before the fft)
    #          are even in the non-expanded array
    ny, ny_big = size(field_hat,2), size(buffer_big_fd,2)
    kx_max = size(field_hat,1) - 2 # [0, 1…kmax, nyquist]
    ky_max = div(ny,2) - 1 # [0, 1…kmax, nyquist, -kmax…1]

    # fft normalization factor 1/(nx*ny) is applied whenever forward transform
    # is performed
    fft_factor = 1 / (size(field_big,1) * size(field_big,2))

    for i in innerindices(field_hat)
        if i[1] == ky_max+2 || i[2] == ky_max+2
            field_hat[i] += 0
        else
            field_hat[i] = i[2] <= 1+ky_max ? buffer_big_fd[i] * fft_factor :
                buffer_big_fd[i[1], i[2] + (ny_big - ny), i[3]] * fft_factor
        end
    end
    field_hat
end

# compute one layer of -(roty[w]*w[w]-rotz[uvp]*v[uvp]) on uvp-nodes
@inline adv_u!(adv_u, v, roty¯, roty⁺, w¯, w⁺, rotz) =
        @. adv_u = rotz * v - 0.5 * (roty¯ * w¯ + roty⁺ * w⁺)

# compute one layer of -(rotz[uvp]*u[uvp]-rotx[w]*w[w]) on uvp-nodes
@inline adv_v!(adv_v, u, rotx¯, rotx⁺, w¯, w⁺, rotz) =
        @. adv_v = 0.5 * (rotx¯ * w¯ + rotx⁺ * w⁺) - rotz * u

# compute one layer of -(rotx[w]*v[uvp]-roty[w]*u[uvp]) on w-nodes
@inline adv_w!(adv_w, u¯, u⁺, rotx, v¯, v⁺, roty) =
        @. adv_w = roty * 0.5 * (u¯ + u⁺) - rotx * 0.5 * (v¯ + v⁺)

function compute_advection_pd!(adv, rot, vel, buffers)

    # TODO: decide whether vel_big should contain buffer space & adapt accordingly
    iz_min, iz_max = (1, size(vel[1],3)) # inner layers

    u_below, v_below, w_above, rotx_above, roty_above = buffers[1:5]
    # TODO: exchange information on boundaries (plus BCs)

    # for the x- and y-advection, we need to interpolate rot[1]/rot[2] and vel[3]
    # from w- up to uvp-nodes. we exclude the last layer as it needs data from
    # the w-layer above
    # for the z-advection, we need to interpolate vel[1] and vel[2] from
    # uvp-nodes down to w-nodes. we exclude the first layer as it needs data from
    # the uvp-layer below
    for k=iz_min:iz_max-1
        @views adv_u!(adv[1][:,:,k],
                      vel[2][:,:,k], rot[2][:,:,k], rot[2][:,:,k+1],
                      vel[3][:,:,k], vel[3][:,:,k+1], rot[3][:,:,k])
        @views adv_v!(adv[2][:,:,k],
                      vel[1][:,:,k], rot[1][:,:,k], rot[1][:,:,k+1],
                      vel[3][:,:,k], vel[3][:,:,k+1], rot[3][:,:,k])
        @views adv_w!(adv[3][:,:,k+1],
                      vel[1][:,:,k], vel[1][:,:,k+1], rot[1][:,:,k+1],
                      vel[2][:,:,k], vel[2][:,:,k+1], rot[2][:,:,k+1])
    end

    # add last layer, using the values above & below as exchanged earlier
    @views adv_u!(adv[1][:,:,iz_max],
                  vel[2][:,:,iz_max], rot[2][:,:,iz_max], roty_above,
                  vel[3][:,:,iz_max], w_above, rot[3][:,:,iz_max])
    @views adv_v!(adv[2][:,:,iz_max],
                  vel[1][:,:,iz_max], rot[1][:,:,iz_max], rotx_above,
                  vel[3][:,:,iz_max], w_above, rot[3][:,:,iz_max])
    @views adv_w!(adv[3][:,:,iz_min],
                  u_below, vel[1][:,:,iz_min], rot[1][:,:,iz_min],
                  v_below, vel[2][:,:,iz_min], rot[2][:,:,iz_min])

    adv
end

function set_advection_fd!(rhs_hat, vel_hat, rot_hat, df, tf_big, to)
    # need: rot_hat[1:3], vel_big[1:3], rot_big[1:3], plan_big_{fwd,bwd},
    # buffer_big_fd, buffer_layers_pd[1:5]

    # compute vorticity in fourier domain
    @timeit to "vorticity" compute_vorticity_fd!(rot_hat, vel_hat, df.dx1, df.dy1, df.dz1)

    @timeit to "rename buffers" begin
    rot_big = tf_big.buffers_pd[1:3]
    vel_big = tf_big.buffers_pd[4:6]
    adv_big = tf_big.buffers_pd[7:9]
    end

    # for each velocity and vorticity, pad the frequencies and transform
    # to physical space (TODO: fix boundaries)
    @timeit to "fwd transforms" for (field_big, field_hat) in zip((rot_big..., vel_big...), (rot_hat..., vel_hat...))
        ifft_dealiased!(field_big, field_hat, tf_big.plan_bwd, tf_big.buffers_fd[1])
    end

    # compute the advection term in physical domain
    @timeit to "non-linear part" compute_advection_pd!(adv_big, rot_big, vel_big, tf_big.buffer_layers_pd[1:5])

    # transform advection term back to frequency domain and set RHS to result,
    # skipping higher frequencies for dealiasing
    @timeit to "bwd transform" for (field_hat, field_big) in zip(rhs_hat, adv_big)
        fft_dealiased!(field_hat, field_big, tf_big.plan_fwd, tf_big.buffers_fd[1])
    end

    rhs_hat
end
