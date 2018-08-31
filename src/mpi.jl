function mpi_println(args...; comm=MPI.COMM_WORLD)
    r, s = MPI.Comm_rank(comm), MPI.Comm_size(comm)
    for i in 0:s-1
        i == r && println(args...)
        MPI.Barrier(comm)
    end
end

prepare_distributed_array(T, dims) = prepare_distributed_array(T, dims...)

function prepare_distributed_array(T, nx, ny, nz)
    # goal: allocate an array that is distributed in vertical direction

    # set up mpi, find rank
    comm = MPI.COMM_WORLD
    r = MPI.Comm_rank(comm)
    s = MPI.Comm_size(comm)

    # find indices of current rank
    # last rank gets fewer layers if not exactly divisible
    nz_per_rank = cld(nz, s)
    iz_min = 1 + r*nz_per_rank
    iz_max = min(nz, (1+r)*nz_per_rank)
    nz_local = iz_max - iz_min + 1

    mpi_println("Rank ", r, " gets indices ", iz_min, " to ", iz_max, ", size ", nz_local)

    zeros(T, nx, ny, nz_local)
end

function load_distributed!(d, full_array)
    # load the local array l into the distributed array d

    # set up mpi, find rank
    comm = MPI.COMM_WORLD
    r = MPI.Comm_rank(comm)
    s = MPI.Comm_size(comm)

    # find indices of current rank
    # last rank gets fewer layers if not exactly divisible
    nz = size(full_array, 3)
    nz_per_rank = cld(nz, s)
    iz_min = 1 + r*nz_per_rank
    iz_max = min(nz, (1+r)*nz_per_rank)
    nz_local = iz_max - iz_min + 1

    if r==0
        #println("size full: ", size(full_array), ", size d: ", size(d), "iz:", iz_min:iz_max)
        @. d = full_array[:,:,iz_min:iz_max] # copy local part
        for i=1:s-1
            iz_min_remote = 1+i*nz_per_rank
            iz_max_remote = min(nz, (1+i)*nz_per_rank)
            MPI.Send(view(full_array, :, :, iz_min_remote:iz_max_remote), i, 0, comm)
        end
    else
        MPI.Recv!(d, 0, 0, comm)
        #println("Rank ", r, " received data with mean = ", sum(d)/length(d))
    end
end

function unload_distributed!(full_array, d)
    # load the distributed array d into the local full array

    # set up mpi, find rank
    comm = MPI.COMM_WORLD
    r = MPI.Comm_rank(comm)
    s = MPI.Comm_size(comm)

    # find indices of current rank
    # last rank gets fewer layers if not exactly divisible
    nz = size(full_array, 3)
    nz_per_rank = cld(nz, s)
    iz_min = 1 + r*nz_per_rank
    iz_max = min(nz, (1+r)*nz_per_rank)
    nz_local = iz_max - iz_min + 1

    if r==0
        #println("size full: ", size(full_array), ", size d: ", size(d), "iz:", iz_min:iz_max)
        @. d = full_array[:,:,iz_min:iz_max] # copy local part
        for i=1:s-1
            iz_min_remote = 1+i*nz_per_rank
            iz_max_remote = min(nz, (1+i)*nz_per_rank)
            #MPI.Send(view(full_array, :, :, iz_min_remote:iz_max_remote), i, 0, comm)
            MPI.Recv!(view(full_array, :, :, iz_min_remote:iz_max_remote), i, 0, comm)
        end
    else
        MPI.Send(d, 0, 0, comm)
        #println("Rank ", r, " sent data with mean = ", sum(d)/length(d))
    end
end

function mpi_wrap_state(state_pd, state_fd, transform)

    # state_pd: replace big arrays with distributed ones
    knew = Symbol[]
    vnew = []
    for k in keys(state_pd)
        if occursin("big", String(k))
            push!(knew, k)
            push!(vnew, prepare_distributed_array(eltype(state_pd[k]), size(state_pd[k])))
        else
            push!(knew, k)
            push!(vnew, state_pd[k])
        end
    end
    state_pd_new = NamedTuple{tuple(knew...)}(tuple(vnew...))

    # state_fd: drop big arrays (no longer needed, buffer used instead)
    knew = Symbol[]
    vnew = []
    for k in keys(state_fd)
        if !occursin("big", String(k))
            push!(knew, k)
            push!(vnew, state_fd[k])
        end
    end
    state_fd_new = NamedTuple{tuple(knew...)}(tuple(vnew...))

    # transform:
    # - replace big fft plans, add distributed version of small plans
    # - add distributed small buffers, replace big FD buffer, drop big PD buffer
    knew = Symbol[]
    vnew = []
    buffer_pd_d = prepare_distributed_array(eltype(transform.buffer_pd),
            size(transform.buffer_pd))
    buffer_fd_d = prepare_distributed_array(eltype(transform.buffer),
            size(transform.buffer))
    buffer_big_fd = prepare_distributed_array(eltype(transform.buffer_big_fd),
            size(transform.buffer_big_fd))
    for k in keys(transform)
        if k ∈ (:plan_fwd, :plan_bwd) # add distributed version of small plans
            push!(knew, k)
            push!(vnew, transform[k])
            push!(knew, Symbol(String(k) * "_d"))
            push!(vnew, (k == :plan_fwd) ? plan_rfft(buffer_pd_d, (1,2)) :
                plan_brfft(buffer_fd_d, size(buffer_pd_d,1), (1,2)))
        elseif k ∈ (:plan_fwd_big, :plan_bwd_big) # replace big plans
            push!(knew, k)
            push!(vnew, (k == :plan_fwd_big) ? plan_rfft(state_pd_new.u_big, (1,2)) :
                plan_brfft(buffer_big_fd, size(state_pd_new.u_big,1), (1,2)))
        elseif k == :buffer_pd
            push!(knew, k)
            push!(vnew, transform[k])
            push!(knew, :buffer_pd_d)
            push!(vnew, buffer_pd_d)
        elseif k == :buffer
            push!(knew, k)
            push!(vnew, transform[k])
            push!(knew, :buffer_fd_d)
            push!(vnew, buffer_fd_d)
        elseif k == :buffer_big_pd
            # drop buffer
        elseif k == :buffer_big_fd
            push!(knew, k)
            push!(vnew, prepare_distributed_array(eltype(transform[k]), size(transform[k])))
        else
            push!(knew, k)
            push!(vnew, transform[k])
        end
    end
    transform_new = NamedTuple{tuple(knew...)}(tuple(vnew...))

    state_pd_new, state_fd_new, transform_new
end

function mpi_wrap_gradients(gradients)
    knew = Symbol[]
    vnew = []
    for k in keys(gradients)
        if occursin("big_hat", String(k))
            # drop these
        elseif occursin("big", String(k)) # replace rot{x,y,z}_big with distributed
            push!(knew, k)
            push!(vnew, prepare_distributed_array(eltype(gradients[k]), size(gradients[k])))
        else
            push!(knew, k)
            push!(vnew, gradients[k])
        end
    end
    NamedTuple{tuple(knew...)}(tuple(vnew...))
end

#=
    - rhs.adv_big_{x,y,z}
    =#

function mpi_wrap_rhs(rhs)
    knew = Symbol[]
    vnew = []
    for k in keys(rhs)
        if !occursin("big", String(k)) # drop big ghost layer
            push!(knew, k)
            push!(vnew, rhs[k])
        end
    end
    for k in (:adv_big_x, :adv_big_y, :adv_big_z)
        push!(knew, k)
        push!(vnew, prepare_distributed_array(eltype(rhs.ghost_big),
                (size(rhs.ghost_big,1), size(rhs.ghost_big,2), size(rhs.rhs_u,3))))
    end
    for k in (:ghost_buffer, :rotx_w_top, :roty_w_top, :u_below, :v_below)
        push!(knew, k)
        push!(vnew, zeros(eltype(rhs.ghost_big), size(rhs.ghost_big)))
    end
    NamedTuple{tuple(knew...)}(tuple(vnew...))
end

@inline setzero!(A) = fill!(A, zero(eltype(A)))

function add_advection_dealiased_mpi!(rhs, state_pd, state_fd, gradients, tf)

    #=
    the products are computed with 3/2 of the frequencies, i.e. on
    1.5 times as many nodes in physical space. this will be done as follows:
    - compute dudy, dudz, dvdz, dvdx, dwdx, dwdy
    - expand velocities to 3/2 of nodes
    - expand curl terms (dwdy-dvdz), (dudz-dwdx), (dvdx-dudy) to 3/2 of nodes
    - compute product and reduce to regular nodes again
    =#

    # get expanded velocities from their frequency domain representations
    map((state_fd.u_hat, state_fd.v_hat, state_fd.w_hat),
        (state_pd.u_big, state_pd.v_big, state_pd.w_big)) do var_hat, var_big
        load_distributed!(tf.buffer_fd_d, var_hat)
        pad_frequencies!(tf.buffer_big_fd, tf.buffer_fd_d)
        LinearAlgebra.mul!(var_big, tf.plan_bwd_big, tf.buffer_big_fd)
    end

    # get expanded vorticities from their physical domain representations
    map((gradients.rotx, gradients.roty, gradients.rotz),
        (gradients.rotx_big, gradients.roty_big, gradients.rotz_big)) do var, var_big
        load_distributed!(tf.buffer_pd_d, var)
        LinearAlgebra.mul!(tf.buffer_fd_d, tf.plan_fwd_d, tf.buffer_pd_d)
        pad_frequencies!(tf.buffer_big_fd, tf.buffer_fd_d)
        LinearAlgebra.mul!(var_big, tf.plan_bwd_big, tf.buffer_big_fd)
    end

    # prepare mpi information
    mpi_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    mpi_size = MPI.Comm_size(MPI.COMM_WORLD)
    nz_local = size(tf.buffer_big_fd,3)

    # compute rot{x,y} * w on lowest w-level and pass down to next mpi rank
    map((rhs.rotx_w_top, rhs.roty_w_top),
       (gradients.rotx_big, gradients.roty_big)) do rot_vel_top, rot_full
        mpi_req = MPI.REQUEST_NULL
        if mpi_rank == 0
            setzero!(rhs.ghost_buffer) # w == 0 at bottom
        else
            rhs.ghost_buffer .= view(rot_full, :, :, 1) .*
                    view(state_pd.w_big, :, :, 1)
            mpi_req = MPI.Isend(rhs.ghost_buffer, mpi_rank-1, 0, MPI.COMM_WORLD)
        end
        if mpi_rank == mpi_size-1
            setzero!(rot_vel_top) # w == 0 at top
        else
            MPI.Recv!(rot_vel_top, mpi_rank+1, 0, MPI.COMM_WORLD)
        end
        MPI.Wait!(mpi_req)
    end

    # pass u & v on highest uvp-level up to next mpi rank
    map((rhs.u_below, rhs.v_below),
        (state_pd.u_big, state_pd.v_big)) do vel_below, vel_full
        mpi_req = MPI.REQUEST_NULL
        if mpi_rank > 0
            mpi_req = MPI.Isend(view(vel_full, :, :, 1), mpi_rank-1, 0, MPI.COMM_WORLD)
        end
        if mpi_rank == mpi_size-1
            @. vel_below = - vel_full[:,:,1] # u,v == 0 at bottom
        else
            MPI.Recv!(vel_below, mpi_rank+1, 0, MPI.COMM_WORLD)
        end
        MPI.Wait!(mpi_req)
    end

    # ADV_X: compute roty[w]*w[w]-rotz[uvp]*v[uvp] (interpolate w up to uvp)
    for i in CartesianIndices(rhs.adv_big_x)
        roty_w_below = gradients.roty_big[i] * state_pd.w_big[i]
        roty_w_above = (i[3] == nz_local) ? rhs.roty_w_top[i[1], i[2]] :
                gradients.roty_big[i[1], i[2], i[3]+1] * state_pd.w_big[i[1], i[2], i[3]+1]
        roty_w = 0.5 * (roty_w_below + roty_w_above)
        rotz_v = gradients.rotz_big[i] * state_pd.v_big[i]
        rhs.adv_big_x[i] = roty_w - rotz_v
    end

    # ADV_Y: compute rotz[uvp]*u[uvp]-rotx[w]*w[w] (interpolate w up to uvp)
    for i in CartesianIndices(rhs.adv_big_y)
        rotz_u = gradients.rotz_big[i] * state_pd.u_big[i]
        rotx_w_below = gradients.rotx_big[i] * state_pd.w_big[i]
        rotx_w_above = (i[3] == nz_local) ? rhs.rotx_w_top[i[1], i[2]] :
                gradients.rotx_big[i[1], i[2], i[3]+1] * state_pd.w_big[i[1], i[2], i[3]+1]
        rotx_w = 0.5 * (rotx_w_below + rotx_w_above)
        rhs.adv_big_y[i] = rotz_u - rotx_w
    end

    # ADV_Z: compute rotx[w]*v[uvp]-roty[w]*u[uvp] (interpolate uvp down to w)
    for i in CartesianIndices(rhs.adv_big_z)
        u_below = (i[3] == 1) ? rhs.u_below[i[1], i[2]] : state_pd.u_big[i[1], i[2], i[3]-1]
        v_below = (i[3] == 1) ? rhs.v_below[i[1], i[2]] : state_pd.v_big[i[1], i[2], i[3]-1]
        u_local = 0.5 * (state_pd.u_big[i] + u_below)
        v_local = 0.5 * (state_pd.v_big[i] + v_below)
        rhs.adv_big_z[i] = (gradients.rotx_big[i] * v_local - gradients.roty_big[i] * u_local)
    end

    # return expanded advection terms to regular nodes and add to RHS
    map((rhs.adv_big_x, rhs.adv_big_y, rhs.adv_big_z),
        (rhs.rhs_u, rhs.rhs_v, rhs.rhs_w)) do adv_big, rhs_var
        LinearAlgebra.mul!(tf.buffer_big_fd, tf.plan_fwd_big, adv_big)
        unpad_frequencies!(tf.buffer_fd_d, tf.buffer_big_fd)
        LinearAlgebra.mul!(tf.buffer_pd_d, tf.plan_bwd_d, tf.buffer_fd_d)
        unload_distributed!(tf.buffer_pd, tf.buffer_pd_d)
        if (MPI.Comm_rank(MPI.COMM_WORLD) == 0)
            @. rhs_var -= tf.buffer_pd
        end
    end

    rhs
end
function update_rhs_mpi!(rhs, state_pd, state_fd, gradients, transform, forcing, ν)

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        # apply pressure forcing first to avoid leftover values in rhs arrays
        fill!(rhs.rhs_u, forcing[1])
        fill!(rhs.rhs_v, forcing[2])
        fill!(rhs.rhs_w, forcing[3])

        # add pressure gradients from last time step
        @. rhs.rhs_u += gradients.dpdx
        @. rhs.rhs_v += gradients.dpdy
        @. rhs.rhs_w += gradients.dpdz
    end

    # add advection terms - (curl u) × u
    #add_advection!(rhs, state_pd, gradients)
    add_advection_dealiased_mpi!(rhs, state_pd, state_fd, gradients, transform)

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        # add viscous term (molecular diffusion)
        @. rhs.rhs_u += ν * gradients.lapu
        @. rhs.rhs_v += ν * gradients.lapv
        @. rhs.rhs_w += ν * gradients.lapw
    end

    rhs
end

function channelflow_mpi(gd::Grid{T}, tspan, u0; verbose = false) where T

    ν = 1e-2 # kinematic viscosity, 1.5e-5 for air

    to = TimerOutput()

    @timeit to "allocations" begin
        state_pd, state_fd, tf = mpi_wrap_state(prepare_state(gd)...)
        gradients = mpi_wrap_gradients(prepare_gradients(gd))
        rhs = mpi_wrap_rhs(prepare_rhs(gd))
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
        if (MPI.Comm_rank(MPI.COMM_WORLD) == 0)
            @timeit to "fourier transforms" update_fd!(state_fd, state_pd, tf)
            @timeit to "gradients" update_velocity_gradients!(gradients, state_pd, state_fd, tf)
        end
        @timeit to "build RHS" update_rhs_mpi!(rhs, state_pd, state_fd, gradients, tf, forcing, ν)
        if (MPI.Comm_rank(MPI.COMM_WORLD) == 0)
            @timeit to "time integration" euler_step!(state_pd, rhs, tspan.dt)
            @timeit to "pressure correction" euler_pressure_correction!(state_pd, state_fd,
            rhs, gradients, tf, tspan.dt)
        end
    end

    verbose && (MPI.Comm_rank(MPI.COMM_WORLD) == 0) && show(to)
    state_pd
end
