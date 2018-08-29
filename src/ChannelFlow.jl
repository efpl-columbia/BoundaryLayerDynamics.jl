module ChannelFlow

import LinearAlgebra, MPI

using FFTW, TimerOutputs

include("mpi.jl")
include("advection.jl")
include("pressure_solver.jl")

# wavenumbers in fft order, with zero for nyquist frequency
wavenumbers(n) = map(i -> i<n/2 ? i : i>n/2 ? i-n : 0 , 0:n-1)

function prepare_state(T::Type{<:Real}, grid)

    # shorthand to initialize arrays
    init_field_pd() = zeros(T, grid.nx, grid.ny, grid.nz)
    init_field_fd() = zeros(Complex{T}, div(grid.nx,2)+1, grid.ny, grid.nz)

    # shorthand to initialize enlarged arrays for dealiasing
    nx_big, ny_big = (3*div(grid.nx,2), 3*div(grid.nx,2))
    init_field_big_pd() = zeros(T, nx_big, ny_big, grid.nz)
    init_field_big_fd() = zeros(Complex{T}, div(nx_big,2)+1, ny_big, grid.nz)

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
    kx = wavenumbers(grid.nx)[1:div(grid.nx,2)+1]
    ky = wavenumbers(grid.ny)

    transform = (
        plan_fwd     = plan_rfft(state_pd.u, (1,2)),
        plan_bwd     = plan_brfft(state_fd.u_hat, grid.nx, (1,2)),
        plan_fwd_big = plan_rfft(state_pd.u_big, (1,2)),
        plan_bwd_big = plan_brfft(state_fd.u_big_hat, nx_big, (1,2)),
        buffer=init_field_fd(),
        buffer_pd=init_field_pd(),
        buffer_big_pd=init_field_big_pd(),
        buffer_big_fd=init_field_big_fd(),
        dx  = Complex{T}[1im * 2*π/grid.lx * k for k=kx],
        dy  = Complex{T}[1im * 2*π/grid.ly * k for k=ky],
        dx_premultiplied = Complex{T}[1im * 2*π/grid.lx * k / (grid.nx*grid.ny) for k=kx],
        dy_premultiplied = Complex{T}[1im * 2*π/grid.ly * k / (grid.nx*grid.ny) for k=ky],
        dx2 = T[1/(grid.nx*grid.ny) * (-4*π*π)/(grid.lx*grid.lx) * k * k
                for k=wavenumbers(grid.nx)[1:div(grid.nx,2)+1]],
        dy2 = T[1/(grid.nx*grid.ny) * (-4*π*π)/(grid.ly*grid.ly) * k * k
                for k=wavenumbers(grid.ny)],
        dd2 = [prepare_laplacian(T, grid, kx, ky) for kx=kx, ky=ky],
        duvdz = (uvp=true, dirichlet=true, dz=convert(T, grid.lz/grid.nz),
            ghost=zeros(T, grid.nx, grid.ny)),
        dpdz = (uvp=true, dirichlet=false, dz=convert(T, grid.lz/grid.nz),
            ghost=zeros(T, grid.nx, grid.ny)),
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


function prepare_gradients(T::Type{<:Real}, grid)
    init_field() = zeros(T, grid.nx, grid.ny, grid.nz)
    nx_big, ny_big = (3*div(grid.nx,2), 3*div(grid.nx,2))
    init_field_big_pd() = zeros(T, nx_big, ny_big, grid.nz)
    init_field_big_fd() = zeros(Complex{T}, div(nx_big,2)+1, ny_big, grid.nz)
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

function prepare_rhs(T::Type{<:Real}, grid)
    nx_big = div(grid.nx,2)*3
    ny_big = div(grid.ny,2)*3
    (
        rhs_u = zeros(T, grid.nx, grid.ny, grid.nz),
        rhs_v = zeros(T, grid.nx, grid.ny, grid.nz),
        rhs_w = zeros(T, grid.nx, grid.ny, grid.nz),
        ghost = zeros(T, grid.nx, grid.ny),
        ghost_big = zeros(T, nx_big, ny_big),
        div_fd = zeros(Complex{T}, div(grid.nx,2)+1, grid.ny, grid.nz)
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

channelflow(grid, tspan, u0; kwargs...) = channelflow(Float64, grid, tspan, u0; kwargs...)

function channelflow(T::Type{<:Real}, grid, tspan, u0; verbose = false)

    ν = 1e-2 # kinematic viscosity, 1.5e-5 for air

    to = TimerOutput()

    @timeit to "allocations" begin
        state_pd, state_fd, tf = prepare_state(T, grid)
        gradients = prepare_gradients(T, grid)
        rhs = prepare_rhs(T, grid)
        forcing = (one(T), zero(T), zero(T))
    end

    # initialize velocity field
    @timeit to "initialization" begin
        dx, dy, dz = grid.lx/grid.nx, grid.ly/grid.ny, grid.lz/grid.nz
        for i in CartesianIndices(state_pd.u)
            state_pd.u[i] = u0((i[1]-1)*dx, i[2]-1*dy, i[3]-1*dz)
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

end # module
