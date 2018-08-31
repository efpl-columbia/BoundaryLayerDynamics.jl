abstract type BoundaryCondition{T} end
struct DirichletBC{T} <: BoundaryCondition{T} value::T end
struct NeumannBC{T} <: BoundaryCondition{T}
    value::T # values has δz premultiplied
    NeumannBC(value::T, δz::T) where T= new{T}(value * δz)
end

struct BigTransform{T<:SupportedReals}
    n::Tuple{Int,Int,Int}
    plan_fwd::FFTW.rFFTWPlan{T,FFTW.FORWARD,false,3}
    plan_bwd::FFTW.rFFTWPlan{Complex{T},FFTW.BACKWARD,false,3}
    buffers_pd::NTuple{9,Array{T,3}}
    buffers_fd::NTuple{1,Array{Complex{T},3}}
    buffer_layers_pd::NTuple{5,Array{T,2}}
    function BigTransform(gd::Grid{T}) where T
        n = (3*div(gd.n[1],2), 3*div(gd.n[2],2), gd.n[3])
        buffers_pd = Tuple(zeros(T, n) for i=1:9)
        buffers_fd = Tuple(zeros(Complex{T}, div(n[1],2)+1, n[2], n[3]) for i=1:1)
        new{T}(n, plan_rfft(buffers_pd[1], (1,2)), plan_brfft(buffers_fd[1], n[1], (1,2)),
            buffers_pd, buffers_fd, Tuple(zeros(T, n[1], n[2]) for i=1:5))
    end
end

struct DerivativeFactors{T<:SupportedReals}
    dx1::Array{Complex{T},3}
    dy1::Array{Complex{T},3}
    dz1::T
    dx2::Array{T,3}
    dy2::Array{T,3}
    dz2::T
    DerivativeFactors(gd::Grid{T}) where T = new{T}(
            reshape(1im * gd.k[1] * (2π/gd.l[1]), length(gd.k[1]), 1, 1),
            reshape(1im * gd.k[2] * (2π/gd.l[2]), 1, length(gd.k[2]), 1),
            1/gd.δ[3],
            reshape( - gd.k[1].^2 * (2π/gd.l[1])^2, length(gd.k[1]), 1, 1),
            reshape( - gd.k[2].^2 * (2π/gd.l[2])^2, 1, length(gd.k[2]), 1),
            1/gd.δ[3]^2,
    )
end

"""
This is a new attempt at how to best implement the channel flow DNS in Julia.
This time, we compute everything in frequency domain, only going back to the
physical domain when necessary. Currently, this is only the case for the
non-linear advection term, for which we need pass through Fourier space anyway
in order to perform the dealiasing.

Currently, there are (almost) complete implementations of the viscous term as
well as the non-linear advection term. We are still missing the initialization,
which could be performed with dealiasing as well, adding the pressure terms to
the RHS, and the pressure solver, as well as the time stepping.
"""
struct ChannelFlowProblem{T<:SupportedReals}

    # problem definition
    grid::Grid{T}
    lower_bc::NTuple{3,BoundaryCondition{T}}
    upper_bc::NTuple{3,BoundaryCondition{T}}
    forcing::NTuple{3,T}

    # state
    vel_hat::NTuple{3,OffsetArray{Complex{T},3,Array{Complex{T},3}}}
    p_hat::OffsetArray{Complex{T},3,Array{Complex{T},3}}

    # buffered values
    rhs_hat::NTuple{3,Array{Complex{T},3}}
    rot_hat::NTuple{3,Array{Complex{T},3}}
    tf_big::BigTransform{T}
    df::DerivativeFactors{T}

    # inner constructor
    ChannelFlowProblem(grid::Grid{T}, lower_bc::NTuple{3,BoundaryCondition{T}},
            upper_bc::NTuple{3,BoundaryCondition{T}}, forcing::NTuple{3,T},
            ) where T = new{T}(grid, lower_bc, upper_bc, forcing,
            Tuple(make_buffered_array_fd(grid) for i=1:3), # vel_hat
            make_buffered_array_fd(grid), # phat
            Tuple(make_array_fd(grid) for i=1:3), # rhs_hat
            Tuple(make_array_fd(grid) for i=1:3), # rot_hat
            BigTransform(grid), DerivativeFactors(grid),
        )
end

noslip(T) = (DirichletBC{T}(zero(T)), DirichletBC{T}(zero(T)),
        DirichletBC{T}(zero(T)))
freeslip(T) = (NeumannBC{T}(zero(T), zero(T)), NeumannBC{T}(zero(T), zero(T)),
        DirichletBC{T}(zero(T)))

ChannelFlowProblem(grid::Grid{T}) where T = ChannelFlowProblem(grid,
        noslip(T), noslip(T), (one(T), zero(T), zero(T)))

initialize!(cf::ChannelFlowProblem, u::Tuple) = initialize!(cf, u...)
initialize!(cf::ChannelFlowProblem{T}, u0) where T =
    initialize!(cf, u0, (x,y,z) -> zero(T))
initialize!(cf::ChannelFlowProblem{T}, u0, v0) where T =
    initialize!(cf, u0, v0, (x,y,z) -> zero(T))

# nodes generally start at zero, vertically direction is centered for uvp-nodes
@inline coord(i, δ, ::Val{:uvp}) = (δ[1] * (i[1]-1),
                                    δ[2] * (i[2]-1),
                                    δ[3] * (2*i[3]-1)/2)
@inline coord(i, δ, ::Val{:w})   = (δ[1] * (i[1]-1),
                                    δ[2] * (i[2]-1),
                                    δ[3] * (i[3]-1))

function initialize!(cf::ChannelFlowProblem, u0, v0, w0)
    δ_big = cf.grid.l ./ cf.tf_big.n
    for (vel_hat, vel_0, nodes) in zip(cf.vel_hat, (u0, v0, w0), (Val(:uvp), Val(:uvp), Val(:w)))
        initialize!(vel_hat, vel_0, δ_big, nodes, cf.tf_big.plan_fwd,
                cf.tf_big.buffers_pd[1], cf.tf_big.buffers_fd[1])
    end
    apply_bcs!(cf)
end

function initialize!(vel_hat, vel_0, δ_big, nodes, plan_big_fwd, buffer_big_pd, buffer_big_fd)
    for i in CartesianIndices(buffer_big_pd)
        buffer_big_pd[i] = vel_0(coord(i, δ_big, nodes)...)
    end
    fft_dealiased!(vel_hat, buffer_big_pd, plan_big_fwd, buffer_big_fd)
end

function build_rhs!(cf)
    set_advection_fd!(cf.rhs_hat, cf.vel_hat, cf.rot_hat, cf.df, cf.tf_big)
    add_laplacian_fd!(cf.rhs_hat, cf.vel_hat, cf.df)
end

make_array_fd(gd::Grid{T}) where T = Array(zeros(Complex{T},
        div(gd.n[1],2)+1, gd.n[2], gd.n[3]))

make_buffered_array_fd(gd::Grid{T}) where T = OffsetArray(zeros(Complex{T},
        div(gd.n[1],2)+1, gd.n[2], gd.n[3]+2),
        1:div(gd.n[1],2)+1, 1:gd.n[2], 0:gd.n[3]+1)

@inline innerindices(A::OffsetArray) = CartesianIndices((1:size(A,1),
        1:size(A,2), 1:size(A,3)-2))
@inline innerindices(A::Array) = CartesianIndices(A)

# uniform Dirichlet boundary conditions for u- and v-velocities
function set_lower_bc!(vel_hat, bc::DirichletBC) # (u[0] + u[1]) / 2 = value
    @views vel_hat[:,:,0] .= -vel_hat[:, :, 1]
    vel_hat[1,1,0] += 2 * bc.value
end
function set_upper_bc!(vel_hat, bc::DirichletBC) # (u[end-1] + u[end]) / 2 = value
    @views vel_hat[:,:,end] .= -vel_hat[:,:,end-1]
    vel_hat[1,1,end] += 2 * bc.value
end

# uniform Neumann boundary conditions for u- and v-velocities
set_lower_bc!(vel_hat, bc::NeumannBC) = @views vel_hat[:,:,0] .=
        vel_hat[:,:,1] - bc.value # value is multiplied by δ[3]
set_upper_bc!(vel_hat, bc::NeumannBC) = @views vel_hat[:,:,end] .=
        vel_hat[:,:,end-1] + bc.value # value is multiplied by δ[3]

# uniform Dirichlet boundary conditions for w-velocity (only Dirichlet supported)
function set_lower_bc_w!(vel_hat, bc::DirichletBC)
    vel_hat[:,:,1] .= zero(eltype(vel_hat))
    vel_hat[1,1,1] = bc.value
end
function set_upper_bc_w!(vel_hat, bc::DirichletBC)
    vel_hat[:,:,end] .= zero(eltype(vel_hat))
    vel_hat[1,1,end] = bc.value
end

function apply_bcs!(cf::ChannelFlowProblem)
    set_lower_bc!(cf.vel_hat[1], cf.lower_bc[1])
    set_upper_bc!(cf.vel_hat[1], cf.upper_bc[1])
    set_lower_bc!(cf.vel_hat[2], cf.lower_bc[2])
    set_upper_bc!(cf.vel_hat[2], cf.upper_bc[2])
    set_lower_bc_w!(cf.vel_hat[3], cf.lower_bc[3])
    set_upper_bc_w!(cf.vel_hat[3], cf.upper_bc[3])
    cf
end

"""
Compute the Laplacian of a velocity and add it to the RHS array,
all in frequency domain. Note that this requires the values at iz=0
and iz=end in the vel_hat array, so they must be set from boundary
conditions and MPI exchanges before calling this function. The lowest
level of w-nodes can be set to NaNs, as the iz=1 level is at the
boundary and should not have a RHS.
"""
function add_laplacian_fd!(rhs_hat, vel_hat, df::DerivativeFactors)
    # for uvp-nodes: rely on values in iz=0 and iz=end in vel_hat for
    # top & bottom layer
    for i in innerindices(vel_hat)
        rhs_hat[i] = vel_hat[i[1], i[2], i[3]-1] * df.dz2 +
                (df.dx2[i[1]] + df.dy2[i[2]] - 2 * df.dz2) +
                vel_hat[i[1], i[2], i[3]+1] * df.dz2
    end
end

function add_laplacian_fd!(rhs_hat::Tuple, vel_hat::Tuple, df::DerivativeFactors)
    for (rh, vh) in zip(rhs_hat, vel_hat)
        add_laplacian_fd!(rh, vh, df)
    end
end

# compute one layer of vorticity (in frequency domain)
# (dw/dy - dv/dz) and (du/dz - dw/dx) on w-nodes, (dv/dx - du/dy) on uvp-nodes
@inline rot_u!(rot_u, v¯, v⁺, w, dy, dz) = @. rot_u = w * dy - (v⁺ - v¯) * dz
@inline rot_v!(rot_v, u¯, u⁺, w, dx, dz) = @. rot_v = (u⁺ - u¯) * dz - w * dx
@inline rot_w!(rot_w, u, v, dx, dy)      = @. rot_w = v * dx - u * dy

function compute_vorticity_fd!(rot_hat, vel_hat, dx, dy, dz)
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
    # -> which nx & ny? should be small values to get correct velocities in
    # physical domain, but needs to be big one to get correct frequencies again
    # when going back to the frequency domain
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

function set_advection_fd!(rhs_hat, vel_hat, rot_hat, df, tf_big)
    # need: rot_hat[1:3], vel_big[1:3], rot_big[1:3], plan_big_{fwd,bwd},
    # buffer_big_fd, buffer_layers_pd[1:5]

    # compute vorticity in fourier domain
    compute_vorticity_fd!(rot_hat, vel_hat, df.dx1, df.dy1, df.dz1)

    rot_big = tf_big.buffers_pd[1:3]
    vel_big = tf_big.buffers_pd[4:6]
    adv_big = tf_big.buffers_pd[7:9]

    # for each velocity and vorticity, pad the frequencies and transform
    # to physical space (TODO: fix boundaries)
    for (field_big, field_hat) in zip((rot_big..., vel_big...), (rot_hat..., vel_hat...))
        ifft_dealiased!(field_big, field_hat, tf_big.plan_bwd, tf_big.buffers_fd[1])
    end

    # compute the advection term in physical domain
    compute_advection_pd!(adv_big, rot_big, vel_big, tf_big.buffer_layers_pd[1:5])

    # transform advection term back to frequency domain and set RHS to result,
    # skipping higher frequencies for dealiasing
    for (field_hat, field_big) in zip(rhs_hat, adv_big)
        fft_dealiased!(field_hat, field_big, tf_big.plan_fwd, tf_big.buffers_fd[1])
    end

    rhs_hat
end
