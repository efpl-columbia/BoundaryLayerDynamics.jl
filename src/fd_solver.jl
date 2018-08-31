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

    grid::Grid{T}
    vel_hat::NTuple{3,OffsetArray{Complex{T},3,Array{Complex{T},3}}}
    p_hat::OffsetArray{Complex{T},3,Array{Complex{T},3}}
    rhs_hat::NTuple{3,OffsetArray{Complex{T},3,Array{Complex{T},3}}}

    ChannelFlowProblem(grid::Grid{T}) where T = new{T}(
            grid,
            map(i -> make_array_fd(grid), (1,2,3)), # vel_hat
            make_array_fd(grid), # phat
            map(i -> make_array_fd(grid), (1,2,3)), # rhs_hat
        )
end

initialize!(cf::ChannelFlowProblem, u::Tuple) = initialize!(cf, u...)
initialize!(cf::ChannelFlowProblem{T}, u0) where T =
    initialize!(cf, u0, v0 = (x,y,z) -> zero(T))
initialize!(cf::ChannelFlowProblem{T}, u0, v0) where T =
    initialize!(cf, u0, v0, w0 = (x,y,z) -> zero(T))

# nodes are centered in horizontal direction, centered in vertical for uvp-nodes
# TODO: decide whether nodes should be centered or start at x=0 and y=0
# - advantages centered: more like finite volume cells, no asymmetry left/right
# - advantages zero: works for different grid spacing (origin is the same)
@inline coord(i, δ, ::Val{uvp}) = (δ[1] * (2*i[1]-1)/2,
                                   δ[2] * (2*i[2]-1)/2,
                                   δ[3] * (2*i[3]-1)/2)
@inline coord(i, δ, ::Val{w})   = (δ[1] * (2*i[1]-1)/2,
                                   δ[2] * (2*i[2]-1)/2,
                                   δ[3] * (i[3]-1))

function initialize!(cf::ChannelFlowProblem{T}, u0, v0, w0)

    broadcast!(initialize!, cf.vel_hat, (u0, v0, w0), δ_big, (Val(:uvp),
            Val(:uvp), Val(:w)), plan_big_fwd, buffer_big_pd, buffer_big_fd)
end

function initialize!(vel_hat, vel_0, δ_big, nodes, plan_big_fwd, buffer_big_pd, buffer_big_fd)
    fill!(vel_hat, zero(eltype(vel_hat)))
    for i in CartesianIndices(buffer_big_pd)
        buffer_big_pd[i] = vel_0(coord(i, δ_big, nodes))
    end
    add_fft_dealiased!(rhs_hat, buffer_big_pd, plan_big_fwd, buffer_big_fd)
end

abstract type BoundaryCondition end
struct DirichletBC{T} <: BoundaryCondition
    val::T
end
struct NeumannBC{T} <: BoundaryCondition
    val::T
    δz::T
end

struct DerivativeFactors{T<:SupportedReals}
    dx2::Array{T,1}
    dy2::Array{T,1}
    dz2::T
    DerivativeFactors(gd::Grid{T}) where T = new{T}(
            - gd.k[1].^2 * (2π/gd.l[1])^2,
            - gd.k[2].^2 * (2π/gd.l[2])^2,
            1/gd.δ[3]^2,
    )
end

make_array_fd(gd::Grid{T}) where T = OffsetArray(zeros(Complex{T},
        div(gd.n[1],2)+1, gd.n[2], gd.n[3]+2),
        1:div(gd.n[1],2)+1, 1:gd.n[2], 0:gd.n[3]+1)

@inline innerindices(A::OffsetArray) = CartesianIndices((1:size(A,1),
        1:size(A,2), 1:size(A,3)-2))

# TODO: make this work properly with non-zero boundary conditions
function set_bc_uvp!(vel_hat, bc_bottom::BoundaryCondition, bc_top::BoundaryCondition)
    if bc_bottom isa DirichletBC
        @. vel_hat[:,:,0] = 2 * bc_bottom.val - vel_hat[:,:,1]
    else
        @. vel_hat[:,:,0] = vel_hat[:,:,1] - bc_bottom.val * bc_bottom.δz
    end
    if bc_top isa DirichletBC
        @. vel_hat[:,:,end] = 2 * bc_top.val - vel_hat[:,:,1]
    else
        @. vel_hat[:,:,end] = vel_hat[:,:,1] + bc_top.val * bc_top.δz
    end
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
    for i in innerindices(buffer_big_fd)
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
frequency domain and remove extra frequencies. The result is added to the output
array rather than replacing the values.
"""
function add_fft_dealiased!(field_hat, field_big, plan_big_fwd, buffer_big_fd)

    LinearAlgebra.mul!(buffer_big_fd, plan_big_fwd, field_big)

    # highest frequencies (excluding nyquist) in non-expanded array
    # warning: this assumes that nx & ny (before the fft)
    #          are even in the non-expanded array
    ny, ny_big = size(field_hat,2), size(buffer_big_fd,2)
    kx_max = size(field_hat,1) - 2 # [0, 1…kmax, nyquist]
    ky_max = div(ny,2) - 1 # [0, 1…kmax, nyquist, -kmax…1]

    for i in innerindices(field_hat)
        if i[1] == ky_max+2 || i[2] == ky_max+2
            field_hat[i] += 0
        else
            field_hat[i] += i[2] <= 1+ky_max ? buffer_big_fd[i] :
                buffer_big_fd[i[1], i[2] + (ny_big - ny), i[3]]
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
    iz_min, iz_max = (1, size(vel_big[1],3)-2) # inner layers

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
        @views adv_z!(adv[3][:,:,k+1],
                      vel[1][:,:,k], vel[1][:,:,k+1], rot[1][:,:,k+1],
                      vel[2][:,:,k], vel[2][:,:,k+1], rot[2][:,:,k+1])
    end

    # add last layer, using the values above & below as exchanged earlier
    @views adv_u!(adv[1][:,:,k],
                  vel[2][:,:,k], rot[2][:,:,k], roty_above,
                  vel[3][:,:,k], w_above, rot[3][:,:,k])
    @views adv_v!(adv[2][:,:,k],
                  vel[1][:,:,k], rot[1][:,:,k], rotx_above,
                  vel[3][:,:,k], w_above, rot[3][:,:,k])
    @views adv_z!(adv[3][:,:,k],
                  u_below, vel[1][:,:,k], rot[1][:,:,k],
                  v_below, vel[2][:,:,k], rot[2][:,:,k])

    adv
end

function add_advection_fd!(rhs_hat, vel_hat, df)
    # need: rot_hat[1:3], vel_big[1:3], rot_big[1:3], plan_big_{fwd,bwd},
    # buffer_big_fd, buffer_layers_pd[1:5]

    # compute vorticity in fourier domain
    compute_vorticity_fd!(rot_hat, vel_hat, df)

    # for each velocity and vorticity, pad the frequencies and transform
    # to physical space (TODO: fix boundaries)
    broadcast(ifft_dealiased!, rot_big, rot_hat, plan_big_bwd, buffer_big_fd)
    broadcast(ifft_dealiased!, vel_big, vel_hat, plan_big_bwd, buffer_big_fd)

    # compute the advection term in physical domain
    compute_advection_pd!(adv_big, rot_big, vel_big, buffer_layers_pd)

    # transform advection term back to frequency domain and add to the
    # RHS, skipping higher frequencies for dealiasing
    broadcast!(add_fft_dealiased!, rhs_hat, adv_big, plan_big_fwd, buffer_big_fd)

    rhs_hat
end
