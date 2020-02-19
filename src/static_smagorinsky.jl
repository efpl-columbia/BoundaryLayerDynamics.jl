struct StaticSmagorinskiModel{T}
    model_constant::T
end

struct RoughWallEquilibriumModel{T}
    roughness_length::T
    von_karman_constant::T
end

struct FilteredAdvectionBuffers{T,P}

    vel::Tuple{Array{T,3},Array{T,3},Array{T,3}}

    vel_dx1::Tuple{Array{T,3},Array{T,3},Array{T,3}}
    vel_dx2::Tuple{Array{T,3},Array{T,3},Array{T,3}}
    vel_dx3::Tuple{Array{T,3},Array{T,3},Array{T,3}}

    vorticity::Tuple{Array{T,3},Array{T,3},Array{T,3}}
    vorticity_bcs::NTuple{2,NTuple{2,UnspecifiedBC{P,T}}}

    strain_rate::NTuple{3,NTuple{3,Array{T,3}}}
    partial_strain_h::Array{T,3}
    partial_strain_v::Array{T,3}
    partial_strain_bcs::NTuple{3,UnspecifiedBC{P,T}}

    sgs_model::StaticSmagorinskiModel{T}
    wall_model::RoughWallEquilibriumModel{T}
    eddy_viscosity_h::Array{T,3}
    eddy_viscosity_v::Array{T,3}
    sgs::NTuple{3,NTuple{3,Array{T,3}}}
    sgs_fd::Tuple{NTuple{3,Array{Complex{T},3}},
                  NTuple{3,Array{Complex{T},3}},
                  Tuple{Array{Complex{T},3},Array{Complex{T},3},Nothing}}
    sgs_bcs::NTuple{5,UnspecifiedBC{P,T}}

    adv::Tuple{Array{T,3},Array{T,3},Array{T,3}}

    filter_width::Tuple{T,T,T}
    grid_spacing::Tuple{T,T,T}

    function FilteredAdvectionBuffers(T, gd::DistributedGrid, domain_size, sgs_model, wall_model)

        # velocity and gradients
        vel = (zeros_pd(T, gd, :H), zeros_pd(T, gd, :H), zeros_pd(T, gd, :V))
        vel_dx1 = (zeros_pd(T, gd, :H), zeros_pd(T, gd, :H), zeros_pd(T, gd, :V))
        vel_dx2 = (zeros_pd(T, gd, :H), zeros_pd(T, gd, :H), zeros_pd(T, gd, :V))
        vel_dx3 = (zeros_pd(T, gd, :V), zeros_pd(T, gd, :V), zeros_pd(T, gd, :H))

        # vorticity
        vorticity = (zeros_pd(T, gd, :V), zeros_pd(T, gd, :V), zeros_pd(T, gd, :H))
        vorticity_bcs = ((UnspecifiedBC(T, gd), UnspecifiedBC(T, gd)),
                         (UnspecifiedBC(T, gd), UnspecifiedBC(T, gd)))

        # strain rate
        S13, S23, S12 = (zeros_pd(T, gd, :V), zeros_pd(T, gd, :V), zeros_pd(T, gd, :H))
        Sij = ((vel_dx1[1], S12, S13), (S12, vel_dx2[2], S23), (S13, S23, vel_dx3[3]))
        S_h, S_v = (zeros_pd(T, gd, :H), zeros_pd(T, gd, :V))
        S_bcs = (UnspecifiedBC(T, gd), UnspecifiedBC(T, gd), UnspecifiedBC(T, gd))

        eddy_viscosity_h = zeros_pd(T, gd, :H)
        eddy_viscosity_v = zeros_pd(T, gd, :V)

        τ11, τ22, τ33 = (zeros_pd(T, gd, :H), zeros_pd(T, gd, :H), zeros_pd(T, gd, :H))
        τ13, τ23, τ12 = (zeros_pd(T, gd, :V), zeros_pd(T, gd, :V), zeros_pd(T, gd, :H))
        sgs = ((τ11, τ12, τ13), (τ12, τ22, τ23), (τ13, τ23, τ33))

        τ11_fd, τ12_fd, τ13_fd = zeros_fd(T, gd, :H), zeros_fd(T, gd, :H), zeros_fd(T, gd, :V)
        τ22_fd , τ23_fd = zeros_fd(T, gd, :H), zeros_fd(T, gd, :V)
        sgs_fd = ((τ11_fd, τ12_fd, τ13_fd), (τ12_fd, τ22_fd, τ23_fd), (τ13_fd, τ23_fd, nothing))
        sgs_bcs = (UnspecifiedBC(T, gd), UnspecifiedBC(T, gd), UnspecifiedBC(T, gd), UnspecifiedBC(T, gd), UnspecifiedBC(T, gd))

        adv = (zeros_pd(T, gd, :H), zeros_pd(T, gd, :H), zeros_pd(T, gd, :V))

        # TODO: check whether this is the correct way of defining the filter width
        filter_width = domain_size ./ (2*gd.nx_fd, gd.ny_fd+1, gd.nz_global)
        grid_spacing = domain_size ./ (gd.nx_pd, gd.ny_pd, gd.nz_global)

        new{T,proc_type()}(vel, vel_dx1, vel_dx2, vel_dx3, vorticity, vorticity_bcs, Sij, S_h, S_v, S_bcs,
                 sgs_model, wall_model, eddy_viscosity_h, eddy_viscosity_v, sgs, sgs_fd, sgs_bcs, adv,
                 filter_width, grid_spacing)
    end
end

StaticSmagorinskiModel(; Cs = 0.1) = StaticSmagorinskiModel(Cs)

RoughWallEquilibriumModel(; z0 = 1e-3, kappa = 0.4) = RoughWallEquilibriumModel(z0, kappa)

dx3!(dx3::AbstractArray{T,2}, vel¯::AbstractArray{T,2}, vel⁺::AbstractArray{T,2}, Δx3) where T =
        @. dx3 = (vel⁺ - vel¯) * Δx3
dx3!(dx3::AbstractArray{T,2}, bc::DirichletBC, vel⁺::AbstractArray{T,2}, Δx3) where T =
        @. dx3 = (vel⁺ - bc.value) * Δx3
dx3!(dx3::AbstractArray{T,2}, vel¯::AbstractArray{T,2}, bc::DirichletBC, Δx3) where T =
        @. dx3 = (bc.value - vel¯) * Δx3

function set_advection!(adv, vel, df::DerivativeFactors{T}, ht::HorizontalTransform,
        lower_bcs::NTuple{3,BoundaryCondition{P,T}}, upper_bcs::NTuple{3,BoundaryCondition{P,T}},
        b::FilteredAdvectionBuffers{T, P}) where {P, T}

    # Compute ui, dui/dx1, dui/dx2 and transform to PD (9 terms).
    get_fields!(b.vel, ht, vel)
    get_fields_dx1!(b.vel_dx1, ht, vel, df)
    get_fields_dx2!(b.vel_dx2, ht, vel, df)

    # Compute vertical velocity gradients in physical domain
    u1ext = layers_expand_half_h(b.vel[1], upper_bcs[1]) # H to H below & above V-nodes
    u2ext = layers_expand_half_h(b.vel[2], upper_bcs[2]) # H to H below & above V-nodes
    u3ext = layers_expand_half_v(b.vel[3], lower_bcs[3], upper_bcs[3]) # V to V below & above H-nodes
    dx3!.(layers(b.vel_dx3[1]), u1ext[1:end-1], u1ext[2:end], df.dz1)
    dx3!.(layers(b.vel_dx3[2]), u2ext[1:end-1], u2ext[2:end], df.dz1)
    dx3!.(layers(b.vel_dx3[3]), u3ext[1:end-1], u3ext[2:end], df.dz1)

    # Compute vorticity in physical domain
    b.vorticity[1] .= b.vel_dx2[3] .- b.vel_dx3[2]
    b.vorticity[2] .= b.vel_dx3[1] .- b.vel_dx1[3]
    b.vorticity[3] .= b.vel_dx1[2] .- b.vel_dx2[1]

    # Compute advection term in physical domain.
    ω1ext = layers_expand_half_v(b.vorticity[1], b.vorticity_bcs[1]...) # V to V below & above H-nodes
    ω2ext = layers_expand_half_v(b.vorticity[2], b.vorticity_bcs[2]...) # V to V below & above H-nodes
    advu!.(layers(b.adv[1]), layers(b.vel[2]), ω2ext[1:end-1], ω2ext[2:end],
                                u3ext[1:end-1], u3ext[2:end], layers(b.vorticity[3]))
    advv!.(layers(b.adv[2]), layers(b.vel[1]), ω1ext[1:end-1], ω1ext[2:end],
                                u3ext[1:end-1], u3ext[2:end], layers(b.vorticity[3]))
    advw!.(layers(b.adv[3]), u1ext[1:end-1], u1ext[2:end], layers(b.vorticity[1]),
                                u2ext[1:end-1], u2ext[2:end], layers(b.vorticity[2]))

    # Compute Sij on its natural nodes in PD (Skk are already set to gradients)
    broadcast!((a,b) -> (a+b)/2, b.strain_rate[2][3], b.vel_dx2[3], b.vel_dx3[2])
    broadcast!((a,b) -> (a+b)/2, b.strain_rate[3][1], b.vel_dx3[1], b.vel_dx1[3])
    broadcast!((a,b) -> (a+b)/2, b.strain_rate[1][2], b.vel_dx1[2], b.vel_dx2[1])

    # Compute S on both sets of nodes in PD, interpolating Sij.
    @. b.partial_strain_h = b.strain_rate[1][1]^2 + b.strain_rate[2][2]^2 +
                            b.strain_rate[3][3]^2 + 2 * b.strain_rate[1][2]^2
    @. b.partial_strain_v = 2 * b.strain_rate[3][1]^2 + 2 * b.strain_rate[2][3]^2
    Sh_partial = layers_expand_half_h(b.partial_strain_h, b.partial_strain_bcs[1]) # unspecified bcs are reused here
    Sv_partial = layers_expand_half_v(b.partial_strain_v, b.partial_strain_bcs[2:3]...)

    # TODO: fill in Sv_partial at the very top & bottom, from S13 & S23 that are computed
    # with other stencils at the top & bottom

    # TODO: this uses the UnspecifiedBCs, so we should make sure these are not accessible
    # from multiple places, e.g. by wrapping the whole Sij computation in a
    # function and only defining the expanded velocities within that function

    Δ = cbrt(prod(b.filter_width))

    # note: this is 2 * (S_direct + (S_below + S_above)/2)
    νT!(νT, S_direct, S_below, S_above) = @. νT = b.sgs_model.model_constant^2 * Δ^2 * sqrt(2 * S_direct + S_below + S_above)

    # νT at first H-nodes doesn’t have S_below specified, need to use one-sided FDs to compute it
    @assert lower_bcs[3] isa DirichletBC # to ensure du3/dx1 and du3/dx2 are zero at lower boundary
    @assert upper_bcs[3] isa DirichletBC # to ensure du3/dx1 and du3/dx2 are zero at upper boundary
    @assert lower_bcs[1] isa DirichletBC # need other interpolation below otherwise
    @assert lower_bcs[2] isa DirichletBC # need other interpolation below otherwise
    @assert upper_bcs[1] isa DirichletBC # need other interpolation below otherwise
    @assert upper_bcs[2] isa DirichletBC # need other interpolation below otherwise
    νT!(νT, S_direct, bc::UnspecifiedBC, _) = νT = (b.sgs_model.model_constant^2 * Δ^2) * sqrt.(2 .* S_direct .+
        (first_layer(b.vel_dx1[3]) ./ 2 .+ # du3/dx1 at first H-node (½ because of interpolation)
            (u1ext[2] .+ 3 .* u1ext[1] .- 4 .* lower_bcs[1].value) .* (df.dz1/3) # du1/dx3 at first H-node
            ).^2 .+ # 2 S₁₃² + 2 S₃₁² = 4 S₁₃² = (du1/dx3 + du3/dx1)²
        (first_layer(b.vel_dx2[3]) ./ 2 .+ # du3/dx2 at first H-node (½ because of interpolation)
            (u2ext[2] .+ 3 .* u2ext[1] .- 4 .* lower_bcs[2].value) .* (df.dz1/3) # du2/dx3 at first H-node
            ).^2 ) # 2 S₂₃² + 2 S₃₂² = 4 S₂₃² = (du2/dx3 + du3/dx2)²

    # νT at last H-nodes doesn’t have S_above specified, need to use one-sided FDs to compute it
    νT!(νT, S_direct, _, bc::UnspecifiedBC) = νT .= (b.sgs_model.model_constant^2 * Δ^2) * sqrt.(2 .* S_direct .+
        (last_layer(b.vel_dx1[3]) ./ 2 .+ # du3/dx1 at last H-node (½ because of interpolation)
            (4 .* upper_bcs[1].value .- 3 .* u1ext[end] .- u1ext[end-1]) .* (df.dz1/3) # du1/dx3 at last H-node
            ).^2 .+ # 2 S₁₃² + 2 S₃₁² = 4 S₁₃² = (du1/dx3 + du3/dx1)²
        (last_layer(b.vel_dx2[3]) ./ 2 .+ # du3/dx2 at last H-node (½ because of interpolation)
            (4 .* upper_bcs[2].value .- 3 .* u2ext[end] .- u2ext[end-1]) .* (df.dz1/3) # du2/dx3 at first H-node
            ).^2 ) # 2 S₂₃² + 2 S₃₂² = 4 S₂₃² = (du2/dx3 + du3/dx2)²
        # TODO: this probably needs extra communication when there’s one layer per process

    # TODO: should rename "value" to "gradient" in NeumannBC to avoid accidentally treating it as
    # a DirichletBC

    map(νT!, layers(b.eddy_viscosity_h), layers(b.partial_strain_h), Sv_partial[1:end-1], Sv_partial[2:end])
    map(νT!, layers(b.eddy_viscosity_v), layers(b.partial_strain_v), Sh_partial[1:end-1], Sh_partial[2:end])

    # Compute τij in PD.
    @. b.sgs[1][1] = 2 * b.eddy_viscosity_h * b.strain_rate[1][1]
    @. b.sgs[2][2] = 2 * b.eddy_viscosity_h * b.strain_rate[2][2]
    @. b.sgs[3][3] = 2 * b.eddy_viscosity_h * b.strain_rate[3][3]
    @. b.sgs[1][2] = 2 * b.eddy_viscosity_h * b.strain_rate[1][2]
    @. b.sgs[1][3] = 2 * b.eddy_viscosity_v * b.strain_rate[3][1]
    @. b.sgs[2][3] = 2 * b.eddy_viscosity_v * b.strain_rate[2][3]

    # Add values of dτi3/dx3 to the resolved part of the non-linear term.
    function apply_wall_model(τ, vel, wm)

        x3ref = 1 / (2 * df.dz1) # height of first H-node, note that dz1 is 1/Δz
        @assert x3ref > wm.roughness_length "Roughness length is larger than distance to first grid point"

        @assert length(τ) >= 2

        first = τ[1]
        last  = τ[end]

        wall_model(u1, u2, u3_below, u3_above, ui) = wm.von_karman_constant^2 * ui / log(x3ref/wm.roughness_length)^2 *
                sqrt(u1^2 + u2^2 + (u3_below/2+u3_above/2)^2)

        if first isa UnspecifiedBC
            @assert lower_bcs[1] isa DirichletBC && lower_bcs[2].value == 0 &&
                    lower_bcs[2] isa DirichletBC && lower_bcs[2].value == 0
            broadcast!(wall_model, first.buffer_pd, first_layer(b.vel[1]), first_layer(b.vel[2]),
                lower_bcs[3].value, first_layer(b.vel[3]), first_layer(vel))
            first = first.buffer_pd
        end

        if last isa UnspecifiedBC
            @assert upper_bcs[1] isa DirichletBC && upper_bcs[2].value == 0 &&
                    upper_bcs[2] isa DirichletBC && upper_bcs[2].value == 0
            # TODO: handle case where the top process doesn’t have any w-velocity
            broadcast!(wall_model, last.buffer_pd, last_layer(b.vel[1]), last_layer(b.vel[2]),
                last_layer(b.vel[3]), upper_bcs[3].value, last_layer(vel))
            last.buffer_pd .*= -1
            last = last.buffer_pd
        end

        return first, τ[2:end-1]..., last

    end

    add_dx3!(f_out, f_below, f_above) = @. f_out += (f_above - f_below) * df.dz1
    add_dx3!(f_out, bc::UnspecifiedBC, _) = @. f_out += error("Cannot compute derivative without BC")
    add_dx3!(f_out, _, bc::UnspecifiedBC) = @. f_out += error("Cannot compute derivative without BC")

    # TODO: rename BCs or use different ones
    τ13 = layers_expand_half_v(b.sgs[1][3], b.sgs_bcs[1], b.sgs_bcs[2])
    τ13 = apply_wall_model(τ13, b.vel[1], b.wall_model)
    map(add_dx3!, layers(b.adv[1]), τ13[1:end-1], τ13[2:end]) # relies on wall model

    τ23 = layers_expand_half_v(b.sgs[2][3], b.sgs_bcs[3], b.sgs_bcs[4])
    τ23 = apply_wall_model(τ23, b.vel[2], b.wall_model)
    map(add_dx3!, layers(b.adv[2]), τ23[1:end-1], τ23[2:end]) # relies on wall model

    τ33 = layers_expand_half_h(b.sgs[3][3], b.sgs_bcs[5])
    map(add_dx3!, layers(b.adv[3]), τ33[1:end-1], τ33[2:end])

    # Transform these terms back to FD (8 terms)
    # NOTE: τ33 is not needed in FD, and τ21 == τ12
    set_fields!(adv, ht, b.adv)
    set_field!(b.sgs_fd[1][1], ht, b.sgs[1][1], NodeSet(:H))
    set_field!(b.sgs_fd[1][2], ht, b.sgs[1][2], NodeSet(:H))
    set_field!(b.sgs_fd[2][2], ht, b.sgs[2][2], NodeSet(:H))
    set_field!(b.sgs_fd[1][3], ht, b.sgs[1][3], NodeSet(:V))
    set_field!(b.sgs_fd[2][3], ht, b.sgs[2][3], NodeSet(:V))

    # 7) Add horizontal contributions of stress divergence to RHS
    @. adv[1] += b.sgs_fd[1][1] * df.dx1 + b.sgs_fd[1][2] * df.dy1
    @. adv[2] += b.sgs_fd[2][1] * df.dx1 + b.sgs_fd[2][2] * df.dy1
    @. adv[3] += b.sgs_fd[3][1] * df.dx1 + b.sgs_fd[3][2] * df.dy1

    # TODO: check computation of time step restriction
    dt_adv = advective_timescale(b.vel, b.grid_spacing)

    return adv, dt_adv
end

# HELPER FUNCTIONS (TODO: merge with other functions in other files)

equivalently(args...) = all(arg === args[1] for arg=args[2:end]) ?
                        args[1] : error("Arguments are not equivalent")

zeros_fd(T, gd, ns::Symbol) = zeros_fd(T, gd, NodeSet(ns))
zeros_pd(T, gd, ns::Symbol) = zeros_pd(T, gd, NodeSet(ns))

first_layer(A) = view(A, :, :, 1)
last_layer(A) = view(A, :, :, size(A, 3))

@inline buffer_for_field(::AbstractArray{T}, bc) where {T<:SupportedReals} = bc.buffer_pd
@inline buffer_for_field(::AbstractArray{Complex{T}}, bc) where {T<:SupportedReals} = bc.buffer_fd

"""
Transform a field from the frequency domain to an extended set of nodes in the
physical domain by adding extra frequencies set to zero.
The function optionally takes an array of prefactors which will be multiplied
with the values before the inverse transform. These prefactors don’t have to be
specified along all dimensions since singleton dimensions are broadcast.
"""
function get_field!(field_pd, ht::HorizontalTransform, field_fd, prefactors, ns::NodeSet)

    # highest frequencies in non-expanded array
    # this assumes that there are no Nyquist frequencies in field_fd
    # while this could be checked in y-direction, the x-direction is ambiguous
    # so we just assume that there is no Nyquist frequency since that’s how the
    # whole code is set up
    kx_max = size(field_fd, 1) - 1 # [0, 1…K]
    ky_max = div(size(field_fd, 2), 2) # [0, 1…K, -K…1]

    # copy frequencies and set the extra frequencies to zero
    buffer = get_buffer_fd(ht, ns)
    @views buffer[1:kx_max+1, 1:ky_max+1, :] .=
        field_fd[:, 1:ky_max+1, :] .*
        prefactors[:, 1:(size(prefactors, 2) == 1 ? 1 : ky_max+1), :]
    @views buffer[1:kx_max+1, end-ky_max+1:end, :] .=
        field_fd[:, ky_max+2:end, :] .*
        prefactors[:, (size(prefactors, 2) == 1 ? 1 : ky_max+2):end, :]
    buffer[1:kx_max+1, ky_max+2:end-ky_max, :] .= 0
    buffer[kx_max+2:end, :, :] .= 0

    # perform ifft (note that this overwrites buffer_big_fd due to the way fftw
    # works, and also note that the factor 1/(nx*ny) is always multiplied during
    # the forward transform and not here)
    LinearAlgebra.mul!(field_pd, get_plan_bwd(ht, ns), buffer)
    field_pd
end

# default when prefactors are missing
get_field!(field_pd, ht::HorizontalTransform, field_fd, ns::NodeSet) =
    get_field!(field_pd, ht, field_fd, ones(eltype(field_pd), (1, 1, 1)), ns::NodeSet)

# convenience functions that transform all three components
map_to_components(f, args...; nodes = (NodeSet(:H), NodeSet(:H), NodeSet(:V))) =
    f.(args..., nodes)
set_fields!(fields_fd, ht, fields_pd) =
    map_to_components(set_field!, fields_fd, (ht,), fields_pd)
get_fields!(fields_pd, ht, fields_fd) =
    map_to_components(get_field!, fields_pd, (ht,), fields_fd)
get_fields_dx1!(fields_dx1_pd, ht, fields_fd, df) =
    map_to_components(get_field!, fields_dx1_pd, (ht,), fields_fd, (df.dx1,))
get_fields_dx2!(fields_dx2_pd, ht, fields_fd, df) =
    map_to_components(get_field!, fields_dx2_pd, (ht,), fields_fd, (df.dy1,))

"""
This function takes a field defined on H-nodes, converts it into layers, and expands
them through communication such that the list includes the layers just above and
below all V-nodes. This means passing data down throughout the domain.
The boundary condition is only used for passing data, the actual values are unused.
"""
function layers_expand_half_h(field::AbstractArray{T}, bc_above::BoundaryCondition{P}) where {T,P}
    #TODO: find a better name for this function

    l = layers(field)

    if P == SingleProc
        l

    elseif P == MinProc
        buffer_above = buffer_for_field(field, bc_above)
        rq = (MPI.Irecv!(buffer_above, bc_above.neighbor_above, 1, MPI.COMM_WORLD), )
        map(MPI.Wait!, rq) # not using Waitall! to avoid allocation of vector
        l..., buffer_above

    elseif P == InnerProc
        buffer_above = buffer_for_field(field, bc_above)
        rq = (MPI.Isend(l[1], bc_above.neighbor_below, 1, MPI.COMM_WORLD),
              MPI.Irecv!(buffer_above, bc_above.neighbor_above, 1, MPI.COMM_WORLD))
        map(MPI.Wait!, rq) # not using Waitall! to avoid allocation of vector
        l..., buffer_above

    # there is a special case for when the top process does not have any V-nodes,
    # which is the case if there is only one layer per process. in this case, we
    # only have to pass data down, but we still return the (single) H-layer for
    # consistency so we always return Nv+1 layers
    elseif P == MaxProc
        rq = (MPI.Isend(l[1], bc_above.neighbor_below, 1, MPI.COMM_WORLD), )
        map(MPI.Wait!, rq) # not using Waitall! to avoid allocation of vector
        l

    else
        @error "Invalid process type: $(P)"
    end
end

"""
This function takes a field defined on V-nodes, converts it into layers, and expands
them through communication such that the list includes the layers just above and
below all H-nodes. This means passing data up throughout the domain.
"""
function layers_expand_half_v(field::AbstractArray{T}, bc_below::BoundaryCondition{P},
                        bc_above::BoundaryCondition{P}) where {T,P}
    #TODO: find a better name for this function

    l = layers(field)
    neighbor_below = equivalently(bc_below.neighbor_below, bc_above.neighbor_below)
    neighbor_above = equivalently(bc_below.neighbor_above, bc_above.neighbor_above)

    if P == SingleProc
        bc_below, l..., bc_above

    elseif P == MinProc
        rq = (MPI.Isend(l[end], neighbor_above, 1, MPI.COMM_WORLD), )
        map(MPI.Wait!, rq) # not using Waitall! to avoid allocation of vector
        bc_below, l...

    elseif P == InnerProc
        buffer_below = buffer_for_field(field, bc_below)
        rq = (MPI.Isend(l[end], neighbor_above, 1, MPI.COMM_WORLD),
              MPI.Irecv!(buffer_below, neighbor_below, 1, MPI.COMM_WORLD))
        map(MPI.Wait!, rq) # not using Waitall! to avoid allocation of vector
        buffer_below, l...

    # there is a special case for when the top process does not have any V-nodes,
    # which is the case if there is only one layer per process. in this case, we
    # only have to pass data down, but we still return the (single) H-layer for
    # consistency so we always return Nv+1 layers
    elseif P == MaxProc
        buffer_below = buffer_for_field(field, bc_below)
        rq = (MPI.Irecv!(buffer_below, neighbor_below, 1, MPI.COMM_WORLD), )
        map(MPI.Wait!, rq) # not using Waitall! to avoid allocation of vector
        buffer_below, l..., bc_above

    else
        @error "Invalid process type: $(P)"
    end
end
