struct RoughWallEquilibriumBC{P,T} <: SolidWallBC{P,T}
    value::T # set to 0, so it can be used just like a DirichletBC
    roughness_length::T
    von_karman_constant::T
    wall_distance::T
    buffer_fd::Array{Complex{T},2}
    buffer_pd::Array{T,2}
    neighbor_below::Int
    neighbor_above::Int
    RoughWallEquilibriumBC(roughness_length::T, von_karman_constant::T, wall_distance::T,
                           nh_fd::Tuple{Int,Int}, nh_pd::Tuple{Int,Int}) where T = begin
        @assert wall_distance > roughness_length "Roughness length is larger than distance to first grid point"
        new{proc_type(),T}(zero(T), roughness_length, von_karman_constant, wall_distance,
                           zeros(Complex{T}, nh_fd...), zeros(T, nh_pd...), proc_below(), proc_above())
    end
end

RoughWallEquilibriumBC(roughness_length::T, gd::DistributedGrid,
                       gm::GridMapping{T}; kappa::T = convert(T, 0.4)) where T =
                       RoughWallEquilibriumBC(roughness_length, kappa,
                       gm.vmap(one(T)/(2*gd.nz_global)),
                       (gd.nx_fd, gd.ny_fd), (gd.nx_pd, gd.ny_pd))

"""
    StaticSmagorinskyModel(<keyword arguments>)

Define a static Smagorinsky model for subgrid-scale stresses, based on [Smagorinsky
(1963)](https://doi.org/10.1175/1520-0493(1963)091%3C0099:GCEWTP%3E2.3.CO;2).
The model is based on the eddy viscosity assumption

``τ_{ij}^{sgs} = - 2 ν_T S_{ij}``

with ``S_{ij}`` denoting the resolved rate of strain tensor. The eddy
viscosity ``ν_T`` is modeled as

``ν_T = C_s^2 Δ^2 |S|``

where ``Δ=(Δ_1+Δ_2+Δ_3)^{1/3}`` is the scale of the grid spacing,
``|S|=(2S_{ij}S_{ij})^{1/2}`` is the total rate of strain, and ``C_s`` is a
non-dimensional model coefficient.

The model can optionally reduce the mixing length close to rough-wall surfaces
using the wall-damping function described in [Mason & Thomson
(1992)](https://doi.org/10.1017/S0022112092002271).

# Arguments

- `Cs=0.1`: the model coefficient ``C_s``
- `wall_damping=false`: if set, a Mason-Thomson wall-damping function is used
  to reduce the mixing length close to rough-wall surfaces
- `wall_damping_exponent=2`: the exponent used in the Mason–Thomson
  wall-damping function
"""
struct StaticSmagorinskyModel{T1,T2}
    model_constant::T1
    wall_damping::Bool
    wall_damping_exponent::T2
end

StaticSmagorinskyModel(; Cs = 0.1, wall_damping=false, wall_damping_exponent=2) =
        StaticSmagorinskyModel(Cs, wall_damping, wall_damping_exponent)

mixing_length(x3, bcs::Tuple{NeumannBC,NeumannBC,DirichletBC}) = Inf
mixing_length(x3, bcs::Tuple{RoughWallEquilibriumBC,RoughWallEquilibriumBC,DirichletBC}) =
        x3 * equivalently(bcs[1].von_karman_constant, bcs[2].von_karman_constant)
mixing_length(x3, x3_lbc, x3_ubc, lbcs, ubcs) = min(mixing_length(x3-x3_lbc, lbcs), mixing_length(x3_ubc-x3, ubcs))

struct StaticSmagorinskyBuffers{T}
    length_scale_h::Array{T,3}
    length_scale_v::Array{T,3}
    StaticSmagorinskyBuffers(sgs_model::StaticSmagorinskyModel{T}, gd, gm, lbcs, ubcs) where T = begin
        # TODO: check whether this is the correct way of defining the filter width
        Δ1 = gm.hsize1 / (2*gd.nx_fd)
        Δ2 = gm.hsize2 / (gd.ny_fd+1)
        Δ3c = gm.Dvmap.(vrange(T, gd, NodeSet(:H))) / gd.nz_global
        Δ3i = gm.Dvmap.(vrange(T, gd, NodeSet(:V))) / gd.nz_global
        Lc = sgs_model.model_constant * cbrt.(Δ1 * Δ2 * Δ3c)
        Li = sgs_model.model_constant * cbrt.(Δ1 * Δ2 * Δ3i)

        if sgs_model.wall_damping
            Lc_wall = broadcast!(mixing_length, similar(Lc), x3(gd, gm, NodeSet(:H)), gm.vmap(zero(T)), gm.vmap(one(T)), (lbcs,), (ubcs,))
            Li_wall = broadcast!(mixing_length, similar(Li), x3(gd, gm, NodeSet(:V)), gm.vmap(zero(T)), gm.vmap(one(T)), (lbcs,), (ubcs,))
            N = sgs_model.wall_damping_exponent
            @. Lc = (Lc^(-N) + Lc_wall^(-N))^(-1/N)
            @. Li = (Li^(-N) + Li_wall^(-N))^(-1/N)
        end

        new{T}(reshape(Lc, (1,1,gd.nz_h)), reshape(Li, (1,1,gd.nz_v)))
    end
end

struct FilteredAdvectionBuffers{T,P}

    vel::Tuple{Array{T,3},Array{T,3},Array{T,3}}

    vel_dx1::Tuple{Array{T,3},Array{T,3},Array{T,3}}
    vel_dx2::Tuple{Array{T,3},Array{T,3},Array{T,3}}
    vel_dx3::Tuple{Array{T,3},Array{T,3},Array{T,3}}

    vorticity::Tuple{Array{T,3},Array{T,3},Array{T,3}}
    vorticity_bcs::NTuple{2,NTuple{2,UnspecifiedBC{P,T}}}

    strain_rate::NTuple{3,NTuple{3,Array{T,3}}}
    strain_bcs::NTuple{2,UnspecifiedBC{P,T}}

    sgs_model::StaticSmagorinskyBuffers{T}
    eddy_viscosity_h::Array{T,3}
    eddy_viscosity_v::Array{T,3}
    sgs::NTuple{3,NTuple{3,Array{T,3}}}
    sgs_fd::Tuple{NTuple{3,Array{Complex{T},3}},
                  NTuple{3,Array{Complex{T},3}},
                  Tuple{Array{Complex{T},3},Array{Complex{T},3},Nothing}}
    sgs_bcs::NTuple{5,UnspecifiedBC{P,T}}

    adv::Tuple{Array{T,3},Array{T,3},Array{T,3}}

    grid_spacing::Tuple{T,T,Array{T,1}}

    function FilteredAdvectionBuffers(gd::DistributedGrid, gm::GridMapping{T}, lower_bcs, upper_bcs, sgs_model) where T

        # velocity and gradients
        vel = zeros_pd.(T, gd, staggered_nodes())
        vel_dx1 = zeros_pd.(T, gd, staggered_nodes())
        vel_dx2 = zeros_pd.(T, gd, staggered_nodes())
        vel_dx3 = zeros_pd.(T, gd, inverted_nodes())

        # vorticity
        vorticity = zeros_pd.(T, gd, inverted_nodes())
        vorticity_bcs = ((UnspecifiedBC(T, gd), UnspecifiedBC(T, gd)),
                         (UnspecifiedBC(T, gd), UnspecifiedBC(T, gd)))

        # strain rate
        S13, S23, S12 = zeros_pd.(T, gd, inverted_nodes())
        Sij = ((vel_dx1[1], S12, S13), (S12, vel_dx2[2], S23), (S13, S23, vel_dx3[3]))
        S_bcs = (UnspecifiedBC(T, gd), UnspecifiedBC(T, gd))

        eddy_viscosity_h = zeros_pd(T, gd, :H)
        eddy_viscosity_v = zeros_pd(T, gd, :V)

        τ11, τ22, τ33 = (zeros_pd(T, gd, :H), zeros_pd(T, gd, :H), zeros_pd(T, gd, :H))
        τ13, τ23, τ12 = (zeros_pd(T, gd, :V), zeros_pd(T, gd, :V), zeros_pd(T, gd, :H))
        sgs = ((τ11, τ12, τ13), (τ12, τ22, τ23), (τ13, τ23, τ33))

        τ11_fd, τ12_fd, τ13_fd = zeros_fd(T, gd, :H), zeros_fd(T, gd, :H), zeros_fd(T, gd, :V)
        τ22_fd , τ23_fd = zeros_fd(T, gd, :H), zeros_fd(T, gd, :V)
        sgs_fd = ((τ11_fd, τ12_fd, τ13_fd), (τ12_fd, τ22_fd, τ23_fd), (τ13_fd, τ23_fd, nothing))
        sgs_bcs = (UnspecifiedBC(T, gd), UnspecifiedBC(T, gd), UnspecifiedBC(T, gd), UnspecifiedBC(T, gd), UnspecifiedBC(T, gd))

        adv = zeros_pd.(T, gd, staggered_nodes())

        grid_spacing = local_grid_spacing(gd, gm)

        new{T,proc_type()}(vel, vel_dx1, vel_dx2, vel_dx3, vorticity, vorticity_bcs, Sij, S_bcs,
                StaticSmagorinskyBuffers(sgs_model, gd, gm, lower_bcs, upper_bcs),
                eddy_viscosity_h, eddy_viscosity_v, sgs, sgs_fd, sgs_bcs, adv, grid_spacing)
    end
end

RoughWallEquilibriumModel(; z0 = 1e-3, kappa = 0.4) = RoughWallEquilibriumModel(z0, kappa)

function set_advection!(adv, vel, df::DerivativeFactors{T}, ht::HorizontalTransform,
        lower_bcs::NTuple{3,BoundaryCondition{P,T}}, upper_bcs::NTuple{3,BoundaryCondition{P,T}},
        b::FilteredAdvectionBuffers{T, P}) where {P, T}

    # Compute ui, dui/dx1, dui/dx2 and transform to PD (9 terms).
    get_field!.(b.vel, ht, vel, staggered_nodes())
    get_field_dx1!.(b.vel_dx1, vel, ht, df, staggered_nodes())
    get_field_dx2!.(b.vel_dx2, vel, ht, df, staggered_nodes())

    # Compute vertical velocity gradients in physical domain
    add_derivative_x3!.(reset!.(b.vel_dx3), b.vel, lower_bcs, upper_bcs, df, staggered_nodes())

    # Compute vorticity in physical domain
    b.vorticity[1] .= b.vel_dx2[3] .- b.vel_dx3[2]
    b.vorticity[2] .= b.vel_dx3[1] .- b.vel_dx1[3]
    b.vorticity[3] .= b.vel_dx1[2] .- b.vel_dx2[1]

    # Compute advection term in physical domain.
    ω1exp = layers_expand_i_to_c(b.vorticity[1], b.vorticity_bcs[1]...) # I to I below & above C-nodes
    ω2exp = layers_expand_i_to_c(b.vorticity[2], b.vorticity_bcs[2]...) # I to I below & above C-nodes
    uexp = layers_expand_half.(b.vel, lower_bcs, upper_bcs, staggered_nodes())
    advu!.(layers(b.adv[1]), layers(b.vel[2]), ω2exp[1:end-1], ω2exp[2:end],
                             uexp[3][1:end-1], uexp[3][2:end], layers(b.vorticity[3]))
    advv!.(layers(b.adv[2]), layers(b.vel[1]), ω1exp[1:end-1], ω1exp[2:end],
                             uexp[3][1:end-1], uexp[3][2:end], layers(b.vorticity[3]))
    advw!.(layers(b.adv[3]), uexp[1][1:end-1], uexp[1][2:end], layers(b.vorticity[1]),
                             uexp[2][1:end-1], uexp[2][2:end], layers(b.vorticity[2]))

    # Compute Sij on its natural nodes in PD (Skk are already set to gradients)
    broadcast!((a,b) -> (a+b)/2, b.strain_rate[2][3], b.vel_dx2[3], b.vel_dx3[2])
    broadcast!((a,b) -> (a+b)/2, b.strain_rate[3][1], b.vel_dx3[1], b.vel_dx1[3])
    broadcast!((a,b) -> (a+b)/2, b.strain_rate[1][2], b.vel_dx1[2], b.vel_dx2[1])

    # Compute eddy viscosity on both sets of nodes
    νT(SijSij, L) = L^2 * sqrt(2*SijSij)
    SijSij!((b.eddy_viscosity_h, b.eddy_viscosity_v), b.strain_rate,
            b.vel, lower_bcs, upper_bcs, b.strain_bcs, df.DD3_h)
    broadcast!(νT, b.eddy_viscosity_h, b.eddy_viscosity_h, b.sgs_model.length_scale_h)
    broadcast!(νT, b.eddy_viscosity_v, b.eddy_viscosity_v, b.sgs_model.length_scale_v)

    # Compute τij in PD.
    @. b.sgs[1][1] = 2 * b.eddy_viscosity_h * b.strain_rate[1][1]
    @. b.sgs[2][2] = 2 * b.eddy_viscosity_h * b.strain_rate[2][2]
    @. b.sgs[3][3] = 2 * b.eddy_viscosity_h * b.strain_rate[3][3]
    @. b.sgs[1][2] = 2 * b.eddy_viscosity_h * b.strain_rate[1][2]
    @. b.sgs[3][1] = 2 * b.eddy_viscosity_v * b.strain_rate[3][1]
    @. b.sgs[2][3] = 2 * b.eddy_viscosity_v * b.strain_rate[2][3]

    # Add values of dτi3/dx3 to the resolved part of the non-linear term.
    τ13 = layers_expand_i_to_c(b.sgs[1][3], b.sgs_bcs[1], b.sgs_bcs[2])
    τ13 = apply_wall_model(τ13, lower_bcs[1], b.vel[1], upper_bcs[1], b.vel)
    for i=1:equivalently(size(b.adv[1], 3), length(τ13)-1)
        add_derivative!(view(b.adv[1], :, :, i), τ13[i:i+1], df.D3_h[i]) # τ13 can be from wall model
    end
    τ23 = layers_expand_i_to_c(b.sgs[2][3], b.sgs_bcs[3], b.sgs_bcs[4])
    τ23 = apply_wall_model(τ23, lower_bcs[2], b.vel[2], upper_bcs[2], b.vel)
    for i=1:equivalently(size(b.adv[2], 3), length(τ23)-1)
        add_derivative!(view(b.adv[2], :, :, i), τ23[i:i+1], df.D3_h[i]) # τ23 can be from wall model
    end
    τ33 = layers_expand_c_to_i(b.sgs[3][3], b.sgs_bcs[5])
    for i=1:equivalently(size(b.adv[3], 3), length(τ33)-1)
        add_derivative!(view(b.adv[3], :, :, i), τ33[i:i+1], df.D3_v[i])
    end

    # Transform these terms back to FD (8 terms)
    # NOTE: τ33 is not needed in FD, and τ21 == τ12
    set_field!.(adv, ht, b.adv, staggered_nodes())
    set_field!(b.sgs_fd[1][1], ht, b.sgs[1][1], NodeSet(:H))
    set_field!(b.sgs_fd[1][2], ht, b.sgs[1][2], NodeSet(:H))
    set_field!(b.sgs_fd[2][2], ht, b.sgs[2][2], NodeSet(:H))
    set_field!(b.sgs_fd[1][3], ht, b.sgs[1][3], NodeSet(:V))
    set_field!(b.sgs_fd[2][3], ht, b.sgs[2][3], NodeSet(:V))

    # Add horizontal contributions of stress divergence to RHS
    @. adv[1] += b.sgs_fd[1][1] * df.D1 + b.sgs_fd[1][2] * df.D2
    @. adv[2] += b.sgs_fd[2][1] * df.D1 + b.sgs_fd[2][2] * df.D2
    @. adv[3] += b.sgs_fd[3][1] * df.D1 + b.sgs_fd[3][2] * df.D2

    # TODO: check computation of time step restriction
    dt_adv = advective_timescale(layers.(b.vel), b.grid_spacing)

    return adv, dt_adv
end

function fix_bcs(Si3::Tuple, lbc, vel, ubc)
    first = Si3[1] isa UnspecifiedBC ? Si3_boundary(Si3[1], Si3[2], lbc, view(vel,:,:,1)) : Si3[1]
    last = Si3[end] isa UnspecifiedBC ? Si3_boundary(Si3[end], Si3[end-1], ubc, view(vel,:,:,size(vel,3)), -1) : Si3[end]
    first, Si3[2:end-1]..., last
end

function Si3_boundary(Si3_bc::UnspecifiedBC, Si3_I1, vel_bc::DirichletBC, vel_C1, sign)
    # one-sided differences for the second-order derivatives du1/dx3 & du2/dx3 at the wall
    # (du/dx)(x=0) = (− u(1.5Δ) + 9 u(0.5Δ) − 8 u(0)) / (3 Δ) + O(Δ²)
    # TODO (a bit complicated because we might need to send the second layer from the second process)
    error("Not implemented")
end

function Si3_boundary(Si3_bc::UnspecifiedBC, Si3_i1, vel_bc::RoughWallEquilibriumBC, vel_c1, sign = 1) # D3 is α/Δx3
    # based on the log law, the derivative at the first node is u / (z log(z/z₀))
    # we can set the value here such that (Si3_boundary + Si3[1])/2 == Si3[1/2]
    # i.e. we have to set it to 2*Si3[1/2] - Si3[1]
    # where Si3[1/2] == 1/2 * u / (z log(z/z₀))
    # so Si3 = u / (z log(z/z₀)) - Si3[1]
    z, z₀ = vel_bc.wall_distance, vel_bc.roughness_length
    @. Si3_bc.buffer_pd = sign * vel_c1 / (z * log(z/z₀)) - Si3_i1
    Si3_bc.buffer_pd
end

function Si3_boundary(_::UnspecifiedBC, _, vel_bc::NeumannBC, _, _)
    # we can return a scalar instead of a whole layer and the broadcasts will still work
    # note: the sign is ignored because the gradient of the BC always has the right direction
    vel_bc.gradient / 2
end

add_interpolated_sq!(out, below, above, count = 1) = @. out += count * (below + above)^2 / 4

function SijSij!((SijSij_h, SijSij_v), S, vel, lbcs, ubcs, temp_bcs, DD3_h)

    # in the following, we assume that du3/dx1 and du3/dx2 vanish at the walls
    @assert lbcs[3] isa DirichletBC && ubcs[3] isa DirichletBC "Horizontal gradients of vertical velocity do not vanish at the walls."

    # initialize output with contributions of (Sij Sij) that are on the correct nodes
    @. SijSij_h = S[1][1]^2 + S[2][2]^2 + S[3][3]^2 + 2 * S[1][2]^2
    @. SijSij_v = 2 * S[3][1]^2 + 2 * S[2][3]^2

    # add contributions of (Sij Sij) that have to be interpolated from H-nodes to V-nodes
    # NOTE: it is always more accurate to interpolate first and square after
    lv = layers(SijSij_v)
    S11 = layers_expand_c_to_i(S[1][1], temp_bcs[1])
    @. add_interpolated_sq!.(lv, S11[1:end-1], S11[2:end])
    S22 = layers_expand_c_to_i(S[2][2], temp_bcs[1])
    @. add_interpolated_sq!.(lv, S22[1:end-1], S22[2:end])
    S33 = layers_expand_c_to_i(S[3][3], temp_bcs[1])
    @. add_interpolated_sq!.(lv, S33[1:end-1], S33[2:end])
    S12 = layers_expand_c_to_i(S[1][2], temp_bcs[1])
    @. add_interpolated_sq!.(lv, S12[1:end-1], S12[2:end], 2)

    # add contributions of (Sij Sij) that have to be interpolated from V-nodes to H-nodes
    # WARNING: the buffers of the temp_bcs are used temporarily when calling fix_bcs,
    #          so be very careful when changing the order of these commands!
    lh = layers(SijSij_h)
    begin # top/bottom layers of S13 might be buffers of temp_bcs in this section
        S13 = layers_expand_i_to_c(S[1][3], temp_bcs[1:2]...)
        S13 = fix_bcs(S13, lbcs[1], vel[1], ubcs[1])
        @. add_interpolated_sq!.(lh, S13[1:end-1], S13[2:end], 2)
    end
    begin # top/bottom layers of S23 might be buffers of temp_bcs in this section
        S23 = layers_expand_i_to_c(S[2][3], temp_bcs[1:2]...)
        S23 = fix_bcs(S23, lbcs[2], vel[2], ubcs[2])
        @. add_interpolated_sq!.(lh, S23[1:end-1], S23[2:end], 2)
    end

    SijSij_h, SijSij_v
end

function wall_model(τi3, vel_bc::RoughWallEquilibriumBC, vel_h1, u1, u2, prefactor = 1)
    @. τi3.buffer_pd = vel_bc.von_karman_constant^2 / log(vel_bc.wall_distance/vel_bc.roughness_length)^2 *
            prefactor * vel_h1 * sqrt(u1^2 + u2^2)
    τi3.buffer_pd
end

function wall_model(τi3, vel_bc::NeumannBC, args...)
    @assert vel_bc.gradient == 0 "Wall model for non-zero Neumann BC not implemented" vel_bc.value
    vel_bc.gradient
end

function apply_wall_model(τi3::Tuple, lbc, vel, ubc, vels)
    first = τi3[1] isa UnspecifiedBC ? wall_model(τi3[1], lbc, first_layer(vel),
            first_layer(vels[1]), first_layer(vels[2])) : τi3[1]
    # TODO: handle case where u3 has no local layer (i.e. last_layer(vels[3]) fails)
    last = τi3[end] isa UnspecifiedBC ? wall_model(τi3[end], ubc, last_layer(vel),
            last_layer(vels[1]), last_layer(vels[2]), -1) : τi3[end]
    first, τi3[2:end-1]..., last
end



