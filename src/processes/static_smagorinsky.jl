export StaticSmagorinskyModel

"""
    StaticSmagorinskyModel(; kwargs...)

Subgrid-scale advective transport of resolved momentum, approximated with the static Smagorinsky model.

# Keywords

- `Cs = 0.1`: The Smagorinsky coefficient ``C_\\mathrm{S}``.
- `dealiasing`: The level of dealiasing, determining the physical-domain resolution at nonlinear operations are performed. The default `nothing` gives a physical-domain resolution that matches the frequency-domain resolution (rounded up to an even value). Alternatively, `dealiasing` can be set to `:quadratic` for a resolution based on the “3/2-rule” or to a tuple of two integers to set the resolution manually. Aliasing errors should be reduced for larger resolutions but they can be expected at any resolution.
- `wall_damping = true`: Adjust the Smagorinsky lenghtscale towards the expected value of ``κ x₃`` towards the wall.
- `wall_damping_exponent = 2`: Exponent of the near-wall adjustment.
"""
struct StaticSmagorinskyModel <: ProcessDefinition
    dealiasing
    model_constant
    wall_damping
    wall_damping_exponent
    StaticSmagorinskyModel(; dealiasing = nothing, Cs = 0.1,
                           wall_damping = true, wall_damping_exponent = 2) =
        new(dealiasing, Cs, wall_damping, wall_damping_exponent)
end

struct DiscretizedStaticSmagorinskyModel{T,B,D} <: DiscretizedProcess
    dims::Tuple{Int,Int}
    eddyviscosity_c::Array{T,3}
    eddyviscosity_i::Array{T,3}
    buffer_c::Array{T,3}
    buffer_i::Array{T,3}
    lengthscale_c::Array{T,3}
    lengthscale_i::Array{T,3}
    boundary_conditions::B
    derivatives::D
end

Base.nameof(::DiscretizedStaticSmagorinskyModel) = "Static Smagorinsky SGS Model"

state_fields(::DiscretizedStaticSmagorinskyModel) = (:vel1, :vel2, :vel3)
physical_domain_terms(sgs::DiscretizedStaticSmagorinskyModel) =
    Tuple(f => sgs.dims for f in (:vel1_1, :vel2_2, :vel3_3, :strain12, :strain13, :strain23, :vel1, :vel2)) # vel1 & vel2 are needed for wall model
physical_domain_rates(sgs::DiscretizedStaticSmagorinskyModel) =
    Tuple(f => sgs.dims for f in (:vel1, :vel1_1, :vel1_2, :vel2, :vel2_1, :vel2_2, :vel3, :vel3_1, :vel3_2))

function init_process(sgs::StaticSmagorinskyModel, domain::Domain{T}, grid) where T
    dims = pdsize(grid, sgs.dealiasing)
    # TODO: use internal BCs where possible
    bcs = Tuple(init_bcs(vel, domain, grid, dims) for vel = (:vel1, :vel2, :vel3))

    # buffers for eddy viscosity
    evc = zeros(T, dims..., grid.n3c)
    evi = zeros(T, dims..., grid.n3i)

    # initialize prefactor: length scale with coefficient
    Δ1 = size(domain, 1) / (2*grid.k1max+2)
    Δ2 = size(domain, 2) / (2*grid.k2max+2)
    Δ3c = domain.Dvmap.(vrange(grid, NodeSet(:C))) / grid.n3global
    Δ3i = domain.Dvmap.(vrange(grid, NodeSet(:I))) / grid.n3global
    lc = convert.(T, sgs.model_constant * cbrt.(Δ1 * Δ2 * Δ3c))
    li = convert.(T, sgs.model_constant * cbrt.(Δ1 * Δ2 * Δ3i))

    # apply wall damping to length scale (Mason & Thomson, 1992)
    if sgs.wall_damping
        x3c = x3range(domain, vrange(grid, NodeSet(:C)))
        x3i = x3range(domain, vrange(grid, NodeSet(:I)))
        x3min, x3max = x3range(domain, (0, 1))
        lc_wall = broadcast!(mixing_length, similar(lc), x3c, x3min, x3max,
                             (domain.lower_boundary,), (domain.upper_boundary,))
        li_wall = broadcast!(mixing_length, similar(li), x3i, x3min, x3max,
                             (domain.lower_boundary,), (domain.upper_boundary,))
        N = sgs.wall_damping_exponent
        @. lc = (lc^(-N) + lc_wall^(-N))^(-1/N)
        @. li = (li^(-N) + li_wall^(-N))^(-1/N)
    end

    # initialize boundary conditions and other values required to apply the
    # wall model for strain (S13, S23) and stress (τ13, τ23) components that
    # need to be defined at the wall
    init_wall(boundary::RoughWall, dx3) = (BoundaryCondition(T, :dynamic => dims, grid, dims),
                                           1 / (2 * dx3 * log(dx3/boundary.roughness)),
                                           - boundary.von_karman_constant^2 / log(dx3/boundary.roughness)^2)
    init_wall(::FreeSlipBoundary, _) = (BoundaryCondition(T, :dirichlet => 0, grid, dims), 0, 0)
    init_wall(opts...) = error("SGS model only supports rough-wall and free-slip boundaries")
    dx3_below = first(x3range(domain, vrange(grid, NodeSet(:C))[1])) - extrema(domain, 3)[1]
    dx3_above = extrema(domain, 3)[end] - first(x3range(domain, vrange(grid, NodeSet(:C))[end]))
    lw = init_wall(domain.lower_boundary, dx3_below)
    uw = init_wall(domain.upper_boundary, dx3_above)
    bcs = (internal = internal_bc(domain, grid, dims),
           wallstress = (lw[1], uw[1]),
           factors_strain = convert.(T, (lw[2], -uw[2])),
           factors_stress = convert.(T, (lw[3], -uw[3])))

    # vertical derivatives of SGS stresses are computed in physical space
    derivatives = (D3c = dx3factors(domain, grid, NodeSet(:C)),
                   D3i = dx3factors(domain, grid, NodeSet(:I)))
    DiscretizedStaticSmagorinskyModel(dims, evc, evi, similar(evc), similar(evi),
                                      reshape(lc, 1, 1, :), reshape(li, 1, 1, :),
                                      bcs, derivatives)
end

mixing_length(x3, ::FreeSlipBoundary) = Inf
mixing_length(x3, wall::RoughWall) = x3 * wall.von_karman_constant
mixing_length(x3, x3min, x3max, lbc, ubc) =
    min(mixing_length(x3-x3min, lbc), mixing_length(x3max-x3, ubc))

function add_rates!(rates, term::DiscretizedStaticSmagorinskyModel, state, t, log)

    # unpack required fields
    state, rates = state[term.dims], rates[term.dims]
    evc, evi, bcs = term.eddyviscosity_c, term.eddyviscosity_i, term.boundary_conditions
    sgsc, sgsi = term.buffer_c, term.buffer_i

    # compute eddy viscosity on both sets of nodes
    strain = (state[:vel1_1], state[:strain12], state[:strain13],
              state[:vel2_2], state[:strain23], state[:vel3_3])
    total_strain!((evc, evi), strain, (state[:vel1], state[:vel2]), bcs)
    evc .*= term.lengthscale_c.^2
    evi .*= term.lengthscale_i.^2
    #νT(strain, L) = L^2 * strain
    #broadcast!(νT, term.eddyviscosity_c, term.eddyviscosity_c, term.lengthscale_c)
    #broadcast!(νT, term.eddyviscosity_i, term.eddyviscosity_i, term.lengthscale_i)

    # τ11 and τ22 can be computed directly on C-nodes
    @. sgsc = - 2 * evc * strain[1]
    log_sample!(log, :sgs11 => sgsc, t)
    rates[:vel1_1] .-= sgsc
    @. sgsc = - 2 * evc * strain[4]
    log_sample!(log, :sgs22 => sgsc, t)
    rates[:vel2_2] .-= sgsc

    # τ33 is computed on C-nodes, and we can directly add dτ33/dx3 to the rates
    # TODO: consider writing a d/dx3 function that adds instead of overwriting
    @. sgsc = - 2 * evc * strain[6]
    log_sample!(log, :sgs33 => sgsc, t)
    dx3_c2i!(sgsi, sgsc, bcs.internal, term.derivatives.D3i)
    rates[:vel3] .-= sgsi

    # τ12==τ21 is required for both horizontal velocity components
    # NOTE: this could be transformed just once to save one FFT, but having
    # rates that are defined in a general way that can be used for many
    # different RHS terms is probably worth more than saving this one FFT
    # TODO: consider checking if vel1_2 == vel2_1 when doing the transforms
    @. sgsc = - 2 * evc * strain[2]
    log_sample!(log, :sgs12 => sgsc, t)
    rates[:vel1_2] .-= sgsc
    rates[:vel2_1] .-= sgsc

    vel1_walls, vel2_walls = layers(state[:vel1])[[1,end]], layers(state[:vel2])[[1,end]]

    # τ13 is computed on I-nodes and needs the wall-model for the boundaries
    @. sgsi = - 2 * evi * strain[3]
    rates[:vel3_1] .-= sgsi
    bcs13 = stress_bc.(bcs.wallstress, vel1_walls, vel2_walls, bcs.factors_stress, 1)
    log_sample!(log, :sgs13 => sgsi, t, bcs = bcs13)
    dx3_i2c!(sgsc, sgsi, bcs13, term.derivatives.D3c)
    rates[:vel1] .-= sgsc

    # τ23 is computed on I-nodes and needs the wall-model for the boundaries
    @. sgsi = - 2 * evi * strain[5]
    rates[:vel3_2] .-= sgsi
    bcs23 = stress_bc.(bcs.wallstress, vel1_walls, vel2_walls, bcs.factors_stress, 2)
    log_sample!(log, :sgs23 => sgsi, t, bcs = bcs23)
    dx3_i2c!(sgsc, sgsi, bcs23, term.derivatives.D3c)
    rates[:vel2] .-= sgsc

    # TODO: check computation of time step restriction
    #dt_adv = advective_timescale(layers.(b.vel), b.grid_spacing)

    rates
end

# τi3 at first C-node is κ² / (log z/z₀)² ui √(u₁²+u₂²)
stress_bc(bc::BoundaryCondition{T}, vel1, vel2, factor, direction) where T <: DynamicValues = begin
    vel = direction == 1 ? vel1 : direction == 2 ? vel2 : error("Invalid direction `$direction`")
    @. bc.type.values = factor * vel * sqrt(vel1^2 + vel2^2)
    bc
end
stress_bc(bc::BoundaryCondition{T}, _, _, _, _) where T <: ConstantValue = bc


function total_strain!((strain_c, strain_i), strain_ij, vel, bcs)

    # unpack tuple with components of strain tensor Sij
    strain11, strain12, strain13, strain22, strain23, strain33 = strain_ij

    # initialize output with contributions of (Sij Sij) that are on the correct nodes
    @. strain_c = strain11^2 + strain22^2 + strain33^2 + 2 * strain12^2
    @. strain_i = 2 * strain13^2 + 2 * strain23^2

    # add contributions of (Sij Sij) that have to be interpolated from C-nodes to I-nodes
    li = layers(strain_i)
    S11 = layers_c2i(strain11, bcs.internal)
    @. add_interpolated_sq!.(li, S11[1:end-1], S11[2:end])
    S22 = layers_c2i(strain22, bcs.internal)
    @. add_interpolated_sq!.(li, S22[1:end-1], S22[2:end])
    S33 = layers_c2i(strain33, bcs.internal)
    @. add_interpolated_sq!.(li, S33[1:end-1], S33[2:end])
    S12 = layers_c2i(strain12, bcs.internal)
    @. add_interpolated_sq!.(li, S12[1:end-1], S12[2:end], 2)

    # add contributions of (Sij Sij) that have to be interpolated from I-nodes to C-nodes
    # WARNING: the buffers of the dynamic BCs are used temporarily when calling strain_bcs,
    #          so be very careful when changing the order of these commands!
    lh = layers(strain_c)
    S13 = layers_i2c(strain13, strain_bc.(bcs.wallstress, layers(vel[1])[[1,end]], bcs.factors_strain)...)
    @. add_interpolated_sq!.(lh, S13[1:end-1], S13[2:end], 2)
    S23 = layers_i2c(strain23, strain_bc.(bcs.wallstress, layers(vel[2])[[1,end]], bcs.factors_strain)...)
    @. add_interpolated_sq!.(lh, S23[1:end-1], S23[2:end], 2)

    # S = √(2 Sij Sij)
    map!(x->sqrt(2*x), strain_c, strain_c), map!(x->sqrt(2*x), strain_i, strain_i)
end

# Si3 at first C-node is ½ ui / (z log z/z₀), assuming du3/dxi is negligible
strain_bc(bc::BoundaryCondition{T}, vel, factor) where T <: DynamicValues =
    (@. bc.type.values = vel * factor; bc)
strain_bc(bc::BoundaryCondition{T}, _, _) where T <: ConstantValue = bc

# NOTE: it is always more accurate to interpolate first and square after
add_interpolated_sq!(out, below, above, count = 1) = @. out += count * (below + above)^2 / 4
add_interpolated_sq!(out, below::ConstantValue, above, count = 1) = @. out += count * (below.value + above)^2 / 4
add_interpolated_sq!(out, below, above::ConstantValue, count = 1) = @. out += count * (below + above.value)^2 / 4

# NOTE: these functions assume that the precomputed dynamic values are already
# interpolated to the target-nodes, and are not values at the wall itself, i.e.
# the function does not interpolate in this case and discards the non-boundary
# values
add_interpolated_sq!(out, below::DynamicValues, above, count = 1) = @. out += count * (below.values)^2
add_interpolated_sq!(out, below, above::DynamicValues, count = 1) = @. out += count * (above.values)^2
