export MomentumAdvection

struct MomentumAdvection <: ProcessDefinition
    dealiasing
    MomentumAdvection(dealiasing = :quadratic) = new(dealiasing)
end

struct DiscretizedMomentumAdvection <: DiscretizedProcess
    dims::Tuple{Int,Int}
    boundary_conditions
end

function init_process(adv::MomentumAdvection, domain::Domain{T}, grid) where T
    # TODO: use internal BCs for vel1 & vel2
    dims = pdsize(grid, adv.dealiasing)
    bcs = Tuple(init_bcs(vel, domain, grid, dims) for vel = (:vel1, :vel2, :vel3))
    DiscretizedMomentumAdvection(dims, bcs)
end

state_fields(::DiscretizedMomentumAdvection) = (:vel1, :vel2, :vel3)
physical_domain_terms(adv::DiscretizedMomentumAdvection) =
    Tuple(f => adv.dims for f in (:vel1 , :vel2, :vel3, :vort1, :vort2, :vort3))
physical_domain_rates(adv::DiscretizedMomentumAdvection) =
    Tuple(f => adv.dims for f in (:vel1 , :vel2, :vel3))

function add_rate!(rates, term::DiscretizedMomentumAdvection, state, t, log)

    # unpack required fields
    state, rates = state[term.dims], rates[term.dims]
    vel = layers.((state[:vel1], state[:vel2], state[:vel3]))
    vort = layers.((state[:vort1], state[:vort2], state[:vort3]))
    rate = layers.((rates[:vel1], rates[:vel2], rates[:vel3]))
    lbc = last.(term.boundary_conditions)
    ubc = first.(term.boundary_conditions)

    # note: boundary conditions do not match velocity indices on purpose,
    # since the velocity BCs are used to derive boundary values for
    # velocity–vorticity products
    add_adv1!(rate[1], vel[2], vort[2], vel[3], vort[3], lbc[1], ubc[1], lbc[3], ubc[3])
    add_adv2!(rate[2], vel[1], vort[1], vel[3], vort[3], lbc[2], ubc[2], lbc[3], ubc[3])
    add_adv3!(rate[3], vel[1], vort[1], vel[2], vort[2], ubc[1], ubc[2])

    # compute smallest time scale for advection term (for CFL condition)
    #dt_adv = advective_timescale((upd, vpd, wpd), b.grid_spacing)

    rates
end

function add_adv1!(rate1::NTuple{I3C},
        vel2::NTuple{I3C}, vort2::NTuple{I3I},
        vel3::NTuple{I3I}, vort3::NTuple{I3C},
        lbc1, ubc1, lbc3, ubc3) where {I3C,I3I}

    vort2_below = layer_below(vort2, lbc1)
    vel3_below = layer_below(vel3, lbc3)

    # TODO: check if this can be simplified with the specialized layers functions
    # inner layers, same for all processes (possibly empty)
    for i = 1:I3C-1
        add_adv1!(rate1[i], vel2[i],
                i>1 ? vort2[i-1] : vort2_below, vort2[i],
                i>1 ?  vel3[i-1] :  vel3_below,  vel3[i],
                vort3[i])
    end

    # last layer, can be different
    add_adv1!(rate1[I3C], vel2[I3C],
            I3C > 1    ? vort2[I3C-1] : vort2_below, # vort2¯
            I3C == I3I ? vort2[I3C]   : nothing,     # vort2⁺
            I3C > 1    ?  vel3[I3C-1] : vel3_below,  # vel3¯
            I3C == I3I ?  vel3[I3C]   : ubc3.type,   # vel3⁺
            vort3[I3C])
end

function add_adv2!(rate2::NTuple{I3C},
        vel1::NTuple{I3C}, vort1::NTuple{I3I},
        vel3::NTuple{I3I}, vort3::NTuple{I3C},
        lbc2, ubc2, lbc3, ubc3) where {I3C,I3I}

    vort1_below = layer_below(vort1, lbc2)
    vel3_below  = layer_below(vel3, lbc3)

    # inner layers, same for all processes (possibly empty)
    for i = 1:I3C-1
        add_adv2!(rate2[i], vel1[i],
                i>1 ? vort1[i-1] : vort1_below, vort1[i],
                i>1 ?  vel3[i-1] :  vel3_below,  vel3[i],
                vort3[i])
    end

    # last layer, can be different
    add_adv2!(rate2[I3C], vel1[I3C],
            I3C > 1    ? vort1[I3C-1] : vort1_below, # vort1¯
            I3C == I3I ? vort1[I3C]   : nothing,     # vort1⁺
            I3C > 1    ?  vel3[I3C-1] : vel3_below,  # vel3¯
            I3C == I3I ?  vel3[I3C]   : ubc3.type,   # vel3⁺
            vort3[I3C])
end

function add_adv3!(rate3::NTuple{I3I},
        vel1::NTuple{I3C}, vort1::NTuple{I3I},
        vel2::NTuple{I3C}, vort2::NTuple{I3I},
        ubc1, ubc2) where {I3C,I3I}

    u_above = layer_above(vel1, ubc1)
    v_above = layer_above(vel2, ubc2)

    # inner layers, same for all processes (possibly empty)
    for i = 1:I3C-1
        add_adv3!(rate3[i], vel1[i], vel1[i+1], vort1[i],
                              vel2[i], vel2[i+1], vort2[i])
    end

    # last layer, does not exist on last node
    I3I == I3C && add_adv3!(rate3[I3I], vel1[I3C], u_above, vort1[I3I],
                                          vel2[I3C], v_above, vort2[I3I])
end

# compute one layer of -(roty[w]*w[w]-rotz[uvp]*v[uvp]) on uvp-nodes
add_adv1!(advu, v, rotv¯, rotv⁺, w¯, w⁺, rotw) =
        @. advu += rotw * v - 0.5 * (rotv¯ * w¯ + rotv⁺ * w⁺)
add_adv1!(advu, v, rotv¯, rotv⁺, lbcw::ConstantValue, w⁺, rotw) = begin
    lbcw.value == 0 || error("Advection for non-zero w at boundary not implemented")
    @. advu += rotw * v - 0.5 * (rotv⁺ * w⁺)
end
add_adv1!(advu, v, rotv¯, rotv⁺, w¯, ubcw::ConstantValue, rotw) = begin
    ubcw.value == 0 || error("Advection for non-zero w at boundary not implemented")
    @. advu += rotw * v - 0.5 * (rotv¯ * w¯)
end
add_adv1!(_, _, _, _, ::ConstantValue, ::ConstantValue, _) =
    error("Simulations with only one vertical layer are not supported")

# compute one layer of -(rotz[uvp]*u[uvp]-rotx[w]*w[w]) on uvp-nodes
add_adv2!(advv, u, rotu¯, rotu⁺, w¯, w⁺, rotw) =
        @. advv += 0.5 * (rotu¯ * w¯ + rotu⁺ * w⁺) - rotw * u
add_adv2!(advv, u, rotu¯, rotu⁺, lbcw::ConstantValue, w⁺, rotw) = begin
    lbcw.value == 0 || error("Advection for non-zero w at boundary not implemented")
    @. advv += 0.5 * (rotu⁺ * w⁺) - rotw * u
end
add_adv2!(advv, u, rotu¯, rotu⁺, w¯, ubcw::ConstantValue, rotw) = begin
    ubcw.value == 0 || error("Advection for non-zero w at boundary not implemented")
    @. advv += 0.5 * (rotu¯ * w¯) - rotw * u
end
add_adv2!(_, _, _, _, ::ConstantValue, ::ConstantValue, _) =
    error("Simulations with only one vertical layer are not supported")

# compute one layer of -(rotx[w]*v[uvp]-roty[w]*u[uvp]) on w-nodes
add_adv3!(advw, u¯, u⁺, rotu, v¯, v⁺, rotv) =
        @. advw += rotv * 0.5 * (u¯ + u⁺) - rotu * 0.5 * (v¯ + v⁺)
