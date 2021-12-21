# compute one layer of -(roty[w]*w[w]-rotz[uvp]*v[uvp]) on uvp-nodes
@inline advu!(advu, v, rotv¯, rotv⁺, w¯, w⁺, rotw) =
        @. advu = rotw * v - 0.5 * (rotv¯ * w¯ + rotv⁺ * w⁺)
@inline advu!(advu, v, rotv¯, rotv⁺, lbcw::DirichletBC, w⁺, rotw) = begin
    lbcw.value == 0 || error("Advection for non-zero w at boundary not implemented")
    @. advu = rotw * v - 0.5 * (rotv⁺ * w⁺)
end
@inline advu!(advu, v, rotv¯, rotv⁺, w¯, ubcw::DirichletBC, rotw) = begin
    ubcw.value == 0 || error("Advection for non-zero w at boundary not implemented")
    @. advu = rotw * v - 0.5 * (rotv¯ * w¯)
end
advu!(_, _, _, _, ::DirichletBC, ::DirichletBC, _) =
    error("Simulations with only one vertical layer are not supported")


# compute one layer of -(rotz[uvp]*u[uvp]-rotx[w]*w[w]) on uvp-nodes
@inline advv!(advv, u, rotu¯, rotu⁺, w¯, w⁺, rotw) =
        @. advv = 0.5 * (rotu¯ * w¯ + rotu⁺ * w⁺) - rotw * u
@inline advv!(advv, u, rotu¯, rotu⁺, lbcw::DirichletBC, w⁺, rotw) = begin
    lbcw.value == 0 || error("Advection for non-zero w at boundary not implemented")
    @. advv = 0.5 * (rotu⁺ * w⁺) - rotw * u
end
@inline advv!(advv, u, rotu¯, rotu⁺, w¯, ubcw::DirichletBC, rotw) = begin
    ubcw.value == 0 || error("Advection for non-zero w at boundary not implemented")
    @. advv = 0.5 * (rotu¯ * w¯) - rotw * u
end
advv!(_, _, _, _, ::DirichletBC, ::DirichletBC, _) =
    error("Simulations with only one vertical layer are not supported")

# compute one layer of -(rotx[w]*v[uvp]-roty[w]*u[uvp]) on w-nodes
@inline advw!(advw, u¯, u⁺, rotu, v¯, v⁺, rotv) =
        @. advw = rotv * 0.5 * (u¯ + u⁺) - rotu * 0.5 * (v¯ + v⁺)

function advu!(advu_layers::NTuple{IZH},
        v_layers::NTuple{IZH}, rotv_layers::NTuple{IZV},
        w_layers::NTuple{IZV}, rotw_layers::NTuple{IZH},
        lower_bcu::BoundaryCondition, upper_bcu::BoundaryCondition,
        lower_bcw::BoundaryCondition, upper_bcw::BoundaryCondition) where {IZH,IZV}

    rotv_below = get_layer_below_pd(rotv_layers, lower_bcu)
    w_below    = get_layer_below_pd(w_layers,    lower_bcw)

    # inner layers, same for all processes (possibly empty)
    for i = 1:IZH-1
        advu!(advu_layers[i], v_layers[i],
                i>1 ? rotv_layers[i-1] : rotv_below, rotv_layers[i],
                i>1 ?    w_layers[i-1] :    w_below,    w_layers[i],
                rotw_layers[i])
    end

    # last layer, can be different
    advu!(advu_layers[IZH], v_layers[IZH],
            IZH > 1    ? rotv_layers[IZH-1] : rotv_below, # rotv¯
            IZH == IZV ? rotv_layers[IZH]   : upper_bcu,  # rotv⁺
            IZH > 1    ?    w_layers[IZH-1] : w_below,    # w¯
            IZH == IZV ?    w_layers[IZH]   : upper_bcw,  # w⁺
            rotw_layers[IZH])
end

function advv!(advv_layers::NTuple{IZH},
        u_layers::NTuple{IZH}, rotu_layers::NTuple{IZV},
        w_layers::NTuple{IZV}, rotw_layers::NTuple{IZH},
        lower_bcv::BoundaryCondition, upper_bcv::BoundaryCondition,
        lower_bcw::BoundaryCondition, upper_bcw::BoundaryCondition) where {IZH,IZV}

    rotu_below = get_layer_below_pd(rotu_layers, lower_bcv)
    w_below    = get_layer_below_pd(w_layers,    lower_bcw)

    # inner layers, same for all processes (possibly empty)
    for i = 1:IZH-1
        advv!(advv_layers[i], u_layers[i],
                i>1 ? rotu_layers[i-1] : rotu_below, rotu_layers[i],
                i>1 ?    w_layers[i-1] :    w_below,    w_layers[i],
                rotw_layers[i])
    end

    # last layer, can be different
    advv!(advv_layers[IZH], u_layers[IZH],
            IZH > 1    ? rotu_layers[IZH-1] : rotu_below, # rotu¯
            IZH == IZV ? rotu_layers[IZH]   : upper_bcv,  # rotu⁺
            IZH > 1    ?    w_layers[IZH-1] : w_below,    # w¯
            IZH == IZV ?    w_layers[IZH]   : upper_bcw,  # w⁺
            rotw_layers[IZH])
end

function advw!(advw_layers::NTuple{IZV},
        u_layers::NTuple{IZH}, rotu_layers::NTuple{IZV},
        v_layers::NTuple{IZH}, rotv_layers::NTuple{IZV},
        upper_bcu::BoundaryCondition, upper_bcv::BoundaryCondition) where {IZH,IZV}

    u_above = get_layer_above(u_layers, upper_bcu)
    v_above = get_layer_above(v_layers, upper_bcv)

    # inner layers, same for all processes (possibly empty)
    for i = 1:IZH-1
        advw!(advw_layers[i], u_layers[i], u_layers[i+1], rotu_layers[i],
                              v_layers[i], v_layers[i+1], rotv_layers[i])
    end

    # last layer, does not exist on last node
    IZV == IZH && advw!(advw_layers[IZV], u_layers[IZH], u_above, rotu_layers[IZV],
                                          v_layers[IZH], v_above, rotv_layers[IZV])
end

function local_grid_spacing(gd, gm::GridMapping{T}) where {T}
    dx1 = gm.hsize1 / gd.nx_pd
    dx2 = gm.hsize2 / gd.nx_pd
    dζ = 1 / gd.nz_global
    ζi = vrange(T, gd, NodeSet(:V))
    dx3dζ = gm.Dvmap.(ζi)
    dx3 = dx3dζ * dζ
    (dx1, dx2, dx3)
end

struct AdvectionBuffers{T}
    rot_fd::NTuple{3,Array{Complex{T},3}}
    vel_pd::NTuple{3,Array{T,3}}
    rot_pd::NTuple{3,Array{T,3}}
    adv_pd::NTuple{3,Array{T,3}}
    grid_spacing::Tuple{T,T,Array{T,1}}

    AdvectionBuffers(gd::DistributedGrid, gm::GridMapping{T}) where T =
        new{T}(
            zeros_fd.(T, gd, inverted_nodes()), # rot_fd
            zeros_pd.(T, gd, staggered_nodes()), # vel_pd
            zeros_pd.(T, gd, inverted_nodes()), # rot_pd
            zeros_pd.(T, gd, staggered_nodes()), # adv_pd
            local_grid_spacing(gd, gm),
        )
end

function advective_timescale(vel_layers, grid_spacing::Tuple{T,T,AbstractArray{T}}) where {T}

    # compute maximum velocity per layer (for CFL condition)
    umax_layer, vmax_layer, wmax_layer = (map(l -> mapreduce(abs, max, l), vel) for vel = vel_layers)

    # compute local timescales
    # NOTE: we use `reduce` instead of `maximum` to handle the case where there are zero local layers
    ts1 = reduce(min, (grid_spacing[1] / umax_layer[i] for i=1:length(umax_layer)), init=convert(T, Inf))
    ts2 = reduce(min, (grid_spacing[2] / vmax_layer[i] for i=1:length(vmax_layer)), init=convert(T, Inf))
    ts3 = reduce(min, (grid_spacing[3][i] / wmax_layer[i] for i=1:length(wmax_layer)), init=convert(T, Inf))

    global_minimum.((ts1, ts2, ts3))
end

function set_advection!(rhs, vel, df::DerivativeFactors, ht::HorizontalTransform,
        lower_bcs::Tuple, upper_bcs::Tuple, b::AdvectionBuffers)

    # compute vorticity in frequency domain
    set_vorticity!(b.rot_fd, vel, lower_bcs, upper_bcs, df)

    # transform velocity and vorticity to physical domain
    get_field!.(b.vel_pd, ht, vel, staggered_nodes())
    get_field!.(b.rot_pd, ht, b.rot_fd, inverted_nodes())

    # compute advection term in physical domain
    upd, vpd, wpd = layers.(b.vel_pd)
    rupd, rvpd, rwpd = layers.(b.rot_pd)
    lbcu, lbcv, lbcw = lower_bcs
    ubcu, ubcv, ubcw = upper_bcs
    advu!(layers(b.adv_pd[1]), vpd, rvpd, wpd, rwpd, lbcu, ubcu, lbcw, ubcw)
    advv!(layers(b.adv_pd[2]), upd, rupd, wpd, rwpd, lbcv, ubcv, lbcw, ubcw)
    advw!(layers(b.adv_pd[3]), upd, rupd, vpd, rvpd, ubcu, ubcv)

    # compute smallest time scale for advection term (for CFL condition)
    dt_adv = advective_timescale((upd, vpd, wpd), b.grid_spacing)

    # transform advection term to frequency domain (overwriting rhs)
    set_field!.(rhs, ht, b.adv_pd, staggered_nodes())

    rhs, dt_adv
end
