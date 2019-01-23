# compute one layer of vorticity (in frequency domain)
# (dw/dy - dv/dz) and (du/dz - dw/dx) on w-nodes, (dv/dx - du/dy) on uvp-nodes
@inline rotu!(rotu, v¯, v⁺, w, dy, dz) = @. rotu = w * dy - (v⁺ - v¯) * dz
@inline rotv!(rotv, u¯, u⁺, w, dx, dz) = @. rotv = (u⁺ - u¯) * dz - w * dx
@inline rotw!(rotw, u, v, dx, dy)      = @. rotw = v * dx - u * dy

function rotu!(rotu_layers::NTuple{IZV}, v_layers::NTuple{IZH}, w_layers::NTuple{IZV},
        df::DerivativeFactors, upper_bcv::BoundaryCondition) where {IZH,IZV}
    v_above = get_layer_above(v_layers, upper_bcv)
    for i=1:IZH-1
        rotu!(rotu_layers[i], v_layers[i], v_layers[i+1], w_layers[i], df.dy1, df.dz1)
    end
    if IZH == IZV # this is not the case for the process at the top of the domain
        rotu!(rotu_layers[IZV], v_layers[IZH], v_above, w_layers[IZV], df.dy1, df.dz1)
    end
end

function rotv!(rotv_layers::NTuple{IZV}, u_layers::NTuple{IZH}, w_layers::NTuple{IZV},
        df::DerivativeFactors, upper_bcu::BoundaryCondition) where {IZH,IZV}
    u_above = get_layer_above(u_layers, upper_bcu)
    for i=1:IZH-1
        rotv!(rotv_layers[i], u_layers[i], u_layers[i+1], w_layers[i], df.dx1, df.dz1)
    end
    if IZH == IZV # this is not the case for the process at the top of the domain
        rotv!(rotv_layers[IZV], u_layers[IZH], u_above, w_layers[IZV], df.dx1, df.dz1)
    end
end

function rotw!(rotw_layers::NTuple{IZH}, u_layers::NTuple{IZH}, v_layers::NTuple{IZH},
        df::DerivativeFactors) where {IZH,IZV}
    for i=1:IZH
        rotw!(rotw_layers[i], u_layers[i], v_layers[i], df.dx1, df.dy1)
    end
end

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

struct AdvectionBuffers{T}
    rotu_fd::Array{Complex{T},3}
    rotv_fd::Array{Complex{T},3}
    rotw_fd::Array{Complex{T},3}
    u_pd::Array{T,3}
    v_pd::Array{T,3}
    w_pd::Array{T,3}
    rotu_pd::Array{T,3}
    rotv_pd::Array{T,3}
    rotw_pd::Array{T,3}
    advu_pd::Array{T,3}
    advv_pd::Array{T,3}
    advw_pd::Array{T,3}

    AdvectionBuffers(T, gd::DistributedGrid) =
        new{T}(
            # rot_fd
            zeros_fd(T, gd, NodeSet(:V)),
            zeros_fd(T, gd, NodeSet(:V)),
            zeros_fd(T, gd, NodeSet(:H)),

            # vel_pd
            zeros_pd(T, gd, NodeSet(:H)),
            zeros_pd(T, gd, NodeSet(:H)),
            zeros_pd(T, gd, NodeSet(:V)),

            # rot_pd
            zeros_pd(T, gd, NodeSet(:V)),
            zeros_pd(T, gd, NodeSet(:V)),
            zeros_pd(T, gd, NodeSet(:H)),

            # adv_pd
            zeros_pd(T, gd, NodeSet(:H)),
            zeros_pd(T, gd, NodeSet(:H)),
            zeros_pd(T, gd, NodeSet(:V)),
        )
end

function set_advection!(rhs, vel, df::DerivativeFactors, ht::HorizontalTransform,
        lower_bcs::Tuple, upper_bcs::Tuple, b::AdvectionBuffers)

    # compute vorticity in frequency domain
    ul, vl, wl = map(layers, vel)
    rotu!(layers(b.rotu_fd), vl, wl, df, upper_bcs[2])
    rotv!(layers(b.rotv_fd), ul, wl, df, upper_bcs[1])
    rotw!(layers(b.rotw_fd), ul, vl, df)

    # transform vorticity to physical domain
    get_field!(b.rotu_pd, ht, b.rotu_fd, NodeSet(:V))
    get_field!(b.rotv_pd, ht, b.rotv_fd, NodeSet(:V))
    get_field!(b.rotw_pd, ht, b.rotw_fd, NodeSet(:H))

    # transform velocity to physical domain
    get_field!(b.u_pd, ht, vel[1], NodeSet(:H))
    get_field!(b.v_pd, ht, vel[2], NodeSet(:H))
    get_field!(b.w_pd, ht, vel[3], NodeSet(:V))

    # compute advection term in physical domain
    upd, vpd, wpd = map(layers, (b.u_pd, b.v_pd, b.w_pd))
    rupd, rvpd, rwpd = map(layers, (b.rotu_pd, b.rotv_pd, b.rotw_pd))
    lbcu, lbcv, lbcw = lower_bcs
    ubcu, ubcv, ubcw = upper_bcs
    advu!(layers(b.advu_pd), vpd, rvpd, wpd, rwpd, lbcu, ubcu, lbcw, ubcw)
    advv!(layers(b.advv_pd), upd, rupd, wpd, rwpd, lbcv, ubcv, lbcw, ubcw)
    advw!(layers(b.advw_pd), upd, rupd, vpd, rvpd, ubcu, ubcv)

    # transform advection term to frequency domain (overwriting rhs)
    set_field!(rhs[1], ht, b.advu_pd, NodeSet(:H))
    set_field!(rhs[2], ht, b.advv_pd, NodeSet(:H))
    set_field!(rhs[3], ht, b.advw_pd, NodeSet(:V))

    rhs
end
