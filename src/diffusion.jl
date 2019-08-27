@inline function add_diffusion_layer!(rhs, vel¯, vel⁰, vel⁺, coeff,
        df::DerivativeFactors, ::NodeSet) # (vel¯ - 2 vel⁰ + vel⁺) / δz²
    @. rhs += coeff * (vel¯ * df.dz2 + vel⁰ * (df.dx2 + df.dy2 - 2 * df.dz2) +
              vel⁺ * df.dz2)
end

@inline function add_diffusion_layer!(rhs, lbc::NeumannBC, vel⁰, vel⁺, coeff,
        df::DerivativeFactors, ::NodeSet{:H}) # (- δz * LBC - vel⁰ + vel⁺) / δz²
    @. rhs += coeff * (vel⁰ * (df.dx2 + df.dy2 - df.dz2) + vel⁺ * df.dz2)
    rhs[1,1] -= coeff * lbc.value * df.dz1
end

@inline function add_diffusion_layer!(rhs, vel¯, vel⁰, ubc::NeumannBC, coeff,
        df::DerivativeFactors, ::NodeSet{:H}) # (vel¯ - vel⁰ + δz * UBC) / δz²
    @. rhs += coeff * (vel⁰ * (df.dx2 + df.dy2 - df.dz2) + vel¯ * df.dz2)
    rhs[1,1] += coeff * ubc.value * df.dz1
end

@inline function add_diffusion_layer!(rhs, lbc::DirichletBC, vel⁰, vel⁺, coeff,
        df::DerivativeFactors, ::NodeSet{:H}) # (8/3 lbc - 4 vel⁰ + 4/3 vel⁺) / δz²
    @. rhs += coeff * (vel⁰ * (df.dx2 + df.dy2 - 4*df.dz2) + vel⁺ * (4*df.dz2/3))
    rhs[1,1] += coeff * lbc.value * (8*df.dz2/3)
end

@inline function add_diffusion_layer!(rhs, vel¯, vel⁰, ubc::DirichletBC, coeff,
        df::DerivativeFactors, ::NodeSet{:H}) # (4/3 vel¯ - 4 vel⁰ + 8/3 ubc) / δz²
    @. rhs += coeff * (vel⁰ * (df.dx2 + df.dy2 - 4*df.dz2) + vel¯ * (4*df.dz2/3))
    rhs[1,1] += coeff * ubc.value * (8*df.dz2/3)
end

@inline function add_diffusion_layer!(rhs, lbc::DirichletBC, vel⁰, vel⁺, coeff,
        df::DerivativeFactors, ::NodeSet{:V}) # (lbc - 2 vel⁰ + vel⁺) / δz²
    @. rhs += coeff * (vel⁰ * (df.dx2 + df.dy2 - 2*df.dz2) + vel⁺ * df.dz2)
    rhs[1,1] += coeff * lbc.value * df.dz2
end

@inline function add_diffusion_layer!(rhs, vel¯, vel⁰, ubc::DirichletBC, coeff,
        df::DerivativeFactors, ::NodeSet{:V}) # (vel¯ - 2 vel⁰ + ubc) / δz²
    @. rhs += coeff * (vel⁰ * (df.dx2 + df.dy2 - 2*df.dz2) + vel¯ * df.dz2)
    rhs[1,1] += coeff * ubc.value * df.dz2
end

function add_diffusion!(rhs::NTuple{NZ}, layers::NTuple{NZ}, lower_bc, upper_bc,
    coeff, df::DerivativeFactors, nodes::NodeSet) where NZ

    layer_below = get_layer_below(layers, lower_bc)
    layer_above = get_layer_above(layers, upper_bc)

    if NZ > 1
        add_diffusion_layer!(rhs[1], layer_below, layers[1], layers[2], coeff, df, nodes)
        for i=2:NZ-1
            add_diffusion_layer!(rhs[i], layers[i-1], layers[i], layers[i+1], coeff, df, nodes)
        end
        add_diffusion_layer!(rhs[end], layers[NZ-1], layers[NZ], layer_above, coeff, df, nodes)
    elseif NZ == 1
        add_diffusion_layer!(rhs[1], layer_below, layers[1], layer_above, coeff, df, nodes)
    end
end

# compute diffusive time scale based on vertical grid spacing
diffusive_timescale(diffusion_coeff, df::DerivativeFactors) =
        (1 / df.dz1)^2 / diffusion_coeff

add_diffusion!(rhs::Array, field::Array, lower_bc::BoundaryCondition,
        upper_bc::BoundaryCondition, coeff::T, df::DerivativeFactors{T},
        nodes::NodeSet) where T = add_diffusion!(layers(rhs), layers(field),
        lower_bc, upper_bc, coeff, df, nodes)

add_diffusion!(rhs::NTuple{3,Array}, fields::NTuple{3,Array},
        lower_bcs::NTuple{3,BoundaryCondition}, upper_bcs::NTuple{3,BoundaryCondition},
        coeff::T, df::DerivativeFactors{T}) where T = (
        add_diffusion!(rhs[1], fields[1], lower_bcs[1], upper_bcs[1], coeff, df, NodeSet(:H));
        add_diffusion!(rhs[2], fields[2], lower_bcs[2], upper_bcs[2], coeff, df, NodeSet(:H));
        add_diffusion!(rhs[3], fields[3], lower_bcs[3], upper_bcs[3], coeff, df, NodeSet(:V));
        (rhs, diffusive_timescale(coeff, df)))
