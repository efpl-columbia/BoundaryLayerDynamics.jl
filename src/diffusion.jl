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

@inline function get_layer_below(layers::Tuple, lower_bc::BoundaryCondition{SingleProc})
    lower_bc
end
@inline function get_layer_below(layers::Tuple, lower_bc::BoundaryCondition{MinProc})
    MPI.Send(layers[end], lower_bc.neighbor_above, 1, MPI.COMM_WORLD)
    lower_bc
end
@inline function get_layer_below(layers::NTuple{N,AbstractArray{Complex{T}}},
        lower_bc::BoundaryCondition{MaxProc,T}) where {N,T}
    MPI.Recv!(lower_bc.buffer_fd, lower_bc.neighbor_below, 1, MPI.COMM_WORLD)
    lower_bc.buffer_fd
end
@inline function get_layer_below(layers::NTuple{N,AbstractArray{Complex{T}}},
        lower_bc::BoundaryCondition{InnerProc,T}) where {N,T}
    r = MPI.Irecv!(lower_bc.buffer_fd, lower_bc.neighbor_below, 1, MPI.COMM_WORLD)
    MPI.Send(layers[end], lower_bc.neighbor_above, 1, MPI.COMM_WORLD)
    MPI.Wait!(r)
    lower_bc.buffer_fd
end

# TODO: consider using a different way of handling pd & fd
@inline function get_layer_below_pd(layers::Tuple, lower_bc::BoundaryCondition{SingleProc})
    lower_bc
end
@inline function get_layer_below_pd(layers::Tuple, lower_bc::BoundaryCondition{MinProc})
    MPI.Send(layers[end], lower_bc.neighbor_above, 1, MPI.COMM_WORLD)
    lower_bc
end
@inline function get_layer_below_pd(layers::NTuple{N,AbstractArray{T}},
        lower_bc::BoundaryCondition{MaxProc,T}) where {N,T}
    # NOTE: it needs to be explicit that the pd version of this method is wanted,
    # since layers can be empty, in which case the type information is lost
    MPI.Recv!(lower_bc.buffer_pd, lower_bc.neighbor_below, 1, MPI.COMM_WORLD)
    lower_bc.buffer_pd
end
@inline function get_layer_below_pd(layers::NTuple{N,AbstractArray{T}},
        lower_bc::BoundaryCondition{InnerProc,T}) where {N,T}
    r = MPI.Irecv!(lower_bc.buffer_pd, lower_bc.neighbor_below, 1, MPI.COMM_WORLD)
    MPI.Send(layers[end], lower_bc.neighbor_above, 1, MPI.COMM_WORLD)
    MPI.Wait!(r)
    lower_bc.buffer_pd
end

@inline function get_layer_above(layers::Tuple, upper_bc::BoundaryCondition{SingleProc})
    upper_bc
end
@inline function get_layer_above(layers::Tuple, upper_bc::BoundaryCondition{MaxProc})
    MPI.Send(layers[1], upper_bc.neighbor_below, 2, MPI.COMM_WORLD)
    upper_bc
end
@inline function get_layer_above(layers::Tuple{}, upper_bc::DirichletBC{MaxProc,T}) where T
    # this is a special case for when the top process does not have any layers,
    # which is the case if there is only one layer per process. in this case, we
    # fill the BC buffer with the boundary condition and pass that down to the
    # process below
    fill!(upper_bc.buffer_fd, zero(T))
    upper_bc.buffer_fd[1,1] = upper_bc.value
    MPI.Send(upper_bc.buffer_fd, upper_bc.neighbor_below, 2, MPI.COMM_WORLD)
    nothing # prevent the caller from trying to use the return value
end
@inline function get_layer_above(layers::NTuple{N,AbstractArray{Complex{T}}},
        upper_bc::BoundaryCondition{MinProc,T}) where {N,T}
    MPI.Recv!(upper_bc.buffer_fd, upper_bc.neighbor_above, 2, MPI.COMM_WORLD)
    upper_bc.buffer_fd
end
@inline function get_layer_above(layers::NTuple{N,AbstractArray{T}},
        upper_bc::BoundaryCondition{MinProc,T}) where {N,T}
    MPI.Recv!(upper_bc.buffer_pd, upper_bc.neighbor_above, 2, MPI.COMM_WORLD)
    upper_bc.buffer_pd
end
@inline function get_layer_above(layers::NTuple{N,AbstractArray{Complex{T}}},
        upper_bc::BoundaryCondition{InnerProc,T}) where {N,T}
    r = MPI.Irecv!(upper_bc.buffer_fd, upper_bc.neighbor_above, 2, MPI.COMM_WORLD)
    MPI.Send(layers[1], upper_bc.neighbor_below, 2, MPI.COMM_WORLD)
    MPI.Wait!(r)
    upper_bc.buffer_fd
end
@inline function get_layer_above(layers::NTuple{N,AbstractArray{T}},
        upper_bc::BoundaryCondition{InnerProc,T}) where {N,T}
    r = MPI.Irecv!(upper_bc.buffer_pd, upper_bc.neighbor_above, 2, MPI.COMM_WORLD)
    MPI.Send(layers[1], upper_bc.neighbor_below, 2, MPI.COMM_WORLD)
    MPI.Wait!(r)
    upper_bc.buffer_pd
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

layers(field::Array{T,3}) where T =
        Tuple(view(field, :, :, iz) for iz=1:size(field,3))

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
