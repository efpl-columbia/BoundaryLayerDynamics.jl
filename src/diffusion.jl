# TODO: reconsider handling of diffusion term for rough walls

# compute diffusive time scale based on vertical grid spacing
# NOTE: df.D3 is 1/Δx₃ (corrected for grid stretching)
diffusive_timescale(diffusion_coeff, df::DerivativeFactors) =
    (Δx3_min = 1 / maximum(df.D3_h); Δx3_min^2 / diffusion_coeff)

add_diffusion!(rhs::NTuple{3,Array}, fields::NTuple{3,Array},
        lower_bcs::NTuple{3,BoundaryCondition}, upper_bcs::NTuple{3,BoundaryCondition},
        coeff::T, df::DerivativeFactors{T}) where T = (add_laplacian!.(rhs, fields,
        lower_bcs, upper_bcs, (df,), staggered_nodes(), coeff), diffusive_timescale(coeff, df))
