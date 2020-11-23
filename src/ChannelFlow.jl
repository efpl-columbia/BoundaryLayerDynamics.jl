module ChannelFlow

__precompile__(false)

import LinearAlgebra, MPI, FFTW, TimerOutputs, OrdinaryDiffEq,
       RecursiveArrayTools, Printf, JSON

export prepare_closed_channel, prepare_open_channel, set_velocity!, integrate!,
       ChannelFlowProblem, FreeSlipBoundary, SmoothWallBoundary, RoughWallBoundary,
       StaticSmagorinskyModel, SinusoidalMapping

include("util.jl")
include("transform.jl")
include("communication.jl")
include("derivatives.jl")
include("advection.jl")
include("diffusion.jl")
include("pressure_solver.jl")
include("output.jl")
include("mean_profiles.jl")
include("mean_spectra.jl")
include("mean_statistics.jl")
include("snapshots.jl")
include("static_smagorinsky.jl")
include("time_integration.jl")
include("integration.jl")

# HIGH-LEVEL INTERFACE

prepare_closed_channel(Re, grid_size; kwargs...) = prepare_channel(Re, grid_size, half_channel=false; kwargs...)
prepare_open_channel(Re, grid_size; kwargs...) = prepare_channel(Re, grid_size, half_channel=true; kwargs...)

function prepare_channel(Re, grid_size;
                         half_channel = false,
                         domain_size = (4π, 2π),
                         initial_velocity = 0,
                         add_noise = false,
                         constant_flux = false,
                         sgs_model = nothing,
                         grid_mapping = nothing,
                         roughness_length = nothing)

    x3max = (half_channel ? 1.0 : 2.0)
    domain = (Float64.(domain_size)..., isnothing(grid_mapping) ? x3max :
              instantiate(grid_mapping, x3max, half_channel))

    cfp = ChannelFlowProblem(grid_size, domain, solid_wall(roughness_length),
                             half_channel ? FreeSlipBoundary() : solid_wall(roughness_length),
                             Float64(1/Re), (1.0, 0.0), constant_flux; sgs_model=sgs_model)
    set_velocity!(cfp, initial_velocity, noise_intensity = (add_noise ? 1e-3 : 0))
    cfp
end

end # module
