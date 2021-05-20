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

@doc (@doc prepare_channel)
prepare_closed_channel(Re, grid_size; kwargs...) = prepare_channel(Re, grid_size, half_channel=false; kwargs...)

@doc (@doc prepare_channel)
prepare_open_channel(Re, grid_size; kwargs...) = prepare_channel(Re, grid_size, half_channel=true; kwargs...)

"""
    prepare_closed_channel(grid_size; <keyword arguments>)
    prepare_open_channel(grid_size; <keyword arguments>)

Set up a channel flow with normalized parameters, either as a closed (full)
channel of height ``2δ`` with no-slip boundary conditions at the bottom and top
or as an open open channel of height ``δ`` with free-slip boundary conditions
at the top. Values are normalized with the friction velocity ``u_τ``, or with
the bulk velocity ``U_b`` if `constant_flux` is set. The simulation is always
run in 64-bit floating point precision.

# Arguments

- `grid_size`: the size of the computational grid, either as a tuple of three
  values (streamwise, spanwise, vertical), a tuple of two values (horizontal,
  vertical), or a single value that is used for all three dimensions. Note that
  in the horizontal directions, the number of Fourier modes is always an odd
  number for symmetry between positive and negative wavenumbers, so if the
  horizontal size is set to e.g. ``64``, the wavenumbers ``κ = -31,…,31`` are
  included and the Nyquist frequency ``κ=±32`` is omitted.
- `Re`: the Reynolds number of the flow, either ``Re_b = U_b δ / ν`` based on
  the bulk velocity ``U_b`` if the simulation is run in constant-flux mode (see
  below), or ``Re_τ = u_τ δ / ν`` otherwise, based on the friction velocity
  ``u_τ``.
- `domain_size=(4π,2π)`: the streamwise and spanwise domain size, normalized by
  ``δ``.
- `initial_velocity=0`: the initial value of the velocity field. Values can be
  provided as functions of the three-dimensional coordinate (e.g.
  `(x1,x2,x3)->x3`) or as a single, real number. If a single value is provided,
  only the streamwise velocity component is set, a tuple of two values sets the
  streamwise and spanwise velocity components, and a tuple of three values sets
  all three velocity components.
- `add_noise=false`: if set, the initial velocity field is perturbed by adding
  random noise of 0.1% of the mean velocity to each Fourier mode.
- `constant_flux=false`: if set, the pressure force driving the flow is
  adjusted dynamically to keep the mass flux constant. This also changes the
  reference velocity scale used for normalization alongside ``δ`` to the bulk
  velocity `U_b` rather than the friction velocity `u_τ`.
- `grid_mapping`: if provided, the vertical grid spacing is set to non-constant
  values using the provided mapping function. Currently the only supported
  maping is the [`SinusoidalMapping`](@ref), but arbitrary mappings can be
  provided when setting up a [`ChannelFlowProblem`](@ref) directly. Note that
  this is not recommended when running with a subgrid-scale model.
- `sgs_model`: if provided, the model is used to compute subgrid-scale
  contributions to the non-linear adveciton term, i.e. performing a large-eddy
  simulation (LES) rather than a direct numerical simulation (DNS). Currently
  the only supported subgrid-scale model is the
  [`StaticSmagorinskyModel`](@ref).
- `roughness_length`: if provided, an equilibrium wall model with the given
  roughness length is used to compute wall stresses. Currently this is required
  when a subgrid-scale model is used (i.e. wall-resolved LES is not supported),
  and no alternative wall models are implemented so far.
- `half_channel=false`: this option is already set when
  `prepare_closed_channel` or `prepare_open_channel` is called and calling
  `prepare_channel` directly is not recommended.
"""
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

"""
    run_simulation()
    run_simulation(config_file)

Set up and run a simulation based on parameters specified in a TOML
configuration file. The path to the file can be provided as a function
argument or as an argument on the command line, such as `julia -e 'using
ChannelFlow; run_simulation()' path/to/config.toml`. In either case, the path
should be absolute or relative to the current working directory.
"""
function run_simulation(config = (length(ARGS) == 1 ? ARGS[1] :
        error("Please provide the configuration of the simulation as argument")))
    error("Not implemented yet!")
end

end # module
