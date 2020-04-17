module ChannelFlow

import LinearAlgebra, MPI, FFTW, TimerOutputs, OrdinaryDiffEq,
       RecursiveArrayTools, Printf, JSON

export closed_channel, open_channel, integrate!, StaticSmagorinskyModel,
       ChannelFlowProblem, FreeSlipBoundary, SmoothWallBoundary, RoughWallBoundary,
       set_velocity!

include("transform.jl")
include("advection.jl")
include("diffusion.jl")
include("pressure_solver.jl")
include("output.jl")
include("mean_profiles.jl")
include("mean_spectra.jl")
include("mean_statistics.jl")
include("snapshots.jl")
include("static_smagorinsky.jl")
include("integration.jl")

end # module
