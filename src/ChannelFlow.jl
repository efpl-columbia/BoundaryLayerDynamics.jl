module ChannelFlow

import LinearAlgebra, MPI, FFTW, TimerOutputs, OrdinaryDiffEq,
       RecursiveArrayTools, Printf, JSON

export ChannelFlowProblem, closed_channel, open_channel, integrate!,
       StaticSmagorinskiModel, RoughWallEquilibriumModel

include("transform.jl")
include("advection.jl")
include("diffusion.jl")
include("pressure_solver.jl")
include("output.jl")
include("mean_profiles.jl")
include("snapshots.jl")
include("static_smagorinsky.jl")
include("integration.jl")

end # module
