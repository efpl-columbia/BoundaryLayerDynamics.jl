module ChannelFlow

import LinearAlgebra, MPI, FFTW, TimerOutputs, OrdinaryDiffEq,
       RecursiveArrayTools, Printf

export ChannelFlowProblem, closed_channel, open_channel, integrate!

include("transform.jl")
include("advection.jl")
include("diffusion.jl")
include("pressure_solver.jl")
include("output.jl")
include("snapshots.jl")
include("integration.jl")

end # module
