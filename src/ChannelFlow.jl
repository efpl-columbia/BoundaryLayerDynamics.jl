module ChannelFlow

import LinearAlgebra, MPI, FFTW, TimerOutputs, OrdinaryDiffEq, RecursiveArrayTools

export ChannelFlowProblem, closed_channel, open_channel, integrate!

include("transform.jl")
include("advection.jl")
include("diffusion.jl")
include("pressure_solver.jl")
include("integration.jl")
include("output.jl")

end # module
