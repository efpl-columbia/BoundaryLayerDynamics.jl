module ChannelFlow

import LinearAlgebra, MPI, FFTW, TimerOutputs

export ChannelFlowProblem, default_channel, integrate

include("transform.jl")
include("advection.jl")
include("diffusion.jl")
include("pressure_solver.jl")
include("integration.jl")

end # module
