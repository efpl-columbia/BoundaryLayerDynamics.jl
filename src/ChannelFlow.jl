module ChannelFlow

import LinearAlgebra, MPI
using OffsetArrays, FFTW, TimerOutputs

include("types.jl")
include("initialization.jl")
include("boundary_conditions.jl")
include("advection.jl")
include("pressure_solver.jl")
include("integration.jl")

# old solvers
include("pd_solver.jl")
include("pd_solver_mpi.jl")

end # module
