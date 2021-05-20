# set up minimal load path (remove system-wide packages)
deleteat!(LOAD_PATH, 1:length(LOAD_PATH))
push!(LOAD_PATH, "@", "@stdlib")

using BenchmarkTools, ChannelFlow, MPI
CF = ChannelFlow # for convenience
MPI.Init()

# set up parameters for benchmarks
# IMPORTANT: the time limit should not be reached, otherwise it is possible
# that the MPI ranks execute the code a different number of times and get stuck
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 3600*24

# set up different combinations of horizontal/vertical resolutions
# NOTE: `union` avoids including the same resolution twice
resolutions = union(
    [(Nh, 64) for Nh = (32, 48, 64, 96, 128, 192, 256)],
    [(64, Nv) for Nv = (32, 48, 64, 96, 128)],
)

# set up & run test suite for each resolution
const suite = BenchmarkGroup()
for N in resolutions
    redirect_stdout(devnull) do # suppress output from ChannelFlow package

        # setup problem
        dns = CF.prepare_closed_channel(1.0, (N[1], N[1], N[2]))
        les = CF.prepare_open_channel(1.0, (N[1], N[1], N[2]),
                sgs_model = StaticSmagorinskyModel(),
                roughness_length = 1e-6) # should be less than Δx₃

        # define tests for individual parts of the code
        bm = BenchmarkGroup()
        # TODO: ChannelFlowProblem.rhs will get removed eventually → replace
        bm["diffusion"] = @benchmarkable CF.add_diffusion!($dns.rhs, $dns.velocity,
            $dns.lower_bcs, $dns.upper_bcs, $dns.diffusion_coeff, $dns.derivatives)
        bm["advection_dns"] = @benchmarkable CF.set_advection!($dns.rhs, $dns.velocity,
            $dns.derivatives, $dns.transform, $dns.lower_bcs, $dns.upper_bcs, $dns.advection_buffers)
        bm["advection_les"] = @benchmarkable CF.set_advection!($les.rhs, $les.velocity,
            $les.derivatives, $les.transform, $les.lower_bcs, $les.upper_bcs, $les.advection_buffers)
        bm["continuity"] = @benchmarkable CF.enforce_continuity!($dns.velocity,
            $dns.lower_bcs, $dns.upper_bcs, $dns.grid, $dns.derivatives, $dns.pressure_solver)

        # run tests
        warmup(bm, verbose=false) # avoid including compilation in samples
        suite[N] = run(bm)
    end
end

const mpir = MPI.Comm_rank(MPI.COMM_WORLD)
const mpis = MPI.Comm_size(MPI.COMM_WORLD)

# print serialized results to standard output
mpir == 1 && BenchmarkTools.save(Base.stdout, suite)
#mpir == 1 && display(suite)

#=
struct Transformable
        A::Array{Float64,3}
        Â::Array{Complex{Float64},3}
plan_fwd::FFTW.rFFTWPlan{Float64,FFTW.FORWARD,false,3}
plan_bwd::FFTW.rFFTWPlan{Complex{Float64},FFTW.BACKWARD,false,3}
        Transformable(N) = begin
                A = zeros(Float64, N)
                Â = zeros(Complex{Float64}, 1+div(N[1],2), N[2], N[3])
                new(A, Â,
                        FFTW.plan_rfft(A, (1,2)),
                        FFTW.plan_brfft(Â, N[1], (1,2)))
        end
end

function fwd(T)
        LinearAlgebra.mul!(T.Â, T.plan_fwd, T.A)
end

function bwd(T)
        LinearAlgebra.mul!(T.Â, T.plan_fwd, T.A)
end

function roundtrip(T)
        LinearAlgebra.mul!(T.Â, T.plan_fwd, T.A)
        LinearAlgebra.mul!(T.A, T.plan_bwd, T.Â)
end
T = Transformable(N)
Ns = 100 # samples
suite["forward"] = @benchmarkable fwd(\$T) evals=1 samples=Ns
suite["backward"] = @benchmarkable bwd(\$T) evals=1 samples=Ns
suite["roundtrip"] = @benchmarkable roundtrip(\$T) evals=1 samples=Ns
=#
