# set up minimal load path (remove system-wide packages)
deleteat!(LOAD_PATH, 1:length(LOAD_PATH))
push!(LOAD_PATH, "@", "@stdlib")

# set up parameters for benchmarks
const STEPS = 1000
const FREQUENCY = 10

using ABL, MPI
MPI.Init()

# set up different combinations of horizontal/vertical resolutions
const resolutions = if length(ARGS) > 2
    nh, nv = parse.(Int, ARGS[1:2])
    [(nh, nv)]
else
    # NOTE: `union` avoids including the same resolution twice
    union([(Nh, 64) for Nh = (32, 48, 64, 96, 128, 192, 256)],
          [(64, Nv) for Nv = (32, 48, 64, 96, 128)])
end

@assert endswith(last(ARGS), ".json") "Last argument needs to be a JSON file path"

function time_integration(variant, nh, nv)
    path = string(last(ARGS)[1:end-5], '-', nh, '-', nv, '-', variant, ".json")

    les_cfg = (sgs_model = StaticSmagorinskyModel(),
               roughness_length = 1e-6)
    abl = openchannelflow(1.0, (nh, nh, nv);
                          (variant == :les ? les_cfg : ())...)

    dt = 1e-12
    evolve!(abl, STEPS * dt, dt = dt, method = AB2(),
            output = ABL.Logging.StepTimer(path = path; frequency = FREQUENCY))
end

for variant in (:les, :dns), (nh, nv) in resolutions
    time_integration(variant, nh, nv)
end
