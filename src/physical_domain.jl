struct ConstantSource <: ProcessDefinition
    field::Symbol
    strength
end

struct ConstantMean <: ProcessDefinition
    field::Symbol
    bulk_velocity
end

struct MolecularDiffusion <: ProcessDefinition
    field::Symbol
    coefficient
end

struct MomentumAdvection <: ProcessDefinition
    dealiasing # nothing, :quadratic, (Npd1, Npd2)
end

struct RoughWall <: BoundaryDefinition
    lengthscale
    value
end

struct FreeSlipBoundary <: BoundardDefinition
end

struct CustomBoundary <: BoundaryDefinition
    todo
end

struct SemiperiodicDomain
    size
    lower_boundary::BoundaryDefinition
    upper_boundary::BoundaryDefinition
    mapping
end

IncompressibleFlow(Re, constant_flux = false) = [
    MomentumAdvection(),
    MolecularDiffusion(:vel1, 1/Re),
    MolecularDiffusion(:vel2, 1/Re),
    MolecularDiffusion(:vel3, 1/Re),
    Pressure(),
    constant_flux ? ConstantMean(:vel1) : ConstantSource(:vel1),
]

PassiveScalar(name, coeff) = [
    MolecularDiffusion(name, coeff),
    ScalarAdvection(name),
]

Temperature(coeff; gravity = 9.81) = [
    MolecularDiffusion(:T, coeff),
    ScalarAdvection(:T),
    Buoyancy(gravity),
]


function run_dns_evaporation()

    N = (341, 512)

    domain_size = (4π, 2π, 2.0)
    lower_bcs = SmoothWallBoundary()
    upper_bcs = SmoothWallBoundary()
    domain = SemiperiodicDomain(domain_size, lower_bcs, upper_bcs,
                                SinusoidalMapping(0.8))

    physics = [
        ConstantFluxForcing(1.0),
        Diffusion(1 / Re),
        Advection(dealiasing = :quadratic),
        PressureSolver(batch_size = 64),
    ]

    flow = DiscretizedFlowSystem(N,
        terms = terms,
        lower_bcs
    )

    integrate!(flow) # or advance! or simulate!
end





function run_dns()

    N = (341, 512)

    domain_size = (4π, 2π, 2.0)
    lower_bcs = SmoothWallBoundary()
    upper_bcs = SmoothWallBoundary()
    domain = SemiperiodicDomain(domain_size, lower_bcs, upper_bcs,
                                SinusoidalMapping(0.8))

    physics = [
        ConstantFluxForcing(1.0),
        Diffusion(1 / Re),
        Advection(resolution = (512, 512)),
        PressureSolver(batch_size = 64),
    ]

    flow = DiscretizedFlowSystem(N,
        terms = terms,
        lower_bcs
    )

    integrate!(flow) # or advance! or simulate!
end

function run_les()

    N = 512

    domain_size = (4π, 2π, 1.0)
    lower_bcs = RoughWallBoundary(1e-3)
    upper_bcs = FreeSlipBoundary()
    domain = SemiperiodicDomain(domain_size, lower_bcs, upper_bcs)

    terms = [ # or physics or mechanisms or models
        ConstantForcing(1.0),
        Diffusion(1 / Re),
        Advection(resolution = (512, 512)),
        SubgridStresses(resolution = (512, 512)),
        PressureSolver(batch_size = 64),
    ]

    flow = DiscretizedFlowSystem(N,
        terms = terms,
        lower_bcs
    )

    # set initial conditions
    reset!(flow, 1.0, add_noise = true)

    integrate!(flow) # or advance! or simulate!
end


struct DiscretizedABL
    # TODO: decide on name
    # e.g. DiscretizedABL, DiscretizedFlowSystem, DiscretizedSystem
    state
    grid
    processes
end

function DiscretizedABL(modes, domain, processes)
    grid = DistributedGrid(modes)
    processes = [init_process(p, grid, domain) for p in processes]
    state = init_state(grid, processes)
    transform = Transform(grid, processes)
end

function init_state(grid, processes)
    fields = unique(Iterator.flatten(necessary_fields.(processes)))
    NamedTuple((f, zeros_fd(grid, nodes(f))) for f in fields)
end

nodes(field) = field == :u3 ? NodeSet(:I) : NodeSet(:C)

# defined in file of each process
state_fields(diff::DiscretizedMolecularDiffusion) = diff.field
state_fields(adv::MomentumAdvectionTerm) = (:u1, :u2, :u3)
state_fields(::PressureTerm) = (:u1, :u2, :u3)


struct DiscretizedMolecularDiffusion <: DiscretizedProcess
    @TODO
end

function init_process(diff::MolecularDiffusion, grid, domain)
    @TODO
end

struct DiscretizedMomentumAdvection <: DiscretizedProcess
    @TODO
end



islinear(_) = true # default value
islinear(::DiscretizedMomentumAdvection) = false

# defined in file of each nonlinear process
function physical_domain_fields(adv::DiscretizedMomentumAdvection)
    NamedTuple(f => adv.size_pd for f in (:vel1, :vel2, :vel3, :vorticity1, :vorticity2, :vorticity3))
end

struct Transform

    state
    fwd_plans
    bwd_plans

    function Transform(grid, processes, )

        processes = filter(p -> !islinear(p), processes)

    end

end

function transform_state!(transform, state)

    state_pd = transform.state_pd

    if haskey(state_pd, :velocity)
        set_field!(...)
    end

    if haskey(state_pd, :duidxj)
        ...
    end



end





function integrate!(flow::DiscretizedFlowSystem{P,T}, tspan;
        # time stepping
        dt = nothing, method = SSPRK33(),
        # output of diagnostics
        verbose=true, output_frequency = max(1, round(Int, (last(tspan) / dt) / 100)),
        # output of snapshots
        snapshot_steps::Array{Int,1}=Int[], snapshot_dir = joinpath(pwd(), "snapshots"),
        # output of profiles
        profiles_frequency = 0, profiles_dir = joinpath(pwd(), "profiles"),
        ) where {P,T}


    # set up logging
    terms = any(isa.(flow.processes, StaticSmagorinskyTerm)) ?
        [:sgs11, :sgs12, :sgs13, :sgs22, :sgs23, :sgs33] : []
    log::FlowLog{P,T} = FlowLog(T, flow.grid,
            profiles_frequency == 0 ? [] : (dt * profiles_frequency):(dt * profiles_frequency):t2,
            profiles_dir, terms)

    # initialize integrator and perform one step to compile functions
    TimerOutputs.@timeit log.timer "Initialization" begin
        u0 = RecursiveArrayTools.ArrayPartition(flow.state...)
        prob = TimeIntegrationProblem(rate!(flow), projection!(flow),
                                      u0, (t1, t2), checkpoint = true)
        reset!(log, t1) # removes samples from initial state and sets correct start time
    end

    # perform the full integration
    TimerOutputs.@timeit log.timer "Time integration" begin
        solve!(prob, method, dt, checkpoints=dt:dt:t2)
    end
end


# generate a function that performs the update of the rate
# based on the current state
rate!(flow) = (rate, state, t; checkpoint = false) -> begin
    # set RHS to zero
    fill!.(rate, 0)

    # add linear terms in frequency domain
    for process in filter(islinear, flow.processes)
        add_term!(rate, process, state, flow.log)
    end

    # add nonlinear terms in physical domain
    physical_domain!(rate, state, flow.transform) do rate, state
        for term in flow.nonlinear_terms
            # TODO: select correct size for each term

            add_term!(rate, term, state, flow.log)
        end
    end

    # perform logging activities
    checkpoint && process_logs!(flow.log, t)
end

# generate a function that performs the projection step
# of the pressure solver
projection!(flow) = (state) -> begin
    enforce_continuity!(state, flow.projection)
    enforce_mass_flux!(state, flow.projection)
end

function physical_domain!(pd_terms!, rate, state, transform)

    # prepare fields in physical domain
    state_pd = transform_state!(transform, state)
    rate_pd = reset_rate!(transform)

    # compute terms in physical domain
    pd_terms!(rate_pd, state_pd)

    # transform results back to physical domain
    transform_rate!(rate, transform)

    rate
end



struct AdvectionTerm
    TODO
end

struct DiffusionTerm
    TODO
end

struct ForcingTerm
    TODO
end

struct StaticSmagorinskyTerm
    TODO
end

# inputs required to compute term in physical domain
state(::AdvectionTerm) = (:velocity, :vorticity)
state(::StaticSmagorinskyTerm) = (:velocity, :duidxj)

# physical-domain outputs produced by the term
rate(::AdvectionTerm) = (:velocity)
rate(::StaticSmagorinskyTerm) = (:velocity, :flux1, :flux2)

function add_term!(rate, term::AdvectionTerm, state, log)

    # unpack state and boundary conditions
    u1, u2, u3 = layers.(state.velocity)
    r1, r2, r3 = layers.(state.vorticity)
    lbc1, lbc2, lbc3 = term.lower_bcs
    ubc1, ubc2, ubc3 = term.upper_bcs

    # compute components of advection term
    adv1!(layers(term.adv[1]), u2, r2, u3, r3, lbc1, ubc1, lbc3, ubc3)
    adv2!(layers(term.adv[2]), u1, r1, u3, r3, lbc2, ubc2, lbc3, ubc3)
    adv3!(layers(term.adv[3]), u1, r1, u2, r2, ubc1, ubc2)

    # compute smallest time scale for advection term (for CFL condition)
    dt_adv = advective_timescale((u1, u2, u3), term.grid_spacing)

    # add advection term to rate
    rate.velocity[1] .+= term.adv[1]
    rate.velocity[2] .+= term.adv[2]
    rate.velocity[3] .+= term.adv[3]

    rate, dt_adv
end





## Running simulation from configuration file

# `julia --project --eval 'using ABL; run_simulation()' setup.toml`

function run_simulation()

    if length(ARGS) > 0
        isfile(ARGS[1]) && return run_simulation(ARGS[1])
    else
        for fn in ("config.toml", "setup.toml", "simulation.toml")
            isfile(fn) && return run_simulation(fn)
        end
    end

    error("Could not find configuration file")
end

function run_simulation(config::AbstractString)
    run_simulation(TOML.parsefile(config))
end

function run_simulation(config::Dict)
    flow = DiscretidzedFlowSystem(config)
    integrate!(flow, config)
end

