function add_forcing!(rhs_hat, forcing)
    @. rhs_hat[1][1,1,:] += forcing[1]
    @. rhs_hat[2][1,1,:] += forcing[2]
end

function force_mean_velocity!(u_hat, u_target, gd, gm::GridMapping{T}) where {T}
    ζc = vrange(T, gd, NodeSet(:H))
    local_flux_u1 = sum(real(u_hat[1][1,1,i]) * gm.Dvmap(ζc[i]) for i=1:gd.nz_h) / gd.nz_global
    local_flux_u2 = sum(real(u_hat[2][1,1,i]) * gm.Dvmap(ζc[i]) for i=1:gd.nz_h) / gd.nz_global
    L3 = gm.vmap(one(T)) - gm.vmap(zero(T))
    u_hat[1][1,1,:] .+= u_target[1] - global_sum(local_flux_u1) / L3
    u_hat[2][1,1,:] .+= u_target[2] - global_sum(local_flux_u2) / L3
end

abstract type BoundaryDefinition end
struct FreeSlipBoundary <: BoundaryDefinition end
struct SmoothWallBoundary <: BoundaryDefinition end
struct RoughWallBoundary <: BoundaryDefinition
    roughness::Real
end

solid_wall(roughness::Nothing) = SmoothWallBoundary()
solid_wall(roughness::Real) = RoughWallBoundary(roughness)

init_bcs(bc::FreeSlipBoundary, gd::DistributedGrid, ::GridMapping{T}) where T =
        (NeumannBC(zero(T), gd), NeumannBC(zero(T), gd), DirichletBC(zero(T), gd))

init_bcs(bc::SmoothWallBoundary, gd::DistributedGrid, ::GridMapping{T}) where T =
        (DirichletBC(zero(T), gd), DirichletBC(zero(T), gd), DirichletBC(zero(T), gd))

init_bcs(bc::RoughWallBoundary, gd::DistributedGrid, gm::GridMapping{T}) where T =
        (RoughWallEquilibriumBC(convert(T, bc.roughness), gd, gm),
         RoughWallEquilibriumBC(convert(T, bc.roughness), gd, gm),
         DirichletBC(zero(T), gd))

init_bcs(bcs::NTuple{3,Union{String,Pair}}, gd::DistributedGrid,
         gm::GridMapping{T}) where T = map(bcs) do bc
    t, val = bc isa String ? (bc, zero(T)) : (bc[1], convert(T, bc[2]))
    t == "Dirichlet" && return DirichletBC(val, gd)
    t == "Neumann" && return NeumannBC(val, gd)
    @error "Invalid boundary condition" bc
end

bc_noslip(::Type{T}, gd) where T = (
        DirichletBC(zero(T), gd),
        DirichletBC(zero(T), gd),
        DirichletBC(zero(T), gd))
bc_freeslip(::Type{T}, gd) where T = (
        NeumannBC(zero(T), gd),
        NeumannBC(zero(T), gd),
        DirichletBC(zero(T), gd))

bc_noslip() = ("Dirichlet", "Dirichlet", "Dirichlet")
bc_freeslip() = ("Neumann", "Neumann", "Dirichlet")

init_advection(gd, gm, lbcs, ubcs, sgs::Nothing) =
        AdvectionBuffers(gd, gm)
init_advection(gd, gm, lbcs, ubcs, sgs::StaticSmagorinskyModel) =
        FilteredAdvectionBuffers(gd, gm, lbcs, ubcs, sgs)

"""
    ChannelFlowProblem(grid_size, …)

Set up a new channel flow with full control over the parameters.
"""
struct ChannelFlowProblem{P,T}
    velocity::NTuple{3,Array{Complex{T},3}}
    rhs::NTuple{3,Array{Complex{T},3}}
    grid::DistributedGrid
    mapping::GridMapping{T}
    transform::HorizontalTransform{T}
    derivatives::DerivativeFactors{T}
    advection_buffers::Union{AdvectionBuffers{T},FilteredAdvectionBuffers{T,P}}
    pressure_solver::PressureSolver{P,T}
    lower_bcs::NTuple{3,BoundaryCondition}
    upper_bcs::NTuple{3,BoundaryCondition}
    diffusion_coeff::T
    forcing::NTuple{2,T}
    constant_flux::Bool

    function ChannelFlowProblem(grid_size::NTuple{3,Int}, domain, lower_bcs, upper_bcs,
        diffusion_coeff::T, forcing::NTuple{2,T}, constant_flux::Bool;
        sgs_model = nothing, pressure_solver_batch_size = 64) where {T<:SupportedReals}

        gd = DistributedGrid(grid_size...)
        gm = GridMapping(domain...)
        ht = HorizontalTransform(T, gd)
        df = DerivativeFactors(gd, gm)
        lbcs = init_bcs(lower_bcs, gd, gm)
        ubcs = init_bcs(upper_bcs, gd, gm)
        adv = init_advection(gd, gm, lbcs, ubcs, sgs_model)
        ps = PressureSolver(gd, gm, pressure_solver_batch_size)

        new{proc_type(),T}(
            zeros_fd.(T, gd, staggered_nodes()),
            zeros_fd.(T, gd, staggered_nodes()),
            gd, gm, ht, df, adv, ps, lbcs, ubcs,
            diffusion_coeff, forcing, constant_flux)
    end
end

prepare_ics(vel::Real) = ((x1,x2,x3)->vel, (x1,x2,x3)->0, (x1,x2,x3)->0)
prepare_ics(velh::Tuple{Real,Real}) = ((x1,x2,x3)->vel[1], (x1,x2,x3)->vel[2], (x1,x2,x3)->0)
prepare_ics(vel::Function) = (vel, (x1,x2,x3)->0, (x1,x2,x3)->0)
prepare_ics(velh::Tuple{Function,Function}) = (velh[1], velh[2], (x1,x2,x3)->0)
prepare_ics(vels::NTuple{3,Function}) = vels

function set_velocity!(cf::ChannelFlowProblem{P,T}, vel; noise_intensity = 0) where {P,T}
    set_field!.(cf.velocity, prepare_ics(vel), cf.grid, cf.mapping, cf.transform, staggered_nodes())
    noise_intensity > 0 && add_noise!.(cf.velocity, convert(T, noise_intensity))
    cf
end

function add_noise!(vel::AbstractArray{Complex{T},3}, intensity::T = one(T) / 8) where T
    intensity == 0 && return vel
    for k=1:size(vel, 3)
        σ = real(vel[1,1,k]) * intensity
        for j=1:size(vel, 2), i=1:size(vel, 1)
            if !(i == j == 1) # do not modify mean flow
                vel[i,j,k] += σ * randn(Complex{T})
            end
        end
    end
    vel
end

open_channel(grid_size, Re; domain = (4π, 2π), ic = zero_ics(Float64),
        constant_flux = false) = ChannelFlowProblem(grid_size, (domain[1], domain[2],
        1.0), ic, bc_noslip(), bc_freeslip(), 1 / Re, (1.0, 0.0), constant_flux)
closed_channel(grid_size, Re; domain = (4π, 2π), ic = zero_ics(Float64),
        constant_flux = false) = ChannelFlowProblem(grid_size, (domain[1], domain[2],
        2.0), ic, bc_noslip(), bc_noslip(), 1 / Re, (1.0, 0.0), constant_flux)

# TODO: consider removing the "get_" from these names
get_velocity(gd::DistributedGrid, ht::HorizontalTransform{T}, velocity) where T = (
        get_field!(zeros_pd(T, gd, NodeSet(:H)), ht, velocity[1], NodeSet(:H)),
        get_field!(zeros_pd(T, gd, NodeSet(:H)), ht, velocity[2], NodeSet(:H)),
        get_field!(zeros_pd(T, gd, NodeSet(:V)), ht, velocity[3], NodeSet(:V)))
get_velocity(cf::ChannelFlowProblem) = get_velocity(cf.grid, cf.transform, cf.velocity)

coords(cf::ChannelFlowProblem, ns::NodeSet) = coords(cf.grid, cf.mapping, ns)

function show_all(io::IO, to::TimerOutputs.TimerOutput)
    if MPI.Initialized()
        r = MPI.Comm_rank(MPI.COMM_WORLD)
        s = MPI.Comm_size(MPI.COMM_WORLD)
        for i=1:s
            if i==r+1
                println(io, "Timing of process ", i, ":")
                show(io, to)
                println(io) # newline is missing in show(to)
            end
            MPI.Barrier(MPI.COMM_WORLD)
        end
    else
        show(io, to)
        println(io) # newline is missing in show(to)
    end
end

load_snapshot!(cf::ChannelFlowProblem, snapshot_dir) =
    read_snapshot!(cf.velocity, snapshot_dir, cf.grid, cf.mapping)

"""
    integrate!(channel_flow, step_size, steps, <keyword arguments>)

Simulate the development of a [`ChannelFlowProblem`] in time, optionally
collecting data about the flow at different points in time.

# Arguments

- `channel_flow`: the flow that is simulated; original flow field is
  overwritten.
- `step_size`: the (constant) size of each time step ``Δt``; needs to be small
  enough to prevent the solution from becoming unstable.
- `steps`: the number of time steps ``N_t`` of size ``Δt`` that are taken
- `snapshot_steps=[]`: array of steps after which a snapshot of the flow
  field is saved as a binary file.
- `snapshot_dir="snapshots"`: the (absolute or relative to the current working
  directory) in which files with snapshots of the velocity field will be saved.
- `output_frequency=div(steps,100)` the number of time steps after which a
  summary of flow statistics is printed to the standard output.
- `profiles_dir="profiles"`: the path (absolute or relative to the current
  working directory) in which files with mean profiles will be saved.
- `profiles_frequency`: the number of time steps after which mean profiles are
  saved and reset. It is recommended to save profiles around 10–20 times per
  simulation to allow removing an initial transient period and evaluate the
  impact of the averaging period during the analysis of the results.
- `method=SSPRK33()`: the method used for time integration. Currently supported
  methods are [`Euler`](@ref), [`AB2`](@ref), [`SSPRK22`](@ref), and
  [`SSPRK33`](@ref).
- `verbose=true`: if set, a summary of flow statistics is printed occasionally
  during the simulation and information on the computational performance is
  printed at the end of the simulation.
"""
function integrate!(cf::ChannelFlowProblem{P,T}, dt, nt;
        snapshot_steps::Array{Int,1}=Int[], snapshot_dir = joinpath(pwd(), "snapshots"),
        output_io = Base.stdout, output_frequency = max(1, round(Int, nt / 100)),
        profiles_dir = joinpath(pwd(), "profiles"), profiles_frequency = 0,
        method = SSPRK33(dt=dt), verbose=true) where {P,T}

    to = TimerOutputs.get_defaulttimer()
    oc = OutputCache(cf.grid, cf.mapping, dt, nt, cf.lower_bcs, cf.upper_bcs,
            cf.diffusion_coeff, snapshot_steps, snapshot_dir, output_io, output_frequency)
    stats = MeanStatistics(T, cf.grid, profiles_dir, profiles_frequency,
            profiles_frequency == 0 ? 0 : div(nt, profiles_frequency))
    dt_adv = (zero(T), zero(T), zero(T))
    dt_dif = zero(T)

    log = (state, tstep, t) -> begin
        TimerOutputs.@timeit to "output" begin
            TimerOutputs.@timeit to "flow statistics" log_statistics!(stats, state.x, cf.lower_bcs, cf.upper_bcs, cf.derivatives, t, tstep)
            TimerOutputs.@timeit to "snapshots" log_state!(oc, state.x, t, (dt, dt_adv, dt_dif), verbose)
        end
    end

    tstep = 0
    rate! = (du, u, t; checkpoint = false) -> begin
        if checkpoint
            tstep += 1
            log(u, tstep, t)
        end
        TimerOutputs.@timeit to "advection" _, dt_adv = set_advection!(du.x, u.x,
            cf.derivatives, cf.transform, cf.lower_bcs, cf.upper_bcs, cf.advection_buffers)
        TimerOutputs.@timeit to "diffusion" _, dt_dif = add_diffusion!(du.x, u.x,
            cf.lower_bcs, cf.upper_bcs, cf.diffusion_coeff, cf.derivatives)
        cf.constant_flux || add_forcing!(du.x, cf.forcing)
    end

    projection! = (u) -> begin
        TimerOutputs.@timeit to "pressure" begin
            enforce_continuity!(u.x, cf.lower_bcs, cf.upper_bcs,
                                cf.grid, cf.derivatives, cf.pressure_solver)
            cf.constant_flux && force_mean_velocity!(u.x, cf.forcing, cf.grid, cf.mapping)
        end
    end

    # initialize integrator and perform one step to compile functions
    TimerOutputs.@timeit to "initialization" begin
        u0 = RecursiveArrayTools.ArrayPartition(cf.velocity...)
        prob = TimeIntegrationProblem(rate!, projection!, u0, (0.0, dt*nt))
        log(prob.u, 0, 0.0)
    end

    # perform the full integration
    TimerOutputs.@timeit to "time stepping" sol = solve(prob, method, checkpoints=1)
    for i=1:3
        cf.velocity[i] .= sol.x[i]
    end
    verbose && show_all(oc.diagnostics_io, to)
    sol
end
