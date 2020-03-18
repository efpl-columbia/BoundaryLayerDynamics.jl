function add_forcing!(rhs_hat, forcing)
    @. rhs_hat[1][1,1,:] += forcing[1]
    @. rhs_hat[2][1,1,:] += forcing[2]
end

function force_mean_velocity!(u_hat, u_target, nz)
    #u_bar = real(global_sum(u_hat[1][1,1,:]) / nz)
    #v_bar = real(global_sum(u_hat[2][1,1,:]) / nz)
    #fx = u_target[1] - u_bar
    #fy = u_target[2] - v_bar
    #@. u_hat[1][1,1,:] += fx
    #@. u_hat[2][1,1,:] += fy
    #println("forcing x-velocity to match ", u_target[1], " (currently ", u_bar, ", adding ", fx, ")")
    #println("forcing y-velocity to match ", u_target[2], " (currently ", v_bar, ", adding ", fy, ")")
    u_hat[1][1,1,:] .+= u_target[1] - real(global_sum(u_hat[1][1,1,:])) / nz
    u_hat[2][1,1,:] .+= u_target[2] - real(global_sum(u_hat[2][1,1,:])) / nz
end

init_velocity_fields(T, gd) = (zeros_fd(T, gd, NodeSet(:H)),
                               zeros_fd(T, gd, NodeSet(:H)),
                               zeros_fd(T, gd, NodeSet(:V)))

function init_velocity(gd, ht::HorizontalTransform{T}, vel0, domain_size) where T
    δ = domain_size ./ (gd.nx_pd, gd.ny_pd, gd.nz_global)
    vel = init_velocity_fields(T, gd)
    set_field!(vel[1], ht, vel0[1], δ, gd.iz_min, NodeSet(:H))
    set_field!(vel[2], ht, vel0[2], δ, gd.iz_min, NodeSet(:H))
    set_field!(vel[3], ht, vel0[3], δ, gd.iz_min, NodeSet(:V))
    vel
end

init_pressure(T, gd) = zeros_fd(T, gd, NodeSet(:H))

abstract type BoundaryDefinition end
struct FreeSlipBoundary <: BoundaryDefinition end
struct SmoothWallBoundary <: BoundaryDefinition end
struct RoughWallBoundary <: BoundaryDefinition
    roughness::Real
end

solid_wall(roughness::Nothing) = SmoothWallBoundary()
solid_wall(roughness::Real) = RoughWallBoundary(roughness)

init_bcs(bc::FreeSlipBoundary, gd::DistributedGrid, ds::NTuple{3,T}) where T =
        (NeumannBC(zero(T), gd), NeumannBC(zero(T), gd), DirichletBC(zero(T), gd))

init_bcs(bc::SmoothWallBoundary, gd::DistributedGrid, ds::NTuple{3,T}) where T =
        (DirichletBC(zero(T), gd), DirichletBC(zero(T), gd), DirichletBC(zero(T), gd))

init_bcs(bc::RoughWallBoundary, gd::DistributedGrid, ds::NTuple{3,T}) where T =
        (RoughWallEquilibriumBC(convert(T, bc.roughness), gd, ds),
         RoughWallEquilibriumBC(convert(T, bc.roughness), gd, ds),
         DirichletBC(zero(T), gd))

init_bcs(bcs::NTuple{3,Union{String,Pair}}, gd::DistributedGrid,
         ds::NTuple{3,T}) where T = map(bcs) do bc
    t, val = bc isa String ? (bc, zero(T)) : (bc[1], convert(T, bc[2]))
    t == "Dirichlet" && return DirichletBC(val, gd)
    t == "Neumann" && return NeumannBC(val, gd)
    @error "Invalid boundary condition" bc
end

bc_noslip() = ("Dirichlet", "Dirichlet", "Dirichlet")
bc_freeslip() = ("Neumann", "Neumann", "Dirichlet")

init_advection(T, gd, ds, lbcs, ubcs, sgs::Nothing) =
        AdvectionBuffers(T, gd, ds)
init_advection(T, gd, ds, lbcs, ubcs, sgs::StaticSmagorinskyModel) =
        FilteredAdvectionBuffers(T, gd, ds, lbcs, ubcs, sgs)

struct ChannelFlowProblem{P,T}
    velocity::NTuple{3,Array{Complex{T},3}}
    pressure::Array{Complex{T},3}
    rhs::NTuple{3,Array{Complex{T},3}}
    grid::DistributedGrid{P}
    domain_size::NTuple{3,T}
    transform::HorizontalTransform{T}
    derivatives::DerivativeFactors{T}
    advection_buffers::Union{AdvectionBuffers{T},FilteredAdvectionBuffers{T,P}}
    lower_bcs::NTuple{3,BoundaryCondition}
    upper_bcs::NTuple{3,BoundaryCondition}
    pressure_bc::UnspecifiedBC
    pressure_solver::DistributedBatchLDLt{P,T,Complex{T}}
    diffusion_coeff::T
    forcing::NTuple{2,T}
    constant_flux::Bool

    ChannelFlowProblem(grid_size::NTuple{3,Int}, domain_size::NTuple{3,T},
        initial_conditions::NTuple{3,Function}, lower_bcs, upper_bcs,
        diffusion_coeff::T, forcing::NTuple{2,T}, constant_flux::Bool;
        sgs_model = nothing) where {T<:SupportedReals} = begin

        pressure_solver_batch_size = 64

        gd = DistributedGrid(grid_size...)
        ht = HorizontalTransform(T, gd)
        df = DerivativeFactors(gd, domain_size)
        lbcs = init_bcs(lower_bcs, gd, domain_size)
        ubcs = init_bcs(upper_bcs, gd, domain_size)

        new{proc_type(),T}(
            init_velocity(gd, ht, initial_conditions, domain_size),
            init_pressure(T, gd),
            init_velocity_fields(T, gd),
            gd, domain_size, ht, df, init_advection(T, gd, domain_size, lbcs, ubcs, sgs_model),
            lbcs, ubcs, UnspecifiedBC(T, gd),
            prepare_pressure_solver(gd, df, pressure_solver_batch_size),
            diffusion_coeff, forcing, constant_flux)
    end
end

zero_ics(T) = (f0 = (x,y,z) -> zero(T); (f0,f0,f0))

# TODO: remove initial conditions from ChannelFlowProblem above

prepare_ics(vel::Real) = ((x1,x2,x3)->vel, (x1,x2,x3)->0, (x1,x2,x3)->0)
prepare_ics(velh::Tuple{Real,Real}) = ((x1,x2,x3)->vel[1], (x1,x2,x3)->vel[2], (x1,x2,x3)->0)
prepare_ics(vel::Function) = (vel, (x1,x2,x3)->0, (x1,x2,x3)->0)
prepare_ics(velh::Tuple{Function,Function}) = (vel[1], vel[2], (x1,x2,x3)->0)
prepare_ics(vels::NTuple{3,Function}) = vels

function set_velocity!(vel, ht::HorizontalTransform, f::NTuple{3,Function}, gd::DistributedGrid, ds::NTuple{3,T}) where T
    δ = ds ./ (gd.nx_pd, gd.ny_pd, gd.nz_global)
    set_field!(vel[1], ht, f[1], δ, gd.iz_min, NodeSet(:H))
    set_field!(vel[2], ht, f[2], δ, gd.iz_min, NodeSet(:H))
    set_field!(vel[3], ht, f[3], δ, gd.iz_min, NodeSet(:V))
    vel
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

function set_velocity!(cf::ChannelFlowProblem{P,T}, vel; noise_intensity = 0) where {P,T}
    set_velocity!(cf.velocity, cf.transform, prepare_ics(vel), cf.grid, cf.domain_size)
    add_noise!.(cf.velocity, convert(T, noise_intensity))
    cf
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

coords(cf::ChannelFlowProblem, ns::NodeSet) = coords(cf.grid, cf.domain_size, ns)

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
    read_snapshot!(cf.velocity, snapshot_dir, cf.grid, cf.domain_size)

function integrate!(cf::ChannelFlowProblem{P,T}, dt, nt;
        snapshot_steps::Array{Int,1}=Int[], snapshot_dir = joinpath(pwd(), "snapshots"),
        output_io = Base.stdout, output_frequency = max(1, round(Int, nt / 100)),
        profiles_dir = joinpath(pwd(), "profiles"), profiles_frequency = 0,
        verbose=true) where {P,T}

    to = TimerOutputs.get_defaulttimer()
    oc = OutputCache(cf.grid, cf.domain_size, dt, nt, cf.lower_bcs, cf.upper_bcs,
            cf.diffusion_coeff, snapshot_steps, snapshot_dir, output_io, output_frequency)
    pp = MeanProfiles(T, cf.grid, profiles_dir, profiles_frequency,
            profiles_frequency == 0 ? 0 : div(nt, profiles_frequency))
    dt_adv = (zero(T), zero(T), zero(T))
    dt_dif = zero(T)

    function pressure_solver!(u)
        TimerOutputs.@timeit to "pressure" begin
            solve_pressure!(cf.pressure, u.x, cf.lower_bcs, cf.upper_bcs,
                cf.pressure_bc, cf.derivatives, cf.pressure_solver)
            subtract_pressure_gradient!(u.x, cf.pressure, cf.derivatives, cf.pressure_bc)
            cf.constant_flux && force_mean_velocity!(u.x, cf.forcing, cf.grid.nz_global)
        end
    end

    # initialize integrator and perform one step to compile functions
    TimerOutputs.@timeit to "initialization" begin

        u0 = RecursiveArrayTools.ArrayPartition(cf.velocity...)
        pressure_solver!(u0)

        # set up time integration as ODE problem, excluding pressure solution
        prob = OrdinaryDiffEq.ODEProblem(u0, (0.0, dt*nt)) do du, u, p, t
            TimerOutputs.@timeit to "advection" _, dt_adv = set_advection!(du.x, u.x,
                cf.derivatives, cf.transform, cf.lower_bcs, cf.upper_bcs, cf.advection_buffers)
            TimerOutputs.@timeit to "diffusion" _, dt_dif = add_diffusion!(du.x, u.x,
                cf.lower_bcs, cf.upper_bcs, cf.diffusion_coeff, cf.derivatives)
            cf.constant_flux || add_forcing!(du.x, cf.forcing)
        end

        # apply pressure solver as stage limiter for SSP stepping
        alg = OrdinaryDiffEq.SSPRK33((u, f, t) -> pressure_solver!(u))

        integrator = OrdinaryDiffEq.init(prob, alg, dt = dt, save_start = false, save_everystep = false)
        TimerOutputs.@timeit to "output" log_state!(oc, integrator.u.x, integrator.t, (dt, dt_adv, dt_dif), verbose)
    end

    # perform the full integration
    tstep = 0
    TimerOutputs.@timeit to "time stepping" for (state, t) in OrdinaryDiffEq.tuples(integrator)
        # this part is run after every step (not before/during)
        tstep += 1
        TimerOutputs.@timeit to "output profiles" log_profiles!(pp, state.x, cf.lower_bcs, cf.upper_bcs, cf.derivatives, t, tstep)
        TimerOutputs.@timeit to "output snapshots" log_state!(oc, state.x, t, (dt, dt_adv, dt_dif), verbose)
    end

    for i=1:3
        cf.velocity[i] .= integrator.sol[end].x[i]
    end
    verbose && show_all(oc.diagnostics_io, to)
    integrator.sol
end
