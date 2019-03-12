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

struct ChannelFlowProblem{P,T}
    velocity::NTuple{3,Array{Complex{T},3}}
    pressure::Array{Complex{T},3}
    rhs::NTuple{3,Array{Complex{T},3}}
    grid::DistributedGrid{P}
    domain_size::NTuple{3,T}
    transform::HorizontalTransform{T}
    derivatives::DerivativeFactors{T}
    advection_buffers::AdvectionBuffers{T}
    lower_bcs::NTuple{3,BoundaryCondition}
    upper_bcs::NTuple{3,BoundaryCondition}
    pressure_bc::DirichletBC
    pressure_solver::DistributedBatchLDLt{P,T,Complex{T}}
    diffusion_coeff::T
    forcing::NTuple{2,T}
    constant_flux::Bool

    ChannelFlowProblem(grid_size::NTuple{3,Int}, domain_size::NTuple{3,T},
        initial_conditions::NTuple{3,Function}, open_channel::Bool,
        diffusion_coeff::T, forcing::NTuple{2,T}, constant_flux::Bool,
        ) where {T<:SupportedReals} = begin

        pressure_solver_batch_size = 64

        gd = DistributedGrid(grid_size...)
        ht = HorizontalTransform(T, gd)
        df = DerivativeFactors(gd, domain_size)

        new{proc_type(),T}(
            init_velocity(gd, ht, initial_conditions, domain_size),
            init_pressure(T, gd),
            init_velocity_fields(T, gd),
            gd, domain_size, ht, df, AdvectionBuffers(T, gd),
            bc_noslip(T, gd), open_channel ? bc_freeslip(T, gd) : bc_noslip(T, gd),
            DirichletBC(gd, zero(T)), prepare_pressure_solver(gd, df, pressure_solver_batch_size),
            diffusion_coeff, forcing, constant_flux)
    end
end

zero_ics(T) = (f0 = (x,y,z) -> zero(T); (f0,f0,f0))

open_channel(grid_size, Re, domain = (4π, 2π), ic = zero_ics(Float64),
        constant_flux = false) = ChannelFlowProblem(grid_size, (domain[1],
        domain[2], 1.0), ic, true, 1 / Re, (1.0, 0.0), constant_flux)
closed_channel(grid_size, Re, domain = (4π, 2π), ic = zero_ics(Float64),
        constant_flux = false) = ChannelFlowProblem(grid_size, (domain[1],
        domain[2], 2.0), ic, false, 1 / Re, (1.0, 0.0), constant_flux)

function show_all(to::TimerOutputs.TimerOutput)
    if MPI.Initialized()
        r = MPI.Comm_rank(MPI.COMM_WORLD)
        s = MPI.Comm_size(MPI.COMM_WORLD)
        for i=1:s
            if i==r+1
                println("Timing of process ", i, ":")
                show(to)
                println() # newline is missing in show(to)
            end
            MPI.Barrier(MPI.COMM_WORLD)
        end
    else
        show(to)
        println() # newline is missing in show(to)
    end
end

function show_progress(p::Integer)
    0 <= p <= 100 || error("Progress has to be a percentage.")
    print("│")
    print(repeat("█", div(p,4)))
    #mod(p,4) == 3 ? print("▓") : mod(p,4) == 2 ? print("▒") : mod(p,4) == 1 ? print("░") : nothing
    mod(p,4) == 3 ? print("▊") : mod(p,4) == 2 ? print("▌") : mod(p,4) == 1 ? print("▎") : nothing
    print(repeat(" ", div(100-p,4)))
    print("│")
    println(" ", p, "%")
end

function integrate!(cf, dt, nt; verbose=true, snapshot_steps::Array{Int,1}=Int[],
                    snapshot_dir = joinpath(pwd(), "snapshots"))

    to = TimerOutputs.TimerOutput()
    oc = OutputCache(cf.grid, cf.domain_size, dt, snapshot_steps, snapshot_dir)

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
            TimerOutputs.@timeit to "advection" set_advection!(du.x, u.x,
                cf.derivatives, cf.transform, cf.lower_bcs, cf.upper_bcs, cf.advection_buffers)
            TimerOutputs.@timeit to "diffusion" add_diffusion!(du.x, u.x,
                cf.lower_bcs, cf.upper_bcs, cf.diffusion_coeff, cf.derivatives)
            cf.constant_flux || add_forcing!(du.x, cf.forcing)
        end

        # apply pressure solver as stage limiter for SSP stepping
        alg = OrdinaryDiffEq.SSPRK33((u, f, t) -> pressure_solver!(u))

        integrator = OrdinaryDiffEq.init(prob, alg, dt = dt, save_start = false, save_everystep = false)
        #log_state(integrator.u.x, integrator.t)
    end

    # perform the full integration
    TimerOutputs.@timeit to "time stepping" for (state, t) in OrdinaryDiffEq.tuples(integrator)
        # this part is run after every step (not before/during)
        log_state!(oc, state.x, t)
    end

    for i=1:3
        cf.velocity[i] .= integrator.sol[end].x[i]
    end
    verbose && show_all(to)
    integrator.sol
end
