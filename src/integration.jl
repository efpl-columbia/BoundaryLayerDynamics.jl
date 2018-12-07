function add_forcing!(rhs_hat, forcing)
    @. rhs_hat[1][1,1,:] += forcing[1]
    @. rhs_hat[2][1,1,:] += forcing[2]
    @. rhs_hat[3][1,1,:] += forcing[3]
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
    transform::HorizontalTransform{T}
    derivatives::DerivativeFactors{T}
    advection_buffers::AdvectionBuffers{T}
    lower_bcs::NTuple{3,BoundaryCondition}
    upper_bcs::NTuple{3,BoundaryCondition}
    pressure_bc::DirichletBC
    pressure_solver::DistributedBatchLDLt{P,T,Complex{T}}
    diffusion_coeff::T
    forcing::NTuple{3,T}

    ChannelFlowProblem(grid_size::NTuple{3,Int}, domain_size::NTuple{3,T}, Re::T,
        open_channel::Bool, forcing::NTuple{3,T}, initial_conditions::NTuple{3,Function},
        ) where {T<:SupportedReals} = begin

        pressure_solver_batch_size = 64

        gd = DistributedGrid(grid_size...)
        ht = HorizontalTransform(T, gd)
        df = DerivativeFactors(gd, domain_size)

        new{proc_type(),T}(
            init_velocity(gd, ht, initial_conditions, domain_size),
            init_pressure(T, gd),
            init_velocity_fields(T, gd),
            gd, ht, df, AdvectionBuffers(T, gd),
            bc_noslip(T, gd), open_channel ? bc_freeslip(T, gd) : bc_noslip(T, gd),
            DirichletBC(gd, zero(T)), prepare_pressure_solver(gd, df, pressure_solver_batch_size),
            1 / Re, forcing)
    end
end

zero_ics(T) = (f0 = (x,y,z) -> zero(T); (f0,f0,f0))

open_channel(grid_size, Re, domain = (4π, 2π, 1.0), ic = zero_ics(Float64)) =
        ChannelFlowProblem(grid_size, domain, Re, true, (1.0, 0.0, 0.0), ic)
closed_channel(grid_size, Re, domain = (4π, 2π, 2.0), ic = zero_ics(Float64)) =
        ChannelFlowProblem(grid_size, domain, Re, false, (1.0, 0.0, 0.0), ic)

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

function integrate!(cf, dt, nt)
    to = TimerOutputs.TimerOutput()
    TimerOutputs.@timeit to "time stepping" for i=1:nt
        TimerOutputs.@timeit to "advection" set_advection!(cf.rhs, cf.velocity,
            cf.derivatives, cf.transform, cf.lower_bcs, cf.upper_bcs, cf.advection_buffers)
        TimerOutputs.@timeit to "diffusion" add_diffusion!(cf.rhs, cf.velocity,
            cf.lower_bcs, cf.upper_bcs, cf.diffusion_coeff, cf.derivatives)
        TimerOutputs.@timeit to "forcing" add_forcing!(cf.rhs, cf.forcing)
        TimerOutputs.@timeit to "pressure" begin
            solve_pressure!(cf.pressure, cf.rhs, cf.lower_bcs, cf.upper_bcs,
                cf.pressure_bc, cf.derivatives, cf.pressure_solver)
            subtract_pressure_gradient!(cf.rhs, cf.pressure, cf.derivatives, cf.pressure_bc)
        end
        TimerOutputs.@timeit to "velocity update" begin
            @. @views cf.velocity[1] += dt * cf.rhs[1]
            @. @views cf.velocity[2] += dt * cf.rhs[2]
            @. @views cf.velocity[3] += dt * cf.rhs[3]
        end
    end
    show_all(to)
end
