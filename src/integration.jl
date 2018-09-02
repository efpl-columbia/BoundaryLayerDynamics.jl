#=
missing steps:
- fix odd/even frequencies (drop Nyquist from the start)
=#

@inline innerindices(A::OffsetArray) = CartesianIndices((1:size(A,1),
        1:size(A,2), 1:size(A,3)-2))
@inline innerindices(A::Array) = CartesianIndices(A)

"""
Compute the Laplacian of a velocity and add it to the RHS array,
all in frequency domain. Note that this requires the values at iz=0
and iz=end in the vel_hat array, so they must be set from boundary
conditions and MPI exchanges before calling this function. The lowest
level of w-nodes can be set to NaNs, as the iz=1 level is at the
boundary and should not have a RHS.
"""
function add_laplacian_fd!(rhs_hat, vel_hat, df::DerivativeFactors)
    # for uvp-nodes: rely on values in iz=0 and iz=end in vel_hat for
    # top & bottom layer
    for i in innerindices(vel_hat)
        rhs_hat[i] = vel_hat[i[1], i[2], i[3]-1] * df.dz2 +
                (df.dx2[i[1]] + df.dy2[i[2]] - 2 * df.dz2) +
                vel_hat[i[1], i[2], i[3]+1] * df.dz2
    end
end

function add_laplacian_fd!(rhs_hat::Tuple, vel_hat::Tuple, df::DerivativeFactors)
    for (rh, vh) in zip(rhs_hat, vel_hat)
        add_laplacian_fd!(rh, vh, df)
    end
end

function add_forcing_fd!(rhs_hat, forcing)
    @. rhs_hat[1][1,1,:] += forcing[1]
    @. rhs_hat[2][1,1,:] += forcing[2]
    @. rhs_hat[3][1,1,:] += forcing[3]
end

function build_rhs!(cf)
    set_advection_fd!(cf.rhs_hat, cf.vel_hat, cf.rot_hat, cf.df, cf.tf_big)
    add_laplacian_fd!(cf.rhs_hat, cf.vel_hat, cf.df)
    add_forcing_fd!(cf.rhs_hat, cf.forcing)
    solve_pressure_fd!(cf.p_hat, cf.p_solver, cf.rhs_hat, cf.lower_bc[3],
            cf.df.dx1, cf.df.dy1, cf.df.dz1)
    subtract_pressure_gradient_fd!(cf.rhs_hat, cf.p_hat, cf.df.dx1, cf.df.dy1, cf.df.dz1)
end

function integrate!(cf, dt, nt; verbose=false)
    to = TimerOutput()
    @timeit to "time stepping" for i=1:nt
        @timeit to "advection" set_advection_fd!(cf.rhs_hat, cf.vel_hat, cf.rot_hat, cf.df, cf.tf_big, to)
        @timeit to "diffusion" add_laplacian_fd!(cf.rhs_hat, cf.vel_hat, cf.df)
        @timeit to "forcing" add_forcing_fd!(cf.rhs_hat, cf.forcing)
        @timeit to "pressure" solve_pressure_fd!(cf.p_hat, cf.p_solver, cf.rhs_hat, cf.lower_bc[3],
                cf.df.dx1, cf.df.dy1, cf.df.dz1)
        @timeit to "pressure" subtract_pressure_gradient_fd!(cf.rhs_hat, cf.p_hat,
                cf.df.dx1, cf.df.dy1, cf.df.dz1)
        @timeit to "velocity update" begin
            @. @views cf.vel_hat[1][:,:,1:cf.grid.n[3]] += dt * cf.rhs_hat[1]
            @. @views cf.vel_hat[2][:,:,1:cf.grid.n[3]] += dt * cf.rhs_hat[2]
            @. @views cf.vel_hat[3][:,:,1:cf.grid.n[3]] += dt * cf.rhs_hat[3]
        end
    end
    verbose && show(to)
    cf
end
