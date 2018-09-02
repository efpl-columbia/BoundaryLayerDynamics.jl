# handle boundary conditions ---------------------------------------------------

# uniform Dirichlet boundary conditions for u- and v-velocities
function set_lower_bc!(vel_hat, bc::DirichletBC) # (u[0] + u[1]) / 2 = value
    @views vel_hat[:,:,0] .= -vel_hat[:, :, 1]
    vel_hat[1,1,0] += 2 * bc.value
end
function set_upper_bc!(vel_hat, bc::DirichletBC) # (u[end-1] + u[end]) / 2 = value
    @views vel_hat[:,:,end] .= -vel_hat[:,:,end-1]
    vel_hat[1,1,end] += 2 * bc.value
end

# uniform Neumann boundary conditions for u- and v-velocities
set_lower_bc!(vel_hat, bc::NeumannBC) = @views vel_hat[:,:,0] .=
        vel_hat[:,:,1] - bc.value # value is multiplied by δ[3]
set_upper_bc!(vel_hat, bc::NeumannBC) = @views vel_hat[:,:,end] .=
        vel_hat[:,:,end-1] + bc.value # value is multiplied by δ[3]

# uniform Dirichlet boundary conditions for w-velocity (only Dirichlet supported)
function set_lower_bc_w!(vel_hat, bc::DirichletBC)
    vel_hat[:,:,1] .= zero(eltype(vel_hat))
    vel_hat[1,1,1] = bc.value
end
function set_upper_bc_w!(vel_hat, bc::DirichletBC)
    vel_hat[:,:,end] .= zero(eltype(vel_hat))
    vel_hat[1,1,end] = bc.value
end

function apply_bcs!(cf::ChannelFlowProblem)
    set_lower_bc!(cf.vel_hat[1], cf.lower_bc[1])
    set_upper_bc!(cf.vel_hat[1], cf.upper_bc[1])
    set_lower_bc!(cf.vel_hat[2], cf.lower_bc[2])
    set_upper_bc!(cf.vel_hat[2], cf.upper_bc[2])
    set_lower_bc_w!(cf.vel_hat[3], cf.lower_bc[3])
    set_upper_bc_w!(cf.vel_hat[3], cf.upper_bc[3])
    cf
end
