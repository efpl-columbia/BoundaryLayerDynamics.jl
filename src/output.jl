max_dt_diffusion(coeff, dx, dy, dz) = min(dx, dy, dz)^2 / coeff

function max_dt_advection(vel, grid_spacing)
    max_vel = map(global_maximum, vel)
    maximum(grid_spacing ./ max_vel)
end

struct OutputCache{T,NP}

    domain_size::NTuple{3,T}
    integration_time::T

    transform::HorizontalTransform{T}
    shift_factors::NTuple{2,Array{Complex{T},3}}

    snapshot_dir::String
    snapshot_counter::Base.RefValue{Int}
    snapshot_timestamps::Array{T,1}
    snapshot_tsdigits::NTuple{2,Int}

    diagnostics_io::IO
    diagnostics_counter::Base.RefValue{Int}
    diagnostics_frequency::Int

    profile_halflives::NTuple{NP,Int}
    profile_hlfactors::NTuple{NP,T}
    profiles_vel::NTuple{3,Array{T,2}}
    profiles_mke::NTuple{3,Array{T,2}}
    profiles_tke::NTuple{3,Array{T,2}}
    wallstress_factors::NTuple{4,Tuple{T,Tuple{Int,Int},NTuple{3,T}}}
    courant::Array{T,1}

    function OutputCache(gd::DistributedGrid, ds::NTuple{3,T}, dt::T, nt::Int,
                         lower_bcs::NTuple{3,BoundaryCondition},
                         upper_bcs::NTuple{3,BoundaryCondition},
                         diffusion_coeff::T,
                         snapshot_steps::Array{Int,1}, snapshot_dir,
                         output_io, output_frequency) where {T}
        timestamps = snapshot_steps .* dt
        halflives = (10, 100, 1000)
        hlfactors = Tuple(2^(-1/hl) for hl=halflives)
        vel, mke, tke = (Tuple(zeros(T, nz, length(halflives) + 1)
                               for nz in (gd.nz_h, gd.nz_h, gd.nz_v)) for i=1:3)
        wsf = wallstress_factors(gd, ds, lower_bcs, upper_bcs, diffusion_coeff)
        courant = zeros(T, 3*length(halflives) + 4)

        new{T,length(halflives)}(ds, dt*nt, HorizontalTransform(T, gd, expand=false),
                shift_factors(T, 2*gd.nx_fd-1, gd.ny_fd),
                snapshot_dir, Ref(0), timestamps, relevant_digits(timestamps),
                output_io, Ref(0), output_frequency,
                halflives, hlfactors, vel, mke, tke, wsf, courant,
                )
    end
end

vel_profile!(vel, vel_fd) = for iz=1:length(vel)
    vel[iz] = real(vel_fd[1,1,iz])
end

mke_profile!(mke, vel_fd) = for iz=1:length(mke)
    mke[iz] = abs2(vel_fd[1,1,iz])
end

tke_profile!(tke, vel_fd) = for iz=1:length(tke)
    tke[iz] = mapreduce(abs2, +, view(vel_fd,:,:,iz)) - abs2(vel_fd[1,1,iz])
end

function update_profiles!(output::OutputCache, state)

    # compute mean profiles from state
    for i=1:3
        vel_profile!(view(output.profiles_vel[i],:,1), state[i])
        mke_profile!(view(output.profiles_mke[i],:,1), state[i])
        tke_profile!(view(output.profiles_tke[i],:,1), state[i])
    end

    # add new profile to moving averages with corresponding half-life
    hlf = output.profile_hlfactors
    for p in (output.profiles_vel..., output.profiles_mke..., output.profiles_tke...)
        for i=1:length(hlf)
            @. @views p[:,1+i] = hlf[i] * p[:,1+i] + (1-hlf[i]) * p[:,1]
        end
    end
end

function wallstress_factors(gd::DistributedGrid, ds::NTuple{3,T},
        lower_bcs, upper_bcs, diffusion_coeff::T) where {T}

    # indices of first, second, second-to-last, and last layer
    # zero for layers that are not on the current process
    iz_min, iz_max, nzg, nzl = gd.iz_min, gd.iz_max, gd.nz_global, gd.nz_h
    lower_indices = (iz_min == 1 ? 1 : 0,
        iz_min == 1 && iz_max > 1 ? 2 : iz_min == 2 ? 1 : 0)
    upper_indices = (iz_max == nzg ? nzl : 0,
        iz_max == nzg && iz_min < nzg ? nzl - 1 : iz_max == nzg - 1 ? nzl : 0)

    # ν u'(0) = ν (-8 u(0) + 9 u(Δz/2) - u(3Δz/2)) / (3Δz)
    factors(bc) = bc isa NeumannBC ? (1, 0, 0) .* diffusion_coeff :
        (-8, 9, -1) ./ (3 * ds[3] / gd.nz_global) .* diffusion_coeff

    # precomputed values: bc, (index1, index2), (factor0, factor1, factor2)
    (   (lower_bcs[1].value, lower_indices, factors(lower_bcs[1])),
        (lower_bcs[2].value, lower_indices, factors(lower_bcs[2])),
        (upper_bcs[1].value, upper_indices, factors(upper_bcs[1])),
        (upper_bcs[2].value, upper_indices, factors(upper_bcs[2])),
        )
end

function wallstress(profiles::Array{T,2}, bc, indices, factors) where {T}

    # send mean velocity of first two grid points to all processes
    velocities = zeros(T, 2, size(profiles, 2))
    for i=1:2
        if indices[i] > 0
            @views velocities[i,:] .= profiles[indices[i],:]
        end
    end
    if MPI.Initialized()
        velocities = MPI.Allreduce!(velocities, zero(velocities), MPI.SUM, MPI.COMM_WORLD)
    end

    # compute wall stress with precomputed finite difference factors
    ws = T[ factors[1] * bc +
            factors[2] * velocities[1,i] +
            factors[3] * velocities[2,i] for i = 1:size(velocities, 2)]
end

function progressbar(p::Integer)
    0 <= p <= 100 || error("Progress has to be a percentage.")
    "│" *
    repeat('█', div(p, 4)) *
    (mod(p,4) == 3 ? "▊" : mod(p,4) == 2 ? "▌" : mod(p,4) == 1 ? "▎" : "") *
    repeat(' ', div(100 - p, 4)) *
    "│" *
    " " * string(p) * "%"
end

map_to_string(f, vals) = join((Printf.@sprintf("% .2e", f(x)) for x=vals), " ")
map_to_string(vals) = map_to_string(identity, vals)

function summary_profile(profiles)
    sum_local = zeros(size(profiles,2) + 1)
    sum_local[1:end-1] .= sum(profiles, dims=1)[:]
    sum_local[end] = size(profiles, 1)
    sum_global = MPI.Initialized() ? MPI.Allreduce!(sum_local, zero(sum_local), MPI.SUM, MPI.COMM_WORLD) : sum_local
    map_to_string(x -> x/sum_global[end], sum_global[1:end-1])
end

function summary_friction_velocity(output, i)
    ws = wallstress(output.profiles_vel[isodd(i) ? 1 : 2], output.wallstress_factors[i]...)
    map_to_string(x -> sqrt(abs(x)), ws)
end

function print_once(io::IO, args...)
    if (!MPI.Initialized() || MPI.Comm_rank(MPI.COMM_WORLD) == 0)
        println(io, args...)
    end
end

function print_diagnostics(io::IO, output::OutputCache, t)
    print_once(io, "Simulation status after ", output.diagnostics_counter[], " steps:")
    print_once(io, " • Bulk Velocity:               ", summary_profile(output.profiles_vel[1]))
    print_once(io, " • Mean Kinetic Energy:         ", summary_profile(output.profiles_mke[1]))
    print_once(io, " • Turbulent Kinetic Energy:    ", summary_profile(output.profiles_tke[1]))
    print_once(io, " • Friction Velocity Below (U): ", summary_friction_velocity(output, 1))
   #print_once(io, " • Friction Velocity Below (V): ", summary_friction_velocity(output, 2))
    print_once(io, " • Friction Velocity Above (U): ", summary_friction_velocity(output, 3))
   #print_once(io, " • Friction Velocity Above (V): ", summary_friction_velocity(output, 4))
    print_once(io, " • Advective Courant Number U:  ", map_to_string(output.courant[1:3:end-1]))
    print_once(io, " • Advective Courant Number V:  ", map_to_string(output.courant[2:3:end-1]))
    print_once(io, " • Advective Courant Number W:  ", map_to_string(output.courant[3:3:end-1]))
    print_once(io, " • Advective Courant Number:    ", map_to_string(sum(output.courant[i:i+2]) for i=1:3:length(output.courant)-1))
    print_once(io, " • Diffusive Courant Number:    ", map_to_string(output.courant[end:end]))
    print_once(io, progressbar(round(Int, 100 * t / output.integration_time)))
    flush(io)
end

function update_timescales!(output::OutputCache, dts)
    dt, dt_adv, dt_dif = dts
    courant_dif = dt / dt_dif
    courant_adv = dt ./ dt_adv
    output.courant[1:3] .= courant_adv
    output.courant[4:3:end-1] .= courant_adv[1] .* output.profile_hlfactors
    output.courant[5:3:end-1] .= courant_adv[2] .* output.profile_hlfactors
    output.courant[6:3:end-1] .= courant_adv[3] .* output.profile_hlfactors
    output.courant[end] = courant_dif
end

function log_state!(output::OutputCache, state, t, dts, show_diagnostics = true)

    update_profiles!(output, state)
    update_timescales!(output, dts)

    # print diagnostic output
    if output.diagnostics_counter[] % output.diagnostics_frequency == 0
        show_diagnostics && print_diagnostics(output.diagnostics_io, output, t)
    end
    output.diagnostics_counter[] += 1

    # write snapshot files
    if next_snapshot_time(output) ≈ t
        write_snapshot(output, state, t)
        if show_diagnostics
            print_once(output.diagnostics_io, "Wrote snapshot at t=", t)
            flush(output.diagnostics_io)
        end
    end
end
