const CBD_MAGIC_NUMBER = hex2bytes("CBDF")
const CBD_VERSION = 1

max_dt_diffusion(coeff, dx, dy, dz) = min(dx, dy, dz)^2 / coeff

function max_dt_advection(vel, grid_spacing)
    max_vel = map(global_maximum, vel)
    maximum(grid_spacing ./ max_vel)
end

function sequential_open(f, filename, mode)
    id, N = proc_id()
    r = nothing
    for i=1:N
        if i == id
            r = open(f, filename, mode)
        end
        MPI.Initialized() && MPI.Barrier(MPI.COMM_WORLD)
    end
    r
end

write_field(filename, field::Array{T}, x, y, z, domain_min, domain_max) where T =
    write_field(eltype(field), filename, field, x, y, z, domain_min, domain_max)

function write_field(T, filename, field, x, y, z, domain_min, domain_max)

    # build first value, describing the version of the file format that is used
    identifier = zeros(UInt8, 8)
    identifier[1:2] = CBD_MAGIC_NUMBER
    identifier[3]   = CBD_VERSION
    identifier[end] = T == Float64 ? 0x8 : T == Float32 ? 0x4 :
            error("Only 64-bit and 32-bit precision is supported.")

    # make sure the dimensions are compatible
    Nx, Ny, Nz = size(field)
    Nz = global_sum(Nz)
    length(x) == Nx || error("Length of x-values incompatible with dimensions of field.")
    length(y) == Ny || error("Length of y-values incompatible with dimensions of field.")
    length(z) == Nz || error("Length of z-values incompatible with dimensions of field.")

    sequential_open(filename, "a") do f
        if proc_id()[1] == 1
            write(f, identifier)
            write(f, collect(UInt64, (Nx, Ny, Nz)))
            write(f, collect(Float64, domain_min))
            write(f, collect(Float64, domain_max))
            write(f, convert(Array{Float64}, x))
            write(f, convert(Array{Float64}, y))
            write(f, convert(Array{Float64}, z))
        end
        write(f, convert(Array{T}, field))
    end
end

function read_field(filename, nodeset::NodeSet)

    grid_points = zeros(UInt64, 3)
    domain_min = zeros(Float64, 3)
    domain_max = zeros(Float64, 3)

    sequential_open(filename, "r") do f

        # parse identifier and select precision
        identifier = read!(f, zeros(UInt8, 8))
        identifier[1:2] == CBD_MAGIC_NUMBER ||
            @error "Not a CBD file (magic number not matching)." identifier[1:2]
        identifier[3] == CBD_VERSION ||
            @error "Unsupported version of CBD format." identifier[3]
        T = identifier[end] == 0x4 ? Float32 : identifier[end] == 0x8 ? Float64 :
            @error "Unsupported data type: Only 32-bit and 64-bit precision is supported."

        read!(f, grid_points)
        read!(f, domain_min)
        read!(f, domain_max)

        x = zeros(Float64, grid_points[1])
        y = zeros(Float64, grid_points[2])
        z = zeros(Float64, grid_points[3])
        read!(f, x)
        read!(f, y)
        read!(f, z)

        iz_min, iz_max = vertical_range(grid_points[3], nodeset)
        data = zeros(T, grid_points[1], grid_points[2], 1+iz_max-iz_min)
        bytes_per_value = T == Float64 ? 8 : T == Float32 ? 4 :
            error("Bytes per value not known for type ", T)
        skip(f, grid_points[1] * grid_points[2] * (iz_min-1) * bytes_per_value)
        read!(f, data)

        Tuple(domain_min), Tuple(domain_max), x, y, z, data
    end
end

function write_field(filename, field::Array{Complex{T}}, ds::NTuple{3},
        ht::HorizontalTransform{T}, sf::NTuple{2}, ns::NodeSet;
        shift=true, output_type=T) where {T}

    buffer_fd = get_buffer_fd(ht, ns)
    buffer_pd = get_buffer_pd(ht, ns)

    if shift
        broadcast!(*, buffer_fd, field, sf[1], sf[2])
    else
        buffer_fd .= field
    end

    shifth, shiftv = (shift, ns isa NodeSet{:H})
    gs = (size(buffer_pd, 1), size(buffer_pd, 2), global_sum(size(buffer_pd, 3)) + (shiftv ? 0 : 1))

    # when the indices are not shifted, the H-nodes start at zero
    # but the V-nodes start at Δz
    x = LinRange(0, ds[1], 2*gs[1]+1)[(shifth ? 2 : 1):2:end-1]
    y = LinRange(0, ds[2], 2*gs[2]+1)[(shifth ? 2 : 1):2:end-1]
    z = LinRange(0, ds[3], 2*gs[3]+1)[(shiftv ? 2 : 3):2:end-1]

    LinearAlgebra.mul!(buffer_pd, get_plan_bwd(ht, ns), buffer_fd)
    write_field(output_type, filename, buffer_pd, x, y, z,
                (zero(T), zero(T), zero(T)), ds)
end

function shift_factors(T, nx_pd, ny_pd)
    nkx = div(nx_pd, 2)
    nky = div(ny_pd, 2)
    kx = 0:nkx
    ky = vcat(0:nky, (isodd(ny_pd) ? -nky : -nky + 1):-1)
    shiftx = Complex{T}[exp(1im * kx * π / nx_pd) for kx=kx, ky=1:1, z=1:1]
    shifty = Complex{T}[exp(1im * ky * π / ny_pd) for kx=1:1, ky=ky, z=1:1]
    shiftx[nkx+1] *= (iseven(nx_pd) ? 0 : 1)
    shifty[nky+1] *= (iseven(ny_pd) ? 0 : 1)
    shiftx, shifty
end

"""
Return a tuple with the number of digits before and after the decimal point
needed to describe `x` in a meaningful way. This algorithm is an attempt at a
heuristic to format decimal numbers in a useful way.
"""
function relevant_digits(x)

    s1, s2 = split(strip(Printf.@sprintf("%.15f", x)[1:end-1], '0'), '.')
    periodic_part = ""
    leading_zeros = 0
    i = 1
    repetitions = 1
    for c=s2
        if isempty(periodic_part)
            if c == '0'
                leading_zeros += 1
            else
                periodic_part *= c
            end
        elseif periodic_part[i] == c
            if i == length(periodic_part)
                repetitions += 1
                i = 1
            else
                i += 1
            end
        else # character is not matching the periodic part
            periodic_part ^= repetitions
            periodic_part *= periodic_part[1:i-1] * c
            repetitions = 1
            i = 1
        end
    end

    Np = length(periodic_part)
    N = leading_zeros + (repetitions == 1 ? (Np < 5 ? Np : 3) : (Np == 1 ? 2 : Np == 2 ? 4 : 3))
    max(1, length(s1)), N
end

relevant_digits(xs::Array) = isempty(xs) ? (0,0) :
        mapreduce(relevant_digits, (x,y) -> max.(x,y), vcat(xs, abs.(diff(xs))))

function format_timestamp(timestamp, tsdigits)
    tint = floor(Int, timestamp)
    tdec = timestamp - tint
    "t" * string(tint, pad=tsdigits[1]) * (tsdigits[2] == 0 ? "" : Printf.@sprintf("%.15f", tdec)[2:end-15+tsdigits[2]])
end

function format_dirname(basedir, i, timestamps, tsdigits)
    prefix = "snapshot"
    id = string(i-1, pad=ndigits(length(timestamps)-1))
    timestamp = format_timestamp(timestamps[i], tsdigits)
    dirname = join((prefix, id, timestamp), '-')
    joinpath(basedir, dirname)
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
        halflives = (10, 100)
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

function next_snapshot_time(output::OutputCache)
    ts = output.snapshot_timestamps
    i = output.snapshot_counter[] + 1
    i > length(ts) ? Inf : ts[i]
end

function write_state(dir, state, domain_size, transform, shift_factors)
    mkpath(dir)
    write_field(joinpath(dir, "u.cbd"), state[1], domain_size, transform, shift_factors, NodeSet(:H))
    write_field(joinpath(dir, "v.cbd"), state[2], domain_size, transform, shift_factors, NodeSet(:H))
    write_field(joinpath(dir, "w.cbd"), state[3], domain_size, transform, shift_factors, NodeSet(:V))
end

function write_snapshot(output::OutputCache, state, t)
    output.snapshot_counter[] += 1
    output.snapshot_timestamps[output.snapshot_counter[]] ≈ t || error("Snapshot output out of sync")
    dir = format_dirname(output.snapshot_dir, output.snapshot_counter[],
                         output.snapshot_timestamps, output.snapshot_tsdigits)
    write_state(dir, state, output.domain_size, output.transform, output.shift_factors)
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
        MPI.Allreduce!(velocities, MPI.SUM, MPI.COMM_WORLD)
    end

    # compute wall stress with precomputed finite difference factors
    ws = T[ factors[1] * bc +
            factors[2] * velocities[1,i] +
            factors[3] * velocities[2,i] for i = 1:size(velocities, 2)]
end

function show_progress(io::IO, p::Integer)
    0 <= p <= 100 || error("Progress has to be a percentage.")
    print("│")
    print(repeat("█", div(p,4)))
    #mod(p,4) == 3 ? print("▓") : mod(p,4) == 2 ? print("▒") : mod(p,4) == 1 ? print("░") : nothing
    mod(p,4) == 3 ? print("▊") : mod(p,4) == 2 ? print("▌") : mod(p,4) == 1 ? print("▎") : nothing
    print(repeat(" ", div(100-p,4)))
    print("│")
    println(" ", p, "%")
end

map_to_string(f, vals) = join((Printf.@sprintf("% .2e", f(x)) for x=vals), " ")
map_to_string(vals) = map_to_string(identity, vals)

function summary_profile(profiles)
    sum_local = zeros(size(profiles,2) + 1)
    sum_local[1:end-1] .= sum(profiles, dims=1)[:]
    sum_local[end] = size(profiles, 1)
    sum_global = MPI.Initialized() ? MPI.Allreduce(sum_local, MPI.SUM, MPI.COMM_WORLD) : sum_local
    map_to_string(x -> x/sum_global[end], sum_global[1:end-1])
end

function summary_friction_velocity(output, i)
    ws = wallstress(output.profiles_vel[isodd(i) ? 1 : 2], output.wallstress_factors[i]...)
    map_to_string(x -> sqrt(abs(x)), ws)
end

function print_diagnostics(io::IO, output::OutputCache, t)
    println(io, "Simulation status after ", output.diagnostics_counter[], " steps:")
    println(io, " • Bulk Velocity:               ", summary_profile(output.profiles_vel[1]))
    println(io, " • Mean Kinetic Energy:         ", summary_profile(output.profiles_mke[1]))
    println(io, " • Turbulent Kinetic Energy:    ", summary_profile(output.profiles_tke[1]))
    println(io, " • Friction Velocity Below (U): ", summary_friction_velocity(output, 1))
    #println(io, " • Friction Velocity Below (V): ", summary_friction_velocity(output, 2))
    println(io, " • Friction Velocity Above (U): ", summary_friction_velocity(output, 3))
    #println(io, " • Friction Velocity Above (V): ", summary_friction_velocity(output, 4))
    println(io, " • Advective Courant Number U:  ", map_to_string(output.courant[1:3:end-1]))
    println(io, " • Advective Courant Number V:  ", map_to_string(output.courant[2:3:end-1]))
    println(io, " • Advective Courant Number W:  ", map_to_string(output.courant[3:3:end-1]))
    println(io, " • Advective Courant Number:    ", map_to_string(sum(output.courant[i:i+2]) for i=1:3:length(output.courant)-1))
    println(io, " • Diffusive Courant Number:    ", map_to_string(output.courant[end:end]))
    show_progress(io, round(Int, 100 * t / output.integration_time))
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
    output.diagnostics_counter[] += 1
    if output.diagnostics_counter[] % output.diagnostics_frequency == 0
        show_diagnostics && print_diagnostics(output.diagnostics_io, output, t)
    end

    # write snapshot files
    if next_snapshot_time(output) ≈ t
        write_snapshot(output, state, t)
        if show_diagnostics
            println(output.diagnostics_io, "Wrote snapshot at t=", t)
            flush(output.diagnostics_io)
        end
    end
end
