module Logging

using TimerOutputs: TimerOutput, @timeit, print_timer
using MPI: MPI
using HDF5: HDF5
using Printf, Dates
using ..Helpers: sequentially
using ..CBD: writecbd
using ..Domains: Domain, x1range, x2range, x3range, scalefactor
using ..Grids: NodeSet, nodes, vrange
using ..PhysicalSpace: Transform2D, get_field, default_size, h1range, h2range
using ..BoundaryConditions: ConstantValue, DynamicValues

struct Log
    timer
    output
    comm
    verbose
    function Log(output, domain, grid, tspan; dt = nothing, verbose = false)
        timer = TimerOutput()
        # add progress monitor and wrap output in array if necessary
        output = [(verbose ? (ProgressMonitor(tstep = dt),) : ())...,
                  (output isa Union{Tuple,AbstractArray} ? output : (output,))...]
        # initialize all outputs
        output = map(output) do o
            if o isa NamedTuple
                kwargs = NamedTuple(k => o[k] for k in keys(o) if k != :output)
                o.output(domain, grid, tspan; kwargs...)
            else
                o
            end
        end
        new(timer, output, grid.comm, verbose)
    end
end

function reset!(log::Log, t)
    @timeit log.timer "Output" for output in log.output
        reset!(output, t)
    end
end
# default for output types and for log=nothing
reset!(opts...) = nothing

function flush!(log::Log)
    @timeit log.timer "Output" for output in log.output
        flush!(output)
    end
    log.verbose && sequentially(log.comm) do
        print_timer(log.timer)
        println()
    end
end
# default for output types and for log=nothing
flush!(opts...) = nothing

function process_log!(log::Log, rates, t)
    @timeit log.timer "Output" for output in log.output
        process_log!(output, rates, t)
    end
end
# default for output types and for log=nothing
process_log!(opts...) = nothing

function log_sample!(log::Log, sample, t; kwargs...)
    @timeit log.timer "Output" for output in log.output
        log_sample!(output, sample, t; kwargs...)
    end
end
# default for output types and for log=nothing
log_sample!(opts...; kwargs...) = nothing

function log_state!(log::Log, state, t)
    @timeit log.timer "Output" for output in log.output
        log_state!(output, state, t)
    end
end
# default for output types and for log=nothing
log_state!(opts...) = nothing

function prepare_samples!(log::Log, t)
    @timeit log.timer "Output" for output in log.output
        prepare_samples!(output, t)
    end
end
# default for output types and for log=nothing
prepare_samples!(opts...) = nothing



"""
Collect time stamps every N steps to measure the compute time per time step.
"""
struct StepTimer
    path::AbstractString
    frequency::Int
    step_counter::Ref{Int}
    init_time::Float64
    timestamps_wall::Vector{Float64}
    timestamps_simulation::Vector{Float64}
    write::Bool

    function StepTimer(domain, grid, tspan; init_time = time(),
            path = "output/timestamps.json", frequency = 1)
        write = isnothing(grid.comm) || MPI.Comm_rank(grid.comm) == 0
        new(path, frequency, Ref(-1), init_time, zeros(0), zeros(0), write)
    end
end

StepTimer(; kwargs...) = (output=StepTimer, init_time=time(), kwargs...)

function process_log!(log::StepTimer, _rates, t)
    log.step_counter[] += 1
    log.step_counter[] % log.frequency == 0 || return
    push!(log.timestamps_wall, time())
    push!(log.timestamps_simulation, t)
end

function flush!(log::StepTimer)
    log.write || return
    mkpath(dirname(log.path))
    open(log.path, "w") do io
        println(io, "{")
        println(io, "  \"frequency\": ", log.frequency, ",")
        println(io, "  \"wallTime\": [",
                join(log.timestamps_wall .- log.init_time, ", "), "],")
        println(io, "  \"simulationTime\": [",
                join(log.timestamps_simulation, ", "), "]")
        println(io, "}")
    end
end


struct ProgressMonitor
    range
    frequency
    description
    timestamp
    progress
    remaining
    showprogress
    sampling
    samples
    prefactors
    boundaries
    comm

    function ProgressMonitor(domain, grid, tspan;
                             frequency = 10,
                             description = "Progress:",
                             tstep = nothing,
                            )
        showprogress = !MPI.Initialized() || MPI.Comm_rank(grid.comm) == 0
        Δζ = 1 / grid.n3global
        Δx3c = [Δζ / scalefactor(domain, 3, ζ) for ζ in vrange(grid, NodeSet(:C))]
        Δx3i = [Δζ / scalefactor(domain, 3, ζ) for ζ in vrange(grid, NodeSet(:I))]
        cweight = Δx3c / size(domain, 3)
        iweight = Δx3i / size(domain, 3)
        prefactors = Any[:cweight => cweight, :iweight => iweight, :Δx3i => Δx3i]
        isnothing(tstep) || push!(prefactors, :tstep => tstep)
        # wall stresses are only collected at the boundaries
        # TODO: use cartesian coordinates here
        s, r = isnothing(grid.comm) ? (1, 1) : (MPI.Comm_size(grid.comm), MPI.Comm_rank(grid.comm) + 1)
        boundaries = (r == 1, s == r)
        new(tspan, frequency, description, Ref(NaN), Ref(NaN), Ref(NaN), showprogress,
            Ref(false), Pair[], NamedTuple(prefactors), boundaries, grid.comm)
    end
end

ProgressMonitor(; kwargs...) = (output=ProgressMonitor, kwargs...)

function prepare_samples!(pm::ProgressMonitor, t)

    progress = (t - pm.range[1]) / (pm.range[2] - pm.range[1])
    #pm.steps[] += 1 # TODO: measure time per step
    now = time()

    # decide whether to show progress on current step
    pm.sampling[] = isnan(pm.timestamp[]) || (now - pm.timestamp[]) >= pm.frequency || progress == 1
    MPI.Initialized() && MPI.Bcast!(pm.sampling, 0, pm.comm)
    if !pm.sampling[]
        pm.showprogress && (print('·'); flush(stdout))
        return
    end
    pm.showprogress && println()

    rate = (progress - pm.progress[]) / (now - pm.timestamp[])
    pm.remaining[] = (1 - progress) / rate
    pm.timestamp[] = now
    pm.progress[] = progress

    pm.sampling[] = true
    empty!(pm.samples)
    push!(pm.samples, "Simulation Time" => t)
end

function largest(field, state)
    for dims in reverse(sort(collect(keys(state))))
        haskey(state[dims], field) && return state[dims][field]
    end
    error("Could not find field `$field`")
end

global_maximum(x, comm) = MPI.Initialized() ? MPI.Reduce(maximum(x, init=-Inf), MPI.MAX, 0, comm) : maximum(x, init=-Inf)
global_minimum(x, comm) = MPI.Initialized() ? MPI.Reduce(minimum(x, init=Inf), MPI.MIN, 0, comm) : minimum(x, init=Inf)
global_sum(x, comm) = MPI.Initialized() ? MPI.Reduce(sum(x), MPI.SUM, 0, comm) : sum(x)

function log_sample!(pm::ProgressMonitor, (field, values)::Pair, t; bcs = nothing)
    pm.sampling[] || return
    if field in (:sgs13, :sgs23)
        bvals = Tuple(pm.boundaries[i] ? hmean(bcs[i].type) : zero(eltype(values)) for i=1:2)
        bvals = global_sum.(bvals, (pm.comm,))
        label = "Wall Stress τ$(field == :sgs13 ? '₁' : '₂')₃"
        push!(pm.samples, label => bvals)
    end
end

function log_state!(pm::ProgressMonitor, state::NamedTuple, t)
    pm.sampling[] || return
    vel1avg = global_sum(real(state.vel1[1,1,:]) .* pm.prefactors.cweight, pm.comm)
    vel2avg = global_sum(real(state.vel2[1,1,:]) .* pm.prefactors.cweight, pm.comm)
    # TODO: include boundary values in mean of vel3 (interpolate to C-nodes)
    vel3avg = global_sum(real(state.vel3[1,1,:]) .* pm.prefactors.iweight, pm.comm)
    push!(pm.samples, "Mean Velocity" => (vel1avg, vel2avg, vel3avg))
end

function log_state!(pm::ProgressMonitor, state::Dict, t)
    pm.sampling[] || return

    vel1 = largest(:vel1, state)
    vel2 = largest(:vel2, state)
    vel3 = largest(:vel3, state)

    # TODO: add vertical component to energy terms
    vel1avg = sum(vel1, dims=(1,2))[:] / prod(size(vel1)[1:2])
    vel2avg = sum(vel2, dims=(1,2))[:] / prod(size(vel2)[1:2])
    mke = vel1avg.^2 .+ vel2avg.^2

    ke   = sum(abs2, vel1, dims=(1,2)) / prod(size(vel1)[1:2])
    ke .+= sum(abs2, vel2, dims=(1,2)) / prod(size(vel2)[1:2])
    tke = ke .- mke

    push!(pm.samples, "Mean KE" => global_sum(mke, pm.comm))
    push!(pm.samples, "Turbulent KE" => global_sum(tke, pm.comm))

    dtmin = global_minimum(pm.prefactors.Δx3i ./ maximum(abs, vel3, dims=(1,2))[:], pm.comm)
    if haskey(pm.prefactors, :tstep) && !isnothing(dtmin) # value only at root process
        push!(pm.samples, "Adv. Courant Number" => pm.prefactors.tstep / dtmin)
    else
        push!(pm.samples, "Advective Timescale" => dtmin)
    end

end

function process_log!(pm::ProgressMonitor, rates, t)
    pm.sampling[] || return
    pm.showprogress || return

    eta = if isfinite(pm.remaining[])
        eta = Second(ceil(pm.remaining[]))
        Dates.format(Time(Nanosecond(eta)), "H:MM:SS")
    else
        "unknown"
    end

    L = 25
    blocks = (' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█')

    # prog to intervals
    ip = round(Int, pm.progress[] * L * 8) # between 0 and 8*L
    bar = prod(blocks[clamp(1+ip-is, 1:9)] for is = 0:8:8*L-1)
    msg = string(pm.description, " ", @sprintf("%3.0f", 100 * pm.progress[]),
                "%▕", bar, "▏ ETA: ", eta)
    wt = length(msg)
    println('─'^wt)
    println(msg)

    wl = maximum(length(k) for k in first.(pm.samples)) + 2
    wr = wt - wl - 1
    println('─'^wl, '┬', '─'^wr)
    for (label, vals) in pm.samples
        vals = join([@sprintf("%.3g", v) for v in vals], ", ")
        println(lpad(label, wl-1), " ╎ ", vals)
    end
    println('─'^wl, '┴', '─'^wr)
    flush(stdout)
end


"""
    MeanProfiles(<keyword arguments>)

Collect profiles of terms, averaged over time and horizontal space.
The time average is computed with the trapezoidal rule using samples
collected before/after each time step. The averages are saved to HDF5
files, which can be written regularly during a simulation.

# Arguments

- `profiles = (:vel1, :vel2, :vel3)`: Terms of which profiles are
  collected. Currently supported: `:vel1`, `:vel2`, `:vel3`, `:adv1`,
  `:adv2`, `:adv3`, `:sgs11`, `:sgs12`, `:sgs13`, `:sgs22`, `:sgs23`,
  `:sgs33`.
- `path = "output/profiles"`: Prefix of the path at which HDF5-files
  with the profiles are saved. The files are numbered, so the first
  file would be written to `output/profiles-01.h5` for the default
  path.
- `output_frequency`: The interval with which the output is written to disk.
  Note that this should be a multiple of the time step, otherwise some outputs
  may be skipped.
"""
struct MeanProfiles{T}
    path
    output_frequency
    means::Dict{Symbol,Vector{T}}
    samples::Dict{Symbol,Tuple{Vector{T},Ref{T}}}
    timespan::Vector{T}
    intervals::Ref{Int}
    boundaries::Tuple{Bool,Bool}
    comm

    function MeanProfiles(domain::Domain{T}, grid, tspan;
            profiles = nothing,
            path = joinpath("output", "profiles"),
            # TODO: support specfiying sample frequency
            output_frequency = nothing) where T

        # for I-nodes, we also include the boundary-values in the profile (set
        # to zero by default, i.e. if terms do not log the boundary conditions)
        s, r = isnothing(grid.comm) ? (1, 1) : (MPI.Comm_size(grid.comm), MPI.Comm_rank(grid.comm) + 1)
        boundaries = (r == 1, s == r)
        vsize(ns) = length(vrange(grid, ns)) + (ns isa NodeSet{:I} ? sum(boundaries) : 0)
        profile(field) = zeros(T, vsize(nodes(field)))

        means = Dict(f => profile(f) for f in profiles)
        samples = Dict(f => (profile(f), Ref(convert(T, NaN))) for f in profiles)

        mkpath(dirname(path))
        new{T}(path, output_frequency, means, samples, [first(tspan), first(tspan)],
               Ref(0), boundaries, grid.comm)
    end
end

MeanProfiles(profiles = (:vel1, :vel2, :vel3); kwargs...) = (output=MeanProfiles, profiles=profiles, kwargs...)

function prepare_samples!(mp::MeanProfiles, t)
    # add first half of trapezoidal integral
    t > mp.timespan[end] || return # only add to mean when time is nonzero
    weight = (t - mp.timespan[end]) / 2
    for (field, (sample, timestamp)) in mp.samples
        @assert timestamp[] == mp.timespan[end] || isnan(timestamp[]) "Sample for `$field` changed between sampling interval"
        mp.means[field] .+= weight .* sample
    end
end

function log_sample!(mp::MeanProfiles, (field, values)::Pair, t; bcs = nothing)

    haskey(mp.samples, field) || return
    sample, timestamp = mp.samples[field]
    timestamp[] < t || isnan(timestamp[]) || return # avoid collecting same sample twice

    # save mean of data iniside computational domain
    @assert values isa AbstractArray
    i3s = if size(values, 3) == length(sample)
        1:length(sample)
    else
        1+Int(mp.boundaries[1]):length(sample)-Int(mp.boundaries[2])
    end
    for i3 = 1:length(i3s)
        sample[i3s[i3]] = hmean(view(values, :, :, i3))
    end

    # also save boundary values if provided
    if !isnothing(bcs)
        lbc, ubc = bcs
        if mp.boundaries[1]
            sample[1] = hmean(lbc.type)
        end
        if mp.boundaries[end]
            sample[end] = hmean(ubc.type)
        end
    end

    timestamp[] = t
    mp
end

hmean(values::AbstractArray) = eltype(values) <: Real ?
    sum(values) / prod(size(values)) : real(values[1,1])
hmean(bc::DynamicValues) = hmean(bc.values)
hmean(bc::ConstantValue) = bc.value

function log_state!(mp::MeanProfiles, state::NamedTuple, t)
    # collect samples of frequency-space fields
    for p in pairs(state)
        log_sample!(mp, p, t)
    end
end

function log_state!(mp::MeanProfiles, state::Dict, t)
    # collect samples of physical-space fields
    for (dims, state) in state
        for p in pairs(state)
            log_sample!(mp, p, t)
        end
    end
end

function process_log!(mp::MeanProfiles, rates, t)

    # add second half of trapezoidal integral
    if t > mp.timespan[end] # only add to mean (and count intervals) when time is nonzero
        weight = (t - mp.timespan[end]) / 2
        for (field, (sample, timestamp)) in mp.samples
            @assert timestamp[] == t "Sample missing for current time step"
            mp.means[field] .+= weight .* sample
        end
        mp.timespan[end] = t
        mp.intervals[] += 1
    end

    # return early if means should not be written to file yet
    mp.timespan[1] == mp.timespan[end] && return # do not save for empty interval (e.g. at t=0)
    t / mp.output_frequency ≈ round(Int, t / mp.output_frequency) || return

    # write output
    flush!(mp)
    reset!(mp)
end

function flush!(mp::MeanProfiles{T}) where T

    # since flush! is always called at the end of a simulation, it might be
    # called twice in a row without new data
    mp.intervals[] == 0 && return

    # gather data from all processes
    metadata = Dict("intervals" => mp.intervals[], "timespan" => mp.timespan)
    profiles = Dict()
    dt = mp.timespan[2] - mp.timespan[1]
    @assert dt > 0 "Time interval for averaging is not positive"
    for k in sort!(collect(keys(mp.means))) # MPI ranks must have same order
        profile = gather_profile(mp.means[k], mp.comm)
        isnothing(profile) && continue # profiles are only available on root process
        profile ./= dt
        profiles[k] = profile
    end

    # write to file (one process only)
    if isnothing(mp.comm) || MPI.Cart_coords(mp.comm)[] == 0

        # determine next available file name
        fn = nothing
        folder = dirname(mp.path)
        prefix = basename(mp.path) * "-"
        suffix = ".h5"
        i = 0
        while isnothing(fn)
            i += 1
            fni = joinpath(folder, prefix * @sprintf("%02d", i) * suffix)
            if !ispath(fni)
                fn = fni
            end
        end

        write_profiles(fn, profiles, metadata)
    end
    isnothing(mp.comm) || MPI.Barrier(mp.comm)
end

function reset!(mp::MeanProfiles)
    mp.intervals[] = 0
    mp.timespan[1] = mp.timespan[2]
    for (field, mean) in mp.means
        mean .= 0
    end
    mp
end

gather_profile(profile, ::Nothing) = profile[:]
function gather_profile(profile, comm)
    counts = MPI.Gather(Cint[length(profile)], 0, MPI.COMM_WORLD)
    if MPI.Comm_rank(comm) == 0
        global_profile = zeros(eltype(profile), sum(counts))
        MPI.Gatherv!(profile, MPI.VBuffer(global_profile, counts), 0, comm)
        global_profile
    else
        MPI.Gatherv!(profile, nothing, 0, comm)
        nothing
    end
end

function write_profiles(fn, profiles, metadata)
    isfile(fn) && error("File `$fn` already exists")
    HDF5.h5open(fn, "w") do h5
        for (k, v) in metadata
            HDF5.write_attribute(h5, k, v)
        end
        for (k, v) in profiles
            HDF5.write_dataset(h5, string(k), v)
        end
    end
end

struct Snapshots{T}
    frequency
    path
    transform
    xlimits
    xranges
    comm
    centered::Bool

    function Snapshots(domain::Domain{T}, grid, tspan;
            path = joinpath("output", "snapshots"),
            frequency = nothing,
            centered = true,
            precision::DataType = T) where T

        dims = default_size(grid)
        transform = Transform2D(precision, dims)
        x1 = x1range(domain, h1range(dims, centered=centered))
        x2 = x2range(domain, h2range(dims, centered=centered))
        x3c = x3range(domain, vrange(grid, NodeSet(:C)))
        x3i = x3range(domain, vrange(grid, NodeSet(:I)))
        xlimits = extrema(domain)
        xranges = (x1, x2, x3c, x3i)

        new{precision}(frequency, path, transform, xlimits, xranges, grid.comm, centered)
    end
end

Snapshots(; kwargs...) = (output=Snapshots, kwargs...)

# only applies to frequency-space state
function log_state!(snaps::Snapshots{Ts}, state::NamedTuple, t) where Ts

    # skip initial state, and only save at specified frequency, but allow for
    # floating-point imprecisions
    t == 0 && return
    n = round(Int, t/snaps.frequency)
    n ≈ t/snaps.frequency || return

    xmin, xmax = snaps.xlimits
    x1, x2, x3c, x3i = snaps.xranges

    for field in keys(state)
        path = joinpath(snaps.path, @sprintf("state-%03d", n), "$field.cbd")
        x3 = nodes(field) isa NodeSet{:C} ? x3c : nodes(field) isa NodeSet{:I} ? x3i : error("Invalid nodes for field `$field`")

        pfield = get_field(snaps.transform, state[field], centered = snaps.centered)
        writecbd(Ts, path, pfield, x1, x2, x3, xmin, xmax, snaps.comm)
    end
end



end # module Logging
