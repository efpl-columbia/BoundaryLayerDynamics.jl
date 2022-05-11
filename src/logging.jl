module Logging

using TimerOutputs: TimerOutput, @timeit
using MPI: MPI
using HDF5: HDF5
using Printf
using ..CBD: writecbd
using ..Domains: ABLDomain as Domain, x1range, x2range, x3range
using ..Grids: NodeSet, nodes, vrange
using ..PhysicalSpace: Transform2D, get_field, default_size, h1range, h2range
using ..BoundaryConditions: ConstantValue, DynamicValues

struct Log
    timer
    output
    function Log(output, domain, grid, t)
        timer = TimerOutput()
        if !(output isa Union{Tuple,AbstractArray})
            output = (output,)
        end
        output = map(output) do o
            kwargs = NamedTuple(k => o[k] for k in keys(o) if k != :output)
            o.output(domain, grid, t; kwargs...)
        end
        new(timer, output)
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
end
# default for output types and for log=nothing
flush!(opts...) = error("TODO")

function process_samples!(log::Log, t, state, pstate)
    @timeit log.timer "Output" for output in log.output
        process_samples!(output, t, state, pstate)
    end
end
# default for output types and for log=nothing
process_samples!(opts...) = nothing

function log_sample!(log::Log, sample, t; kwargs...)
    @timeit log.timer "Output" for output in log.output
        log_sample!(output, sample, t; kwargs...)
    end
end
# default for output types and for log=nothing
log_sample!(opts...; kwargs...) = nothing


function prepare_samples!(log::Log, t)
    @timeit log.timer "Output" for output in log.output
        prepare_samples!(output, t)
    end
end
# default for output types and for log=nothing
prepare_samples!(opts...) = nothing


struct MeanProfiles{T}
    path
    output_frequency
    means::Dict{Symbol,Vector{T}}
    samples::Dict{Symbol,Tuple{Vector{T},Ref{T}}}
    timespan::Vector{T}
    intervals::Ref{Int}
    boundaries::Tuple{Bool,Bool}
    comm

    function MeanProfiles(domain::Domain{T}, grid, t0;
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

        new{T}(path, output_frequency, means, samples, [t0, t0], Ref(0), boundaries, grid.comm)
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

function process_samples!(mp::MeanProfiles, t, state, pstate)

    # collect samples of frequency-space fields
    for p in pairs(state)
        log_sample!(mp, p, t)
    end

    # collect samples of physical-space fields
    for (dims, state) in pstate
        for p in pairs(state)
            log_sample!(mp, p, t)
        end
    end

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

    i = 0
    fn = nothing
    folder = dirname(mp.path)
    prefix = basename(mp.path) * "-"
    suffix = ".h5"
    while isnothing(fn)
        i += 1
        fni = joinpath(folder, prefix * @sprintf("%02d", i) * suffix)
        if !ispath(fni)
            fn = fni
        end
    end

    # gather data from all processes
    metadata = Dict("intervals" => mp.intervals[], "timespan" => mp.timespan)
    profiles = Dict()
    for k in sort!(collect(keys(mp.means))) # MPI ranks must have same order
        profile = gather_profile(mp.means[k], mp.comm)
        isnothing(profile) && continue
        dt = mp.timespan[2] - mp.timespan[1]
        @assert dt > 0 "Time interval for averaging is not positive"
        profile ./= dt
        profiles[k] = profile
    end

    # write to file (one process only)
    if isnothing(mp.comm) || MPI.Cart_coords(mp.comm)[] == 0
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

    function Snapshots(domain::Domain{T}, grid, t0;
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

function process_samples!(snaps::Snapshots{Ts}, t, state, pstate) where Ts
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
