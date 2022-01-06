# This file contains the new logging functionality, that should eventually
# include collecting flow statistics, saving snapshots of the flow field, and
# providing data for monitoring ongoing simulations.

struct FlowLog{P<:ProcType, T<:SupportedReals}

    mean_profiles::Dict{Symbol, Tuple{Array{T,1}, Ref{Int}}}

    files::Array{Pair{T, String}, 1}
    samples::Ref{Int}
    start::Ref{T}

    FlowLog(::Type{T}, terms, saveat, path) where {T} = new{proc_type(),T}(
        Dict(k => (zeros(T, n), Ref(0)) for (k, n) in terms),
        [t => joinpath(path * Printf.@sprintf("profiles-%03d.h5", i-1)) for (i, t) in enumerate(sort(saveat))],
        Ref(0), Ref(zero(T)))
end

# compact definition of terms to save profiles of
# with their corresponding node sets
const profile_nodes = Dict(
)

function FlowLog(::Type{T}, gd::DistributedGrid, saveat, path, terms = []) where T

    nz(term) = get_nz(gd, NodeSet(profile_nodes[term]))

    FlowLog(T, [t => nz(t) for t=terms], saveat, path)
end

log!(::Nothing, _, _, _) = return

function process_logs!(log::FlowLog, t)

    # check that all logs have been updated
    log.samples[] += 1
    for (k, (_, n)) in log.mean_profiles
        n[] == log.samples[] || error("Samples are incomplete for `$k`")
    end

    # write profiles to file necessary
    next_time(log) ≈ t && write_next!(log)

    # check that no file output was skipped
    t < next_time(log) || error("Output time was skipped")
end

next_time(log::FlowLog) = isempty(log.files) ? Inf : log.files[1].first

function write_next!(log::FlowLog{P}) where {P<:ProcType}

    # gather data from all processes
    t, fn = popfirst!(log.files)
    profiles = gather_profiles(P, log.mean_profiles)
    metadata = Dict("samples" => log.samples[], "timespan" => [log.start[], t])

    # write to file (one process only)
    P <: LowestProc && write_profiles(fn, profiles, metadata)

    # reset profiles
    reset!(log, t)
end

function reset!(log::FlowLog, t)
    log.samples[] = 0
    log.start[] = t
    for (p, n) in log.mean_profiles
        n[] = 0
        p .= 0
    end
    log
end

# TODO: avoid mixing ProcType and explicit MPI ranks
gather_profile(::Type{SingleProc}, profile) = profile
function gather_profile(::Type{MinProc}, profile)
    counts = MPI.Gather(Cint[length(profile)], 0, MPI.COMM_WORLD)
    global_profile = zeros(eltype(profile), sum(counts))
    Gatherv!(profile, VBuffer(global_profile, counts), 0, MPI.COMM_WORLD)
    global_profile
end
function gather_profile(::Type{P}, profile) where {P<:ProcType}
    MPI.Gather(Cint[length(profile)], 0, MPI.COMM_WORLD)
    Gatherv!(profile, nothing, 0, MPI.COMM_WORLD)
    nothing
end

function gather_profiles(::Type{P}, profiles::Dict{Symbol,A}) where {P<:ProcType, A}
    global_profiles = Dict{Symbol, (P<:LowestProc) ? A : Nothing}()
    for k in sort!(collect(keys(profiles))) # MPI ranks must have same order
        profile, _ = profiles[k]
        global_profiles[k] = gather_profile(P, profile)
    end
    global_profiles
end

function write_profiles(fn, profiles, metadata)
    isfile(fn) && error("File `$fn` already exists")
    HDF5.h5open(fn, "w") do h5
        for (k, v) in metadata
            HDF5.write_attribute(h5, k, v)
        end
        for (k, v) in profiles
            HDF5.write_dataset(fn, string(k), v)
        end
    end
end
