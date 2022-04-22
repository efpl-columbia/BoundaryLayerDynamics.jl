module Logging

using TimerOutputs: TimerOutput, @timeit
using HDF5: HDF5
using Printf
using ..CBD: writecbd
using ..Domains: ABLDomain as Domain, x1range, x2range, x3range
using ..Grids: NodeSet, nodes, vrange
using ..PhysicalSpace: Transform2D, get_field, default_size, h1range, h2range

struct Log
    timer
    output
    function Log(output, domain, grid)
        timer = TimerOutput()
        output = map(output) do o
            kwargs = NamedTuple(k => o[k] for k in keys(o) if k != :output)
            o.output(domain, grid; kwargs...)
        end
        new(timer, output)
    end
end

reset!(log::Log, t) = log

process_samples!(::Nothing, opts...) = nothing
function process_samples!(log::Log, t, state, pstate)
    @timeit "Output" begin
        for output in log.output
            process_samples!(output, t, state, pstate)
        end
    end
end

struct MeanProfiles
    args
    MeanProfiles(domain, grid; kwargs...) = new(kwargs)
end

MeanProfiles(; kwargs...) = (output=MeanProfiles, kwargs...)

function process_samples!(s::MeanProfiles, t, state, pstate)
    # TODO
end

struct Snapshots{T}
    frequency
    path
    transform
    xlimits
    xranges
    comm
    centered::Bool

    function Snapshots(domain::Domain{T}, grid;
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
