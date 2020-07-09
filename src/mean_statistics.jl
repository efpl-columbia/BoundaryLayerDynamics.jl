struct MeanStatistics{T}
    profiles::MeanProfiles{T}
    spectra::MeanSpectra{T}
    write_path::String
    write_frequency::Int
    write_count::Ref{Int}
    write_maxcount::Int
    start_t::Ref{T}
    start_it::Ref{Int}
    counter::Ref{Int}

    MeanStatistics(T, gd::DistributedGrid, path::String, frequency::Int,
                 total_count::Int) = new{T}(MeanProfiles(T, gd.nz_h, gd.nz_v),
        MeanSpectra(T, gd.nx_fd-1, div(gd.ny_fd-1, 2), gd.nz_h),
        path, frequency, Ref(0), total_count, Ref(zero(T)), Ref(0), Ref(0))
end

function log_statistics!(stats::MeanStatistics, vel, lower_bcs, upper_bcs, derivatives, t, it; flush = false)
    collect_statistics!(stats, vel, lower_bcs, upper_bcs, derivatives)
    if flush || ready_to_write(stats)
        write_statistics(stats, t, it)
        reset_statistics!(stats, t, it)
    end
end

function collect_statistics!(stats::MeanStatistics, vel, lbcs, ubcs, df)

    # TODO: use same boundary data for profiles & spectra
    TimerOutputs.@timeit "exchange boundary data" begin
        u1a = hlayers_above(layers(vel[1]), ubcs[1])
        u1b = hlayers_below(layers(vel[1]), lbcs[1])
        u2a = hlayers_above(layers(vel[2]), ubcs[2])
        u2b = hlayers_below(layers(vel[2]), lbcs[2])
        u3a = vlayers_above(layers(vel[3]), ubcs[3])
        u3b = vlayers_below(layers(vel[3]), lbcs[3])
    end

    add_profiles!(stats.profiles, vel, (u1b, u2b, u3b), (u1a, u2a, u3a), df)
    add_spectra!(stats.spectra, layers(vel[1]), layers(vel[2]),
                 layers_expand_i_to_c(vel[3], lbcs[3], ubcs[3]))
    stats.counter[] += 1
end

function ready_to_write(stats::MeanStatistics)
    stats.counter[] == stats.write_frequency
end

function write_statistics(stats, t, it)

    profiles = gather_profiles(stats.profiles)
    spectra = gather_spectra(stats.spectra)

    pad = ndigits(stats.write_maxcount)
    fn = joinpath(stats.write_path, "statistics-" *
                  string(stats.write_count[] + 1, pad = pad) *
                  ".json")

    if proc_type() <: LowestProc

        # only normalize profiles on process writing them
        for p in values(profiles)
            p[:] /= stats.counter[]
        end
        for s in values(spectra)
            s[:] /= stats.counter[]
        end

        mkpath(dirname(fn))
        open(fn, "w") do f
            JSON.print(f, Dict(
                "simulation_time" => (stats.start_t[], t),
                "simulation_steps" => (stats.start_it[] + 1, it),
                "mean_profiles" => profiles,
                "mean_spectra" => spectra,
                ))
        end
    end

    stats.write_count[] += 1
    fn
end

function reset_statistics!(stats::MeanStatistics, t, it)
    reset!(stats.profiles)
    reset!(stats.spectra)
    stats.start_t[] = t
    stats.start_it[] = it
    stats.counter[] = 0
    stats
end

# -------------------- EXCHANGE BOUNDARY DATA ----------------------------

# produce a tuple of layers that has the same number of entries as there are v-nodes
# and contains the h-layers just above/below these
function hlayers_below(hlayers, bc_below::BoundaryCondition{P}) where P
    P <: HighestProc ? hlayers[1:end-1] : hlayers
end
function hlayers_above(hlayers, bc_above::BoundaryCondition{SingleProc})
    hlayers[2:end]
end
function hlayers_above(hlayers, bc_above::BoundaryCondition{MinProc})
    MPI.Recv!(bc_above.buffer_fd, bc_above.neighbor_above, 1, MPI.COMM_WORLD)
    hlayers[2:end]..., bc_above.buffer_fd
end
function hlayers_above(hlayers, bc_above::BoundaryCondition{MaxProc})
    MPI.Send(hlayers[1], bc_above.neighbor_below, 1, MPI.COMM_WORLD)
    hlayers[2:end]
end
function hlayers_above(hlayers, bc_above::BoundaryCondition{InnerProc})
    r = MPI.Irecv!(bc_above.buffer_fd, bc_above.neighbor_above, 1, MPI.COMM_WORLD)
    MPI.Send(hlayers[1], bc_above.neighbor_below, 1, MPI.COMM_WORLD)
    MPI.Wait!(r)
    hlayers[2:end]..., bc_above.buffer_fd
end

# produce a tuple of layers that has the same number of entries as there are h-nodes
# and contains the v-layers (or BC) just above/below these
function vlayers_above(vlayers, bc_above::BoundaryCondition{P}) where P
    P <: HighestProc ? (vlayers..., bc_above) : vlayers
end
function vlayers_below(vlayers, bc_below::BoundaryCondition{SingleProc})
    bc_below, vlayers...
end
function vlayers_below(vlayers, bc_below::BoundaryCondition{MinProc})
    MPI.Send(vlayers[end], bc_below.neighbor_above, 1, MPI.COMM_WORLD)
    bc_below, vlayers[1:end-1]...
end
function vlayers_below(vlayers, bc_below::BoundaryCondition{MaxProc})
    MPI.Recv!(bc_below.buffer_fd, bc_below.neighbor_below, 1, MPI.COMM_WORLD)
    bc_below.buffer_fd, vlayers...
end
function vlayers_below(vlayers, bc_below::BoundaryCondition{InnerProc})
    r = MPI.Irecv!(bc_below.buffer_fd, bc_below.neighbor_below, 1, MPI.COMM_WORLD)
    MPI.Send(vlayers[end], bc_below.neighbor_above, 1, MPI.COMM_WORLD)
    MPI.Wait!(r)
    bc_below.buffer_fd, vlayers[1:end-1]...
end
