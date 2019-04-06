struct MeanProfiles{T}
    write_path::String
    write_frequency::Int
    write_count::Ref{Int}
    write_maxcount::Int
    start_t::Ref{T}
    start_it::Ref{Int}
    counter::Ref{Int}
    u::Array{T,1}
    v::Array{T,1}
    w::Array{T,1}
    uu::Array{T,1}
    vv::Array{T,1}
    ww::Array{T,1}
    uv::Array{T,1}
    uwa::Array{T,1}
    uwb::Array{T,1}
    vwa::Array{T,1}
    vwb::Array{T,1}

    MeanProfiles(T, gd::DistributedGrid, path::String, frequency::Int, total_count::Int) =
        new{T}(path, frequency, Ref(0), total_count, Ref(zero(T)), Ref(0), Ref(0),
        zeros(T, gd.nz_h), zeros(T, gd.nz_h), zeros(T, gd.nz_v), # u, v, w
        zeros(T, gd.nz_h), zeros(T, gd.nz_h), zeros(T, gd.nz_v), zeros(gd.nz_h), # uu, vv, ww, uv
        zeros(T, gd.nz_h), zeros(T, gd.nz_h), zeros(T, gd.nz_h), zeros(gd.nz_h), # uwa, uwb, vwa, vwb
        )
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

avg_from_fd(vel::AbstractArray{Complex{T},2}) where T = real(vel[1,1])
avg_sq_from_fd(vel::AbstractArray{Complex{T},2}) where T =
        2 * mapreduce(abs2, +, vel) - abs2(vel[1,1])
avg_prod_from_fd(vel1::AbstractArray{Complex{T},2}, vel2::AbstractArray{Complex{T},2}) where T =
        (prod2((z1, z2)) = real(z1) * real(z2) + imag(z1) * imag(z2);
         2 * mapreduce(prod2, +, zip(vel1, vel2)) - prod2((vel1[1,1], vel2[1,1])))
avg_prod_from_fd(vel::AbstractArray{Complex{T},2}, bc::DirichletBC) where T =
        2 * bc.value * mapreduce(real, +, vel) - bc.value * real(vel[1,1])

function save_profiles!(profiles::MeanProfiles, vel, lower_bcs, upper_bcs)
    wa = vlayers_above(layers(vel[3]), upper_bcs[3])
    wb = vlayers_below(layers(vel[3]), lower_bcs[3])
    profiles.u[:]   .+= avg_from_fd.(layers(vel[1]))
    profiles.v[:]   .+= avg_from_fd.(layers(vel[2]))
    profiles.w[:]   .+= avg_from_fd.(layers(vel[3]))
    profiles.uu[:]  .+= avg_sq_from_fd.(layers(vel[1]))
    profiles.vv[:]  .+= avg_sq_from_fd.(layers(vel[2]))
    profiles.ww[:]  .+= avg_sq_from_fd.(layers(vel[3]))
    profiles.uv[:]  .+= avg_prod_from_fd.(layers(vel[1]), layers(vel[2]))
    profiles.uwa[:] .+= avg_prod_from_fd.(layers(vel[1]), wa)
    profiles.uwb[:] .+= avg_prod_from_fd.(layers(vel[1]), wb)
    profiles.vwa[:] .+= avg_prod_from_fd.(layers(vel[2]), wa)
    profiles.vwb[:] .+= avg_prod_from_fd.(layers(vel[2]), wb)
    profiles.counter[] += 1
end

function write_profiles(fn, profiles::MeanProfiles, t, it)

    # NOTE: calls to "Gatherv" need an array of Int32s
    nz_h, nz_v = Int32(length(profiles.u)), Int32(length(profiles.w))
    counts_h = MPI.Initialized() ? MPI.Allgather(nz_h, MPI.COMM_WORLD) : Int32[nz_h]
    counts_v = MPI.Initialized() ? MPI.Allgather(nz_v, MPI.COMM_WORLD) : Int32[nz_v]

    gather_profile(p, c) = MPI.Initialized() ? MPI.Gatherv(p, c, 0, MPI.COMM_WORLD) : p

    global_profiles = Dict(
        "u"   => gather_profile(profiles.u,   counts_h),
        "v"   => gather_profile(profiles.v,   counts_h),
        "w"   => gather_profile(profiles.w,   counts_v),
        "uu"  => gather_profile(profiles.uu,  counts_h),
        "vv"  => gather_profile(profiles.vv,  counts_h),
        "ww"  => gather_profile(profiles.ww,  counts_v),
        "uv"  => gather_profile(profiles.uv,  counts_h),
        "uwa" => gather_profile(profiles.uwa, counts_h),
        "uwb" => gather_profile(profiles.uwb, counts_h),
        "vwa" => gather_profile(profiles.vwa, counts_h),
        "vwb" => gather_profile(profiles.vwb, counts_h),
    )

    interval_t  = (profiles.start_t[], t)
    interval_it = (profiles.start_it[] + 1, it)

    if proc_type() <: LowestProc

        # only normalize profiles on process writing them
        for p in values(global_profiles)
            p[:] /= profiles.counter[]
        end

        mkpath(dirname(fn))
        open(fn, "w") do f
            JSON.print(f, Dict(
                "simulation_time" => interval_t,
                "simulation_steps" => interval_it,
                "mean_profiles" => global_profiles,
                ))
        end
    end
end

function reset_profiles!(profiles::MeanProfiles, t, it)
    for p in (profiles.u, profiles.v, profiles.w,
            profiles.uu, profiles.vv, profiles.ww, profiles.uv,
            profiles.uwa, profiles.uwb, profiles.vwa, profiles.vwb)
        p[:] .= 0
    end
    profiles.start_t[] = t
    profiles.start_it[] = it
    profiles.counter[] = 0
    profiles
end

function log_profiles!(profiles::MeanProfiles, vel, lower_bcs, upper_bcs, t, it)
    save_profiles!(profiles, vel, lower_bcs, upper_bcs)
    if profiles.counter[] == profiles.write_frequency
        profiles.write_count[] += 1
        pad = ndigits(profiles.write_maxcount)
        fn = "profiles-" * string(profiles.write_count[], pad = pad) * ".json"
        write_profiles(joinpath(profiles.write_path, fn), profiles, t, it)
        reset_profiles!(profiles, t, it)
    end
end
