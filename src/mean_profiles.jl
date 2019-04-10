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
    uxux::Array{T,1}
    vxvx::Array{T,1}
    wxwx::Array{T,1}
    uyuy::Array{T,1}
    vyvy::Array{T,1}
    wywy::Array{T,1}
    uzuz::Array{T,1}
    vzvz::Array{T,1}
    wzwz::Array{T,1}

    MeanProfiles(T, gd::DistributedGrid, path::String, frequency::Int, total_count::Int) =
        new{T}(path, frequency, Ref(0), total_count, Ref(zero(T)), Ref(0), Ref(0),
        zeros(T, gd.nz_h), zeros(T, gd.nz_h), zeros(T, gd.nz_v), # u, v, w
        zeros(T, gd.nz_h), zeros(T, gd.nz_h), zeros(T, gd.nz_v), zeros(gd.nz_h), # uu, vv, ww, uv
        zeros(T, gd.nz_h), zeros(T, gd.nz_h), zeros(T, gd.nz_h), zeros(gd.nz_h), # uwa, uwb, vwa, vwb
        zeros(T, gd.nz_h), zeros(T, gd.nz_h), zeros(T, gd.nz_v), # uxux, vxvx, wxwx
        zeros(T, gd.nz_h), zeros(T, gd.nz_h), zeros(T, gd.nz_v), # uyuy, vyvy, wywy
        zeros(T, gd.nz_v), zeros(T, gd.nz_v), zeros(T, gd.nz_h), # uzuz, vzvz, wzwz
        )
end

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

# NOTE: We can compute products directly from Fourier coefficients since we
# only need the (0,0) wavenumber of the product. The contributions to this are
# the products vel1(kx,ky)*vel2(-kx,-ky) or equivalently (from the properties
# of DFTs of real-valued data) vel1(kx,ky)*conj(vel2(kx,ky)). However, we have
# to account for the compact representation of these coefficients, i.e. we have
# to count the values for kx=1..kx_max twice since the negative kx-frequencies
# are not saved explicitly (because they are just the complex conjugate of the
# frequencies with opposite sign).
avg_from_fd(vel::AbstractArray{Complex{T},2}) where T = real(vel[1,1])
avg_sq_from_fd(vel::AbstractArray{Complex{T},2}) where T =
        @views mapreduce(abs2, +, vel[1,:]) + 2 * mapreduce(abs2, +, vel[2:end,:])
avg_prod_from_fd(vel1::AbstractArray{Complex{T},2}, vel2::AbstractArray{Complex{T},2}) where T =
        (prod2((z1, z2)) = real(z1) * real(z2) + imag(z1) * imag(z2);
        @views mapreduce(prod2, +, zip(vel1[1,:], vel2[1,:])) +
           2 * mapreduce(prod2, +, zip(vel1[2:end,:], vel2[2:end,:])))
avg_prod_from_fd(vel::AbstractArray{Complex{T},2}, bc::DirichletBC) where T =
        bc.value * real(vel[1,1])

function avg_sq_dx_from_fd(vel::AbstractArray{Complex{T},2}, dx) where T
    sum = zero(T)
    for j = 1:size(vel,2)
        sum += abs2(vel[1,j] * dx[1]) # dx[i=1] should be zero
        for i = 2:size(vel,1)
            sum += 2 * abs2(vel[i,j] * dx[i])
        end
    end
    sum
end
function avg_sq_dy_from_fd(vel::AbstractArray{Complex{T},2}, dy) where T
    sum = zero(T)
    for j = 1:size(vel,2) # dy[j=1] should be zero
        sum += abs2(vel[1,j] * dy[j])
        for i = 2:size(vel,1)
            sum += 2 * abs2(vel[i,j] * dy[j])
        end
    end
    sum
end
function avg_sq_dz_from_fd(vel_below::AbstractArray{Complex{T},2},
                           vel_above::AbstractArray{Complex{T},2}, dz) where T
    sum = zero(T)
    for j = 1:size(vel_below,2)
        sum += abs2((vel_above[1,j] - vel_below[1,j]) * dz)
        for i = 2:size(vel_below,1)
            sum += 2 * abs2((vel_above[i,j] - vel_below[i,j]) * dz)
        end
    end
    sum
end
avg_sq_dz_from_fd(bc_below::DirichletBC, vel_above::AbstractArray{Complex{T},2}, dz) where T =
        abs2((vel_above[1,1] - bc_below.value) * dz)
avg_sq_dz_from_fd(vel_below::AbstractArray{Complex{T},2}, bc_above::DirichletBC, dz) where T =
        abs2((bc_above.value - vel_below[1,1]) * dz)

function save_profiles!(profiles::MeanProfiles, vel, lower_bcs, upper_bcs, df)
    ua = hlayers_above(layers(vel[1]), upper_bcs[1])
    ub = hlayers_below(layers(vel[1]), lower_bcs[1])
    va = hlayers_above(layers(vel[2]), upper_bcs[2])
    vb = hlayers_below(layers(vel[2]), lower_bcs[2])
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
    profiles.uxux[:] .+= avg_sq_dx_from_fd.(layers(vel[1]), (df.dx1,))
    profiles.vxvx[:] .+= avg_sq_dx_from_fd.(layers(vel[2]), (df.dx1,))
    profiles.wxwx[:] .+= avg_sq_dx_from_fd.(layers(vel[3]), (df.dx1,))
    profiles.uyuy[:] .+= avg_sq_dy_from_fd.(layers(vel[1]), (df.dy1,))
    profiles.vyvy[:] .+= avg_sq_dy_from_fd.(layers(vel[2]), (df.dy1,))
    profiles.wywy[:] .+= avg_sq_dy_from_fd.(layers(vel[3]), (df.dy1,))
    profiles.uzuz[:] .+= avg_sq_dz_from_fd.(ub, ua, (df.dz1,))
    profiles.vzvz[:] .+= avg_sq_dz_from_fd.(vb, va, (df.dz1,))
    profiles.wzwz[:] .+= avg_sq_dz_from_fd.(wb, wa, (df.dz1,))
    profiles.counter[] += 1
end

function write_profiles(fn, profiles::MeanProfiles, t, it)

    # NOTE: calls to "Gatherv" need an array of Int32s
    nz_h, nz_v = Int32(length(profiles.u)), Int32(length(profiles.w))
    counts_h = MPI.Initialized() ? MPI.Allgather(nz_h, MPI.COMM_WORLD) : Int32[nz_h]
    counts_v = MPI.Initialized() ? MPI.Allgather(nz_v, MPI.COMM_WORLD) : Int32[nz_v]

    gather_profile(p, c) = MPI.Initialized() ? MPI.Gatherv(p, c, 0, MPI.COMM_WORLD) : p

    global_profiles = Dict(
        "u"    => gather_profile(profiles.u,    counts_h),
        "v"    => gather_profile(profiles.v,    counts_h),
        "w"    => gather_profile(profiles.w,    counts_v),
        "uu"   => gather_profile(profiles.uu,   counts_h),
        "vv"   => gather_profile(profiles.vv,   counts_h),
        "ww"   => gather_profile(profiles.ww,   counts_v),
        "uv"   => gather_profile(profiles.uv,   counts_h),
        "uwa"  => gather_profile(profiles.uwa,  counts_h),
        "uwb"  => gather_profile(profiles.uwb,  counts_h),
        "vwa"  => gather_profile(profiles.vwa,  counts_h),
        "vwb"  => gather_profile(profiles.vwb,  counts_h),
        "uxux" => gather_profile(profiles.uxux, counts_h),
        "vxvx" => gather_profile(profiles.vxvx, counts_h),
        "wxwx" => gather_profile(profiles.wxwx, counts_v),
        "uyuy" => gather_profile(profiles.uyuy, counts_h),
        "vyvy" => gather_profile(profiles.vyvy, counts_h),
        "wywy" => gather_profile(profiles.wywy, counts_v),
        "uzuz" => gather_profile(profiles.uzuz, counts_v),
        "vzvz" => gather_profile(profiles.vzvz, counts_v),
        "wzwz" => gather_profile(profiles.wzwz, counts_h),
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
            profiles.uwa, profiles.uwb, profiles.vwa, profiles.vwb,
            profiles.uxux, profiles.vxvx, profiles.wxwx,
            profiles.uyuy, profiles.vyvy, profiles.wywy,
            profiles.uzuz, profiles.vzvz, profiles.wzwz)
        p[:] .= 0
    end
    profiles.start_t[] = t
    profiles.start_it[] = it
    profiles.counter[] = 0
    profiles
end

function log_profiles!(profiles::MeanProfiles, vel, lower_bcs, upper_bcs, derivatives, t, it)
    save_profiles!(profiles, vel, lower_bcs, upper_bcs, derivatives)
    if profiles.counter[] == profiles.write_frequency
        profiles.write_count[] += 1
        pad = ndigits(profiles.write_maxcount)
        fn = "profiles-" * string(profiles.write_count[], pad = pad) * ".json"
        write_profiles(joinpath(profiles.write_path, fn), profiles, t, it)
        reset_profiles!(profiles, t, it)
    end
end
