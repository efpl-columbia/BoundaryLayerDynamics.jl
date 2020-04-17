struct MeanProfiles{T}
    vel::NTuple{3,Array{T,1}}
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

    # TODO: reorganize these profiles with better names & structures
    MeanProfiles(T, n3h, n3v) = new{T}(
        (zeros(T, n3h), zeros(T, n3h), zeros(T, n3v)), # (u1, u2, u3)
        zeros(T, n3h), zeros(T, n3h), zeros(T, n3v), zeros(n3h), # uu, vv, ww, uv
        zeros(T, n3h), zeros(T, n3h), zeros(T, n3h), zeros(n3h), # uwa, uwb, vwa, vwb
        zeros(T, n3h), zeros(T, n3h), zeros(T, n3v), # uxux, vxvx, wxwx
        zeros(T, n3h), zeros(T, n3h), zeros(T, n3v), # uyuy, vyvy, wywy
        zeros(T, n3v), zeros(T, n3v), zeros(T, n3h), # uzuz, vzvz, wzwz
    )
end

function add_profiles!(profiles::MeanProfiles, vel, (u1b, u2b, u3b), (u1a, u2a, u3a), df)
    TimerOutputs.@timeit "save velocity" begin
        profiles.vel[1][:] .+= avg_from_fd.(layers(vel[1]))
        profiles.vel[2][:] .+= avg_from_fd.(layers(vel[2]))
        profiles.vel[3][:] .+= avg_from_fd.(layers(vel[3]))
    end
    TimerOutputs.@timeit "save kinetic energy" begin
        profiles.uu[:]  .+= avg_sq_from_fd.(layers(vel[1]))
        profiles.vv[:]  .+= avg_sq_from_fd.(layers(vel[2]))
        profiles.ww[:]  .+= avg_sq_from_fd.(layers(vel[3]))
    end
    TimerOutputs.@timeit "save velocity products" begin
        profiles.uv[:]  .+= avg_prod_from_fd.(layers(vel[1]), layers(vel[2]))
        profiles.uwa[:] .+= avg_prod_from_fd.(layers(vel[1]), u3a)
        profiles.uwb[:] .+= avg_prod_from_fd.(layers(vel[1]), u3b)
        profiles.vwa[:] .+= avg_prod_from_fd.(layers(vel[2]), u3a)
        profiles.vwb[:] .+= avg_prod_from_fd.(layers(vel[2]), u3b)
    end
    TimerOutputs.@timeit "save diffusion terms" begin
        profiles.uxux[:] .+= avg_sq_dx_from_fd.(layers(vel[1]), (df.dx1,))
        profiles.vxvx[:] .+= avg_sq_dx_from_fd.(layers(vel[2]), (df.dx1,))
        profiles.wxwx[:] .+= avg_sq_dx_from_fd.(layers(vel[3]), (df.dx1,))
        profiles.uyuy[:] .+= avg_sq_dy_from_fd.(layers(vel[1]), (df.dy1,))
        profiles.vyvy[:] .+= avg_sq_dy_from_fd.(layers(vel[2]), (df.dy1,))
        profiles.wywy[:] .+= avg_sq_dy_from_fd.(layers(vel[3]), (df.dy1,))
        profiles.uzuz[:] .+= avg_sq_dz_from_fd.(u1b, u1a, (df.dz1,))
        profiles.vzvz[:] .+= avg_sq_dz_from_fd.(u2b, u2a, (df.dz1,))
        profiles.wzwz[:] .+= avg_sq_dz_from_fd.(u3b, u3a, (df.dz1,))
    end
    profiles
end

function gather_profiles(profiles::MeanProfiles)

    # NOTE: calls to "Gatherv" need an array of Cints
    nz_h, nz_v = Cint(equivalently(length(profiles.vel[1]), length(profiles.vel[2]))),
                 Cint(length(profiles.vel[3]))
    counts_h = MPI.Initialized() ? MPI.Gather(nz_h, 0, MPI.COMM_WORLD) : Cint[nz_h]
    counts_v = MPI.Initialized() ? MPI.Gather(nz_v, 0, MPI.COMM_WORLD) : Cint[nz_v]

    gather_profile(p, c) = MPI.Initialized() ? MPI.Gatherv(p, c, 0, MPI.COMM_WORLD) : p

    Dict("u1"     => gather_profile(profiles.vel[1], counts_h),
         "u2"     => gather_profile(profiles.vel[2], counts_h),
         "u3"     => gather_profile(profiles.vel[3], counts_v),
         "u1u1"   => gather_profile(profiles.uu,     counts_h),
         "u2u2"   => gather_profile(profiles.vv,     counts_h),
         "u3u3"   => gather_profile(profiles.ww,     counts_v),
         "u1u2"   => gather_profile(profiles.uv,     counts_h),
         "u1u3a"  => gather_profile(profiles.uwa,    counts_h),
         "u1u3b"  => gather_profile(profiles.uwb,    counts_h),
         "u2u3a"  => gather_profile(profiles.vwa,    counts_h),
         "u2u3b"  => gather_profile(profiles.vwb,    counts_h),
         "u1x1sq" => gather_profile(profiles.uxux,   counts_h),
         "u2x1sq" => gather_profile(profiles.vxvx,   counts_h),
         "u3x1sq" => gather_profile(profiles.wxwx,   counts_v),
         "u1x2sq" => gather_profile(profiles.uyuy,   counts_h),
         "u2x2sq" => gather_profile(profiles.vyvy,   counts_h),
         "u3x2sq" => gather_profile(profiles.wywy,   counts_v),
         "u1x3sq" => gather_profile(profiles.uzuz,   counts_v),
         "u2x3sq" => gather_profile(profiles.vzvz,   counts_v),
         "u3x3sq" => gather_profile(profiles.wzwz,   counts_h),)
end

function reset!(profiles::MeanProfiles)
    for p in (profiles.vel...,
            profiles.uu, profiles.vv, profiles.ww, profiles.uv,
            profiles.uwa, profiles.uwb, profiles.vwa, profiles.vwb,
            profiles.uxux, profiles.vxvx, profiles.wxwx,
            profiles.uyuy, profiles.vyvy, profiles.wywy,
            profiles.uzuz, profiles.vzvz, profiles.wzwz)
        p[:] .= 0
    end
    profiles
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
