function test_advection_exact(NZ)

    # we can test the contributions from u*du/dy, u*du/dz, v*dv/dx, v*dv/dz
    # exactly with functions that are linear in the vertical (necessary for
    # interpolation) and trigonometric functions in the horizontal, up to the
    # highest frequency that can be represented. with dealiasing, even that
    # frequency should be represented exactly. for the vertical velocity, there
    # is a complication in that the boundary conditions w(z=0)=w(z=top)=0 do not
    # allow linear functions other than w(z)=0. we can still test the horizontal
    # variation as long as we skip the highest and lowest H-nodes, since those
    # rely on interpolations using the boundary condition.

    T = Float64

    gd = CF.DistributedGrid(12, 14, NZ)
    ht = CF.HorizontalTransform(T, gd)
    ds = (2*π, 2*π, 1.0)
    gs = ds ./ (gd.nx_pd, gd.ny_pd, gd.nz_global)
    df = CF.DerivativeFactors(gd, ds)

    x = LinRange(0, 2π, 13)[1:12]
    y = LinRange(0, 2π, 15)[1:14]
    z_h = 1/(2*NZ):1/NZ:1-1/(2*NZ)
    z_v = 1/NZ:1/NZ:1-1/NZ

    lbcs = CF.bc_noslip(T, gd)
    ubcs = CF.bc_noslip(T, gd)

    vel = (CF.zeros_fd(T, gd, CF.NodeSet(:H)),
           CF.zeros_fd(T, gd, CF.NodeSet(:H)),
           CF.zeros_fd(T, gd, CF.NodeSet(:V)))
    rhs = (CF.zeros_fd(T, gd, CF.NodeSet(:H)),
           CF.zeros_fd(T, gd, CF.NodeSet(:H)),
           CF.zeros_fd(T, gd, CF.NodeSet(:V)))
    b = CF.AdvectionBuffers(T, gd)
    advh = CF.zeros_fd(T, gd, CF.NodeSet(:H))
    advv = CF.zeros_fd(T, gd, CF.NodeSet(:V))
    u, v, w = vel

    Random.seed!(142645812) # same seed for each process

    for Nk = 6:7 # with Ny=14, ky=±6 is the highest frequency that can be represented
        u0z = rand(2)
        u0s = rand(Nk)
        u0c = rand(Nk)
        u0(x, y, z) = u0z[2] * z + u0z[1] * (1 - z) + sum(u0s[i] * sin(i * y) + u0c[i] * cos(i * y) for i=1:Nk)
        du0(x, y, z) = sum(u0s[i] * i * cos(i * y) - u0c[i] * i * sin(i * y) for i=1:Nk)
        CF.set_field!(u, ht, u0, gs, gd.iz_min, CF.NodeSet(:H))
        v .= 0
        w .= 0
        CF.set_advection!(rhs, vel, df, ht, lbcs, ubcs, b)
        CF.set_field!(advh, ht, (x,y,z) -> du0(x,y,z) * u0(x,y,z), gs, gd.iz_min, CF.NodeSet(:H))
        @test rhs[1] .+ 1 ≈ ones(eltype(u), size(u))
        if Nk == 6
            @test rhs[2] ≈ advh
        else
            @test !(rhs[2] ≈ advh)
        end
        CF.set_field!(advv, ht, (x,y,z) -> (u0z[2] - u0z[1]) * u0(x,y,z), gs, gd.iz_min, CF.NodeSet(:V))
        @test rhs[3] ≈ advv
    end

    for Nk = 5:6 # with Nx=12, kx=±5 is the highest frequency that can be represented
        v0z = rand(2)
        v0s = rand(Nk)
        v0c = rand(Nk)
        v0(x, y, z) = v0z[2] * z + v0z[1] * (1 - z) + sum(v0s[i] * sin(i * x) + v0c[i] * cos(i * x) for i=1:Nk)
        dv0(x, y, z) = sum(v0s[i] * i * cos(i * x) - v0c[i] * i * sin(i * x) for i=1:Nk)
        CF.set_field!(v, ht, v0, gs, gd.iz_min, CF.NodeSet(:H))
        u .= 0
        w .= 0
        CF.set_advection!(rhs, vel, df, ht, lbcs, ubcs, b)
        CF.set_field!(advh, ht, (x,y,z) -> dv0(x,y,z) * v0(x,y,z), gs, gd.iz_min, CF.NodeSet(:H))
        if Nk == 5
            @test rhs[1] ≈ advh
        else
            @test !(rhs[1] ≈ advh)
        end
        @test rhs[2] .+ 1 ≈ ones(eltype(v), size(v))
        CF.set_field!(advv, ht, (x,y,z) -> (v0z[2] - v0z[1]) * v0(x,y,z), gs, gd.iz_min, CF.NodeSet(:V))
        @test rhs[3] ≈ advv
    end

    for Nk = 5:6 # with Nx=12, kx=±5 is the highest frequency that can be represented
        w0s = rand(Nk)
        w0c = rand(Nk)
        w0(x, y, z) = sum(w0s[i] * sin(i * x) + w0c[i] * cos(i * x) for i=1:Nk)
        dw0(x, y, z) = sum(w0s[i] * i * cos(i * x) - w0c[i] * i * sin(i * x) for i=1:Nk)
        CF.set_field!(w, ht, w0, gs, gd.iz_min, CF.NodeSet(:V))
        u .= 0
        v .= 0
        CF.set_advection!(rhs, vel, df, ht, lbcs, ubcs, b)
        CF.set_field!(advh, ht, (x,y,z) -> dw0(x,y,z) * w0(x,y,z), gs, gd.iz_min, CF.NodeSet(:H))
        @test rhs[2] .+ 1 ≈ ones(eltype(u), size(u))
        @test rhs[3] .+ 1 ≈ ones(eltype(w), size(w))
        P = CF.proc_type()
        iz = (P <: CF.LowestProc ? 2 : 1):(P <: CF.HighestProc ? size(u,3)-1 : size(u,3))
        if Nk == 5
            @test rhs[1][:,:,iz] ≈ advh[:,:,iz]
        else
            # do not test if all iz-indices are skipped
            @test length(iz) == 0 ? true : !(rhs[1][:,:,iz] ≈ advh[:,:,iz])
        end
    end

    for Nk = 6:7 # with Ny=14, ky=±6 is the highest frequency that can be represented
        w0s = rand(Nk)
        w0c = rand(Nk)
        w0(x, y, z) = sum(w0s[i] * sin(i * y) + w0c[i] * cos(i * y) for i=1:Nk)
        dw0(x, y, z) = sum(w0s[i] * i * cos(i * y) - w0c[i] * i * sin(i * y) for i=1:Nk)
        CF.set_field!(w, ht, w0, gs, gd.iz_min, CF.NodeSet(:V))
        u .= 0
        v .= 0
        CF.set_advection!(rhs, vel, df, ht, lbcs, ubcs, b)
        CF.set_field!(advh, ht, (x,y,z) -> dw0(x,y,z) * w0(x,y,z), gs, gd.iz_min, CF.NodeSet(:H))
        @test rhs[1] .+ 1 ≈ ones(eltype(u), size(u))
        @test rhs[3] .+ 1 ≈ ones(eltype(w), size(w))
        P = CF.proc_type()
        iz = (P <: CF.LowestProc ? 2 : 1):(P <: CF.HighestProc ? size(u,3)-1 : size(u,3))
        if Nk == 6
            @test rhs[2][:,:,iz] ≈ advh[:,:,iz]
        else
            # do not test if all iz-indices are skipped
            @test length(iz) == 0 ? true : !(rhs[2][:,:,iz] ≈ advh[:,:,iz])
        end
    end
end

function advection_error_convergence(Nh, Nv)

    T = Float64
    ds = (2*π, 2*π, 1.0)

    # define horizontal function, periodic in [0, 2π]
    fh(x) = 2^sin(x) # periodic in [0, 2π]
    dfh(x) = 2^sin(x) * log(2) * cos(x)

    # define vertical function, zero at 0 & 1
    fv(x) = cos(π*x-π/2)
    dfv(x) = -π*sin(π*x-π/2)

    # define velocity fields based on these functions
    Random.seed!(406595721) # same seed for each process
    ucoeff = rand(3)
    vcoeff = rand(3)
    wcoeff = rand(3)
    u0(x, y, z) = (ucoeff[1] * fh(x) + ucoeff[2] * fh(y) + ucoeff[3] * fh(x) * fh(y)) * fv(z)
    v0(x, y, z) = (vcoeff[1] * fh(x) + vcoeff[2] * fh(y) + vcoeff[3] * fh(x) * fh(y)) * fv(z)
    w0(x, y, z) = (wcoeff[1] * fh(x) + wcoeff[2] * fh(y) + wcoeff[3] * fh(x) * fh(y)) * fv(z)

    # define functions for exact solution of advection term
    dudy(x, y, z) = (ucoeff[2] * dfh(y) + ucoeff[3] * fh(x) * dfh(y)) * fv(z)
    dudz(x, y, z) = (ucoeff[1] *  fh(x) + ucoeff[2] * fh(y) + ucoeff[3] * fh(x) * fh(y)) * dfv(z)
    dvdx(x, y, z) = (vcoeff[1] * dfh(x) + vcoeff[3] * dfh(x) * fh(y)) * fv(z)
    dvdz(x, y, z) = (vcoeff[1] *  fh(x) + vcoeff[2] * fh(y) + vcoeff[3] * fh(x) * fh(y)) * dfv(z)
    dwdx(x, y, z) = (wcoeff[1] * dfh(x) + wcoeff[3] * dfh(x) * fh(y)) * fv(z)
    dwdy(x, y, z) = (wcoeff[2] * dfh(y) + wcoeff[3] * fh(x) * dfh(y)) * fv(z)
    rotx(x, y, z) = dwdy(x, y, z) - dvdz(x, y, z)
    roty(x, y, z) = dudz(x, y, z) - dwdx(x, y, z)
    rotz(x, y, z) = dvdx(x, y, z) - dudy(x, y, z)
    advx(x, y, z) = - roty(x, y, z) * w0(x, y, z) + rotz(x, y, z) * v0(x, y, z)
    advy(x, y, z) = - rotz(x, y, z) * u0(x, y, z) + rotx(x, y, z) * w0(x, y, z)
    advz(x, y, z) = - rotx(x, y, z) * v0(x, y, z) + roty(x, y, z) * u0(x, y, z)


    function get_errors(Nx, Ny, Nz)

        gd = CF.DistributedGrid(Nx, Ny, Nz)
        ht = CF.HorizontalTransform(T, gd)
        gs = ds ./ (gd.nx_pd, gd.ny_pd, gd.nz_global)
        df = CF.DerivativeFactors(gd, ds)
        ab = CF.AdvectionBuffers(T, gd)

        lbcs = CF.bc_noslip(T, gd)
        ubcs = CF.bc_noslip(T, gd)

        vel = (CF.zeros_fd(T, gd, CF.NodeSet(:H)),
               CF.zeros_fd(T, gd, CF.NodeSet(:H)),
               CF.zeros_fd(T, gd, CF.NodeSet(:V)))
        rhs = (CF.zeros_fd(T, gd, CF.NodeSet(:H)),
               CF.zeros_fd(T, gd, CF.NodeSet(:H)),
               CF.zeros_fd(T, gd, CF.NodeSet(:V)))
        adv = (CF.zeros_fd(T, gd, CF.NodeSet(:H)),
               CF.zeros_fd(T, gd, CF.NodeSet(:H)),
               CF.zeros_fd(T, gd, CF.NodeSet(:V)))

        CF.set_field!(vel[1], ht, u0, gs, gd.iz_min, CF.NodeSet(:H))
        CF.set_field!(vel[2], ht, v0, gs, gd.iz_min, CF.NodeSet(:H))
        CF.set_field!(vel[3], ht, w0, gs, gd.iz_min, CF.NodeSet(:V))
        CF.set_advection!(rhs, vel, df, ht, lbcs, ubcs, ab)

        CF.set_field!(adv[1], ht, advx, gs, gd.iz_min, CF.NodeSet(:H))
        CF.set_field!(adv[2], ht, advy, gs, gd.iz_min, CF.NodeSet(:H))
        CF.set_field!(adv[3], ht, advz, gs, gd.iz_min, CF.NodeSet(:V))

        ε1 = CF.global_maximum(abs.(rhs[1] .- adv[1]))
        ε2 = CF.global_maximum(abs.(rhs[2] .- adv[2]))
        ε3 = CF.global_maximum(abs.(rhs[3] .- adv[3]))

        ε1, ε2, ε3
    end

    εx = zeros(length(Nh), 3)
    εy = zeros(length(Nh), 3)
    εz = zeros(length(Nv), 3)

    for i=1:length(Nh)-1
        εx[i,:] .= get_errors(Nh[i], Nh[end], Nv[end])
        εy[i,:] .= get_errors(Nh[end], Nh[i], Nv[end])
    end

    for i=1:length(Nv)-1
        εz[i,:] .= get_errors(Nh[end], Nh[end], Nv[i])
    end

    ε  = get_errors(Nh[end], Nh[end], Nv[end])
    εx[end, :] .= ε
    εy[end, :] .= ε
    εz[end, :] .= ε

    εx, εy, εz
end

function test_advection_convergence(Nz_min)

    Nh = [2*n for n=1:5]
    Nv = [2^n for n=ceil(Int, log2(Nz_min)):8]

    ε1, ε2, ε3 = advection_error_convergence(Nh, Nv)

    for i=1:3
        test_convergence(Nh[1:end-1], ε1[1:end-1,i], exponential=true)
        test_convergence(Nh[1:end-1], ε2[1:end-1,i], exponential=true)
        test_convergence(Nv, ε3[:,i], order=2)
    end

    Nh, Nv, ε1, ε2, ε3
end

test_advection_exact(16)

# also test the parallel version with one layer per process
MPI.Initialized() && test_advection_exact(MPI.Comm_size(MPI.COMM_WORLD))
test_advection_convergence(MPI.Initialized() ? MPI.Comm_size(MPI.COMM_WORLD) : 8)
