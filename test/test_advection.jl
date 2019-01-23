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

test_advection_exact(16)
# also test the parallel version with one layer per process
MPI.Initialized() && test_advection_exact(MPI.Comm_size(MPI.COMM_WORLD))
