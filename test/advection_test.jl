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

    dims = (12, 14, NZ)
    domain = Domain((2*π, 2*π, 1.0), SmoothWall(), SmoothWall())
    abl = DiscretizedABL(dims, domain, [MomentumAdvection()])
    pddims = ABL.PhysicalSpace.pdsize(abl.grid, :quadratic)

    x1 = LinRange(0, 2π, 13)[1:12]
    x2 = LinRange(0, 2π, 15)[1:14]
    x3c = 1/(2*NZ):1/NZ:1-1/(2*NZ)
    x3i = 1/NZ:1/NZ:1-1/NZ

    rhs = deepcopy(abl.state)
    adv = deepcopy(abl.state)

    #u, v, w = vel

    Random.seed!(142645812) # same seed for each process

    # test u1(x2, x3)
    for Nk = 6:7 # with Ny=14, ky=±6 is the highest frequency that can be represented

        # define exact velocity field
        u0z = rand(2)
        u0s = rand(Nk)
        u0c = rand(Nk)
        u0(x, y, z) = u0z[2] * z + u0z[1] * (1 - z) + sum(u0s[i] * sin(i * y) + u0c[i] * cos(i * y) for i=1:Nk)
        du0(x, y, z) = sum(u0s[i] * i * cos(i * y) - u0c[i] * i * sin(i * y) for i=1:Nk)

        # compute numerical and exact solution
        initialize!(abl, vel1 = u0,  vel2 = 0, vel3 = 0)
        ABL.Processes.rate!(rhs, abl.state, 0.0, abl.processes, abl.physical_spaces)
        ABL.set_field!((x,y,z) -> du0(x,y,z) * u0(x,y,z), adv[:vel2],
                       abl.physical_spaces[pddims].transform, abl.domain, abl.grid, ABL.nodes(:vel2))
        ABL.set_field!((x,y,z) -> (u0z[2] - u0z[1]) * u0(x,y,z), adv[:vel3],
                       abl.physical_spaces[pddims].transform, abl.domain, abl.grid, ABL.nodes(:vel3))

        # compare solutions
        @test rhs.vel1 .+ 1 ≈ ones(eltype(rhs.vel1), size(rhs.vel1))
        if Nk == 6
            @test rhs.vel2 ≈ adv.vel2
        else
            @test !(rhs.vel2 ≈ adv.vel2)
        end
        @test rhs[3] ≈ adv.vel3
    end

    # test u2(x1, x3)
    for Nk = 5:6 # with Nx=12, kx=±5 is the highest frequency that can be represented

        # define exact velocity field
        v0z = rand(2)
        v0s = rand(Nk)
        v0c = rand(Nk)
        v0(x, y, z) = v0z[2] * z + v0z[1] * (1 - z) + sum(v0s[i] * sin(i * x) + v0c[i] * cos(i * x) for i=1:Nk)
        dv0(x, y, z) = sum(v0s[i] * i * cos(i * x) - v0c[i] * i * sin(i * x) for i=1:Nk)

        # compute numerical and exact solution
        initialize!(abl, vel1 = 0, vel2 = v0, vel3 = 0)
        ABL.Processes.rate!(rhs, abl.state, 0.0, abl.processes, abl.physical_spaces)
        ABL.set_field!((x,y,z) -> dv0(x,y,z) * v0(x,y,z), adv[:vel1],
                       abl.physical_spaces[pddims].transform, abl.domain, abl.grid, ABL.nodes(:vel1))
        ABL.set_field!((x,y,z) -> (v0z[2] - v0z[1]) * v0(x,y,z), adv[:vel3],
                       abl.physical_spaces[pddims].transform, abl.domain, abl.grid, ABL.nodes(:vel3))

        # compare solutions
        if Nk == 5
            @test rhs.vel1 ≈ adv.vel1
        else
            @test !(rhs.vel1 ≈ adv.vel1)
        end
        @test rhs.vel2 .+ 1 ≈ ones(eltype(rhs.vel2), size(rhs.vel2))
        @test rhs.vel3 ≈ adv.vel3
    end

    # test u3(x1)
    for Nk = 5:6 # with Nx=12, kx=±5 is the highest frequency that can be represented

        # define exact velocity field
        w0s = rand(Nk)
        w0c = rand(Nk)
        w0(x, y, z) = sum(w0s[i] * sin(i * x) + w0c[i] * cos(i * x) for i=1:Nk)
        dw0(x, y, z) = sum(w0s[i] * i * cos(i * x) - w0c[i] * i * sin(i * x) for i=1:Nk)

        # compute numerical and exact solution
        initialize!(abl, vel1 = 0, vel2 = 0, vel3 = w0)
        ABL.Processes.rate!(rhs, abl.state, 0.0, abl.processes, abl.physical_spaces)
        ABL.set_field!((x,y,z) -> dw0(x,y,z) * w0(x,y,z), adv[:vel1],
                       abl.physical_spaces[pddims].transform, abl.domain, abl.grid, ABL.nodes(:vel1))

        # boundary conditions do not match exact solution, so skip the border
        # layers in test
        i3 = (abl.grid.i3min == 1 ? 2 : 1):(size(rhs.vel1, 3) - (abl.grid.i3max == dims[3] ? 1 : 0))
        if Nk == 5
            @test rhs.vel1[:,:,i3] ≈ adv.vel1[:,:,i3]
        else
            # do not test if all i3-indices are skipped
            @test length(i3) == 0 ? true : !(rhs.vel1[:,:,i3] ≈ adv.vel1[:,:,i3])
        end
        @test rhs.vel2 .+ 1 ≈ ones(eltype(rhs.vel2), size(rhs.vel2))
        @test rhs.vel3 .+ 1 ≈ ones(eltype(rhs.vel3), size(rhs.vel3))
    end

    # test u3(x2)
    for Nk = 6:7 # with Ny=14, ky=±6 is the highest frequency that can be represented

        # define exact velocity field
        w0s = rand(Nk)
        w0c = rand(Nk)
        w0(x, y, z) = sum(w0s[i] * sin(i * y) + w0c[i] * cos(i * y) for i=1:Nk)
        dw0(x, y, z) = sum(w0s[i] * i * cos(i * y) - w0c[i] * i * sin(i * y) for i=1:Nk)

        # compute numerical and exact solution
        initialize!(abl, vel1 = 0, vel2 = 0, vel3 = w0)
        ABL.Processes.rate!(rhs, abl.state, 0.0, abl.processes, abl.physical_spaces)
        ABL.set_field!((x,y,z) -> dw0(x,y,z) * w0(x,y,z), adv[:vel2],
                       abl.physical_spaces[pddims].transform, abl.domain, abl.grid, ABL.nodes(:vel2))

        # boundary conditions do not match exact solution, so skip the border
        # layers in test
        i3 = (abl.grid.i3min == 1 ? 2 : 1):(size(rhs.vel2, 3) - (abl.grid.i3max == dims[3] ? 1 : 0))
        @test rhs.vel1 .+ 1 ≈ ones(eltype(rhs.vel1), size(rhs.vel1))
        if Nk == 6
            @test rhs.vel2[:,:,i3] ≈ adv.vel2[:,:,i3]
        else
            # do not test if all i3-indices are skipped
            @test length(i3) == 0 ? true : !(rhs.vel2[:,:,i3] ≈ adv.vel2[:,:,i3])
        end
        @test rhs.vel3 .+ 1 ≈ ones(eltype(rhs.vel3), size(rhs.vel3))
    end
end

function advection_error_convergence(Nh, Nv)

    domain = Domain((2*π, 2*π, 1.0), SmoothWall(), SmoothWall())
    processes = [MomentumAdvection()]

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
    adv1(x, y, z) = - roty(x, y, z) * w0(x, y, z) + rotz(x, y, z) * v0(x, y, z)
    adv2(x, y, z) = - rotz(x, y, z) * u0(x, y, z) + rotx(x, y, z) * w0(x, y, z)
    adv3(x, y, z) = - rotx(x, y, z) * v0(x, y, z) + roty(x, y, z) * u0(x, y, z)


    function get_errors(Nx, Ny, Nz)

        abl = DiscretizedABL((Nx, Ny, Nz), domain, processes)
        rhs = deepcopy(abl.state)

        # create arrays with exact solution (in frequency space)
        adv = deepcopy(abl.state)
        pddims = ABL.PhysicalSpace.pdsize(abl.grid, :quadratic)
        ABL.set_field!(adv1, adv[:vel1], abl.physical_spaces[pddims].transform,
                       abl.domain, abl.grid, ABL.nodes(:vel1))
        ABL.set_field!(adv2, adv[:vel2], abl.physical_spaces[pddims].transform,
                       abl.domain, abl.grid, ABL.nodes(:vel2))
        ABL.set_field!(adv3, adv[:vel3], abl.physical_spaces[pddims].transform,
                       abl.domain, abl.grid, ABL.nodes(:vel3))

        # set up velocity field and compute advection term
        initialize!(abl, vel1 = u0, vel2 = v0, vel3 = w0)
        ABL.Processes.rate!(rhs, abl.state, 0.0, abl.processes, abl.physical_spaces)

        # measure error in frequency space
        ε1 = global_maximum(abs.(rhs[:vel1] .- adv[:vel1]))
        ε2 = global_maximum(abs.(rhs[:vel2] .- adv[:vel2]))
        ε3 = global_maximum(abs.(rhs[:vel3] .- adv[:vel3]))

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
        #test_convergence(Nh[1:end-1], ε1[1:end-1,i], exponential=true) # TODO: check convergence
        #test_convergence(Nh[1:end-1], ε2[1:end-1,i], exponential=true) # TODO: check convergence
        test_convergence(Nv, ε3[:,i], order=2)
    end

    Nh, Nv, ε1, ε2, ε3
end

test_advection_exact(16)

# also test the parallel version with one layer per process
MPI.Initialized() && test_advection_exact(MPI.Comm_size(MPI.COMM_WORLD))
test_advection_convergence(MPI.Initialized() ? MPI.Comm_size(MPI.COMM_WORLD) : 8)
