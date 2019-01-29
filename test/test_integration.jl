"""
Analytical solution for transient Poiseuille flow. The solution is normalized
as poiseuille(y/δ, tν/δ²) = uν/Gδ², where G is the pressure gradient -dp/dx.
"""
function poiseuille(y, t; kmax=100_000)
    t == 0 && return 0 .* y # t=0 would require many modes, but u(0) = 0
    u = (y .- 0.5 * y.^2) # initialize with steady-state solution
    t == Inf && return u
    for k=1:2:kmax # add modes until the contribution is negligible
        k_factor = 16/(k^3*π^3) * exp(-0.25*k^2*π^2*t)
        8*k_factor < eps() && break # 8 is a safety factor
        u -= k_factor * sin.(0.5*k*π*y)
    end
    u
end

"""
Compute the maximum relative error of a Poiseuille flow at time T with Nv
vertical grid points. The constant time step is computed such that the CFL is
respected based on the viscous velocity ν/dz.
"""
function poiseuille_error(T, Nt, Nv, Nh, Re)
    uref = poiseuille(LinRange(0, 2, 2*Nv+1)[2:2:end-1], T)
    cf = closed_channel((Nh,Nh,Nv), Re)
    integrate!(cf, T/Nt, Nt, verbose=false)
    u = global_vector(real(cf.velocity[1][1,1,:]))
    maximum(abs.((u .- uref) ./ uref)) # max. relative error
end

"""
Compute the number of steps necessary to integrate up to time T based on the
vertical grid spacing and the viscous stability criterion.
"""
Nt_viscous(T, dz, CFL, Re) = ceil(Int, T / (CFL * dz^2 * Re)) # dt = CFL dz²/ν

"""
Test the integration error of a transient Poiseuille flow. The default tolerance
is computed based on the theoretical order of convergence O(dz²) and a constant
factor that should work for Nv≥4 based on earlier convergence tests.
"""
test_poiseuille(Nv; Nh = 4, Re = 1.0, CFL = 0.5, T = 1/Re, tol=Nv^(-2)/4) =
    @test poiseuille_error(T, Nt_viscous(T, 2/Nv, CFL, Re), Nv, Nh, Re) < tol

"""
Test the order of convergence of a transient Poiseuille flow is a least O(dz²).
The vertical resolutions are given as powers of 2.
"""
function test_poiseuille_convergence(logNv; Nh = 4, Re = 1.0, CFL = 0.5, T = 1/Re)
    Nv = [2^logN for logN=logNv]
    Nt = Nt_viscous(T, 2/Nv[end], CFL, Re)
    ε = [poiseuille_error(T, Nt, Nv, Nh, Re) for Nv=Nv]
    test_convergence(Nv, ε, order=2)
end

# test the correct integration of a poiseuille flow at t=1 for Nz=16
# (also test the parallel version with one layer per process)
test_poiseuille(16)
MPI.Initialized() && test_poiseuille(MPI.Comm_size(MPI.COMM_WORLD))

# test order of convergence for Nz=16,32,64,128,256 is at least 2
test_poiseuille_convergence(4:8, T=0.01)
