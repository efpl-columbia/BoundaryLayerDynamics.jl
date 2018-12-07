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
function poiseuille_error(T, Nv, Nh, Re, CFL)
    uref = poiseuille(LinRange(0, 2, 2*Nv+1)[2:2:end-1], T)
    cf = closed_channel((Nh,Nh,Nv), Re)
    Nt = ceil(Int, T * Nv^2 / (CFL * 4 * Re)) # dt = CFL dz²/ν = CFL (2/Nv)² Re
    integrate!(cf, T/Nt, Nt, verbose=false)
    u = global_vector(real(cf.velocity[1][1,1,:]))
    maximum(abs.((u .- uref) ./ uref)) # max. relative error
end

"""
Test the integration error of a transient Poiseuille flow. The default tolerance
is computed based on the theoretical order of convergence O(dz²) and a constant
factor that should work for Nv≥4 based on earlier convergence tests.
"""
test_poiseuille(Nv; Nh = 8, Re = 1.0, CFL = 0.25, T = 1/Re, tol=Nv^(-2)/3.8) =
    @test poiseuille_error(T, Nv, Nh, Re, CFL) < tol

"""
Test the order of convergence of a transient Poiseuille flow is a least O(dz²).
The vertical resolutions are given as powers of 2.
"""
function test_poiseuille_convergence(logNv; Nh = 8, Re = 1.0, CFL = 0.25, T = 1/Re)
    ε = [poiseuille_error(T, 2^logN, Nh, Re, CFL) for logN=logNv]
    cov(x,y) = (N = length(x); N == length(y) || error("Length does not match");
            μx = sum(x)/N; μy = sum(y)/N; sum((x.-μx) .* (y.-μy)) ./ (N-1))
    order = - cov(logNv, log2.(ε)) / cov(logNv, logNv)
    @test order > 2
end

# test the correct integration of a poiseuille flow at t=1 for Nz=16
# (also test the parallel version with one layer per process)
test_poiseuille(16)
MPI.Initialized() && test_poiseuille(MPI.Comm_size(MPI.COMM_WORLD))

# test order of convergence for Nz=4,8,16,32 is at least 2
test_poiseuille_convergence(2:5)
