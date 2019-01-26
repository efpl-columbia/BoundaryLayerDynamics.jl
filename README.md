# Direct Numerical Simulation of a Channel Flow

## Current State

The code has all parts to run a DNS with MPI, using any number of  layers per
process. Parts of the code have been tested thoroughly and the combined solver
is tested for laminar Poiseuille flow. For turbulent flows, the solver is poorly
tested so it is not even clear whether it will reliably compute a solution, and
if it does the solution might be very wrong.

### Next Steps

- compare timing in more challenging setting
- batch size: play in interactive trials, implement automatic search for best value
- improve handling of errors during MPI testing
- consider more complete handling of boundary conditions in advection term
- clean up code: reorganize files, remove inconsistencies, consider improvements
- add transient Couette flow to integration tests
- test DNS of turbulent channel flow

## Organization of Code & Data

Some notes on what data and variables are needed in the code and how they could
be organized. *This may be out of date!*

### Variables & Buffers

problem definition (channel setup/physics):
- size of domain (lx, ly, lz)
- boundary conditions u/v/w top & bottom (only Dirichlet for w is okay)
- pressure forcing
- Reynolds number

time integration:
- initial conditions: where to we start? what time does this correspond to? (usually t=0)
- time: how long do we integrate for
- stepping: constant step size (temporal solution method)

spatial solution method:
- discretization: number of horizontal frequencies, number of vertical points (frequencies are odd number, i.e. positive & negative frequencies 1 to kmax plus DC content k=0; in physical space we use 3*(kmax+1) points, according to 3/2 rule for dealiasing)
- approximation of derivatives: implicit & not adaptable (Fourier & finite differences)

problem state:
- 3 velocities fd
- pressure fd
- current time

buffers & precomputed values:
- 3 rhs fd -> buffer of time stepping
- 3 rot fd -> buffer of advection term
- plan fwd & bwd large for both node sets
- buffers large pd & fd for both node sets
- buffers large for neighbor layers pd
- precomputed factors for first & second x/y/z-derivatives
- precomputed arrays for pressure solver

### Modules in Serial Version

- transform: switch between physical & frequency domain, set values with functions, get values to array or even file (?)
- advection: set array in frequency domain to non-linear advection term, performing product in physical domain
- diffusion: add laplacian to other array, all in frequency domain
- pressure solver: compute a field the gradient of which can be added to a vector field to make it divergence free, provide functions to compute this gradient and add it to a vector field
- build rhs: compute rhs without pressure, project with pressure solver, add pressure gradient to rhs
- combined euler step: compute rhs without pressure, perform fractional step, project velocities with pressure solver, add pressure gradient to velocities

### Modules in Parallel Version

- transform: no parallelism needed, just have to know which part of the domain you have to set up the velocity field
- advection: here, we need some communication between interpolating the slices above & below -> needs more work here
- diffusion: need to pass over data from above & below
- pressure solver: need to perform pipelined solve
- rest should not need communication

## Distributed Symmetric Tridiagonal Thomas Algorithm

Here are some notes on the implementation of the parallel, pipelined algorithm
for solving a symmetric tridiagonal system of equations.

### Algorithm from Quarteroni, Sacco & Saleri

```
function [x] = modthomas (a,b,c,f)
%MODTHOMAS modified version of the Thomas algorithm
% X=MODTHOMAS(A,B,C,F) solves the system T*X=F where T
% is the tridiagonal matrix T=tridiag(B,A,C).
n=length(a);
b=[0; b];
c=[c; 0];
gamma(1)=1/a(1);
for i=2:n
    gamma(i)=1/(a(i)-b(i)*gamma(i-1)*c(i-1));
end
y(1)=gamma(1)*f (1);
for i =2:n
    y(i)=gamma(i)*(f(i)-b(i)*y(i-1));
end
x(n,1)=y(n);
for i=n-1:-1:1
    x(i,1)=y(i)-gamma(i)*c(i)*x(i+1,1);
end
return
```

- a1 to an is the diagonal
- b2 to bn is the lower off-diagonal
- c1 to cn-1 is the upper off-diagonal
- the modification of b & c adds zeros to have vectors of length n where the
  valid indices are as just described
- gamma is the modified diagonal

### Symmetric version of the algorithm

```
"""
symthomas is a modified version of the Thomas algorithm
x = symthomas(dv, ev, b) solves the system A * x = b where A is the symmetric
tridiagonal matrix A = SymTridiagonal(dv, ev).
"""
function symthomas(dv, ev, B)

    n = length(dv)

    # preparation pass, independent of b
    γ = similar(dv)
    γ[1] = 1 / dv[1]
    for i = 2:n
        γ[i] = 1 / (dv[i] - ev[i-1] * γ[i-1] * ev[i-1])
    end

    # forward pass, only changes b (γ & ev are inputs)
    b[1] = γ[1] * b[1]
    for i = 2:n
        b[i] = γ(i) * (b[i] - ev[i-1] * b[i-1])
    end

    # backward pass
    for i = n-1:-1:1
        b[i] = b[i] - γ[i] * ev[i] * b[i+1]
    end
    b
end
```

- we first translate the code to Julia syntax
- in our case, the system is symmetric, so c1==b2, …, cn-1=bn
- instead of extending the off-diagonals, we can adapt the index to directly
  use the values of the (single) off-diagonal ev
- this way we can get rid of the intermediate variables for the off-diagonals
- we can write the result directly into the rhs vector, avoiding allocation
  of two extra vectors for the forward and backward pass

### Data Dependency of Preparation Pass

- the preparation pass computes the modified diagonal vector γ
- each iteration depends on the previous iteration, i.e. γ[i] needs γ[i-1]
- the iteration for γ[i] also needs dv[i], and ev[i-1] if i>1
- for the parallel version, this means that we start at the bottom and pass
  up ev[i_top]^2*γ[i_top]
- we only need the local bits of dv & ev, where ev is one shorter for the last
  process and the same length for all the others

### Data Dependency of Forward Pass

- the forward pass overwrites the right-hand-side vector b, with increasing indices
- each iteration depends on the previous iteration, i.e. b[i] needs b[i-1]
- the iteration for b[i] also needs b[i], and ev[i-1] if i>1
- for the parallel version, this means that we start at the bottom and pass up
  ev[i_top] * b[i_top]
- we only need the local bits of b (before overwriting) & ev

### Data Dependency of Backward Pass

- the backward pass overwrites the right-hand-side vector b again, this time
  in backward order with decreasing indices
- each iteration depends on the previous iteration, i.e. b[i] needs b[i+1]
- for the parallel version, this means that we start at the top and pass down
  b[i+1]
- we only need the local bits of b (before overwriting), ev, and γ
