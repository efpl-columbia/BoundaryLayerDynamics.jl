# Turbulence-Resolving Simulations of Atmospheric Boundary-Layer Flows

BoundaryLayerDynamics.jl can simulate a range of physical problems, but its
primary purpose is to simulate the evolution of turbulent flows in simplified
representations of the atmospheric boundary layer.

It can be used to perform *direct numerical simulation (DNS)* and *large-eddy
simulation (LES)* of flows in channel-like geometries.

- governing equations: incompressible Navier–Stokes equations
- spatial discretization: spectral Fourier collocation method in horizontal
  directions (periodic boundary conditions), second-order finite differences in
  vertical direction (various boundary conditions supported)
- pressure handling: direct linear pressure solver
- time integration: various explicit methods supported
- LES subgrid-scale model: static Smagorinsky model
- LES wall-model: analytic rough-wall equilibrium model
- parallelization: MPI (one process per vertical grid point)

BoundaryLayerDynamics.jl was created by [Manuel F. Schmid](mailto:mfs2173@columbia.edu)
in collaboration with Marco G. Giometto and Marc B. Parlange.

## Installation

The package is currently not part of any registry. The easiest way of adding it
to a Julia project is with the `Pkg.develop` command, passing the path to the
directory of the package.

## Usage

To run a simulation, first create a `DiscretizedABL` specifying the dimensions
and boundary conditions of the domain, the number of modes and grid points used
for the discretization, and the physical processes that are simulated. Then
(optionally) specify initial conditions with `initialize!` and simulate the
flow dynamics with `evolve!`.

If MPI has been initialized with `MPI.init()` before the model is created, the
simulation is distributed among the available processes.

## Documentation

The documentation in the `docs` folder relies on
[Documenter.jl](https://github.com/JuliaDocs/Documenter.jl).

To build the documentation, run `julia docs/make.jl`.
This will create the HTML documentation in the `docs/build` directory.
The `make.jl` takes care of setting up the package environment so it can be run
from anywhere as long as it has access to `Pkg` from the standard library.

To work on the documentation, it is recommended to use
[LiveServer.jl](https://github.com/tlienart/LiveServer.jl).
If it is installed in the global environment, a development server can be
launched with `julia -e 'using LiveServer; servedocs()'` from the project root.
