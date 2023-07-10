# Setup & Workflow

## Recommended Project Structure

It is recommended to create a separate project folder for each simulation or set of simulations and set it up as a [Julia environment](https://pkgdocs.julialang.org/v1/environments/).
In practice, this means launching Julia with `julia --project=path/to/project` both in interactive mode and when launching simulations.

It is further recommended to create a Julia script that defines and runs the simulation, as explained [below](@ref script).
The same approach can be used for post-processing results, although it may be more convenient to use a [Pluto.jl](https://github.com/fonsp/Pluto.jl) notebook.

Both for reproducibility and convenience, it is recommended to save the commands with which simulations, notebooks, etc. are run somewhere in the project, for example in a README file, a launch script, a [Makefile](https://www.gnu.org/software/make/), a [justfile](https://just.systems/), or something similar.


## Adding `BoundaryLayerDynamics` to the Project

To set up and run a simulation, the `BoundaryLayerDynamics` package needs to be in the Julia load path.
This is best achieved by adding the package to the project-specific environment.

```juliarepl
julia> import Pkg
julia> Pkg.add("BoundaryLayerDynamics")
```

## Making Project-Specific Modifications

Simulations may require functionality that is not currently implemented in the `BoundaryLayerDynamics` package.
New [physical processes](@ref Physical-Processes), [time-integration methods](@ref Time-Integration-Methods), and [output modules](@ref Output-Modules) can be defined in the user code and simply passed to the package functions, although this may require adding a few methods to unexported functions.

!!! note
    Consider contributing on [GitHub](https://github.com/efpl-columbia/BoundaryLayerDynamics.jl) for functionality that may be of interest to others as well.
    It is recommended to open an issue to discuss new functionality before implementing it, as 1) there may already be partial implementations from other people, 2) we can discuss how the functionality is best integrated into the existing code, and 3) the functionality may be outside the scope of BoundaryLayerDynamics.jl.

If a project requires significant changes to the code, it may be easier to directly modify the code of BoundaryLayerDynamics.jl.
There are a number of options to use such a modified version of the package in simulations:

1. Clone the package repository and use that folder/project both for package modifications and simulation setup code.
1. Push a clone with the required modifications to a separate GitHub repository and provide the full URL when [`add`](https://pkgdocs.julialang.org/v1/repl/#repl-add)ing the package to the simulation environment.
1. Make modifications in a local clone of the package repository that is added to the simulation environment with [`develop`](https://pkgdocs.julialang.org/v1/repl/#repl-develop).
   This way, any changes that are made to the `BoundaryLayerDynamics` code are reflected immediately and you do not have to run [`update`](https://pkgdocs.julialang.org/v1/repl/#repl-update) unless you want to update other dependencies.
   The clone can be created by `develop` (with the `--shared` or `--local` arguments) or manually at a convenient location, possibly as a Git submodule or in a folder shared between multiple simulations.

!!! note
    Adding the parent folder of the `BoundaryLayerDynamics` package to the load path using
    `push!(LOAD_PATH, ...)` is possible but not recommended, as this requires
    that all dependencies of `BoundaryLayerDynamics` (such as the `FFTW` package) are installed
    manually.


## Interactive Use

Once BoundaryLayerDynamics.jl is part of the project environment, it can be loaded with `using BoundaryLayerDynamics` as usual.
Interactive use is only recommended for exploration and to access the built-in documentation.
For running simulations and post-processing, it is better to [write a script](@ref script) or [create a notebook](@ref notebook) so the work is reproducible.


## [Simulation Scripts](@id script)

To run simulations, it is recommended to write a script that configures the model and runs the time integration.
An example is given here:

```julia
using BoundaryLayerDynamics, MPI

function run_dns()

    ν = 1e-4
    domain = Domain((4π, 2π, 2), SmoothWall(), SmoothWall())
    grid_size = (256, 192, 128)
    processes = incompressible_flow(ν, constant_flux = 1)
    abl = Model(grid_size, domain, processes)

    u0(x,y,z) = (1 - (z-1)^2)*3/2
    initialize!(abl, vel1 = u0)
    BoundaryLayerDynamics.State.add_noise!(abl.state.vel1, 1e-2)

    turnovers = 25
    dt = 1e-4
    steps_per_turnover = 5_000
    nt = turnovers * steps_per_turnover

    outdir = abspath("$(dirname(PROGRAM_FILE))/data")
    sfreq = div(steps_per_turnover, 10) * dt
    snapshots = Snapshots(path = joinpath(outdir, "snapshots"),
                          frequency = sfreq, precision = Float32)

    evolve!(abl, dt * nt, dt = dt, output = snapshots, verbose = true)
end

MPI.Init()
run_dns()
```

If this file is saved as `simulation.jl`, you can run the simulation for example with:

```shell
$ mpiexec -n 128 julia --project=path/to/project simulation.jl
```

!!! warning
    If a simulation is run without `mpiexec` (or similar) or there is no call to `MPI.Init` before the simulation, the simulation is run with a single process, or possibly with many processes all running the same serial simulation.

## [Notebook Use](@id notebook)

Notebooks are mainly useful for code development and post-processing, as they enable rapid, interactive iteration.
[Jupyter](https://jupyter.org/) is probably the most well-known software for this, but for Julia code it is worth considering [Pluto.jl](https://github.com/fonsp/Pluto.jl) instead.
You can add Pluto to the project dependencies with `Pkg.add("Pluto")` and launch it with `julia --project=path/to/project --eval 'using Pluto; Pluto.run()'`.
On a remote system, you probably want to run with `Pluto.run(launch_browser=false)` instead.

By default, Pluto creates a separate, internal project environment for each notebook.
When working with BoundaryLayerDynamics.jl simulations, it is usually better to deactivate this functionality and use a single, shared project environment – mainly because Pluto does not have the equivalent of `Pkg.develop`, but also to ensure that simulations and post-processing use identical package versions.
You can use the following code as the first notebook cell:

```julia
begin
    LOAD_PATH = ["@", "@stdlib"] # to avoid loading globally-installed packages
    using Pkg; Pkg.activate(".") # disables the built-in package management
    using BoundaryLayerDynamics # add additional dependencies here (can also use `import`)
end
```


## High-Performance Computing

Running simulations on HPC systems should work as expected by launching Julia from a batch script.
If the system does not have Julia installed, [downloading](https://julialang.org/downloads/) and unpacking the latest version should generally work.

However, there are a number of issues that may arise when working with MPI parallelization and HPC systems.

### Choose MPI Library

[MPI.jl](https://github.com/JuliaParallel/MPI.jl) provides its own MPI libraries by default.
On most HPC systems, it is better to use the system-provided libraries instead.
As documented [here](https://juliaparallel.org/MPI.jl/stable/configuration/#using_system_mpi), this can be achieved by running `MPI.MPIPreferences.use_system_binary()`.

!!! warning
    Recent versions of the HDF5 library load their own MPI dependencies in a way that does may not match the system MPI and can result in crashes. See [this issue](https://github.com/JuliaIO/HDF5.jl/issues/1079) for details and workarounds.

### Avoid Precompilation Errors

The Julia package manager can run into errors, when multiple MPI processes try to do package operations at the same time.
As documented [here](https://juliaparallel.org/MPI.jl/stable/knownissues/#Julia-module-precompilation), it is therefore best to either trigger the precompilation from a single process before launching MPI processes or to run Julia with `--compiled-modules=no`.

### Change Package Depot Path

By default, Julia saves downloaded packages inside the `$HOME/.julia` directory.
On some systems, the home directory might not be available during simulations or its use may be discouraged.
The [`JULIA_DEPOT_PATH`](https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_DEPOT_PATH) environment variable can be used to change this path, for example by running `export JULIA_DEPOT_PATH=$SCRATCH/julia-depot`.
