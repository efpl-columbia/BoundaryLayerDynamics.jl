# Usage

```@meta
CurrentModule = ChannelFlow
```

A simulation is usually configured in a short Julia script that is then run with `julia --project simulation.jl` or `mpiexec -n NPROC julia --project simulation.jl` (cf. [Parallel Processing with MPI](@ref)).
For more on project environments and dependencies, see [Installation](@ref).

A simulation script loads the package (`using ChannelFlow`) and contains the two commands to set up and run a simulation.
In a first step, a channel flow is set up as described in [Flow Setup](@ref).
This creates the initial flow field as well as the data structures necessary to compute all terms of the governing equations.
In a second step, the development of this flow over a certain time period is simulated as described in [Flow Simulation](@ref).
During the simulation, flow data can be collected in the form of snapshots of the velocity field an different flow statistics.

Alternatively, you can also specify the parameters of a simulation in a TOML configuration file, as described in [Simulation Configuration File](@ref).

## Flow Setup

The functions `prepare_open_channel` and `prepare_closed_channel` are used to set up a standard channel flow using non-dimensional parameters.
For more control over the setup, you can directly construct a [`ChannelFlowProblem`](@ref) as described in [Advanced Flow Setup](@ref).

```@docs
prepare_channel
```

## Flow Simulation

Once a flow is set up, it can be advanced in time with the `integrate!` function.

```@docs
integrate!
Euler
AB2
SSPRK22
SSPRK33
```

## Parallel Processing with MPI

For parallel processing, the domain can be split vertically into sections that are handled by different MPI processes.
This only requires running `MPI.Init()` before setting up a flow, and the domain will be distributed amongst the available MPI processes.
It is recommended that the vertical grid size is an exact multiple of the number of processes, and the number of processes should not exceed the vertical grid size.

To run a parallel simulation, you can use a command like `mpiexec -n 256 julia --project simulation.jl`.

!!! warning

    The package [MPI.jl](https://github.com/JuliaParallel/MPI.jl) used for
    MPI operations does not use the system’s MPI libraries by default. When
    running on an HPC cluster, you probably want to use the provided system
    libraries, and you should always ensure that the `mpiexec` binary matches
    the MPI library. For more on this, refer to the [documentation of
    MPI.jl](https://juliaparallel.github.io/MPI.jl/latest/configuration/).

!!! warning

    Julia’s package precompilation mechanism does not interact well with MPI
    and there can be issues if several MPI processes try to precompile
    a package at the same time. To avoid issues, precompilation should be run
    by a single Julia process before running a parallel simulation, or
    precompilation should be deactivated completely by passing
    `--compiled-modules=no` to the `julia` command. More information can be
    found in the [documentation of
    MPI.jl](https://juliaparallel.github.io/MPI.jl/latest/knownissues/).

## Variable Grid Spacing

By default, an equidistant staggered grid is used in vertical direction.
For direct numerical simulation, it is often preferable to have a smaller grid spacing at the wall and a larger spacing in the free-flow region.
The grid transformation can be specified with the `grid_mapping` option of the [`prepare_*_channel`](@ref prepare_channel) function, where currently the only supported option is [`SinusoidalMapping`](@ref).
For more control over the grid transform, refer to the `domain` option of [`ChannelFlowProblem`](@ref).

```@docs
SinusoidalMapping
```

## Large-Eddy Simulation

If the grid spacing is not fine enough to resolve all relevant scales of
motion, a subgrid-scale model is required to add the missing contributions from
unresolved motions to the non-linear advection term.
This can be specified with the `sgs_model` option of the [`prepare_*_channel`](@ref prepare_channel) function, where currently the only supported option is [`StaticSmagorinskyModel`](@ref).

```@docs
StaticSmagorinskyModel
```

## Simulation Configuration File

Instead of passing simulation parameters as function arguments, they can be provided in a TOML configuration file.
This can be helpful when running a series of simulations, since different configuration files can easily be generated programmatically.

!!! note

    The configuration file interface is a bit less flexible since it does not
    support arbitrary Julia syntax.

```@docs
run_simulation
```

!!! compat

    This functionality is not yet implemented!

## Advanced Flow Setup

To construct the full set of flow configurations that are supported, the `ChannelFlowProblem` can be constructed directly.

!!! note

    The recommended usage is to create a wrapper function that takes the non-dimensional parameters of the problem of interest and sets up the `ChannelFlowProblem` accordingly.

```@docs
ChannelFlowProblem
```

