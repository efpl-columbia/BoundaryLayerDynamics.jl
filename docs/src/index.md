# ChannelFlow.jl Documentation

```@meta
CurrentModule = ChannelFlow
```

## Quick Start

Set up and run a simple large-eddy simulation interactively:

```juliarepl
julia> cf = prepare_open_channel(64, Re=1e6, roughness_length=1e-3,
       sgs_model=StaticSmagorinskyModel())
julia> integrate!(cf, 1e-3, 1000)
```

Set up a direct numerical simulation and run with 32 MPI processes (roughly corresponding to the low-resolution case of [Lee & Moser (2015)](https://doi.org/10.1017/jfm.2015.268)):

```julia
# contents of `simulation.jl`
using ChannelFlow, MPI
MPI.Init()
cf = prepare_closed_channel((256, 192, 96),
        Re = 2857,
        constant_flux = true,
        grid_mapping = SinusoidalMapping(0.95))
integrate!(cf, 5e-3, 20*3000, profiles_frequency=3000)
```

```shell
$ mpiexec -n 32 julia simulation.jl
```
