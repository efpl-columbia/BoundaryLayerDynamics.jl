# Installation

To run a simulation, the `ChannelFlow` package needs to be in the Julia load path.

!!! note

    `ChannelFlow` is currently not registered in the Julia package registry, so
    running `Pkg.add("ChannelFlow")` won’t work.

First, a copy of the ChannelFlow.jl repository should be obtained with `git clone`.
In the following, it is assumed that the code is available in a local directory
`path/to/ChannelFlow.jl`.

The easiest way of using the code is by setting the Julia project environment to the directory of the package, i.e.

```shell
$ julia --project=path/to/ChannelFlow.jl
```

This will use the exact versions of all dependencies that have been used to develop the latest version of `ChannelFlow`.
Note that if you are running this for the firs time on a new machine or with a new version of Julia, you need to run `Pkg.instantiate()` to download the dependencies.

If you would like to add other dependencies or use the latest versions of all dependencies, it is best to create a new project environment with `--project=path/to/some/other/directory` and add the `ChannelFlow` package as a development dependency:

```julia
import Pkg
Pkg.develop(path="path/to/ChannelFlow.jl")
```

In both cases, any changes that are made to the `ChannelFlow` code are reflected immediately and you do not have to run `Pkg.update()`, unless you want to update the dependencies.

If you are using `ChannelFlow` for the first time

!!! note

    Adding the parent folder of the `ChannelFlow` package to the load path
    using `push!(LOAD_PATH, ...)` is possible but not recommended, as this
    requires that all dependencies of `ChannelFlow` (such as the `FFTW`
    package) are installed manually.
