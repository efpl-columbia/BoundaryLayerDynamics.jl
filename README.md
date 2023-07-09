# <img alt="BoundaryLayerDynamics.jl" src="./docs/src/assets/logo-vector.svg" height=120 />

*Turbulence-resolving simulations of boundary-layer flows.*

BoundaryLayerDynamics.jl can simulate a range of physical problems, but its
primary purpose is to simulate the evolution of turbulent flows in simplified
representations of the [atmospheric boundary layer](https://en.wikipedia.org/wiki/Planetary_boundary_layer).
It can be used to perform [direct numerical simulation (DNS)](https://en.wikipedia.org/wiki/Direct_numerical_simulation) and [large-eddy simulation (LES)](https://en.wikipedia.org/wiki/Large_eddy_simulation) of flows in channel-like geometries.
A detailed description of the physical and mathematical models is provided in [the documentation](https://docs.mfsch.dev/BoundaryLayerDynamics.jl/) and in [a preprint](https://doi.org/10.5194/egusphere-2023-1071) describing the code.

## Documentation

[![Online Documentation](https://img.shields.io/badge/🕮-Online_Documentation-2C6BAC)](https://docs.mfsch.dev/BoundaryLayerDynamics.jl/)

The documentation in the `docs` folder relies on
[Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) and can be accessed [online](https://docs.mfsch.dev/BoundaryLayerDynamics.jl/) or by building it locally.

To build the documentation, run `julia docs/make.jl`.
This will create the HTML documentation in the `docs/build` directory.
The `make.jl` script takes care of setting up the package environment so it can be run
from anywhere as long as it has access to `Pkg` from the standard library.

To work on the documentation, it is recommended to use
[LiveServer.jl](https://github.com/tlienart/LiveServer.jl).
If it is installed in the global environment, a development server can be
launched with `julia -e 'using LiveServer; servedocs()'` from the project root.

## Attribution & License

[![Preprint](https://img.shields.io/badge/Preprint-10.5194%2Fegusphere--2023--1071-8E0F56)](https://doi.org/10.5194/egusphere-2023-1071)
[![MIT License](https://img.shields.io/badge/License-MIT-D2D2C0)](./LICENSE.md)

BoundaryLayerDynamics.jl was created by [Manuel F. Schmid](https://orcid.org/0000-0002-7880-9913) in collaboration with [Marco G. Giometto](https://orcid.org/0000-0001-9661-0599) and [Marc B. Parlange](https://orcid.org/0000-0001-6972-4371).
A paper describing the motivation for the project, the mathematical and physical models, their validation, as well as some performance measurements is currently under review.
Please cite [this preprint](https://doi.org/10.5194/egusphere-2023-1071) if you use BoundaryLayerDynamics.jl in your research.

BoundaryLayerDynamics.jl is freely available under the terms of the [MIT License](./LICENSE.md).
