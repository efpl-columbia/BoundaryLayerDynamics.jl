# Subgrid-Scale Advection

Large-eddy simulation (LES) simulates the dynamics of a spatially-filtered quantity ``q = \widetilde{Q} = \int G(\bm{r}, \bm{x}) Q(\bm{x}−\bm{r}) \mathrm{d}\bm{r}``, where ``G`` defines the filtering operation applied to the “real”, unfiltered quantity ``Q``.

## Momentum Advection

If the filtering operation is defined as a projection (i.e. ``\widetilde{\widetilde{U_i}} = \widetilde{U_i}``) that commutes with derivation in time and space, applying the filter to the momentum and continuity equations gives equations for the filtered velocity ``\widetilde{U_i}``.
Linear processes retain the same mathematical form, but the non-linear advection term produces additional contributions, i.e.

```math
\begin{aligned}
\frac{∂\widetilde{U_i}}{∂t} &= …
− \frac{∂ \widetilde{U_i U_j}}{∂x_j}
\\ &= …
− \frac{∂ \widetilde{U_i} \widetilde{U_j}}{∂x_j}
− \frac{∂ τ_{ij}^\mathrm{R}}{∂x_j}
\quad &&\text{with} \quad
τ_{ij}^\mathrm{R} = \widetilde{U_i U_j} − \widetilde{U_i} \widetilde{U_j}
\\ &= …
− \frac{∂ \widetilde{U_i} \widetilde{U_j}}{∂x_j}
− \frac{∂ τ_{ij}^\mathrm{SGS}}{∂x_j}
− \frac{∂}{∂x_i} \frac{τ_{jj}^\mathrm{R}}{3}
\quad &&\text{with} \quad
τ_{ij}^\mathrm{SGS} =
τ_{ij}^\mathrm{R} − \frac{1}{3} τ_{ii}^\mathrm{R} δ_{ij} \,.
\end{aligned}
```

If we set ``u_i = \widetilde{U_i}``, the first contribution can be handled with the regular advection term and the last term can be included in a modified pressure term.
The second term, however, needs to be modeled based on the resolved velocity field.
This is the purpose of the subgrid-scale advection process, which adds a contribution

```math
\frac{∂u_i}{∂t} = …
− \frac{∂ τ_{ij}^\mathrm{SGS}}{∂x_j}
```

to the simulated momentum dynamics.
Currently the [`StaticSmagorinskyModel`](@ref) is the only implemented model for ``τ_{ij}^\mathrm{SGS}``.

## Boundary Conditions

The subgrid-scale transport requires vertical boundary conditions for ``τ_{13}^\mathrm{SGS}`` and  ``τ_{23}^\mathrm{SGS}``.

- For a [`FreeSlipBoundary`](@ref), the subgrid-scale fluxes are assumed to vanish at the boundary.
  - For a [`RoughWall`](@ref) boundary, a local equilibrium layer is assumed in the near-wall region and the wall-stress is estimated with ``τ_{i3}^\mathrm{w} = κ² \sqrt{u₁²+u₂²} \, u_i / \log²(x₃/z₀)``, evaluated at the first grid point at ``ζ=½/N₃``. The roughness length ``z₀`` and the magnitude of the von Kármán constant ``κ`` are model parameters. See [Algebraic Equilibrium Rough-Wall Model](@ref) for more details.
- The [`CustomBoundary`](@ref) is currently not supported for subgrid-scale advection, as it is not clear how the subgrid-scale fluxes should be evaluated at such a boundary.


## Static Smagorinsky Model

The static Smagorinsky subgrid-scale model relies on the approximation

```math
τ_{ij}^\mathrm{SGS} = − 2 \, l_\mathrm{S}^2 \, \mathcal{S} \, S_{ij}
\quad \text{with} \quad
l_\mathrm{S} = C_\mathrm{S} \, \left(Δx₁Δx₂Δx₃\right)^{1/3} \,,
```

where the Smagorinsky coefficient ``C_\mathrm{S}`` is typically set to a value around ``C_\mathrm{S} ≈ 0.1``.
The resolved strain rate ``S_{ij}`` and the total strain rate ``\mathcal{S}`` are defined as

```math
S_{ij} ≡ \frac{1}{2} \left( \frac{∂u_i}{∂x_j} + \frac{∂u_j}{∂x_i} \right)
\quad \text{and} \quad
\mathcal{S} ≡ \sqrt{2 S_{ij} S_{ij}} \,.
```

The computation of ``S_{33}`` near the boundary relies on the homogeneous Dirichlet boundary conditions for ``u₃``, which are assumed for all supported domain boundaries.
Additionally, the computation of ``\mathcal{S}`` on the first ``ζ_C``-nodes relies on the near-wall behavior of ``S_{13}`` and ``S_{23}``, which is handled differently depending on the type of boundary:

- For free-slip boundaries, the strain rates are assumed to vanish at the boundary and can be interpolated from the values at the first ``ζ_I``-nodes.
- For rough-wall boundaries, the values are derived from the wall model as ``S_{i3} = u_i / (2 x₃ \log(x₃/z₀))`` (evaluated at the first ``ζ_C``-node, analogous for upper boundary).

Since the model relies on non-linear expressions, those are computed in the physical domain.

```@docs
StaticSmagorinskyModel
```


## Contributions to Budget Equations

Contribution to the instantaneous momentum equation:

```math
\frac{\partial}{\partial t} u_i = …
− \frac{\partial \tau_{ij}^\mathrm{SGS}}{\partial x_j}
```

Contribution to the mean momentum equation:

```math
\frac{\partial}{\partial t} \overline{u_i} = …
− \frac{\partial \overline{\tau_{ij}^\mathrm{SGS}}}{\partial x_j}
```

Contribution to turbulent momentum equation:

```math
\frac{\partial}{\partial t} u_i^\prime = …
− \frac{\partial {\tau_{ij}^\mathrm{SGS}}^\prime}{\partial x_j}
```

Contribution to the mean kinetic energy equation:

```math
\frac{\partial}{\partial t} \frac{\overline{u_i}^2}{2} = …
+ \frac{\partial}{\partial x_j} \left(
  − \overline{u_i} \overline{\tau_{ij}^\mathrm{SGS}}
\right)
+ \overline{\tau_{ij}^\mathrm{SGS}} \frac{\partial \overline{u_i}}{\partial x_j}
```

Contribution to the turbulent kinetic energy equation (could use $S_{ij}^\prime$ for the last term, since the SGS-tensor is symmetric):

```math
\frac{\partial}{\partial t} \frac{\overline{u_i^\prime u_i^\prime}}{2} = …
+ \frac{\partial}{\partial x_j} \left(
  − \overline{u_i^\prime {\tau_{ij}^\mathrm{SGS}}^\prime}
\right)
+ \overline{
{\tau_{ij}^\mathrm{SGS}}^\prime
\frac{\partial u_i^\prime}{\partial x_j}
}
```
