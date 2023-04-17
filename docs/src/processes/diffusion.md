# Uniform Diffusion

!!! note
    This page describes a “process”. Refer to [Physical Processes](@ref) for general information about processes and their implementation.

The molecular diffusion process represents transport along a gradient of some quantity ``q``, also known as [Fickian diffusion](https://en.wikipedia.org/wiki/Fick%27s_laws_of_diffusion).
The diffusion coefficient ``D`` is assumed to be constant in space and time, resulting in

```math
\frac{∂q}{∂t} = … + D \frac{∂²q}{∂x_i²} \,.
```

The process is discretized with

```math
\begin{aligned}
f_i(\hat{q}^{κ₁κ₂ζ}) =
&D \left( ∂₁(κ₁)² + ∂₂(κ₂)² \right) \hat{q}^{κ₁κ₂ζ}
+ \\
&D
∂₃\left(ζ\right)
\left(
∂₃\left(ζ⁺\right)
\left(\hat{q}^{κ₁κ₂(ζ+δζ)} - \hat{q}^{κ₁κ₂ζ}\right)
−
∂₃\left(ζ¯\right)
\left(\hat{q}^{κ₁κ₂ζ} − \hat{q}^{κ₁κ₂(ζ−δζ)}\right)
\right) \,.
\end{aligned}
```

This requires boundary conditions for ``q`` at ``ζ=0`` and ``ζ=1``.

For a vector quantity like the velocity/momentum field, an instance of the diffusion process should be included for each component ``u_i``.

```@docs
MolecularDiffusion
```

## Contributions to Budget Equations

Contribution to the instantaneous momentum equation:

```math
\frac{\partial}{\partial t} u_i = …
+ \nu \frac{\partial^2 u_i}{\partial x_j^2}
```

Contribution to the mean momentum equation:

```math
\frac{\partial}{\partial t} \overline{u_i} = …
+ \nu \frac{\partial^2 \overline{u_i}}{\partial x_j^2}
```

Contribution to the turbulent momentum equation:

```math
\frac{\partial}{\partial t} u_i^\prime = …
+ \nu \frac{\partial^2 u_i^\prime}{\partial x_j^2}
```

Contribution to the mean kinetic energy equation:

```math
\begin{aligned}
\frac{\partial}{\partial t} \frac{\overline{u_i}^2}{2} &= …
+ \nu \frac{\partial^2}{\partial x_j^2} \frac{\overline{u_i}^2}{2}
- \frac{\partial \overline{u_i}}{\partial x_j}
  \frac{\partial \overline{u_i}}{\partial x_j}
\\ &= …
+ \frac{\partial}{\partial x_j} \left(
2 \nu \overline{u_i} \overline{S_{ij}}
\right)
- 2 \nu \overline{S_{ij}} \, \overline{S_{ij}}
\end{aligned}
```

Contribution to the turbulent kinetic energy equation:

```math
\begin{aligned}
\frac{\partial}{\partial t} \frac{\overline{u_i^\prime u_i^\prime}}{2} &= …
+ \nu \frac{\partial^2}{\partial x_j^2} \frac{\overline{u_i^\prime u_i^\prime}}{2}
\underbrace{ - \nu \overline{
  \frac{\partial u_i^\prime}{\partial x_j}
  \frac{\partial u_i^\prime}{\partial x_j} } }_{\text{pseudo-dissipation}}
\\ &= …
+ \nu \frac{\partial^2}{\partial x_j^2} \frac{\overline{u_i^\prime u_i^\prime}}{2}
+ \nu \frac{\partial^2 \overline{u_i^\prime u_j^\prime}}{\partial x_i \partial x_j}
- 2 \nu \overline{S_{ij}^\prime S_{ij}^\prime}
\\ &= …
+ \frac{\partial}{\partial x_j} \left(
  2 \nu \overline{u_i^\prime S_{ij}^\prime}
\right)
\underbrace{ - 2 \nu \overline{S_{ij}^\prime S_{ij}^\prime} }_{\text{dissipation}}
\end{aligned}
```
