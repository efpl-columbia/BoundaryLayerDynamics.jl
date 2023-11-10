# Momentum Advection

!!! note
    This page describes a “process”. Refer to [Physical Processes](@ref) for general information about processes and their implementation.

The momentum advection process represents the advective transport of ``u_i`` (i.e. momentum normalized by density) in rotation form, i.e.

```math
\frac{∂u_i}{∂t} = … − u_j \left(\frac{∂u_i}{∂x_j} − \frac{∂u_j}{∂x_i}\right)
− \frac{∂}{∂x_i} \frac{u_j u_j}{2} \,.
```

Only the first term is computed here while the second term is assumed to become part of a modified pressure handled separately.

The process is discretized by computing the vorticity in the frequency domain, then computing the velocity–vorticity product in the physical domain, and finally transforming the result back to a frequency-domain representation.
If other processes (e.g. a [`StaticSmagorinskyModel`](@ref)) require physical-domain velocity gradients at the same resolution, the vorticity is computed from those to minimize the number of transforms.

If the level of `dealiasing` is set to at least `:quadratic`, the physical-domain products are equivalent to frequency-domain convolution and no additional aliasing errors are introduced.

The process relies on boundary conditions for ``u₃``.

```@docs
MomentumAdvection
```


## Discretization

To compute the non-linear advection term, the velocity components $u_i$ as well as the vorticity components $ω_i$ are transformed to the physical domain.

The contributions of the advection term become

```math
\begin{aligned}
\frac{∂}{∂t} u₁(ζ_C) &= … + u₂(ζ_C) \, ω₃(ζ_C) - \frac{u₃(ζ_C⁻) \, ω₂(ζ_C⁻) + u₃(ζ_C⁺) \, ω₂(ζ_C⁺)}{2}
\\
\frac{∂}{∂t} u₂(ζ_C) &= … + \frac{u₃(ζ_C⁻) \, ω₁(ζ_C⁻) + u₃(ζ_C⁺) \, ω₁(ζ_C⁺)}{2} - u₁(ζ_C) \, ω₃(ζ_C)
\\
\frac{∂}{∂t} u₃(ζ_I) &= … + \frac{u₁(ζ_I⁻) + u₁(ζ_I⁺)}{2} \, ω₂(ζ_I) - \frac{u₂(ζ_I⁻) + u₂(ζ_I⁺)}{2} \, ω₁(ζ_I)
\end{aligned}
```

where the values that are defined at the “wrong” set of nodes have been interpolated.


## Contributions to Budget Equations

Contribution to the instantaneous momentum equation:

```math
\begin{aligned}
  \frac{\partial}{\partial t} u_i &= …
  - u_j \frac{\partial u_i}{\partial x_j}
  \quad &\text{Convection Form}
  \\ &= …
  - \frac{\partial u_i u_j}{\partial x_j}
  \quad &\text{Divergence Form}
  \\ &= …
  - \frac{1}{2} \left(
    \frac{\partial u_i u_j}{\partial x_j}
    + u_j \frac{\partial u_i}{\partial x_j}
  \right)
  \quad &\text{Skew-Symmetric Form}
  \\ &= …
  - u_j \left( \frac{\partial u_i}{\partial x_j} - \frac{\partial u_j}{\partial x_i} \right)
  - \frac{\partial}{\partial x_i} \frac{u_j u_j}{2}
  \quad &\text{Rotation Form}
\end{aligned}
```

Contribution to the mean momentum equation:

```math
\begin{aligned}
  \frac{\partial}{\partial t} \overline{ u_i } = …
  - \overline{ u_j \left( \frac{\partial u_i}{\partial x_j} - \frac{\partial u_j}{\partial x_i} \right) }
  - \frac{\partial}{\partial x_i} \frac{\overline{u_j u_j}}{2}
\end{aligned}
```

```math
\begin{aligned}
  \frac{\partial}{\partial t} \overline{ u_i } = …
  + \frac{\partial}{\partial x_j} \left(
  − \overline{u_i}\,\overline{u_j}
  − \overline{u_i^\prime u_j^\prime}
  \right)
\end{aligned}
```

```math
\begin{aligned}
  \frac{\partial}{\partial t} \overline{ u_i } = …
  - \frac{\partial \overline{u_i}\,\overline{u_j}}{\partial x_j}
  - \frac{\partial \overline{u_i^\prime u_j^\prime}}{\partial x_j}
\end{aligned}
```

Contribution to turbulent momentum equation:

```math
\begin{aligned}
  \frac{\partial}{\partial t} u_i^\prime = …
  - \frac{\partial}{\partial x_j} \left(
    u_i^\prime \overline{u_j}
    + \overline{u_i} u_j^\prime
    + u_i^\prime u_j^\prime
    - \overline{u_i^\prime u_j^\prime}
  \right)
\end{aligned}
```

Contribution to the mean kinetic energy equation:

```math
\begin{aligned}
  \frac{\partial}{\partial t} \frac{\overline{u_i}^2}{2} &= …
  - \overline{u_i} \frac{\partial \overline{u_i}\,\overline{u_j}}{\partial x_j}
  - \overline{u_i} \frac{\partial \overline{u_i^\prime u_j^\prime}}{\partial x_j}
  \\ &= …
  \frac{\partial}{\partial x_j} \left(
    - \frac{\overline{u_i}^2}{2} \overline{u_j}
    - \overline{u_i} \overline{u_i^\prime u_j^\prime}
  \right)
  \underbrace{+ \overline{u_i^\prime u_j^\prime} \frac{\partial \overline{u_i}}{\partial x_j}}_{\text{TKE production}}
\end{aligned}
```

Contribution to the turbulent kinetic energy equation:

```math
\begin{aligned}
  \frac{\partial}{\partial t} \frac{\overline{u_i^\prime u_i^\prime}}{2} = …
  + \frac{\partial}{\partial x_j} \left(
    - \frac{\overline{u_i^\prime u_i^\prime}}{2} \overline{u_j}
    - \frac{1}{2} \overline{u_i^\prime u_i^\prime u_j^\prime}
  \right)
  \underbrace{- \overline{u_i^\prime u_j^\prime} \frac{\partial \overline{u_i}}{\partial x_j}}_{\text{TKE production}}
\end{aligned}
```
