# Pressure/Continuity

!!! note
    This page describes a “process”. Refer to [Physical Processes](@ref) for general information about processes and their implementation.

The pressure solver enforces the continuity equation by subtracting the gradient of a pressure-like variable ``ϕ`` from the momentum equation, i.e.

```math
\frac{∂u_i}{∂t} = … - \frac{∂ϕ}{∂x_i}
\quad \text{such that} \quad
\frac{∂u_i}{∂x_i} = 0
```

with the additional constraint that ``\int\int ϕ(x₁, x₂, x₃(0)) \, \mathrm{d}x₁ \mathrm{d}x₂ = 0`` to fix the absolute value of ``ϕ``.

After discretization, the computation becomes

```math
\begin{aligned}
f_i(\hat{u}_1^{κ₁κ₂ζ_C}) &= - ∂₁(κ₁) \, \hat{ϕ}^{κ₁κ₂ζ_C}
\\
f_i(\hat{u}_2^{κ₁κ₂ζ_C}) &= - ∂₂(κ₂) \, \hat{ϕ}^{κ₁κ₂ζ_C}
\\
f_i(\hat{u}_3^{κ₁κ₂ζ_I}) &= - ∂₃(ζ_I) \left(\hat{ϕ}^{κ₁κ₂ζ_I^+} - \hat{ϕ}^{κ₁κ₂ζ_I^−}\right)
\end{aligned}
```

such that

```math
∂₁(κ₁) \, \hat{u}_1^{κ₁κ₂ζ_C} +
∂₁(κ₁) \, \hat{u}_2^{κ₁κ₂ζ_C} +
∂₃(ζ_C) \left( \hat{u}₃^{κ₁κ₂ζ_C^+} - \hat{u}₃^{κ₁κ₂ζ_C^−} \right)
= 0
```

with the constraint that ``\hat{ϕ}⁰⁰⁰=0``.

The process is implemented as a [Projection](@ref Projections).
It relies on boundary conditions for ``u₃``.

```@docs
Pressure
```


## Contributions to Budget Equations

Contribution to the instantaneous momentum equation:

```math
\frac{\partial}{\partial t} u_i = …
- \frac{\partial \phi}{\partial x_i}
```

Contribution to the mean momentum equation:

```math
\frac{\partial}{\partial t} \overline{u_i} = …
- \frac{\partial \overline{\phi}}{\partial x_i}
```

Contribution to the turbulent momentum equation:

```math
\frac{\partial}{\partial t} u_i^\prime = …
- \frac{\partial \phi^\prime}{\partial x_i}
```

Contribution to the mean kinetic energy equation:

```math
\frac{\partial}{\partial t} \frac{\overline{u_i}^2}{2} = …
- \frac{\partial \overline{u_i} \overline{\phi}}{\partial x_i}
```

Contribution to the turbulent kinetic energy equation:

```math
\frac{\partial}{\partial t} \frac{\overline{u_i^\prime u_i^\prime}}{2} = …
- \frac{\partial \overline{u_i^\prime \phi^\prime}}{\partial x_i}
```
