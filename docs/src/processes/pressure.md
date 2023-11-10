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


## Discrete Conservation Properties

!!! tip "Conservation of Kinetic Energy"
    The pressure term (including contributions from the rotational advection term) preserves the total kinetic energy to machine precision both without and with grid stretching (in the absence of time-integration errors).

Analytically, we can show that the pressure term conserves energy using integration by parts (with vanishing ``u₃`` at the boundary) to show that

```math
\frac{∂}{∂t} \int_V \frac{u_i^2}{2} \mathrm{d}V
= … − \int_V u_i \frac{∂ϕ}{∂x_i} \mathrm{d}V
= … + \int_V ϕ \frac{∂u_i}{∂x_i} \mathrm{d}V
\;,
```

which is zero due to the continuity equation.

For the discretized equations we have

```math
\sum_{ζ_C} u₁(ζ_C) \left.\frac{∂ϕ}{∂x₁}\right|_{ζ_C} \left.\frac{∂x₃}{∂ζ}\right|_{ζ_C} +
\sum_{ζ_C} u₂(ζ_C) \left.\frac{∂ϕ}{∂x₂}\right|_{ζ_C} \left.\frac{∂x₃}{∂ζ}\right|_{ζ_C} +
\sum_{ζ_I} u₃(ζ_I) \left.\frac{∂ϕ}{∂x₃}\right|_{ζ_I} \left.\frac{∂x₃}{∂ζ}\right|_{ζ_I}
= 0 \;.
```

For the horizontal derivatives, the integration by parts holds exactly since no approximations are made when computing derivatives or evaluating the horizontal mean.
For the vertical derivatives, we can sum over ``ζ_C`` instead of ``ζ_I`` to get

```math
\sum_{ζ_I} u₃(ζ_I) \left(\frac{ϕ(ζ_I⁺) - ϕ(ζ_I⁻)}{Δζ} \left.\frac{∂ζ}{∂x₃}\right|_{ζ_I} \right) \left.\frac{∂x₃}{∂ζ}\right|_{ζ_I}
= - \sum_{ζ_C} ϕ(ζ_C) \left( \frac{u₃(ζ_C⁺) - u₃(ζ_C⁻)}{Δζ} \left.\frac{∂ζ}{∂x₃}\right|_{ζ_C} \right) \left.\frac{∂x₃}{∂ζ}\right|_{ζ_C}
```

for any wavenumber (the grid-stretching terms cancel on both sides).
The contributions therefore add up to

```math
- \sum_{ζ_C} ϕ(ζ_C) \left(
\left.\frac{∂u₁}{∂x₁}\right|_{ζ_C} +
\left.\frac{∂u₂}{∂x₂}\right|_{ζ_C} +
\frac{u₃(ζ_C⁺) - u₃(ζ_C⁻)}{Δζ} \left.\frac{∂ζ}{∂x₃}\right|_{ζ_C}
\right) \left.\frac{∂x₃}{∂ζ}\right|_{ζ_C}
= 0
```

as the part in the parentheses is zero due to the discrete continuity equation.
