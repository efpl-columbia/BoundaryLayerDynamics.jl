# Computational Model

```@meta
CurrentModule = BoundaryLayerDynamics
```

The first step of a simulation consists of setting up a [`Model`](@ref).
This struct holds the current state of the (discretized) quantities ``qᵢ`` as well as all the data that is required to compute their current rate of change ``∂qᵢ/∂t``.

Once a [`Model`](@ref) has been set up, its dynamics can be simulated as discussed in [Evolution in Time](@ref).

```@docs
Model
```


## Domain

The [`Domain`](@ref) describes the mathematical domain in which the governing equations are assumed to hold and its relation to the Cartesian domain in which the computations are performed.

Along the first two coordinate directions ``x₁`` and ``x₂`` (usually streamwise and cross-stream), the domain is assumed to be periodic with a repeating unit of size ``L₁×L₂``, i.e.

```math
q(x₁, x₂, x₃) = q(x₁+L₁, x₂, x₃) = q(x₁, x₂+L₂, x₃) \,.
```

Along the third coordinate direction ``x₃`` (usually wall-normal/vertical), the domain is assumed to span an range of size ``L₃`` with boundary conditions specified at the lower and upper limits.
The ``x₃``-range is represented as a function of ``ζ ∈ [0, 1]`` such as ``x₃(ζ) = L₃ζ``.
By specifying a custom, non-linear function, the computational domain can be deformed such that a discretization that is equidistant in ``ζ`` results in grid stretching/refinement in the ``x₃``-domain.

The mathematical domain is therefore defined as

```math
(x₁, x₂, x₃) ∈ (-∞, +∞) × (-∞, +∞) × \left[x₃(ζ=0), x₃(ζ=1)\right] \,,
```

where boundary conditions have to be specified at ``x_3(\zeta=0)`` and ``x_3(\zeta=1)``.

```@docs
Domain
```


### Domain Boundaries

While the domain is assumed to be periodic along the first two coordinate directions, the behaviors at the minimum and maximum ``x₃``-value need to be prescribed.
While it is possible to specify the mathematical boundary conditions of each state variable using a `CustomBoundary`, the boundary definitions `SmoothWall`, `RoughWall`, and `FreeSlipBoundary` provide a convenient shorthand for common types of boundaries and also allow processes to adapt their behavior to the physical properties of those boundaries.

```@docs
SmoothWall
RoughWall
FreeSlipBoundary
CustomBoundary
```


### Domain Mappings

The coordinates along the third dimension can be specified as an arbitrary function ``x₃(ζ)``, where ``ζ∈[0,1]``.
This coordinate transform can be used to reduce the discretization errors of values that are computed on a grid that is uniform in ``ζ`` (i.e. grid stretching).
To specify a custom mapping, both the function ``x₃(ζ)`` and its derivative ``\mathrm{d}x₃/\mathrm{d}ζ`` have to be provided so ``x₃``-derivatives can be computed with the chain rule.
The [`SinusoidalMapping`](@ref) provides a predefined coordinate transform that automatically adjusts to the specified domain size and boundaries.

```@docs
SinusoidalMapping
```


## Discretization

Within a semiperiodic [`Domain`](@ref), a quantity ``q`` can be represented as the Fourier series

```math
q(x₁, x₂, x₃) =
\sum_{\substack{-∞ < κ₁ < ∞ \\ -∞ < κ₂ < ∞}}
\hat{q}^{κ₁κ₂}\!\left(x₃(ζ)\right)
\,\mathrm{e}^{i 2 π κ₁ x₁ / L₁}
\,\mathrm{e}^{i 2 π κ₂ x₂ / L₂}
\,.
```

These continuous fields are discretized by resolving ``N₁ × N₂`` Fourier modes at ``N₃`` grid points.

Along the first two coordinate directions (``i=1,2``), the normalized (integer) wavenumbers ``|κᵢ| < Nᵢ / 2`` are resolved.
Since this range is symmetric around ``κᵢ=0``, there are always an odd number of resolved wavenumbers.
When specifying ``N₁`` and ``N₂``, an even number is therefore equivalent to the next-smaller odd number.

Along the third coordinate direction, the ``ζ``-range ``[0, 1]`` is split into ``N₃`` sections.
Numerical solutions are then computed either at the center ``ζ_C`` of each section or at the interfaces ``ζ_I`` between the sections, with

```math
ζ_C ∈ \left\{
\frac{1/2}{N₃}, \frac{3/2}{N₃}, …, \frac{N₃-1/2}{N₃}
\right\} \quad \text{and} \quad
ζ_I ∈ \left\{
\frac{1}{N₃}, \frac{2}{N₃}, …, \frac{N₃-1}{N₃}
\right\} \,.
```

This staggered arrangement can reduce discretization errors and avoid instabilities due to odd–even decoupling.
To simplify the notation, we also introduce the definitions
```math
Δζ ≡ \frac{1}{N₃}
\quad \text{and} \quad
ζ¯ ≡ ζ - \frac{Δζ}{2}
\quad \text{and} \quad
ζ⁺ ≡ ζ + \frac{Δζ}{2}
```
to refer to the grid spacing and the (staggered) neighbors of a ``ζ``-location.

A quantity ``q(x₁, x₂, x₃)`` is therefore represented by a vector ``\bm{\hat{q}}`` of the ``N₁×N₂×N₃`` modal/nodal values ``\hat{q}^{κ₁κ₂ζ_C}``, or the ``N₁×N₂×(N₃-1)`` values ``\hat{q}^{κ₁κ₂ζ_I}`` for quantities that are discretized on ``ζ_I`` nodes.
The elements of ``\bm{\hat{q}}`` are complex numbers but since ``q`` is real-valued, the values for the wavenumbers ``(-κ₁, -κ₂)`` are the complex conjugate of the values for ``(κ₁, κ₂)``.


## Physical Processes

The [`Model`](@ref) represents the dynamics of a number of quantities ``q_i`` that are each represented by a vector ``\bm{\hat{q}}_i`` as described in [Discretization](@ref).
Together they make up the state $\bm{\hat{s}} = \begin{pmatrix} \bm{\hat{q}₁}, \bm{\hat{q}₂}, … \end{pmatrix}^\intercal$. The state usually consists of the three components of the velocity field plus perhaps a number of scalar fields.

The evolution of this state is governed by an equation in the form of

```math
\frac{∂ \bm{\hat{s}}}{∂ t} =
\sum_i f_i(\bm{\hat{s}}, t) \,,
```

where $f_i$ are functions that describe the contribution of each physical process (molecular diffusion, advection, etc.) to the rate of change of the state.
The [`Model`](@ref) is configured with the desired subset of the [Available Processes](@ref), which then determine the dynamical behavior of a simulation.

While processes are free to choose how they compute their contribution ``f_i``, they generally rely on a shared approach for [discrete derivatives](@ref Derivatives) and [non-linear expressions](@ref Non-linear-Processes).
Some processes are implemented as a [Projections](@ref), i.e. they directly modify the state ``\bm{\hat{s}}`` instead of contributing to its rate of change.
These can be used to enforce invariants of the state, such as enforcing incompressibility with the [`Pressure`](@ref) process.


### Available Processes

Most processes have an argument `field::Symbol` that specifies the name of the quantity they act on.
By convention, the names used for the velocity components are `:vel1`, `:vel2`, and `:vel3`.

- [`MomentumAdvection`](@ref)
- [`MolecularDiffusion`](@ref)
- [`Pressure`](@ref)
- [`ConstantSource`](@ref)
- [`ConstantMean`](@ref)
- [`StaticSmagorinskyModel`](@ref)


### Derivatives

In principle, each process is free to define its own way of handling spatial
derivatives, as long as they can be applied to the [discretized form](@ref
Discretization) of the state.
However, the [available processes](@ref Available-Processes) all rely on exact
frequency-domain derivatives along the ``x₁``- and ``x₂``-directions and
second-order finite differences along the ``x₃``-direction.

To simplify the notation for discretized processes, the definitions
```math
∂₁(κ₁) ≡ \frac{i 2 π κ₁}{L₁}
\quad \text{and} \quad
∂₂(κ₂) ≡ \frac{i 2 π κ₂}{L₂}
\quad \text{and} \quad
∂₃(ζ) ≡ \left( Δζ \frac{\mathrm{d}x₃}{\mathrm{d}ζ} \bigg|_{ζ} \right)^{-1}
```
are introduced for the “derivation factors” that appear in expressions of discretized spatial derivatives.

This gives the following expressions for the discrete computation of derivatives:

```math
\widehat{\frac{∂^n q}{∂x₁^n}}^{κ₁κ₂ζ} = ∂₁(κ₁)^n \, \hat{q}^{κ₁κ₂ζ}
\quad \text{and} \quad
\widehat{\frac{∂^n q}{∂x₂^n}}^{κ₁κ₂ζ} = ∂₂(κ₂)^n \, \hat{q}^{κ₁κ₂ζ}
```

```math
\widehat{\frac{∂q}{∂x₃}}^{κ₁κ₂ζ} =
∂₃(ζ) \left( \hat{q}^{κ₁κ₂ζ⁺} − \hat{q}^{κ₁κ₂ζ¯} \right)
+ \mathcal{O}(Δζ²)
```

```math
\widehat{\frac{∂²q}{∂x₃²}}^{κ₁κ₂ζ} = ∂₃(ζ) \left(
∂₃(ζ⁺) \left( \hat{q}^{κ₁κ₂(ζ+Δζ)} − \hat{q}^{κ₁κ₂ζ} \right) −
∂₃(ζ¯) \left( \hat{q}^{κ₁κ₂ζ} − \hat{q}^{κ₁κ₂(ζ−Δζ)} \right)
\right) + \mathcal{O}(Δζ²)
```

Near the boundary, one-sided finite differences with the same order of accuracy are used where necessary.
To compute second derivatives at the first ``ζ_C``-location, ``q`` is extrapolated to ``ζ=\frac{−Δζ}{2}`` with

```math
\hat{q}^{κ₁κ₂(−Δζ/2)} =
\frac{16}{5}\, \hat{q}^{κ₁κ₂0}
− 3\, \hat{q}^{κ₁κ₂(Δζ/2)}
+ 1\, \hat{q}^{κ₁κ₂(3Δζ/2)}
− \frac{1}{5}\, \hat{q}^{κ₁κ₂(5Δζ/2)}
```

where ``\hat{q}^{κ₁κ₂0}`` is a Dirichlet boundary condition.
The upper boundary at ``ζ=1`` is treated in an analogous manner.

### Non-linear Processes

Performing non-linear operations on frequency-domain representations of
a quantity can be challenging.
Non-linear processes therefore rely on a physical-domain representation computed at ``N₁^\mathrm{PD} × N₂^\mathrm{PD}`` points.
These values can be obtained with an inverse FFT of the appropriate size.
After computing non-linear expressions at these points, the result is transformed back to a frequency-domain representation with a forward FFT.
Note that unless the physical-domain resolution is fine enough to represent all wavenumbers produced by the non-linear operations, such a computation will produce *aliasing errors* in addition to the truncation errors that are inherent to the frequency-domain discretization.


### Projections

Projection processes are defined as a linear function ``f_i(\bm{\hat{s}}, t) = F_i \, λ(\bm{\hat{s}}, t)`` of some quantity ``λ`` that enforces an affine constraint of the form ``C_i \bm{\hat{s}} = \bm{c}_i`` on the state ``\bm{\hat{s}}``.
The main role of projections is to accommodate the [`Pressure`](@ref) process for incompressible flows, but they can also be used for other functionality such as the [`ConstantMean`](@ref) source term.

In keeping with the discussion of [time-integration methods](@ref Time-Integration-Methods), the explicit form of ``f_i`` can be written as
```math
f_i(\bm{\hat{s}}, t)
= − F_i (C_i F_i)^{−1} C_i \sum_{j ≠ i} f_j(\bm{\hat{s}}, t) \,,
```
but the implementation relies on a projection function
```math
P_i(\bm{\hat{s}}) = \bm{\hat{s}} − F_i (C_i F_i)^{−1} \left(C_i \bm{\hat{s}} − \bm{c}_i\right)
```
instead.

!!! warning
    There is no special functionality to simultaneously enforce multiple
    constraints. If more than one projections are enabled, the
    ``P_i(\bm{\hat{s}})`` are applied consecutively. This is valid as long as
    all projections are independent with ``C_i F_j = 0`` for all combinations
    ``(P_i, P_j)`` of enabled projection.
