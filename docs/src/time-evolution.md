# Evolution in Time

Once a [`Model`](@ref) is set up, it can be advanced in time with the `evolve!` function.

```@docs
evolve!
```

## Time-Integration Methods

The time-integration methods are based on an extension of the standard formulation ``\mathrm{d}\bm{\hat{s}}/\mathrm{d}t = f(\bm{\hat{s}}, t)`` that accommodates the role that [pressure](@ref Pressure/Continuity) plays in incompressible flows.
The methods solve a semi-discretized problem

```math
\frac{\mathrm{d}\bm{\hat{s}}}{\mathrm{d}t} =
f_r(\bm{\hat{s}}, t) + f_p(\bm{\hat{s}}, t)
```

where the right-hand-side term ``f_r`` is computed directly from the state ``\bm{\hat{s}}`` while ``f_p`` is computed indirectly through a projection ``P(\bm{\hat{s}})`` that can be used to enforce invariants of the state ``\bm{\hat{s}}``.

!!! warning
    None of the currently implemented processes are time-dependent, i.e.
    ``f(\bm{\hat{s}}, t) = f(\bm{\hat{s}})``. It is likely that there are still errors in the handling of time-dependent processes and careful testing is highly recommended when implementing such processes.

### Explicit Linear Multi-Step Methods

Explicit linear multi-step methods take the form
```math
\bm{\hat{s}}^{(n+1)} =
\sum_{i=1}^{m} \left(
α_i \bm{\hat{s}}^{(n+1-i)} +
Δt β_i f(\bm{\hat{s}}^{(n+1-i)}, t_i)
\right) \,,
```
where each method is defined by its coefficients ``m``, ``α_i``, and ``β_i`` and ``\sum_{i=1}^m α_i = 1`` is required for consistency.
The projection-based formulation of such a method is
```math
\bm{\hat{s}}^{(n+1)} =
P \left(
\sum_{i=1}^{m} \left(
α_i \bm{\hat{s}}^{(n+1-i)} +
Δt β_i f_r(\bm{\hat{s}}^{(n+1-i)}, t_i)
\right) \right) \,.
```

```@docs
Euler
AB2
```

### Explicit Runge–Kutta Methods

Explicit ``m``-stage Runge–Kutta methods can be written as
```math
\bm{\hat{s}}^{(n, i)} =
\sum_{k=0}^{i-1} \left(
α_{ik} \bm{\hat{s}}^{(n,k)} +
Δt β_{ik} f\left(\bm{\hat{s}}^{(n,k)}, t^{(n,k)}\right)
\right)
```
for ``i=1,…,m``, with ``\bm{\hat{s}}^{(n,0)} = \bm{\hat{s}}^{(n)}`` and ``\bm{\hat{s}}^{(n+1)} = \bm{\hat{s}}^{(n,m)}``.
``\sum_{k=0}^{i-1} α_{ik} = 1`` is required for consistency.
The projection-based formulation of such a method is
```math
\bm{\hat{s}}^{(n, i)} =
P \left(
\sum_{k=0}^{i-1} \left(
α_{ik} \bm{\hat{s}}^{(n,k)} +
Δt β_{ik} f_r\left(\bm{\hat{s}}^{(n,k)}, t^{(n,k)}\right)
\right)
\right) \,.
```

```@docs
SSPRK22
SSPRK33
```

## Output Modules

Output modules can collect information about a simulation while it is running.
At each time step, they are provided with the state as well as any other fields
that [processes](@ref Physical-Processes) provide to the output modules by
calling the `log_sample!` function.

```@docs
ProgressMonitor
StepTimer
MeanProfiles
Snapshots
```
