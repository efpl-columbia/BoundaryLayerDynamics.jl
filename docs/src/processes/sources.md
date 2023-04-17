# Uniform Sources/Sinks

!!! note
    Refer to [Physical Processes](@ref) for general information about processes and their implementation.

The uniform sources add a constant value ``S`` to the rate of change of a quantity ``q``.
This contribution is uniform in space and therefore only affects the ``(0, 0)`` wavenumber pairs.
If ``S`` is negative, the term acts as a sink rather than a source.

## Source/Sink with Constant Strength

Source term for a scalar quantity ``q`` with source strength ``S`` that is constant in space and time, i.e.

```math
\frac{∂q}{∂t} = … + S \,.
```

The process is discretized with

```math
f_i(\hat{q}^{00ζ}) = S \,.
```

```@docs
ConstantSource
```


## Source/Sink to Maintain Constant Mean

```@docs
ConstantMean
```

!!! note
    `ConstantMean` is currently only implemented for ``ζ_C``-nodes.

## Contributions to Budget Equations

Contribution to the instantaneous momentum equation:

```math
\frac{\partial}{\partial t} u_i = … + f_i
```

Contribution to the mean momentum equation:

```math
\frac{\partial}{\partial t} \overline{u_i} = … + \overline{f_i}
```

Contribution to the turbulent momentum equation:

```math
\frac{\partial}{\partial t} u_i^\prime = … + f_i^\prime
```

Contribution to the mean kinetic energy equation:

```math
\frac{\partial}{\partial t} \frac{\overline{u_i}^2}{2} = …
+ \overline{u_i} \overline{f_i}
```

Contribution to the turbulent kinetic energy equation:

```math
\frac{\partial}{\partial t} \frac{\overline{u_i^\prime u_i^\prime}}{2} = …
+ \overline{u_i^\prime f_i^\prime}
```

!!! note
    For the `ConstantSource` process, ``\overline{f_i}=S`` and ``f_i^\prime=0``, but for the `ConstantMean` process this may depend on the exact definition of the averaging operation.
