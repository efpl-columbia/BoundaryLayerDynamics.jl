# Wall Models

## Algebraic Equilibrium Rough-Wall Model

The current rough-wall model assumes that near the wall the components
``\tau_{13}^\mathrm{w}`` and ``\tau_{23}^\mathrm{w}`` are constant in wall-normal direction and aligned
with the velocity components ``u_1`` and ``u_2``, i.e.

``\tau_{i3}^\mathrm{w}(x_1, x_2) = \alpha \, u_i(x_1, x_2, x_3) \quad \text{for } i=1,2 \text{ and small } x₃,``

and can be described reasonably accurately with a (local) log-law relation

```math
u_h(x_1, x_2, x_3) = \frac{u_\tau(x_1, x_2)}{\kappa} \, \mathrm{log} \left( \frac{x_3}{z_0} \right),
```

where ``u_h = \sqrt{u_1^2 + u_2^2}`` and ``u_\tau^2 = \sqrt{(\tau_{13}^\mathrm{w})^2 + (\tau_{23}^\mathrm{w})^2}``.
Note that this defines the domain as starting at ``x_3 = 0``, but the
relation only holds from ``x_3 ≥ z_0``.
If we define the computational domain to start at the level of the roughness
length, the log-law relation would change to
``\mathrm{log} \left( \frac{x_3 - z_0}{z_0} \right)``.
However, this definition generalizes poorly to the case where additional scalar
quantities are included as each quantity might have a different roughness
length.

With this, we can obtain expressions for ``\alpha`` and thus for the two
components of the wall stress, ``\tau_{13}^\mathrm{w} = u_\tau^2 \frac{u_1}{u_h}``, and
``\tau_{23}^\mathrm{w} = u_\tau^2 \frac{u_2}{u_h}``.

```math
\begin{aligned}
  \tau_{13}^\mathrm{w}(x_1, x_2) &=
  \frac{\kappa^2 u_h(x_1, x_2, x_3)}{\mathrm{log}^2(x_3/z_0)} \, u_1(x_1, x_2, x_3)
  \\
  \tau_{23}^\mathrm{w}(x_1, x_2) &=
  \frac{\kappa^2 u_h(x_1, x_2, x_3)}{\mathrm{log}^2(x_3/z_0)} \, u_2(x_1, x_2, x_3)
\end{aligned}
```

By selecting a reference height ``x_3^\mathrm{ref}``, generally the first grid point, we
have an algebraic relation for the wall stress based on the (resolved)
near-wall velocity.
Ideally the resulting wall stress is insensitive to the exact value of the
reference height.

Since the velocity gradients become very large near the wall, they are poorly
resolved and should also be modeled, which can be done based on the same
assumptions with

```math
\frac{\partial u_1}{\partial x_3} =
\frac{u_1}{x_3 \, \mathrm{log}(x_3/z_0)}
\quad \text{and} \quad
\frac{\partial u_2}{\partial x_3} =
\frac{u_2}{x_3 \, \mathrm{log}(x_3/z_0)} \, .
```

These can be obtained from the fact that ``u_i/u_h`` is (assumed) constant in
wall-normal direction (equal to ``\tau_{i3}^\mathrm{w}/u_\tau^2``).
