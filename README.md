# kellax — pseudo-arclength continuation in JAX

Trace the solution branch of **R(x, p) = 0** through folds. Supply a residual;
Jacobians (dR/dx and dR/dp) are automatic; the bordered Keller system stays
non-singular where naive parameter stepping jumps or stalls; turning points
are detected and reported.

```python
import jax.numpy as jnp
from kellax import arclength_continuation

R = lambda x, p: jnp.array([x[0]**3 - x[0] + p])        # canonical S-curve
br = arclength_continuation(R, x0=jnp.array([-1.2]), p0=0.7, ds=0.05)
br.p[br.turning_points]                                  # folds at +-2/(3*sqrt(3))
```

`Branch` carries the states, parameter values, unit-tangent p-components
(sign flips mark folds), and accepted step sizes. A plain dense `newton`
(fixed p) is included for branch seeding.

## Why

Continuation died with LOCA (the standalone library behind Sandia's Tramonto)
in the Python-adjacent world; Julia has BifurcationKit; **JAX has nothing** —
despite JAX being exactly the substrate where the Jacobians, the bordered
solves, and (roadmap) the *differentiability of the branch itself* come for
free. kellax originated as the isotherm engine of the `cdft` classical-DFT
toolbox (capillary condensation loops, spinodals) and is model-agnostic: any
smooth R(x, p) — steady states of PDEs, reaction networks, neural-ODE
equilibria.

## Shipped

- Moore–Spence fold tracking (`refine_fold`, `track_fold`) — sharpen a
  detected turning point to Newton precision, then continue the fold itself
  in a second parameter (arclength on the augmented system, so cusps pass).
- Generic bordered constraints (`bordered_newton`) — one dense Newton, k
  border rows: arclength, mass constraints, phase conditions.

## Roadmap

Matrix-free bordered solves (jvp + GMRES) for large N;
**differentiable continuation** — gradients of fold locations with respect to
model parameters, enabling optimisation and learning directly against
bifurcation diagrams; branch switching.

## Install / test

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e .
.venv/bin/python tests/test_kellax.py
```
