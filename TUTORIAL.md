# kellax tutorial

A fast, API-first tour. For the longer, worked-problem treatment see the book,
[*kellax by solved problems*](book/README.md).

kellax traces the solution branch of a nonlinear system **R(x, p) = 0** as the
scalar parameter *p* varies — through **folds** (turning points), where the branch
reverses and naive parameter-stepping fails. You supply only the residual; every
Jacobian is automatic (JAX).

## Install

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python matplotlib   # for the examples' figures
```

kellax needs float64 — enable it once at the top of every script:

```python
import jax; jax.config.update("jax_enable_x64", True)
```

## 60 seconds: trace a fold

```python
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as np
from kellax import arclength_continuation

R = lambda x, p: np.array([x[0]**3 - x[0] + p])           # the cubic S-curve
br = arclength_continuation(R, x0 = np.array([-1.2]), p0 = 0.7, ds = 0.05, direction = -1.0)

print(br.p[br.turning_points])        # parameter values at the folds: ~ [-0.385, +0.385]
```

`arclength_continuation` returns a `Branch` with `.x` (M×N states), `.p` (M
parameter values), `.tan_p` (the tangent's p-component; sign flips mark folds),
`.ds` (accepted step sizes), and `.turning_points` (indices of detected folds).

## The API, in one screen

| Call | Does | Returns |
|---|---|---|
| `arclength_continuation(R, x0, p0, …)` | trace a branch through folds (dense) | `Branch` |
| `mf_arclength_continuation(R, x0, p0, …, precond=, p_stop=)` | same, **matrix-free** (GMRES on JVPs) for large N | `Branch` |
| `refine_fold(R, x0, p0, v0=)` | pin a detected fold to Newton precision (Moore–Spence) | `(x, p, v, res)` |
| `track_fold(R2, x0, p0, q0, …)` | continue a fold in a second parameter `q` | `FoldBranch` |
| `bordered_newton(R, constraints, x0, aux0)` | generic (N+k) bordered Newton | `(x, aux, res)` |
| `newton(R, x0, p)` | plain dense Newton at fixed p (branch seeding) | `(x, res)` |

Key optional arguments to the continuation drivers: `ds` (initial step),
`ds_min`/`ds_max` (adaptive bounds), `n_steps`, `p_min`/`p_max` (stop when p
leaves the window), `direction` (+1 toward increasing p, −1 decreasing),
`newton_tol`, `accept_res`.

## Sharpen a fold

Continuation only *brackets* a fold between accepted points. `refine_fold` solves
the Moore–Spence augmented system to locate it exactly:

```python
from kellax import refine_fold
i = br.turning_points[0]
x, p, v, res = refine_fold(R, np.array(br.x[i]), float(br.p[i]))
#   p -> -0.3849001795 (exact -2/(3*sqrt 3)), residual ~ 1e-16
```

## Track a fold in a second parameter

A fold moves as a second parameter `q` changes. `track_fold` continues it:

```python
from kellax import track_fold
R2 = lambda x, p, q: np.array([x[0]**3 - q*x[0] + p])
fb = track_fold(R2, np.array([0.6]), p0 = 0.4, q0 = 1.0, ds = 0.1, q_min = 0.02, q_max = 2.5)
#   fb.q, fb.p trace the fold curve p*(q); fb.x, fb.v the fold states and null vectors
```

Because the fold curve is itself parametrised by arclength, `track_fold` passes
through **cusps** (where `dq/ds = 0`) and reports them as turning points of the
augmented branch — see [book chapter 2](book/02-the-cusp.md).

## Scale up: matrix-free

For a discretised 2D/3D field the Jacobian cannot be formed. `mf_arclength_continuation`
solves every bordered system by preconditioned GMRES on Jacobian–vector products
(from `jax.linearize`), so N can reach $10^4$–$10^6$:

```python
from kellax import mf_arclength_continuation
br = mf_arclength_continuation(R, x0, p0, precond = Minv, p_stop = target_p)
```

`precond` is a callable `Minv(v)` approximating the inverse of the state-block
Jacobian (e.g. a spectral/FFT preconditioner); `p_stop` sizes the final step to
land exactly on a target parameter value.

## Where to go next

Work the book chapters in order — [the fold](book/01-the-fold.md) →
[the cusp](book/02-the-cusp.md) → [Bratu–Gelfand](book/03-bratu.md) →
[matrix-free scaling](book/04-matrix-free.md) → [snaking](book/05-snaking.md) —
then the applied problems: [CSTR hysteresis](book/06-cstr.md),
[predator–prey](book/07-predator-prey.md), and [2-D Bratu](book/08-bratu-2d.md).
Each is a runnable script in [`examples/`](examples) that regenerates its figure.
