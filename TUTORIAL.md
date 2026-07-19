# kellax tutorial

This tutorial is a fast, API-first tour of the library. For the longer
treatment through worked problems see the book,
[*kellax by solved problems*](book/README.md).

kellax traces the solution branch of a nonlinear system **R(x, p) = 0** as the
scalar parameter *p* varies. The branch is followed through **folds** (turning
points) at which the branch reverses and the naive stepping of the parameter
fails. You supply only the residual. Every Jacobian is obtained from it by
automatic differentiation (JAX). A dense engine and a matrix-free engine are
provided and both are driven by the same residual. The traced branch is then
read as a dynamical object: its spectrum classifies folds, branch points and
Hopf candidates. Each detected point can be refined to Newton precision. The
location of a fold can even be differentiated with respect to the parameters
of the model.

## Install

We install into a fresh virtual environment:

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python matplotlib   # for the examples' figures
```

The second line adds matplotlib, which is needed only for the figures of the
examples. kellax needs float64. Enable it once at the top of every script:

```python
import jax; jax.config.update("jax_enable_x64", True)
```

That is the only piece of global configuration.

## 60 seconds: trace a fold

We trace the S-curve of the cubic through both of its folds.

```python
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as np
from kellax import arclength_continuation

R = lambda x, p: np.array([x[0]**3 - x[0] + p])           # the cubic S-curve
br = arclength_continuation(R, x0 = np.array([-1.2]), p0 = 0.7, ds = 0.05, direction = -1.0)

print(br.p[br.turning_points])        # parameter values at the folds: ~ [-0.385, +0.385]
```

The two printed values are the locations of the folds of the cubic. The call
itself returns a `Branch`. The field `.x` holds the M×N states and the field
`.p` holds the M parameter values. The p-component of the tangent is stored in
`.tan_p` and a sign flip of it marks a fold. The accepted step sizes are
stored in `.ds`. Finally, `.turning_points` holds the indices of the detected
folds. We shall use this branch again below.

## The API, in one screen

The whole of the API fits in the table below.

| Call | Does | Returns |
|---|---|---|
| `arclength_continuation(R, x0, p0, …)` | trace a branch through folds (dense) | `Branch` |
| `mf_arclength_continuation(R, x0, p0, …, precond=, p_stop=)` | same, **matrix-free** (GMRES on JVPs) for large N | `Branch` |
| `refine_fold(R, x0, p0, v0=)` | pin a detected fold to Newton precision (Moore–Spence) | `(x, p, v, res)` |
| `track_fold(R2, x0, p0, q0, …)` | continue a fold in a second parameter `q` | `FoldBranch` |
| `bordered_newton(R, constraints, x0, aux0)` | generic (N+k) bordered Newton | `(x, aux, res)` |
| `newton(R, x0, p)` | plain dense Newton at fixed p (branch seeding) | `(x, res)` |
| `analyze_branch(R, br, sign=)` | spectrum along the branch; classify folds / branch points / Hopf candidates | `BranchAnalysis` |
| `refine_hopf(R, x0, p0)` | pin a Hopf point exactly (3N+2 augmented system) | `(x, p, omega, q, res)` |
| `deflated_newton(R, x0, p, known)` / `deflated_search(…)` | converge to a solution NOT already known / collect all of them | `(x, res)` / `[x…]` |
| `branch_off(R, x_bp, p_bp)` | jump onto the bifurcating branch at a simple branch point | `[(x, p)…]` seeds |
| `bifurcation_diagram(R, x0, p0, max_depth=)` | trace → classify → switch → recurse (equilibria) | `[DiagramBranch…]` |
| `fold_sensitivity(R3, x0, p0, theta0)` | fold location AND its exact gradient d p*/d theta | `(x, p, v, dp, res)` |

The continuation drivers share their optional arguments. The initial step is
`ds`. The bounds `ds_min` and `ds_max` limit the adaptive step and `n_steps`
limits the number of steps. The trace stops when p leaves the window between
`p_min` and `p_max`. The flag `direction` selects the sense of the trace: +1
toward increasing p and −1 toward decreasing p. The tolerances of the Newton
solves are set by `newton_tol` and `accept_res`. All of them can be left at
their defaults.

Notice that the table also lists the bifurcation layer of the library. The
call `analyze_branch` computes the spectrum along a traced branch and
classifies the crossings into folds and branch points and Hopf candidates.
The call `refine_hopf` then pins a Hopf point exactly. Deflation allows us to
converge to solutions not already known and `branch_off` jumps onto the
bifurcating branch at a simple branch point. The driver `bifurcation_diagram`
composes these steps into a recursive trace of the equilibria. Finally,
`fold_sensitivity` returns the location of a fold together with its exact
gradient with respect to the parameters of the model.

## Sharpen a fold

Continuation only *brackets* a fold between accepted points. The call
`refine_fold` then solves the Moore–Spence augmented system and locates the
fold exactly:

```python
from kellax import refine_fold
i = br.turning_points[0]
x, p, v, res = refine_fold(R, np.array(br.x[i]), float(br.p[i]))
#   p -> -0.3849001795 (exact -2/(3*sqrt 3)), residual ~ 1e-16
```

Notice that the printed residual is at machine precision. The exact location
of the fold is printed alongside it for comparison. The refined fold can then
serve as a seed for `track_fold` below.

## Track a fold in a second parameter

A fold moves as a second parameter `q` changes. The call `track_fold`
continues it:

```python
from kellax import track_fold
R2 = lambda x, p, q: np.array([x[0]**3 - q*x[0] + p])
fb = track_fold(R2, np.array([0.6]), p0 = 0.4, q0 = 1.0, ds = 0.1, q_min = 0.02, q_max = 2.5)
#   fb.q, fb.p trace the fold curve p*(q); fb.x, fb.v the fold states and null vectors
```

The returned object is a `FoldBranch`. The fields `fb.q` and `fb.p` trace the
curve of the fold and the fields `fb.x` and `fb.v` hold the states and the
null vectors. The curve of the fold is itself parametrised by arclength.
Therefore, `track_fold` passes through **cusps** (where `dq/ds = 0`) and
reports them as turning points of the augmented branch. This is worked out in
[book chapter 2](book/02-the-cusp.md).

## Scale up: matrix-free

For a discretised 2D/3D field the Jacobian cannot be formed. The driver
`mf_arclength_continuation` solves every bordered system by preconditioned
GMRES on Jacobian–vector products (from `jax.linearize`). The engine never
forms a matrix. Thus, N can reach $10^4$–$10^6$:

```python
from kellax import mf_arclength_continuation
br = mf_arclength_continuation(R, x0, p0, precond = Minv, p_stop = target_p)
```

The argument `precond` is a callable `Minv(v)`. It approximates the inverse of
the state-block Jacobian and a spectral/FFT preconditioner is a typical
choice. The argument `p_stop` sizes the final step so as to land exactly on a
target parameter value.

## Where to go next

Work the book chapters in order. Begin with [the fold](book/01-the-fold.md)
and [the cusp](book/02-the-cusp.md). Continue to
[Bratu–Gelfand](book/03-bratu.md) and to
[matrix-free scaling](book/04-matrix-free.md). The method part of the book
ends with [snaking](book/05-snaking.md). Then come the applied problems:
[CSTR hysteresis](book/06-cstr.md), [predator–prey](book/07-predator-prey.md),
and [2-D Bratu](book/08-bratu-2d.md). Each of them is a runnable script in
[`examples/`](examples) that regenerates its figure. Finally,
[differentiable continuation](book/09-differentiable.md) computes the exact
gradient of a located fold.
