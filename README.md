# kellax — pseudo-arclength continuation in JAX

[![CI](https://github.com/pyatsysh/kellax/actions/workflows/ci.yml/badge.svg)](https://github.com/pyatsysh/kellax/actions/workflows/ci.yml)

Trace the solution branch of **R(x, p) = 0** through folds. Supply a residual;
every Jacobian is automatic; the bordered Keller system stays non-singular
where naive parameter stepping jumps or stalls; turning points are detected,
refined to Newton precision, and can themselves be continued in a second
parameter.

```python
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as np
from kellax import arclength_continuation

R = lambda x, p: np.array([x[0]**3 - x[0] + p])          # canonical S-curve
br = arclength_continuation(R, x0 = np.array([-1.2]), p0 = 0.7, ds = 0.05)
br.p[br.turning_points]                                   # folds at +-2/(3*sqrt(3))
```

![Swift–Hohenberg snaking](figures/swift_hohenberg.png)

## Why

Numerical continuation is how one maps multiplicity and hysteresis — phase
transitions, ignition thresholds, pattern selection — and its mature software
lives outside the modern Python stack: Fortran (AUTO-07p), MATLAB (MatCont,
COCO, pde2path), C++ (LOCA/Trilinos). The one modern, autodiff-native tool is
Julia's BifurcationKit.jl — considerably more complete than kellax (periodic
orbits, Hopf, branch switching); kellax does not claim to displace it. In the
Python/JAX ecosystem, however, there is no comparable tool: the available
options are finite-difference and small, stale wrappers around AUTO, or
matrix-free but autodiff-less. kellax asks only for the residual —
`jax.jacfwd` and `jax.linearize` supply every derivative, and `jit`, GPU
execution, and float64 come with the substrate.

kellax began as the isotherm engine of a classical density-functional-theory
code, tracing capillary-condensation loops and locating spinodals as external
conditions vary. It is model-agnostic: any smooth R(x, p) — steady states of
discretised PDEs, reaction networks, phase equilibria, fixed points of
learned dynamics.

## What ships (0.3.0)

- `arclength_continuation` — dense Keller trace with adaptive steps and fold
  detection; returns a `Branch` (states, parameters, tangents, turning points).
- `mf_arclength_continuation` — the matrix-free twin: preconditioned GMRES on
  `jax.linearize` JVPs, a `precond` hook, `p_stop` landing on a target
  parameter; for the 10^4–10^6-dof fields where the Jacobian cannot be formed.
- `refine_fold` / `track_fold` — Moore–Spence refinement of a detected fold to
  Newton precision, and continuation of the fold itself in a second parameter
  (arclength on the augmented system, so cusps are passed and reported).
- `bordered_newton` — the generic (N+k) bordered primitive: arclength, mass
  constraints, phase conditions.
- `newton` — plain dense Newton at fixed p, for branch seeding.

Every claim is validated in the test suite and the examples: the cubic folds
to 1e-10, the two-parameter fold law to 1e-8, Bratu ignition at
lambda* = 3.5138 (1-D) and 6.808 (2-D square), the CSTR ignition/extinction
pair to the Uppal–Ray–Poore values, MatCont's predator–prey folds to five
digits, and a 38-fold Swift–Hohenberg snaking branch traced in one
continuation.

## The book

[*kellax by solved problems*](book/README.md) — 8 chapters, each one worked
problem: [the fold](book/01-the-fold.md), [the cusp](book/02-the-cusp.md),
[Bratu–Gelfand](book/03-bratu.md), [matrix-free scaling](book/04-matrix-free.md),
[homoclinic snaking](book/05-snaking.md), [CSTR hysteresis](book/06-cstr.md),
[a predator–prey fold pair](book/07-predator-prey.md), and
[Bratu in 2-D](book/08-bratu-2d.md). Each chapter's script lives in
[`examples/`](examples) and regenerates its figure; the printed numbers are
real output. The [TUTORIAL](TUTORIAL.md) is the faster, API-first tour.

## Install / test

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[test,examples]"
.venv/bin/python tests/test_kellax.py
python examples/cubic_fold.py          # -> figures/cubic_fold.png
```

kellax needs float64: `jax.config.update("jax_enable_x64", True)` once at the
top of every script.

## Roadmap

Matrix-free fold refinement; branch switching at simple bifurcation points;
Hopf detection. The long-term goal is **differentiable continuation** — fold
locations as differentiable functions of model parameters, by implicit
differentiation of the converged bordered system rather than by unrolling
Newton — so that a model can be optimised or learned directly against its
bifurcation diagram.

## Citing

See [`CITATION.cff`](CITATION.cff). License: Apache-2.0.
