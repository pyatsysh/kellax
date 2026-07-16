# kellax — pseudo-arclength continuation and bifurcation analysis in JAX

*The first continuation package with gradients through the bifurcation diagram.*

[![CI](https://github.com/pyatsysh/kellax/actions/workflows/ci.yml/badge.svg)](https://github.com/pyatsysh/kellax/actions/workflows/ci.yml)

Trace the solution branch of **R(x, p) = 0** through folds. Supply a residual;
every Jacobian is automatic; the bordered Keller system stays non-singular
where naive parameter stepping jumps or stalls. The branch is then read as a
dynamical object: its spectrum classifies folds, branch points and Hopf
points, each refinable to Newton precision and — the part no other
continuation package ships — **differentiable with respect to the model**.

```python
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as np
from kellax import arclength_continuation, fold_sensitivity

R = lambda x, p: np.array([x[0]**3 - x[0] + p])          # canonical S-curve
br = arclength_continuation(R, x0 = np.array([-1.2]), p0 = 0.7, ds = 0.05)
br.p[br.turning_points]                                   # folds at +-2/(3*sqrt(3))

R3 = lambda x, p, th: np.array([x[0]**3 - th[0]*x[0] + p])
x, p, v, dp, res = fold_sensitivity(R3, np.array([0.6]), 0.4, np.array([1.0]))
dp                                                        # d(fold)/d(theta) = sqrt(1/3), exactly
```

![Swift–Hohenberg snaking](figures/swift_hohenberg.png)

## Why

Numerical continuation is how one maps multiplicity and hysteresis — phase
transitions, ignition thresholds, pattern selection — and its mature software
lives outside the modern Python stack: Fortran (AUTO-07p), MATLAB (MatCont,
COCO, pde2path), C++ (LOCA/Trilinos). The one modern, autodiff-native tool is
Julia's BifurcationKit.jl — considerably more complete than kellax on
periodic orbits (where roughly half its source is invested); kellax does not
claim to displace it. In the Python/JAX ecosystem, however, there is no
comparable tool: the available options are finite-difference and small, stale
wrappers around AUTO, or matrix-free but autodiff-less. kellax asks only for
the residual.

kellax began as the isotherm engine of a classical density-functional-theory
code, tracing capillary-condensation loops and locating spinodals as external
conditions vary. It is model-agnostic: any smooth R(x, p) — steady states of
discretised PDEs, reaction networks, phase equilibria, fixed points of
learned dynamics.

## Why JAX, specifically

The substrate is the feature. Continuation is built out of derivatives, and
JAX supplies every order of them from the residual alone:

- **First derivatives** drive the engines: `jax.jacfwd` for the dense
  bordered solves, `jax.linearize` JVPs for the matrix-free GMRES path.
- **Second derivatives** make the exact bifurcation systems free: the
  Moore–Spence fold block d(R_x v)/dx and the Hopf system's Jacobian are
  autodiff of code written once, for the residual. In classical packages this
  is hand-assembled operator calculus.
- **The deflation operator's product-rule Jacobian** — the reason deflated
  Newton implementations grow long — is one more `jax.jacfwd`.
- **Implicit differentiation of the converged Moore–Spence system** turns the
  fold location into a differentiable function of the model parameters
  (`fold_sensitivity`): one linear solve against the Jacobian Newton already
  used. Gradients *through* a bifurcation diagram — so a model can be
  optimised or learned against it — ship here as a library call; in the Julia
  ecosystem this capability lives in a separate research package.

The whole predictor-corrector jit-compiles and runs unchanged on CPU or GPU,
in float64 throughout.

## What ships (0.4.0)

- `arclength_continuation` — dense Keller trace with adaptive steps and fold
  detection; returns a `Branch` (states, parameters, tangents, turning points).
- `mf_arclength_continuation` — the matrix-free twin: preconditioned GMRES on
  `jax.linearize` JVPs, a `precond` hook, `p_stop` landing; for the
  10^4–10^6-dof fields where the Jacobian cannot be formed.
- `refine_fold` / `track_fold` — Moore–Spence refinement of a detected fold to
  Newton precision, and continuation of the fold itself in a second parameter
  (arclength on the augmented system, so cusps are passed and reported).
- `analyze_branch` / `branch_eigenvalues` — the spectrum along the branch:
  stability, and classification of every axis crossing into fold, branch
  point, or Hopf candidate.
- `refine_hopf` — a Hopf point pinned exactly by the (3N+2) standard
  augmented system, second derivatives by autodiff.
- `deflated_newton` / `deflated_search` — converge to the solutions you have
  NOT already found (Farrell-style deflation); the route to disconnected
  branches.
- `branch_off` / `bifurcation_diagram` — jump onto the bifurcating branch at
  a simple branch point by deflation, and a bounded-depth recursive driver:
  trace, classify, switch, recurse (equilibria).
- `fold_sensitivity` — **differentiable continuation**: the exact gradient of
  a fold location with respect to model parameters, by implicit
  differentiation of the converged Moore–Spence system.
- `bordered_newton` / `newton` — the generic (N+k) bordered primitive, and
  plain Newton for seeding.

Every claim is validated against an analytic anchor or the literature: the
cubic folds to 1e-10; the two-parameter fold law to 1e-8; the fold gradient
to 1e-9 against the closed-form law d p*/d theta = sqrt(theta/3); the Hopf
normal form and the Brusselator at exactly b = 1 + a^2, omega = a; the
pitchfork diagram with both arms on x^2 = p to 1e-8; Bratu ignition at
lambda* = 3.5138 (1-D) and 6.808 (2-D); the CSTR ignition/extinction pair at
the Uppal–Ray–Poore values with its fold-curve tangents reproduced by
autodiff; MatCont's predator–prey folds to five digits; and a 38-fold
Swift–Hohenberg snaking branch traced in one continuation.

## The book

[*kellax by solved problems*](book/README.md) — 9 chapters, each one worked
problem: [the fold](book/01-the-fold.md), [the cusp](book/02-the-cusp.md),
[Bratu–Gelfand](book/03-bratu.md), [matrix-free scaling](book/04-matrix-free.md),
[homoclinic snaking](book/05-snaking.md), [CSTR hysteresis](book/06-cstr.md),
[a predator–prey fold pair](book/07-predator-prey.md),
[Bratu in 2-D](book/08-bratu-2d.md), and
[differentiable continuation](book/09-differentiable.md). Each chapter's
script lives in [`examples/`](examples) and regenerates its figure; the
printed numbers are real output. The [TUTORIAL](TUTORIAL.md) is the faster,
API-first tour.

## Install / test

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[test,examples]"
.venv/bin/python tests/test_kellax.py && .venv/bin/python tests/test_bifurcations.py
python examples/cubic_fold.py          # -> figures/cubic_fold.png
```

kellax needs float64: `jax.config.update("jax_enable_x64", True)` once at the
top of every script.

## Roadmap

Periodic-orbit continuation from Hopf points (shooting first) — the front
where BifurcationKit.jl is the benchmark; Hopf tracking in a second
parameter; matrix-free eigenvalues (Arnoldi on JVPs) so the classification
layer reaches 10^4+ dof; deflated continuation; a `jax.custom_jvp` fold
location, so `jax.grad` composes through the diagram directly. The long-term
goal stands: models optimised and learned against their bifurcation diagrams.

## Citing

See [`CITATION.cff`](CITATION.cff). License: Apache-2.0.
