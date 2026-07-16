# kellax — pseudo-arclength continuation and bifurcation analysis in JAX

[![CI](https://github.com/pyatsysh/kellax/actions/workflows/ci.yml/badge.svg)](https://github.com/pyatsysh/kellax/actions/workflows/ci.yml)

kellax traces solution branches of **R(x, p) = 0** through folds. The user
supplies the residual. Every derivative the method needs is obtained from it
by automatic differentiation, and the bordered Keller formulation stays
non-singular where naive parameter stepping jumps or stalls. The traced branch
is then read as a dynamical object: its spectrum classifies folds, branch
points and Hopf candidates. Each detected point can be refined to Newton
precision and — for the first time in a continuation package — differentiated
with respect to the parameters of the model.

```python
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as np
from kellax import arclength_continuation, refine_fold, fold_sensitivity

R = lambda x, p: np.array([x[0]**3 - x[0] + p])          # canonical S-curve
br = arclength_continuation(R, x0 = np.array([-1.2]), p0 = 0.7, ds = 0.05,
                            ds_max = 0.05, direction = -1.0)   # through both folds
for i in br.turning_points:
    x_f, p_f, v_f, res = refine_fold(R, br.x[i], float(br.p[i]))
    print(p_f)                                            # -+2/(3*sqrt(3)), to 1e-10

R3 = lambda x, p, th: np.array([x[0]**3 - th[0]*x[0] + p])
x, p, v, dp, res = fold_sensitivity(R3, np.array([0.6]), 0.4, np.array([1.0]))
dp                                                        # d(fold)/d(theta) = sqrt(1/3), exactly
```

![Swift–Hohenberg snaking](figures/swift_hohenberg.png)

*The homoclinic snaking branch of the Swift–Hohenberg equation: 38 folds
passed in one continuation.
[`examples/swift_hohenberg.py`](examples/swift_hohenberg.py) regenerates the
figure.*

## Motivation

Numerical continuation is the standard tool for mapping out multiplicity and
hysteresis: phase transitions, ignition thresholds, pattern selection. Its
mature implementations live outside the modern Python stack — AUTO-07p in
Fortran; MatCont, COCO and pde2path in MATLAB; LOCA/Trilinos in C++. The one
modern autodiff-native package is BifurcationKit.jl in Julia. It remains
considerably more complete than kellax on periodic orbits and kellax does not
attempt to displace it. In the Python/JAX ecosystem a comparable tool is
elusive. The options are young finite-difference codes (pycont-lite), stale
wrappers around AUTO (PyDSTool/PyCont) and the matrix-free but autodiff-less
pacopy. kellax fills this gap.

kellax began as the isotherm engine of a classical density-functional-theory
code. It traced capillary condensation loops and located spinodals as the
external conditions varied. Nothing in the library is specific to that origin:
any smooth R(x, p) will do, from steady states of discretised PDEs to reaction
networks, phase equilibria and fixed points of learned dynamics.

## Why JAX

Continuation is built out of derivatives, and JAX supplies every order of them
from the residual alone. The first derivatives drive the two engines:
`jax.jacfwd` assembles the dense bordered solves, and `jax.linearize` provides
the JVPs for the matrix-free GMRES path. The exact fold and Hopf systems
require second derivatives. These are equally automatic: the Moore–Spence
block d(R_x v)/dx and the Jacobian of the Hopf system are autodiff of code
written once for the residual. In the classical packages the same objects are
hand-assembled operator calculus. The Jacobian of the deflation operator is
one more `jax.jacfwd` call; that product-rule term is the reason
deflated-Newton implementations grow long. Finally, implicit differentiation
of the converged Moore–Spence system turns the fold location into a
differentiable function of the model parameters (`fold_sensitivity`). It costs
one extra linear solve against the Jacobian Newton has already used. In return
the fold acquires an exact gradient: a model can be optimised or learned
against its own bifurcation diagram. The whole predictor–corrector
jit-compiles and runs unchanged on CPU or GPU, in float64 throughout.

## The toolbox (v0.4.0)

- `arclength_continuation` — the dense Keller trace with adaptive steps and
  fold detection. Returns a `Branch` (states, parameters, tangents, turning
  points).
- `mf_arclength_continuation` — the matrix-free counterpart: preconditioned
  GMRES over `jax.linearize` JVPs with a `precond` hook and `p_stop` landing.
  For fields of 10^4–10^6 dof the Jacobian is never formed.
- `refine_fold` / `track_fold` — Moore–Spence refinement of a detected fold to
  Newton precision, and continuation of the fold itself in a second parameter.
  The augmented system is continued by arclength, so cusps are passed and
  reported.
- `analyze_branch` / `branch_eigenvalues` — the spectrum along the branch:
  stability, and the classification of every axis crossing into fold, branch
  point or Hopf candidate.
- `refine_hopf` — a Hopf point pinned by the standard (3N+2) augmented system,
  with second derivatives supplied by autodiff.
- `deflated_newton` / `deflated_search` — Farrell-style deflation: Newton
  converges away from the solutions already found. This is the route to
  disconnected branches.
- `branch_off` / `bifurcation_diagram` — switching onto the bifurcating branch
  at a simple branch point, and a bounded-depth recursive driver: trace,
  classify, switch, recurse (equilibria only).
- `fold_sensitivity` — the exact gradient of a fold location with respect to
  the model parameters, by implicit differentiation of the converged
  Moore–Spence system.
- `bordered_newton` / `newton` — the generic (N+k) bordered primitive and
  plain Newton for seeding.

Every claim above is validated against an exact result or the literature. The
cubic folds are recovered to 1e-10 and the two-parameter fold law to 1e-8. The
fold gradient matches the closed-form law d p*/d theta = sqrt(theta/3) to
1e-9. The Hopf machinery is pinned by the normal form and by the Brusselator
at exactly b = 1 + a^2, omega = a. The pitchfork diagram lands both arms on
x^2 = p to 1e-8. Bratu ignition is reproduced at lambda* = 3.5138 in 1-D and
6.808 in 2-D. The CSTR ignition/extinction pair sits at the Uppal–Ray–Poore
values and its fold-curve tangents are reproduced by autodiff. MatCont's
predator–prey folds are matched to five digits. The snaking branch above
passes its 38 folds in one continuation.

## The book

[*kellax by solved problems*](book/README.md) — nine chapters, each a single
worked problem: [the fold](book/01-the-fold.md), [the cusp](book/02-the-cusp.md),
[Bratu–Gelfand](book/03-bratu.md), [matrix-free scaling](book/04-matrix-free.md),
[homoclinic snaking](book/05-snaking.md), [CSTR hysteresis](book/06-cstr.md),
[a predator–prey fold pair](book/07-predator-prey.md),
[Bratu in 2-D](book/08-bratu-2d.md), and
[differentiable continuation](book/09-differentiable.md). Each chapter's
script lives in [`examples/`](examples) and regenerates its figure. The
printed numbers are real output. The [TUTORIAL](TUTORIAL.md) is the quick tour
of the API.

## Install / test

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[test,examples]"
.venv/bin/python tests/test_kellax.py && .venv/bin/python tests/test_bifurcations.py
python examples/cubic_fold.py          # -> figures/cubic_fold.png
```

kellax requires float64: `jax.config.update("jax_enable_x64", True)` once at
the top of every script.

## Roadmap

- Periodic-orbit continuation from Hopf points (shooting first).
- Hopf tracking in a second parameter.
- Matrix-free eigenvalues (Arnoldi over JVPs) so the classification layer
  reaches 10^4+ dof.
- Deflated continuation.
- A `jax.custom_jvp` rule for the fold location, so that `jax.grad` composes
  through the diagram directly.

The long-term goal is unchanged: models optimised or learned against their
bifurcation diagrams.

## Citing

See [`CITATION.cff`](CITATION.cff). License: Apache-2.0.
