# Changelog

## 0.4.0 — 2026-07-18

The bifurcation layer, and the first public tagged release.

- `analyze_branch` / `branch_eigenvalues` — the spectrum along a traced branch;
  every axis crossing classified into fold, branch point or Hopf candidate,
  crossing locations interpolated.
- `refine_hopf` — the standard (3N+2) augmented Hopf system, second derivatives
  by autodiff; pinned by the normal form and the Brusselator at b* = 1 + a^2,
  omega = a.
- `deflated_newton` / `deflated_search` — Farrell-style deflation; `jacfwd`
  differentiates the deflation operator.
- `branch_off` / `bifurcation_diagram` — switching onto the bifurcating branch
  at a simple branch point, and a bounded-depth recursive driver (equilibria
  only).
- `fold_sensitivity` — the exact gradient of a fold location with respect to
  the model parameters, by implicit differentiation of the converged
  Moore-Spence system; matches the closed-form cubic law to 1e-9.
- The book grew to nine chapters (differentiable continuation); the JOSS paper
  draft moved into `paper/`; a landing page at `docs/` (GitHub Pages).

## 0.3.0 — 2026-07-15

Matrix-free engine and publication prep.

- `mf_arclength_continuation` — preconditioned GMRES over `jax.linearize` JVPs
  with a `precond` hook and `p_stop` landing; the Jacobian is never formed.
  Fixed a corrector bug that rejected an already-converged seed.
- Eight worked examples with validated figures, the eight-chapter book and the
  TUTORIAL; canonical cases cross-checked against AUTO, MatCont and
  BifurcationKit.jl (Bratu lambda* = 3.5138, SH23 snaking with 38 folds, CSTR
  at the Uppal-Ray-Poore values).
- Version single-sourced from `kellax.__version__`; `examples`/`test`/`dev`
  extras; three-job CI (test matrix 3.11-3.13, examples smoke test, build
  check); `CITATION.cff`.

## 0.2.0 — 2026-07-08

- `refine_fold` / `track_fold` — Moore-Spence refinement of detected folds to
  Newton precision, and continuation of the fold in a second parameter through
  cusps.
- `bordered_newton` — the generic (N+k) bordered primitive.

## 0.1.0 — 2026-07-08

- `arclength_continuation` — the dense Keller pseudo-arclength trace with
  adaptive steps and fold detection, autodiff Jacobians throughout; `newton`
  for seeding.
