# Changelog

## 0.5.0 — unreleased

The ladder's inner solvers, consolidated from the cdft programme. The
continuation layer is unchanged; this release is what sits underneath it.

- `fixed_point_solve` — damped Picard with an Anderson/DIIS globaliser for
  self-consistent maps, on flat-vector states. Returns the residual of the
  state it actually returns, not of the iterate before it.
- `newton_krylov` — inexact Newton-Krylov: J v from `jax.linearize`, each
  system by preconditioned GMRES, globalised by an infinity-norm trust cap and
  a NaN-safe Armijo line search. `make_step` and `make_step_bordered` are the
  step factories; the bordered one carries a scalar constraint (mass, a phase
  condition) through ONE GMRES on the joint pytree, so the border row
  regularises a near-null Jacobian mode inside the Krylov space rather than
  cancelling catastrophically the way an explicit Schur split does.
- `hessian_vector_product` / `hessian_spectrum` / `smallest_eigenvalue` /
  `morse_index` — autodiff Hessian spectra by matrix-free Lanczos, never
  forming H. Dense diagonalisation where Lanczos cannot run (k reaching N,
  and `eigsh` raising for N <= 2); `morse_index` doubles k until it sees a
  non-negative eigenvalue or the whole spectrum, so the count is exact rather
  than silently floored at k.
- `ift_injection` — differentiable solutions by the implicit-function theorem,
  dense or Krylov-adjoint.
- The GMRES forcing is now stated once, in `_krylov.forced_gmres`, and in the
  preconditioned norm. jax's `gmres` tests the preconditioned residual against
  `tol * |b|` of the raw right-hand side, so an exactly normalised
  preconditioner reported convergence at iterate 0 and returned a zero step;
  both engines took the defect separately before it was consolidated.
- scipy is declared as the `spectra` extra rather than inherited from jax.
- Two new chapters and their scripts. Chapter 10, the Lorenz equilibria
  (`examples/lorenz.py`), works the classification layer end to end: a
  pitchfork that fold detection cannot see, `branch_off` onto the convection
  pair, `refine_hopf` at r = 470/19 to 1e-15, and `deflated_search` recovering
  all three equilibria from one guess. Chapter 11, the inner solvers
  (`examples/allen_cahn.py`), is the first worked problem for this release's
  layer: the Picard/Anderson/Newton-Krylov ladder on an Allen-Cahn phase
  field, Morse indices for its critical points, a prescribed-mean constrained
  state through `make_step_bordered`, and `ift_injection` recovering the
  surface tension as dF/d(eps).

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
