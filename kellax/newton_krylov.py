"""Matrix-free inexact Newton-Krylov: the ladder's endgame, generic.

Consolidated from the cdft toolbox verbatim (its DFT wrappers stayed
behind): solve ``residual(x) = 0`` with the Jacobian never formed — J v from
``jax.linearize``, each Newton system by preconditioned GMRES (jax.scipy),
globalised by an infinity-norm trust cap plus a NaN-safe backtracking Armijo
line search on ||R||_2 (an overflow in a trial step simply rejects it). The
GMRES forcing follows eta = clip(sqrt(||R||_inf), eta_min, eta_max) — loose
far out, tight at the end (Eisenstat-Walker; see Knoll & Keyes 2004).

``make_step_bordered`` is the constrained variant: R(x, lam, *args) = 0 with
a scalar constraint(x, *args) = 0, solved as ONE GMRES on the joint
(x, lam) pytree — every border block from jax.linearize of the joint
residual, so the border row regularises a near-null Jacobian mode INSIDE the
Krylov space (the textbook Schur split forms b ~ J^{-1}1 explicitly, which
blows up along a soft mode and cancels catastrophically; the calibration
case was a near-critical droplet's volume mode).

Note: ``matrixfree.arclength_continuation`` carries its own copy of this
step pattern with the arclength phase condition baked in (same _TRUST_MAX /
_ARMIJO conventions); delegating it through ``make_step_bordered`` is the
internal-dedup item on the v0.5 review list.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres

_ARMIJO = 1e-4
_TRUST_MAX = 64.0


def _l2(*rs):
    return jnp.sqrt(sum(jnp.sum(r * r) for r in rs))


def _gmres(jvp, b, Minv, eta, x0, restart, maxiter):
    # gmres tests the PREconditioned residual against tol*|b| of the raw rhs,
    # so a well-normalised Minv (|Minv b| << |b|) "converges" at iterate 0
    # and returns a zero step. State the forcing in the preconditioned norm:
    # tol = 0 and atol = eta * |Minv b|, which reduces to the old tol = eta
    # when Minv is None and is invariant to scaling Minv.
    bM = b if Minv is None else Minv(b)
    d, _ = gmres(jvp, b, x0=x0, M=Minv, tol=0.0, atol=eta * _l2(bM),
                 restart=restart, maxiter=maxiter, solve_method="batched")
    return jnp.where(jnp.isfinite(d), d, 0.0)          # Krylov breakdown guard


def _trust_update(trust, ok, t, capped, dx_max):
    """Adaptive trust radius: a clean full step that hit the cap doubles it
    (an exactly-linear far field is walked in O(log) steps instead of
    crawling), a backtrack halves it (never below the configured dx_max)."""
    return jnp.where(ok & (t >= 1.0) & capped,
                     jnp.minimum(trust * 2.0, _TRUST_MAX),
                     jnp.where(t < 1.0, jnp.maximum(trust * 0.5, dx_max),
                               trust))


def make_step(residual, Minv, *, dx_max, restart, maxiter, eta_min, eta_max,
              ls_max):
    """R(x, *args) = 0 Newton step: linearize -> GMRES -> trust cap -> Armijo.
    Returns (x_new, trust_new, res_inf_new, t, ok)."""

    @jax.jit
    def step(x, trust, *args):
        R, jvp = jax.linearize(lambda xx: residual(xx, *args), x)
        m0 = _l2(R)
        res0 = jnp.max(jnp.abs(R))
        eta = jnp.clip(jnp.sqrt(res0), eta_min, eta_max)
        d = _gmres(jvp, -R, Minv, eta, jnp.zeros_like(x), restart, maxiter)
        raw = jnp.max(jnp.abs(d))
        d = d * jnp.minimum(1.0, trust / (raw + 1e-300))

        def cond(c):
            t, r, k = c
            return (k < ls_max) & ~(_l2(r) <= (1.0 - _ARMIJO * t) * m0)

        def body(c):
            t, _, k = c
            t = 0.5 * t
            return t, residual(x + t * d, *args), k + 1

        t, r, _ = jax.lax.while_loop(cond, body,
                                     (1.0, residual(x + d, *args), 0))
        ok = _l2(r) <= (1.0 - _ARMIJO * t) * m0        # NaN-safe: NaN -> False
        x_new = jnp.where(ok, x + t * d, x)
        res = jnp.where(ok, jnp.max(jnp.abs(r)), res0)
        return x_new, _trust_update(trust, ok, t, raw > trust, dx_max), res, t, ok

    return step


def make_step_bordered(residual, constraint, Minv, *, dx_max, restart,
                       maxiter, eta_min, eta_max, ls_max):
    """Newton step for the bordered system R(x, lam, *args) = 0,
    constraint(x, *args) = 0, solved as ONE GMRES on the (x, lam) pytree.
    Returns (x, lam, trust, res_inf, t, ok)."""

    def joint(u, *args):
        return (residual(u[0], u[1], *args), constraint(u[0], *args))

    Mb = None if Minv is None else (lambda v: (Minv(v[0]), v[1]))

    @jax.jit
    def step(x, lam, trust, *args):
        (R, Rc), jvp = jax.linearize(lambda u: joint(u, *args), (x, lam))
        m0 = _l2(R, Rc)
        res0 = jnp.maximum(jnp.max(jnp.abs(R)), jnp.abs(Rc))
        eta = jnp.clip(jnp.sqrt(res0), eta_min, eta_max)
        rhs = (-R, -Rc)
        rhsM = rhs if Mb is None else Mb(rhs)          # preconditioned-norm
        d, _ = gmres(jvp, rhs,                         # forcing, as in _gmres
                     x0=(jnp.zeros_like(x), jnp.zeros_like(lam)), M=Mb,
                     tol=0.0, atol=eta * _l2(rhsM[0], rhsM[1]),
                     restart=restart, maxiter=maxiter, solve_method="batched")
        dx = jnp.where(jnp.isfinite(d[0]), d[0], 0.0)
        dlam = jnp.where(jnp.isfinite(d[1]), d[1], 0.0)
        raw = jnp.maximum(jnp.max(jnp.abs(dx)), jnp.abs(dlam))
        s = jnp.minimum(1.0, trust / (raw + 1e-300))
        dx, dlam = dx * s, dlam * s

        def merit_state(t):
            return (residual(x + t * dx, lam + t * dlam, *args),
                    constraint(x + t * dx, *args))

        def cond(cst):
            t, (r, rc), k = cst
            return (k < ls_max) & ~(_l2(r, rc) <= (1.0 - _ARMIJO * t) * m0)

        def body(cst):
            t, _, k = cst
            t = 0.5 * t
            return t, merit_state(t), k + 1

        t, (r, rc), _ = jax.lax.while_loop(cond, body, (1.0, merit_state(1.0), 0))
        ok = _l2(r, rc) <= (1.0 - _ARMIJO * t) * m0    # NaN-safe
        x_new = jnp.where(ok, x + t * dx, x)
        lam_new = jnp.where(ok, lam + t * dlam, lam)
        res = jnp.where(ok, jnp.maximum(jnp.max(jnp.abs(r)), jnp.abs(rc)), res0)
        return (x_new, lam_new, _trust_update(trust, ok, t, raw > trust, dx_max),
                res, t, ok)

    return step


def newton_krylov(residual, x0, *, precond: Callable = None, tol: float = 1e-10,
                  max_newton: int = 30, dx_max: float = 4.0, restart: int = 40,
                  maxiter: int = 25, eta_min: float = 1e-4,
                  eta_max: float = 1e-1, ls_max: int = 25,
                  verbose: bool = False):
    """Generic matrix-free Newton-Krylov: solve ``residual(x) = 0``.

    Returns (x, res_inf, n_newton, converged). ``precond`` is an approximate
    inverse-Jacobian callable (e.g. a spectral symbol built by probing the
    linearised operator).
    """
    step = make_step(residual, precond, dx_max=dx_max, restart=restart,
                     maxiter=maxiter, eta_min=eta_min, eta_max=eta_max,
                     ls_max=ls_max)
    x = jnp.asarray(x0)
    trust = jnp.asarray(dx_max, dtype=x.dtype)
    res = float(jnp.max(jnp.abs(residual(x))))
    for k in range(max_newton):
        if res < tol:
            return x, res, k, True
        x, trust, r, t, ok = step(x, trust)
        res = float(r)
        if verbose:
            print(f"  [newton] it {k + 1:2d}  res_inf {res:.3e}  t {float(t):.4f}")
        if not bool(ok):
            return x, res, k + 1, False
    return x, res, max_newton, res < tol
