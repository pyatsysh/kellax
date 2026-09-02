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

Note on the sibling engine, and why it is not delegated here.
``matrixfree.arclength_continuation`` runs the same step pattern with the
arclength phase condition baked in. Everything mechanical is now shared
rather than copied: the GMRES forcing, the Armijo backtracking and the trust
update all come from ``_krylov``. What is left is not duplication but a
deliberate difference, and routing one through the other would erase it.

  * This step reports ``max(|R|, |constraint|)`` as its residual. The
    arclength step reports ``|R|`` alone, because its corrector tests that
    against ``newton_tol`` and the DENSE engine tests the same quantity
    (inside ``keller.arclength_continuation``'s corrector). The two
    continuation engines are documented as having identical semantics, and
    that is the line which makes it true.
  * ``constraint`` here is a function of x only. An arclength phase condition
    reads the parameter as well, so delegating would mean broadening this
    signature for a single internal caller.

So the v0.5 review's internal-dedup item is closed as "shared what is common,
kept what differs on purpose" rather than by delegation.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as np

from ._krylov import ARMIJO as _ARMIJO, backtrack, forced_gmres, trust_update


def _l2(*rs):
    return np.sqrt(sum(np.sum(r * r) for r in rs))


def make_step(residual, Minv, *, dx_max, restart, maxiter, eta_min, eta_max,
              ls_max):
    """R(x, *args) = 0 Newton step: linearize -> GMRES -> trust cap -> Armijo.
    Returns (x_new, trust_new, res_inf_new, t, ok)."""

    @jax.jit
    def step(x, trust, *args):
        R, jvp = jax.linearize(lambda xx: residual(xx, *args), x)
        m0 = _l2(R)
        res0 = np.max(np.abs(R))
        eta = np.clip(np.sqrt(res0), eta_min, eta_max)
        d = forced_gmres(jvp, -R, Minv, eta, np.zeros_like(x), restart, maxiter)
        raw = np.max(np.abs(d))
        d = d * np.minimum(1.0, trust / (raw + 1e-300))

        t, r, _ = backtrack(lambda tt: residual(x + tt * d, *args), _l2,
                            m0, ls_max)
        ok = _l2(r) <= (1.0 - _ARMIJO * t) * m0        # NaN-safe: NaN -> False
        x_new = np.where(ok, x + t * d, x)
        res = np.where(ok, np.max(np.abs(r)), res0)
        return x_new, trust_update(trust, ok, t, raw > trust, dx_max), res, t, ok

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
        res0 = np.maximum(np.max(np.abs(R)), np.abs(Rc))
        eta = np.clip(np.sqrt(res0), eta_min, eta_max)
        dx, dlam = forced_gmres(jvp, (-R, -Rc), Mb, eta,
                                (np.zeros_like(x), np.zeros_like(lam)),
                                restart, maxiter)
        raw = np.maximum(np.max(np.abs(dx)), np.abs(dlam))
        s = np.minimum(1.0, trust / (raw + 1e-300))
        dx, dlam = dx * s, dlam * s

        def merit_state(t):
            return (residual(x + t * dx, lam + t * dlam, *args),
                    constraint(x + t * dx, *args))

        t, (r, rc), _ = backtrack(merit_state, lambda rs: _l2(*rs), m0, ls_max)
        ok = _l2(r, rc) <= (1.0 - _ARMIJO * t) * m0    # NaN-safe
        x_new = np.where(ok, x + t * dx, x)
        lam_new = np.where(ok, lam + t * dlam, lam)
        res = np.where(ok, np.maximum(np.max(np.abs(r)), np.abs(rc)), res0)
        return (x_new, lam_new, trust_update(trust, ok, t, raw > trust, dx_max),
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
    step = make_step(residual, precond, dx_max = dx_max, restart = restart,
                     maxiter = maxiter, eta_min = eta_min, eta_max = eta_max,
                     ls_max = ls_max)
    x = np.asarray(x0)
    trust = np.asarray(dx_max, dtype = x.dtype)
    res = float(np.max(np.abs(residual(x))))
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
