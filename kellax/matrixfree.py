"""Matrix-free Keller pseudo-arclength continuation.

The scale-up path promised in ``keller.py``: the SAME bordered predictor-
corrector, but the Jacobian is never formed. Each bordered (N+1) Keller system
is solved by preconditioned GMRES (jax.scipy), every block coming from
``jax.linearize`` of the residual -- so N can be 10^4-10^6 (2D/3D fields)
where the dense ``keller.arclength_continuation`` (jax.jacfwd + jnp.linalg.solve,
N up to a few thousand) cannot form the Jacobian at all.

Identical semantics to the dense engine: trace R(x, p) = 0 through folds by
arclengthing the (x, p) system, tangent from the same bordered matrix against
e_{N+1}, folds flagged by sign changes of the tangent's p-component. The border
row regularises the near-null Jacobian mode at a fold INSIDE the Krylov space
(the textbook Schur split forms J^{-1} of that soft mode explicitly and cancels
catastrophically -- see the DFT toolbox's "solver zoo, Scar 3").

Extras over the dense API:
  * ``precond`` -- a callable Minv(v) approximating the inverse of the x-block
    Jacobian (e.g. an RPA spectral preconditioner). Applied to the N-block of
    the bordered system; the scalar border row is left unpreconditioned.
  * ``p_stop`` -- march to EXACTLY this parameter value and stop (the final
    step is sized to land on it), for targeting a specific state on the branch.

dR/dp and J v are both obtained from one ``jax.linearize`` of the joint map,
so an analytic parameter-derivative is never needed (it is ``-1`` for a DFT
Euler-Lagrange residual, but the engine stays model-agnostic).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres

from .keller import Branch

_TRUST_MAX = 64.0
_ARMIJO = 1e-4


def _make_step(residual, precond, restart, maxiter, eta_min, eta_max, ls_max,
               dx_max, w):
    """One bordered-Newton step of  R(x,p)=0,  w tx.(x-x0)+tp(p-p0)-ds = 0,
    solved as ONE GMRES on the (x, p) pytree with a NaN-safe Armijo search.
    jax.linearize of the joint map supplies J, dR/dp and the border in one shot.
    ``w`` weights the state part of the arclength norm (use ~1/N so a
    high-dimensional field and the scalar parameter are commensurate; else the
    unit tangent is all state and p crawls at ds/sqrt(N))."""
    Mb = None if precond is None else (lambda v: (precond(v[0]), v[1]))

    @jax.jit
    def step(x, p, tx, tp, x0, p0, ds, trust):
        def joint(u):
            xx, pp = u
            R = residual(xx, pp)
            phase = w * jnp.vdot(tx, xx - x0) + tp * (pp - p0) - ds
            return (R, phase)

        (R, ph), jvp = jax.linearize(joint, (x, p))
        m0 = jnp.sqrt(jnp.vdot(R, R) + ph * ph)
        res0 = jnp.maximum(jnp.max(jnp.abs(R)), jnp.abs(ph))
        eta = jnp.clip(jnp.sqrt(res0), eta_min, eta_max)
        d, _ = gmres(jvp, (-R, -ph), x0=(jnp.zeros_like(x), jnp.zeros_like(p)),
                     M=Mb, tol=eta, atol=0.0, restart=restart, maxiter=maxiter,
                     solve_method="batched")
        dx = jnp.where(jnp.isfinite(d[0]), d[0], 0.0)
        dp = jnp.where(jnp.isfinite(d[1]), d[1], 0.0)
        raw = jnp.maximum(jnp.max(jnp.abs(dx)), jnp.abs(dp))
        s = jnp.minimum(1.0, trust / (raw + 1e-300))
        dx, dp = dx * s, dp * s

        def merit(t):
            xx, pp = x + t * dx, p + t * dp
            R2 = residual(xx, pp)
            ph2 = w * jnp.vdot(tx, xx - x0) + tp * (pp - p0) - ds
            return R2, ph2

        def cond(c):
            t, (r, rc), k = c
            return (k < ls_max) & ~(jnp.sqrt(jnp.vdot(r, r) + rc * rc)
                                    <= (1.0 - _ARMIJO * t) * m0)

        def body(c):
            t, _, k = c
            t = 0.5 * t
            return t, merit(t), k + 1

        t, (r, rc), _ = jax.lax.while_loop(cond, body, (1.0, merit(1.0), 0))
        ok = jnp.sqrt(jnp.vdot(r, r) + rc * rc) <= (1.0 - _ARMIJO * t) * m0
        x_new = jnp.where(ok, x + t * dx, x)
        p_new = jnp.where(ok, p + t * dp, p)
        resR = jnp.where(ok, jnp.max(jnp.abs(r)), jnp.max(jnp.abs(R)))
        trust_new = jnp.where(ok & (t >= 1.0) & (raw > trust),
                              jnp.minimum(trust * 2.0, _TRUST_MAX),
                              jnp.where(t < 1.0, jnp.maximum(trust * 0.5, dx_max),
                                        trust))
        return x_new, p_new, resR, t, ok, trust_new

    return step


def _make_tangent(residual, precond, restart, maxiter, w):
    """Unit tangent (in the w-weighted norm) from [J dR/dp; w tx0  tp0] t = e_{N+1},
    oriented by (tx0, tp0)."""
    Mb = None if precond is None else (lambda v: (precond(v[0]), v[1]))

    @jax.jit
    def tangent(x, p, tx0, tp0):
        _, jvp = jax.linearize(lambda u: residual(u[0], u[1]), (x, p))

        def op(u):
            vx, vp = u
            return (jvp((vx, vp)), w * jnp.vdot(tx0, vx) + tp0 * vp)

        sol, _ = gmres(op, (jnp.zeros_like(x), jnp.ones_like(p)),
                       x0=(jnp.zeros_like(x), jnp.zeros_like(p)), M=Mb,
                       tol=1e-8, atol=0.0, restart=restart, maxiter=maxiter,
                       solve_method="batched")
        tx, tp = sol
        nrm = jnp.sqrt(w * jnp.vdot(tx, tx) + tp * tp)
        tx, tp = tx / nrm, tp / nrm
        sgn = jnp.sign(w * jnp.vdot(tx, tx0) + tp * tp0)
        sgn = jnp.where(sgn == 0, 1.0, sgn)
        return tx * sgn, tp * sgn

    return tangent


def arclength_continuation(residual: Callable, x0, p0: float, ds: float = 0.3,
                           n_steps: int = 2000, ds_min: float = 2e-3,
                           ds_max: float = 1.0, newton_tol: float = 1e-8,
                           newton_max: int = 15, p_min: float = -np.inf,
                           p_max: float = np.inf, direction: float = 1.0,
                           accept_res: float = 1e-6, p_stop: float = None,
                           precond: Callable = None, state_scale: float = None,
                           gmres_restart: int = 60,
                           gmres_maxiter: int = 50, dx_max: float = 4.0,
                           eta_min: float = 1e-6, eta_max: float = 1e-1,
                           ls_max: int = 30, verbose: bool = False) -> Branch:
    """Matrix-free Keller arclength trace of ``residual(x, p) = 0`` from (x0, p0).

    Mirrors ``keller.arclength_continuation`` (same Branch, same fold rule) but
    every linear solve is preconditioned GMRES on ``jax.linearize`` JVPs. Set
    ``precond`` for the x-block and, optionally, ``p_stop`` to land on a target p.

    ``state_scale`` weights the state part of the arclength norm; the default
    ``1/N`` keeps the field and the scalar parameter commensurate at large N (an
    unweighted norm makes the unit tangent almost all state, so p advances at
    only ds/sqrt(N) per step). Pass an explicit value to tune the state/parameter
    trade-off.
    """
    x0 = jnp.asarray(x0, dtype=float)
    w = float(1.0 / x0.shape[0] if state_scale is None else state_scale)
    step = _make_step(residual, precond, gmres_restart, gmres_maxiter,
                      eta_min, eta_max, ls_max, dx_max, w)
    tangent = _make_tangent(residual, precond, gmres_restart, gmres_maxiter, w)

    def corrector(xp, pp, xprev, pprev, tx, tp, dstep):
        x, p, trust = xp, pp, jnp.asarray(dx_max)
        res = jnp.inf
        for _ in range(newton_max):
            x, p, res, t, ok, trust = step(x, p, tx, tp, xprev, pprev, dstep, trust)
            if not bool(jnp.isfinite(res)):
                return x, p, float(res), False
            if float(res) < newton_tol:      # converged: accept even if the line
                break                        # search could not improve an already-
            if not bool(ok):                 # exact point; only a stall *before*
                return x, p, float(res), False   # convergence is a real failure
        return x, p, float(res), bool(np.isfinite(float(res)))

    # initial point: pin p to p0 (tx=0, tp=1, ds=0) and Newton R -> 0
    x, p, r0, ok0 = corrector(x0, float(p0), x0, float(p0),
                              jnp.zeros_like(x0), 1.0, 0.0)
    if (not ok0) or r0 > accept_res:
        raise RuntimeError(f"initial point did not converge (res={r0:.2e})")
    tx, tp = tangent(x, p, jnp.zeros_like(x), jnp.asarray(float(direction)))
    if float(tp) * float(direction) < 0:
        tx, tp = -tx, -tp

    xs, ps, tps, dss = [np.asarray(x)], [float(p)], [float(tp)], [0.0]
    turning: list = []
    cur = float(ds)
    for stp in range(n_steps):
        land = False
        if p_stop is not None and float(tp) != 0.0:
            d2s = (p_stop - float(p)) / float(tp)
            if 0.0 < d2s <= cur:
                cur, land = float(d2s), True
        xc, pc, res, ok = corrector(x + cur * tx, p + cur * tp, x, p, tx, tp, cur)
        if (not ok) or (not np.isfinite(res)) or res > accept_res:
            if cur <= ds_min * 1.001:
                if verbose:
                    print(f"  step {stp}: stalled at ds_min (res={res:.1e}); stop")
                break
            cur = max(cur * 0.5, ds_min)
            continue
        tpp = tp
        tx, tp = tangent(xc, pc, tx, tp)
        x, p = xc, float(pc)
        xs.append(np.asarray(x)); ps.append(p); tps.append(float(tp)); dss.append(cur)
        if float(tp) * float(tpp) < 0:
            turning.append(len(ps) - 1)
            if verbose:
                print(f"  step {stp}: turning point at p={p:.6f}")
        if verbose and (stp % 20 == 0 or land):
            print(f"  step {stp:4d}: p={p:+.6f} ds={cur:.3f} res={res:.1e} tp={float(tp):+.3f}")
        if land and abs(p - p_stop) < max(ds_min, 1e-9):
            break
        if p > p_max or p < p_min:
            break
        cur = min(cur * 1.3, ds_max)

    return Branch(x=np.array(xs), p=np.array(ps), tan_p=np.array(tps),
                  ds=np.array(dss), turning_points=turning)
