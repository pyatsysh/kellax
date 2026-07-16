"""Matrix-free Keller pseudo-arclength continuation.

The scale-up path promised in ``keller.py``: the SAME bordered predictor-
corrector, but the Jacobian is never formed. Each bordered (N+1) Keller
system is solved by preconditioned GMRES (jax.scipy), every block coming
from ``jax.linearize`` of the residual — so N can be 10^4-10^6 (2D/3D
fields) where the dense ``keller.arclength_continuation`` (jax.jacfwd +
np.linalg.solve, N up to a few thousand) cannot form the Jacobian at all.

Identical semantics to the dense engine: trace R(x, p) = 0 through folds by
arclengthing the (x, p) system, tangent from the same bordered matrix
against e_{N+1}, folds flagged by sign changes of the tangent's
p-component. The border row regularises the near-null Jacobian mode at a
fold INSIDE the Krylov space — the textbook Schur split solves against the
soft mode of J explicitly and cancels catastrophically near the fold; the
bordered operator never does, which is the whole point of Keller's method
carried into the Krylov setting.

Extras over the dense API:
  (1) ``precond`` — a callable Minv(v) approximating the inverse of the
      x-block Jacobian (e.g. a spectral preconditioner built from the
      linearised operator). Applied to the N-block of the bordered system;
      the scalar border row is left unpreconditioned.
  (2) ``p_stop`` — march to EXACTLY this parameter value and stop (the
      final step is sized to land on it), for targeting a specific state
      on the branch.

dR/dp and J v are both obtained from one ``jax.linearize`` of the joint
map, so an analytic parameter-derivative is never needed. Inexact-Newton
machinery is standard Jacobian-free Newton-Krylov practice: an
Eisenstat-Walker forcing schedule for the GMRES tolerance, a NaN-safe
Armijo line search, a trust region on the step. See Knoll & Keyes (2004).
"""
import jax
import jax.numpy as np
import numpy as onp
from jax.scipy.sparse.linalg import gmres

from .keller import Branch

_TRUST_MAX = 64.0
_ARMIJO = 1e-4

# 2Do: expose the converged (x, p) -> branch map to implicit differentiation
# (jax.custom_vjp on the bordered solve) — the differentiable-continuation
# roadmap item; the machinery here already keeps every block as a closure


def _make_step(residual, precond, restart, maxiter, eta_min, eta_max, ls_max,
               dx_max, w, verbose = False):
    """One bordered-Newton step of  R(x,p)=0,  w tx.(x-x0)+tp(p-p0)-ds = 0,
    solved as ONE GMRES on the (x, p) pytree with a NaN-safe Armijo search.
    jax.linearize of the joint map supplies J, dR/dp and the border in one
    shot. ``w`` weights the state part of the arclength norm (use ~1/N so a
    high-dimensional field and the scalar parameter are commensurate; else
    the unit tangent is all state and p crawls at ds/sqrt(N))."""
    Mb = None if precond is None else (lambda v: (precond(v[0]), v[1]))

    @jax.jit
    def step(x, p, tx, tp, x0, p0, ds, trust):
        if verbose: print('jitting mf step..')

        def joint(u):
            xx, pp = u
            R = residual(xx, pp)
            phase = w * np.vdot(tx, xx - x0) + tp * (pp - p0) - ds
            return (R, phase)

        (R, ph), jvp = jax.linearize(joint, (x, p))
        m0 = np.sqrt(np.vdot(R, R) + ph * ph)
        res0 = np.maximum(np.max(np.abs(R)), np.abs(ph))
        eta = np.clip(np.sqrt(res0), eta_min, eta_max)    # Eisenstat-Walker
        d, _ = gmres(jvp, (-R, -ph), x0 = (np.zeros_like(x), np.zeros_like(p)),
                     M = Mb, tol = eta, atol = 0.0, restart = restart,
                     maxiter = maxiter, solve_method = "batched")
        dx = np.where(np.isfinite(d[0]), d[0], 0.0)
        dp = np.where(np.isfinite(d[1]), d[1], 0.0)
        raw = np.maximum(np.max(np.abs(dx)), np.abs(dp))
        s = np.minimum(1.0, trust / (raw + 1e-300))       # trust-region clip
        dx, dp = dx * s, dp * s

        def merit(t):
            xx, pp = x + t * dx, p + t * dp
            R2 = residual(xx, pp)
            ph2 = w * np.vdot(tx, xx - x0) + tp * (pp - p0) - ds
            return R2, ph2

        def cond(c):
            t, (r, rc), k = c
            return (k < ls_max) & ~(np.sqrt(np.vdot(r, r) + rc * rc)
                                    <= (1.0 - _ARMIJO * t) * m0)

        def body(c):
            t, _, k = c
            t = 0.5 * t
            return t, merit(t), k + 1

        t, (r, rc), _ = jax.lax.while_loop(cond, body, (1.0, merit(1.0), 0))
        ok = np.sqrt(np.vdot(r, r) + rc * rc) <= (1.0 - _ARMIJO * t) * m0
        x_new = np.where(ok, x + t * dx, x)
        p_new = np.where(ok, p + t * dp, p)
        resR = np.where(ok, np.max(np.abs(r)), np.max(np.abs(R)))
        trust_new = np.where(ok & (t >= 1.0) & (raw > trust),
                             np.minimum(trust * 2.0, _TRUST_MAX),
                             np.where(t < 1.0, np.maximum(trust * 0.5, dx_max),
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
            return (jvp((vx, vp)), w * np.vdot(tx0, vx) + tp0 * vp)

        sol, _ = gmres(op, (np.zeros_like(x), np.ones_like(p)),
                       x0 = (np.zeros_like(x), np.zeros_like(p)), M = Mb,
                       tol = 1e-8, atol = 0.0, restart = restart,
                       maxiter = maxiter, solve_method = "batched")
        tx, tp = sol
        nrm = np.sqrt(w * np.vdot(tx, tx) + tp * tp)
        tx, tp = tx / nrm, tp / nrm
        sgn = np.sign(w * np.vdot(tx, tx0) + tp * tp0)
        sgn = np.where(sgn == 0, 1.0, sgn)
        return tx * sgn, tp * sgn

    return tangent


def arclength_continuation(residual, x0, p0, ds = 0.3, n_steps = 2000,
        ds_min = 2e-3, ds_max = 1.0, newton_tol = 1e-8, newton_max = 15,
        p_min = -onp.inf, p_max = onp.inf, direction = 1.0,
        accept_res = 1e-6, p_stop = None, precond = None,
        state_scale = None, gmres_restart = 60, gmres_maxiter = 50,
        dx_max = 4.0, eta_min = 1e-6, eta_max = 1e-1, ls_max = 30,
        verbose = False):
    """Matrix-free Keller arclength trace of ``residual(x, p) = 0`` from (x0, p0).

    Mirrors ``keller.arclength_continuation`` (same Branch, same fold rule)
    but every linear solve is preconditioned GMRES on ``jax.linearize``
    JVPs. Set ``precond`` for the x-block and, optionally, ``p_stop`` to
    land on a target p.

    ``state_scale`` weights the state part of the arclength norm; the
    default ``1/N`` keeps the field and the scalar parameter commensurate
    at large N (an unweighted norm makes the unit tangent almost all state,
    so p advances at only ds/sqrt(N) per step). Pass an explicit value to
    tune the state/parameter trade-off.
    """
    x0 = np.asarray(x0, dtype = float)
    w = float(1.0 / x0.shape[0] if state_scale is None else state_scale)
    step = _make_step(residual, precond, gmres_restart, gmres_maxiter,
                      eta_min, eta_max, ls_max, dx_max, w, verbose)
    tangent = _make_tangent(residual, precond, gmres_restart, gmres_maxiter, w)

    def corrector(xp, pp, xprev, pprev, tx, tp, dstep):
        x, p, trust = xp, pp, np.asarray(dx_max)
        res = np.inf
        for _ in range(newton_max):
            x, p, res, t, ok, trust = step(x, p, tx, tp, xprev, pprev, dstep, trust)
            if not bool(np.isfinite(res)):
                return x, p, float(res), False
            if float(res) < newton_tol:      # converged: accept even if the line
                break                        # search could not improve an already-
            if not bool(ok):                 # exact point; only a stall *before*
                return x, p, float(res), False   # convergence is a real failure
        return x, p, float(res), bool(onp.isfinite(float(res)))

    # --------------------------------------------------------------
    # INITIAL POINT: pin p to p0 (tx=0, tp=1, ds=0) and Newton R -> 0
    # --------------------------------------------------------------
    x, p, r0, ok0 = corrector(x0, float(p0), x0, float(p0),
                              np.zeros_like(x0), 1.0, 0.0)
    if (not ok0) or r0 > accept_res:
        raise RuntimeError(f"initial point did not converge (res={r0:.2e})")
    tx, tp = tangent(x, p, np.zeros_like(x), np.asarray(float(direction)))
    if float(tp) * float(direction) < 0:
        tx, tp = -tx, -tp

    xs, ps, tps, dss = [onp.asarray(x)], [float(p)], [float(tp)], [0.0]
    turning = []
    cur = float(ds)

    # --------------------------------------------------------------
    # PREDICTOR-CORRECTOR MARCH
    # --------------------------------------------------------------
    for stp in range(n_steps):
        land = False
        if p_stop is not None and float(tp) != 0.0:
            d2s = (p_stop - float(p)) / float(tp)
            if 0.0 < d2s <= cur:                          # size the last step to land
                cur, land = float(d2s), True
        xc, pc, res, ok = corrector(x + cur * tx, p + cur * tp, x, p, tx, tp, cur)
        if (not ok) or (not onp.isfinite(res)) or res > accept_res:
            if cur <= ds_min * 1.001:
                if verbose:
                    print(f"  step {stp}: stalled at ds_min (res={res:.1e}); stop")
                break
            cur = max(cur * 0.5, ds_min)
            continue
        tpp = tp
        tx, tp = tangent(xc, pc, tx, tp)
        x, p = xc, float(pc)
        xs.append(onp.asarray(x)); ps.append(p); tps.append(float(tp)); dss.append(cur)
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

    return Branch(x = onp.array(xs), p = onp.array(ps), tan_p = onp.array(tps),
                  ds = onp.array(dss), turning_points = turning)
