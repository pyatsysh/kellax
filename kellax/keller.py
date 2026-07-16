"""Keller pseudo-arclength continuation in JAX — model-agnostic.

Trace the solution branch of R(x, p) = 0 (x in R^N the state, p a scalar
control parameter) through folds, where dR/dx is singular and naive
p-stepping jumps or stalls. Keller (1977) parametrises the branch by
arclength s and solves the bordered (N+1) system

    R(x, p) = 0
    tangent . (x - x_prev, p - p_prev) - ds = 0

whose bordered Jacobian stays non-singular through turning points. One
continuation step is (1) predictor: step along the unit tangent, (2)
corrector: Newton on the bordered system, (3) next tangent: solve the same
bordered matrix against e_{N+1}. Both Jacobians dR/dx and dR/dp are
automatic (jax.jacfwd) — supply only the residual.

E.G. the canonical S-curve, folds at p = +-2/(3 sqrt 3):

    import jax.numpy as np
    from kellax import arclength_continuation
    R = lambda x, p: np.array([x[0]**3 - x[0] + p])
    br = arclength_continuation(R, x0 = np.array([-1.2]), p0 = 0.7, ds = 0.05)
    br.p[br.turning_points]

Folds are detected by sign changes of the tangent's p-component and
reported in Branch.turning_points; pin them to Newton precision with
folds.refine_fold. Dense linear algebra (N up to a few thousand); for
large fields (2D/3D) use the GMRES-on-JVP twin,
matrixfree.arclength_continuation. Background: Keller (1977); Seydel,
Practical Bifurcation and Stability Analysis; Govaerts (2000).
"""
from dataclasses import dataclass

import jax
import jax.numpy as np
import numpy as onp

# 2Do:
# 1. branch switching at simple bifurcation points (second null vector of
#    the bordered matrix gives the bifurcating tangent)
# 2. lax.scan the accepted-step loop so the whole trace jits end-to-end
#    (each step is already jitted; the Python loop only orchestrates)


@dataclass
class Branch:
    """A traced branch: states, parameter values, tangents, fold indices."""
    x: onp.ndarray             # (M, N) states along the branch
    p: onp.ndarray             # (M,) control parameter
    tan_p: onp.ndarray         # (M,) p-component of the unit tangent (sign flip = fold)
    ds: onp.ndarray            # (M,) accepted step sizes
    turning_points: list       # indices where the branch folds


def newton(residual, x0, p, tol = 1e-9, max_iter = 25):
    """Plain dense Newton on R(x, p) = 0 at fixed p. Returns (x, res).

    For seeding a branch from a rough initial guess before arclengthing.
    """
    Rf = jax.jit(residual)
    Rx = jax.jit(jax.jacfwd(residual, argnums = 0))

    @jax.jit
    def run(x):
        def cond(s):
            _, it, res = s
            return (it < max_iter) & (res > tol)

        def body(s):
            x, it, _ = s
            R = Rf(x, p)
            x = x - np.linalg.solve(Rx(x, p), R)
            return x, it + 1, np.max(np.abs(R))

        x, _, _ = jax.lax.while_loop(cond, body, (x, 0, np.inf))
        return x, np.max(np.abs(Rf(x, p)))

    return run(x0)


def arclength_continuation(residual, x0, p0, ds = 0.4, n_steps = 400,
        ds_min = 2e-3, ds_max = 1.5, newton_tol = 1e-9, newton_max = 25,
        p_min = -onp.inf, p_max = onp.inf, direction = 1.0,
        accept_res = 1e-6, verbose = False):
    """Trace the branch of ``residual(x, p) = 0`` from a point near (x0, p0).

    ``direction`` = +1 to start toward increasing p, -1 for decreasing.
    Steps adapt: shrink on corrector rejection, grow when Newton converges
    fast. Stops after ``n_steps`` or when p leaves [p_min, p_max].
    Returns a Branch.
    """
    N = int(x0.shape[0])
    Rf = jax.jit(residual)
    Rx = jax.jit(jax.jacfwd(residual, argnums = 0))
    Rp = jax.jit(jax.jacfwd(residual, argnums = 1))

    def _bordered(Jx, Rpv, tx, tp):
        # [ dR/dx  dR/dp ]        the (N+1) Keller matrix; last row is the
        # [  tx      tp  ]        arclength normalisation
        top = np.concatenate([Jx, Rpv[:, None]], axis = 1)
        bot = np.concatenate([tx, np.reshape(tp, (1,))])[None, :]
        return np.concatenate([top, bot], axis = 0)

    @jax.jit
    def tangent(x, p, tx0, tp0):
        M = _bordered(Rx(x, p), Rp(x, p), tx0, tp0)
        v = np.linalg.solve(M, np.concatenate([np.zeros(N), np.ones(1)]))
        tx, tp = v[:N], v[N]
        nrm = np.sqrt(tx @ tx + tp * tp)
        tx, tp = tx / nrm, tp / nrm
        s = np.sign(tx @ tx0 + tp * tp0)      # orient along the previous tangent
        return tx * s, tp * s

    @jax.jit
    def corrector(xp, pp, xprev, pprev, tx, tp, dstep):
        if verbose: print('jitting corrector..')

        def cond(s):
            _, _, it, res = s
            return (it < newton_max) & (res > newton_tol)

        def body(s):
            x, p, it, _ = s
            R = Rf(x, p)
            nval = tx @ (x - xprev) + tp * (p - pprev) - dstep
            M = _bordered(Rx(x, p), Rp(x, p), tx, tp)
            d = np.linalg.solve(M, -np.concatenate([R, np.reshape(nval, (1,))]))
            return x + d[:N], p + d[N], it + 1, np.max(np.abs(R))

        x, p, it, _ = jax.lax.while_loop(cond, body, (xp, pp, 0, np.inf))
        return x, p, np.max(np.abs(Rf(x, p))), it

    # --------------------------------------------------------------
    # INITIAL CONVERGED POINT + TANGENT
    # --------------------------------------------------------------
    x, res0 = newton(residual, x0, float(p0), tol = newton_tol, max_iter = newton_max)
    if float(res0) > accept_res:
        raise RuntimeError(f"initial point did not converge (res={float(res0):.2e})")
    p = float(p0)
    tx, tp = tangent(x, p, np.zeros(N), np.asarray(float(direction)))

    xs, ps, tps, dss = [onp.asarray(x)], [p], [float(tp)], [0.0]
    turning = []

    # --------------------------------------------------------------
    # PREDICTOR-CORRECTOR MARCH
    # --------------------------------------------------------------
    for step in range(n_steps):
        xpred = x + ds * tx
        ppred = p + ds * tp
        xc, pc, res, iters = corrector(xpred, ppred, x, p, tx, tp, ds)
        res = float(res)

        if not onp.isfinite(res) or res > accept_res:     # reject -> shrink & retry
            if ds <= ds_min * 1.001:
                if verbose:
                    print(f"  step {step}: stalled at ds_min (res={res:.1e}); stop")
                break
            ds = max(ds * 0.5, ds_min)
            continue

        tp_prev = tp
        tx, tp = tangent(xc, pc, tx, tp)
        x, p = xc, float(pc)
        xs.append(onp.asarray(x)); ps.append(p)
        tps.append(float(tp)); dss.append(ds)

        if float(tp) * float(tp_prev) < 0:                # fold detected
            turning.append(len(ps) - 1)
            if verbose:
                print(f"  step {step}: turning point at p={p:.6f}")

        if p > p_max or p < p_min:
            break
        if int(iters) <= 4:                               # grow step when easy
            ds = min(ds * 1.3, ds_max)

    return Branch(x = onp.array(xs), p = onp.array(ps), tan_p = onp.array(tps),
                  ds = onp.array(dss), turning_points = turning)
