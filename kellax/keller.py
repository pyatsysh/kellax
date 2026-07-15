"""Keller pseudo-arclength continuation in JAX — model-agnostic.

Trace the solution branch of R(x, p) = 0 (x in R^N the state, p a scalar
control parameter) through folds, where dR/dx is singular and naive
p-stepping jumps or stalls. Keller's method parametrises the branch by
arclength s and solves the bordered (N+1) system

    R(x, p) = 0
    tangent . (x - x_prev, p - p_prev) - ds = 0

whose bordered Jacobian stays non-singular through turning points. Predictor
= tangent step; corrector = Newton on the bordered system; the next tangent
solves the same bordered matrix against e_{N+1}. Both Jacobians dR/dx and
dR/dp are automatic (jax.jacfwd) — supply only the residual.

Fold locations are detected by sign changes of the tangent's p-component and
reported in Branch.turning_points. Dense linear algebra (suited to N up to a
few thousand); for larger fields (2D/3D) use ``kellax.mf_arclength_continuation``
(matrixfree.py), the GMRES-on-JVP twin of this engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp


@dataclass
class Branch:
    """A traced branch: states, parameter values, tangents, fold indices."""
    x: np.ndarray              # (M, N) states along the branch
    p: np.ndarray              # (M,) control parameter
    tan_p: np.ndarray          # (M,) p-component of the unit tangent (sign flip = fold)
    ds: np.ndarray             # (M,) accepted step sizes
    turning_points: list       # indices where the branch folds


def newton(residual: Callable, x0: jnp.ndarray, p: float,
           tol: float = 1e-9, max_iter: int = 25):
    """Plain dense Newton on R(x, p) = 0 at fixed p. Returns (x, res)."""
    Rf = jax.jit(residual)
    Rx = jax.jit(jax.jacfwd(residual, argnums=0))

    @jax.jit
    def run(x):
        def cond(s):
            _, it, res = s
            return (it < max_iter) & (res > tol)

        def body(s):
            x, it, _ = s
            R = Rf(x, p)
            x = x - jnp.linalg.solve(Rx(x, p), R)
            return x, it + 1, jnp.max(jnp.abs(R))

        x, _, _ = jax.lax.while_loop(cond, body, (x, 0, jnp.inf))
        return x, jnp.max(jnp.abs(Rf(x, p)))

    return run(x0)


def arclength_continuation(residual: Callable, x0: jnp.ndarray, p0: float,
                           ds: float = 0.4, n_steps: int = 400,
                           ds_min: float = 2e-3, ds_max: float = 1.5,
                           newton_tol: float = 1e-9, newton_max: int = 25,
                           p_min: float = -np.inf, p_max: float = np.inf,
                           direction: float = 1.0, accept_res: float = 1e-6,
                           verbose: bool = False) -> Branch:
    """Trace the branch of ``residual(x, p) = 0`` from a point near (x0, p0).

    ``direction`` = +1 to start toward increasing p, -1 for decreasing.
    Steps adapt: shrink on corrector rejection, grow when Newton converges
    fast. Stops after ``n_steps`` or when p leaves [p_min, p_max].
    """
    N = int(x0.shape[0])
    Rf = jax.jit(residual)
    Rx = jax.jit(jax.jacfwd(residual, argnums=0))
    Rp = jax.jit(jax.jacfwd(residual, argnums=1))

    def _bordered(Jx, Rpv, tx, tp):
        top = jnp.concatenate([Jx, Rpv[:, None]], axis=1)
        bot = jnp.concatenate([tx, jnp.reshape(tp, (1,))])[None, :]
        return jnp.concatenate([top, bot], axis=0)

    @jax.jit
    def tangent(x, p, tx0, tp0):
        M = _bordered(Rx(x, p), Rp(x, p), tx0, tp0)
        v = jnp.linalg.solve(M, jnp.concatenate([jnp.zeros(N), jnp.ones(1)]))
        tx, tp = v[:N], v[N]
        nrm = jnp.sqrt(tx @ tx + tp * tp)
        tx, tp = tx / nrm, tp / nrm
        s = jnp.sign(tx @ tx0 + tp * tp0)
        return tx * s, tp * s

    @jax.jit
    def corrector(xp, pp, xprev, pprev, tx, tp, dstep):
        def cond(s):
            _, _, it, res = s
            return (it < newton_max) & (res > newton_tol)

        def body(s):
            x, p, it, _ = s
            R = Rf(x, p)
            nval = tx @ (x - xprev) + tp * (p - pprev) - dstep
            M = _bordered(Rx(x, p), Rp(x, p), tx, tp)
            d = jnp.linalg.solve(M, -jnp.concatenate([R, jnp.reshape(nval, (1,))]))
            return x + d[:N], p + d[N], it + 1, jnp.max(jnp.abs(R))

        x, p, it, _ = jax.lax.while_loop(cond, body, (xp, pp, 0, jnp.inf))
        return x, p, jnp.max(jnp.abs(Rf(x, p))), it

    # --- initial converged point + tangent -------------------------------
    x, res0 = newton(residual, x0, float(p0), tol=newton_tol, max_iter=newton_max)
    if float(res0) > accept_res:
        raise RuntimeError(f"initial point did not converge (res={float(res0):.2e})")
    p = float(p0)
    tx, tp = tangent(x, p, jnp.zeros(N), jnp.asarray(float(direction)))

    xs, ps, tps, dss = [np.asarray(x)], [p], [float(tp)], [0.0]
    turning: list = []

    for step in range(n_steps):
        xpred = x + ds * tx
        ppred = p + ds * tp
        xc, pc, res, iters = corrector(xpred, ppred, x, p, tx, tp, ds)
        res = float(res)

        if not np.isfinite(res) or res > accept_res:      # reject -> shrink & retry
            if ds <= ds_min * 1.001:
                if verbose:
                    print(f"  step {step}: stalled at ds_min (res={res:.1e}); stop")
                break
            ds = max(ds * 0.5, ds_min)
            continue

        tp_prev = tp
        tx, tp = tangent(xc, pc, tx, tp)
        x, p = xc, float(pc)
        xs.append(np.asarray(x)); ps.append(p)
        tps.append(float(tp)); dss.append(ds)

        if float(tp) * float(tp_prev) < 0:                # fold detected
            turning.append(len(ps) - 1)
            if verbose:
                print(f"  step {step}: turning point at p={p:.6f}")

        if p > p_max or p < p_min:
            break
        if int(iters) <= 4:                               # grow step when easy
            ds = min(ds * 1.3, ds_max)

    return Branch(x=np.array(xs), p=np.array(ps), tan_p=np.array(tps),
                  ds=np.array(dss), turning_points=turning)
