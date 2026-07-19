"""Accelerated fixed-point driver: damped Picard warm-up -> Anderson (DIIS).

The globaliser of the solver ladder, consolidated here from the cdft toolbox
(where it drove every DFT self-consistency map): given g(x), iterate

    warm-up (k < warmup):  x <- (1-a) x + a g(x)
    Anderson  (k >= warmup, depth m): least-squares extrapolation over the
        last m residual differences (Walker–Ni form, Tikhonov-regularised so
        unfilled history columns are harmless), with the same damping a as
        mixing parameter and an optional positivity floor.

Runs under lax.while_loop with a residual tolerance — iterations stop when
converged instead of burning a fixed budget. Anderson typically cuts the
iteration count by 10-50x when plain Picard's spectral radius approaches 1
(near-spinodal density maps were the calibration case). Set m = 0 to recover
pure damped Picard.

``clamp`` floors every new iterate — right for density-like states (the cdft
default is 1e-30); pass ``clamp=None`` for signed states (e.g. an
Ornstein-Zernike gamma iteration), where a floor would corrupt the map.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def fixed_point_solve(gmap, x0, tol: float, max_steps: int,
                      damping: float, m: int = 5, warmup: int = 50,
                      beta: float = 1.0, clamp: float | None = 1e-30):
    """Solve x = gmap(x). Returns (x, res, iters). `damping` mixes the
    Picard warm-up; `beta` is the Anderson mixing (~1: the least-squares
    extrapolation supplies the stability that warm-up damping supplied);
    `clamp` floors each iterate (None: no floor, signed states allowed)."""
    a = damping
    N = x0.shape[0]
    m = int(m)

    def _floor(v):
        return v if clamp is None else jnp.maximum(v, clamp)

    if m == 0:
        def cond(c):
            x, k, res = c
            return (k < max_steps) & (res > tol)

        def body(c):
            x, k, _ = c
            g = gmap(x)
            res = jnp.max(jnp.abs(g - x))
            x = _floor((1.0 - a) * x + a * g)
            return x, k + 1, res

        x, k, res = jax.lax.while_loop(
            cond, body, (x0, 0, jnp.asarray(jnp.inf)))
        return x, res, k

    def cond(c):
        x, dX, dF, f_prev, x_prev, k, res, res_prev, k_restart = c
        return (k < max_steps) & (res > tol)

    def body(c):
        x, dX, dF, f_prev, x_prev, k, _, res_prev, k_restart = c
        g = gmap(x)
        f = g - x                                     # residual
        res = jnp.max(jnp.abs(f))

        # SAFEGUARD: a growing residual means the last extrapolation misfired
        # (e.g. an effectively-undamped step on a stiff map) — flush the
        # history and fall back to damped Picard while it refills.
        grew = res > 1.5 * res_prev
        k_restart = jnp.where(grew, k, k_restart)

        # update circular difference history (skip the very first iterate)
        slot = (k - 1) % m
        have_prev = (k > 0) & ~grew
        dX = jnp.where(grew, jnp.zeros_like(dX),
                       jnp.where(have_prev, dX.at[slot].set(x - x_prev), dX))
        dF = jnp.where(grew, jnp.zeros_like(dF),
                       jnp.where(have_prev, dF.at[slot].set(f - f_prev), dF))

        # plain damped step
        x_picard = (1.0 - a) * x + a * g

        # Anderson step: gamma = argmin ||f - dF^T gamma||, RELATIVE
        # regularisation so a small history cannot degenerate the step into
        # undamped Picard (gamma -> 0 with an absolute lambda would).
        A = dF @ dF.T
        lam = 1e-10 * (jnp.trace(A) / m) + 1e-300
        gamma = jnp.linalg.solve(A + lam * jnp.eye(m), dF @ f)
        x_aa = x + beta * f - (dX + beta * dF).T @ gamma

        # AA only after warm-up AND once the history has refilled post-restart
        use_aa = (k >= warmup) & (k - k_restart > m) & ~grew
        x_new = _floor(jnp.where(use_aa, x_aa, x_picard))
        return x_new, dX, dF, f, x, k + 1, res, res, k_restart

    z = jnp.zeros((m, N))
    x, _, _, _, _, k, res, _, _ = jax.lax.while_loop(
        cond, body,
        (x0, z, z, jnp.zeros(N), x0, 0, jnp.asarray(jnp.inf),
         jnp.asarray(jnp.inf), -m - 1))
    return x, res, k
