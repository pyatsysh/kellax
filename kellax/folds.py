"""Moore-Spence fold refinement and two-parameter fold tracking in JAX.

A fold (turning point) of R(x, p) = 0 is a solution where dR/dx is singular.
Arclength continuation only brackets it between accepted points; the
Moore-Spence augmented system pins it exactly: solve, for (x, p, v),

    R(x, p)     = 0        (N rows)
    R_x(x, p) v = 0        (N rows: v spans the null space of dR/dx)
    c . v - 1   = 0        (1 row: normalisation, fixes the scale of v)

a square (2N+1) system whose Jacobian is non-singular at a generic (quadratic)
fold, so plain Newton converges quadratically from a nearby point.
``refine_fold`` does exactly this; jax.jacfwd supplies every block, including
the directional second derivative d(R_x v)/dx, automatically.

``track_fold`` continues the fold itself in a second parameter q: the
Moore-Spence system for residual2(x, p, q) defines a curve (x, p, v)(q), and
that curve is traced by the very same pseudo-arclength predictor-corrector as
the primary branch — ``arclength_continuation`` applied to the augmented
residual with q as the continuation parameter. Parametrisation is therefore
arclength in (x, p, v, q), not naive q-stepping, so cusps of the fold curve
(where dq/ds = 0) are passed and reported as turning points of the augmented
branch. The normalisation vector c is frozen at the initial refined null
vector; re-seed if v rotates far along the curve.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

from .keller import Branch, arclength_continuation


@dataclass
class FoldBranch:
    """A traced fold curve: fold states, fold locations p(q), null vectors."""
    x: np.ndarray              # (M, N) states on the fold curve
    p: np.ndarray              # (M,) fold location in the primary parameter
    q: np.ndarray              # (M,) second (tracking) parameter
    v: np.ndarray              # (M, N) null vectors of dR/dx along the curve
    branch: Branch             # raw augmented-system branch (tangents, steps,
                               # turning points = cusps of the fold curve)


def refine_fold(residual: Callable, x0: jnp.ndarray, p0: float,
                v0: jnp.ndarray | None = None,
                tol: float = 1e-10, max_iter: int = 25):
    """Sharpen a point near a fold of ``residual(x, p) = 0`` to the exact fold.

    Newton on the (2N+1) Moore-Spence system in (x, p, v); seed (x0, p0) from
    e.g. a ``Branch.turning_points`` entry. ``v0`` seeds the null vector (e.g.
    the arclength tangent's x-part); default is the right singular vector of
    dR/dx(x0, p0) for its smallest singular value. The normalisation vector c
    is the unit-scaled v0. Returns (x, p, v, res) with p a float and res the
    max-abs augmented residual.
    """
    x0 = jnp.asarray(x0)
    N = int(x0.shape[0])
    Rx = jax.jacfwd(residual, argnums=0)

    if v0 is None:
        _, _, Vt = jnp.linalg.svd(Rx(x0, jnp.asarray(float(p0), dtype=x0.dtype)))
        v0 = Vt[-1]
    v0 = jnp.asarray(v0, dtype=x0.dtype)
    v0 = v0 / jnp.linalg.norm(v0)
    c = v0

    def G(z):
        x, p, v = z[:N], z[N], z[N + 1:]
        return jnp.concatenate([residual(x, p), Rx(x, p) @ v,
                                jnp.reshape(c @ v - 1.0, (1,))])

    Gf = jax.jit(G)
    Gz = jax.jit(jax.jacfwd(G))

    @jax.jit
    def run(z):
        def cond(s):
            _, it, res = s
            return (it < max_iter) & (res > tol)

        def body(s):
            z, it, _ = s
            g = Gf(z)
            z = z - jnp.linalg.solve(Gz(z), g)
            return z, it + 1, jnp.max(jnp.abs(g))

        z, _, _ = jax.lax.while_loop(cond, body, (z, 0, jnp.inf))
        return z, jnp.max(jnp.abs(Gf(z)))

    z0 = jnp.concatenate([x0, jnp.reshape(jnp.asarray(float(p0), dtype=x0.dtype), (1,)), v0])
    z, res = run(z0)
    return z[:N], float(z[N]), z[N + 1:], res


def track_fold(residual2: Callable, x0: jnp.ndarray, p0: float, q0: float,
               ds: float = 0.1, n_steps: int = 200,
               ds_min: float = 2e-3, ds_max: float = 0.5,
               newton_tol: float = 1e-10, newton_max: int = 25,
               q_min: float = -np.inf, q_max: float = np.inf,
               direction: float = 1.0, accept_res: float = 1e-8,
               verbose: bool = False) -> FoldBranch:
    """Continue a fold of ``residual2(x, p, q) = 0`` in the second parameter q.

    The fold near (x0, p0) at q = q0 is first refined (Moore-Spence Newton),
    then the augmented state y = (x, p, v) is traced in q with
    ``arclength_continuation`` on the Moore-Spence residual — every accepted
    point is a converged fold, so accuracy is Newton-limited, not
    step-limited. ``direction`` = +1 to start toward increasing q. Parameters
    mirror ``arclength_continuation`` with q_min/q_max bounding q.
    """
    x0 = jnp.asarray(x0)
    N = int(x0.shape[0])
    Rx = jax.jacfwd(residual2, argnums=0)

    x, p, v, res = refine_fold(lambda x_, p_: residual2(x_, p_, float(q0)),
                               x0, p0, tol=newton_tol, max_iter=newton_max)
    if float(res) > accept_res:
        raise RuntimeError(f"initial fold did not converge (res={float(res):.2e})")
    v = v / jnp.linalg.norm(v)                        # refresh c at the refined fold
    c = v

    def ms_residual(y, q):
        x, p, v = y[:N], y[N], y[N + 1:]
        return jnp.concatenate([residual2(x, p, q), Rx(x, p, q) @ v,
                                jnp.reshape(c @ v - 1.0, (1,))])

    y0 = jnp.concatenate([x, jnp.reshape(jnp.asarray(p, dtype=x.dtype), (1,)), v])
    br = arclength_continuation(ms_residual, y0, float(q0), ds=ds,
                                n_steps=n_steps, ds_min=ds_min, ds_max=ds_max,
                                newton_tol=newton_tol, newton_max=newton_max,
                                p_min=q_min, p_max=q_max, direction=direction,
                                accept_res=accept_res, verbose=verbose)
    return FoldBranch(x=br.x[:, :N], p=br.x[:, N], q=br.p,
                      v=br.x[:, N + 1:], branch=br)
