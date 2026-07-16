"""Moore-Spence fold refinement and two-parameter fold tracking in JAX.

A fold (turning point) of R(x, p) = 0 is a solution where dR/dx is singular.
Arclength continuation only brackets it between accepted points; the
Moore-Spence augmented system pins it exactly: solve, for (x, p, v),

    (1) R(x, p)     = 0        (N rows)
    (2) R_x(x, p) v = 0        (N rows: v spans the null space of dR/dx)
    (3) c . v - 1   = 0        (1 row: normalisation, fixes the scale of v)

a square (2N+1) system whose Jacobian is non-singular at a generic
(quadratic) fold, so plain Newton converges quadratically from a nearby
point. ``refine_fold`` does exactly this; jax.jacfwd supplies every block,
including the directional second derivative d(R_x v)/dx, automatically.

``track_fold`` continues the fold itself in a second parameter q: the
Moore-Spence system for residual2(x, p, q) defines a curve (x, p, v)(q),
and that curve is traced by the very same pseudo-arclength
predictor-corrector as the primary branch — ``arclength_continuation``
applied to the augmented residual with q as the continuation parameter.
Parametrisation is therefore arclength in (x, p, v, q), not naive
q-stepping, so cusps of the fold curve (where dq/ds = 0) are passed and
reported as turning points of the augmented branch.
"""
from dataclasses import dataclass

import jax
import jax.numpy as np
import numpy as onp

from .keller import Branch, arclength_continuation

# 2Do: re-seed the frozen normalisation vector c when v rotates far along
# the fold curve (Newton can degrade there; not seen in practice yet, and
# the fix is one refine_fold call at the rotated point)


@dataclass
class FoldBranch:
    """A traced fold curve: fold states, fold locations p(q), null vectors."""
    x: onp.ndarray             # (M, N) states on the fold curve
    p: onp.ndarray             # (M,) fold location in the primary parameter
    q: onp.ndarray             # (M,) second (tracking) parameter
    v: onp.ndarray             # (M, N) null vectors of dR/dx along the curve
    branch: Branch             # raw augmented-system branch (tangents, steps,
                               # turning points = cusps of the fold curve)


def refine_fold(residual, x0, p0, v0 = None, tol = 1e-10, max_iter = 25):
    """Sharpen a point near a fold of ``residual(x, p) = 0`` to the exact fold.

    Newton on the (2N+1) Moore-Spence system in (x, p, v); seed (x0, p0) from
    e.g. a ``Branch.turning_points`` entry. ``v0`` seeds the null vector (e.g.
    the arclength tangent's x-part); default is the right singular vector of
    dR/dx(x0, p0) for its smallest singular value. The normalisation vector c
    is the unit-scaled v0. Returns (x, p, v, res) with p a float and res the
    max-abs augmented residual.
    """
    x0 = np.asarray(x0)
    N = int(x0.shape[0])
    Rx = jax.jacfwd(residual, argnums = 0)

    if v0 is None:
        _, _, Vt = np.linalg.svd(Rx(x0, np.asarray(float(p0), dtype = x0.dtype)))
        v0 = Vt[-1]
    v0 = np.asarray(v0, dtype = x0.dtype)
    v0 = v0 / np.linalg.norm(v0)
    c = v0

    def G(z):
        x, p, v = z[:N], z[N], z[N + 1:]
        return np.concatenate([residual(x, p), Rx(x, p) @ v,
                               np.reshape(c @ v - 1.0, (1,))])

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
            z = z - np.linalg.solve(Gz(z), g)
            return z, it + 1, np.max(np.abs(g))

        z, _, _ = jax.lax.while_loop(cond, body, (z, 0, np.inf))
        return z, np.max(np.abs(Gf(z)))

    z0 = np.concatenate([x0, np.reshape(np.asarray(float(p0), dtype = x0.dtype), (1,)), v0])
    z, res = run(z0)
    return z[:N], float(z[N]), z[N + 1:], res


def track_fold(residual2, x0, p0, q0, ds = 0.1, n_steps = 200,
        ds_min = 2e-3, ds_max = 0.5, newton_tol = 1e-10, newton_max = 25,
        q_min = -onp.inf, q_max = onp.inf, direction = 1.0,
        accept_res = 1e-8, verbose = False):
    """Continue a fold of ``residual2(x, p, q) = 0`` in the second parameter q.

    The fold near (x0, p0) at q = q0 is first refined (Moore-Spence Newton),
    then the augmented state y = (x, p, v) is traced in q with
    ``arclength_continuation`` on the Moore-Spence residual — every accepted
    point is a converged fold, so accuracy is Newton-limited, not
    step-limited. ``direction`` = +1 to start toward increasing q. Parameters
    mirror ``arclength_continuation`` with q_min/q_max bounding q. Returns a
    FoldBranch.
    """
    x0 = np.asarray(x0)
    N = int(x0.shape[0])
    Rx = jax.jacfwd(residual2, argnums = 0)

    x, p, v, res = refine_fold(lambda x_, p_: residual2(x_, p_, float(q0)),
                               x0, p0, tol = newton_tol, max_iter = newton_max)
    if float(res) > accept_res:
        raise RuntimeError(f"initial fold did not converge (res={float(res):.2e})")
    v = v / np.linalg.norm(v)                        # refresh c at the refined fold
    c = v

    def ms_residual(y, q):
        x, p, v = y[:N], y[N], y[N + 1:]
        return np.concatenate([residual2(x, p, q), Rx(x, p, q) @ v,
                               np.reshape(c @ v - 1.0, (1,))])

    y0 = np.concatenate([x, np.reshape(np.asarray(p, dtype = x.dtype), (1,)), v])
    br = arclength_continuation(ms_residual, y0, float(q0), ds = ds,
                                n_steps = n_steps, ds_min = ds_min, ds_max = ds_max,
                                newton_tol = newton_tol, newton_max = newton_max,
                                p_min = q_min, p_max = q_max, direction = direction,
                                accept_res = accept_res, verbose = verbose)
    return FoldBranch(x = br.x[:, :N], p = br.x[:, N], q = br.p,
                      v = br.x[:, N + 1:], branch = br)
