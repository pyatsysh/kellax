"""Generic bordered Newton in JAX — one solver, k border rows.

Solve the square (N+k) system

    residual(x, aux)    = 0        (N rows)
    constraints(x, aux) = 0        (k rows)

for (x in R^N, aux in R^k) by dense Newton with jax.jacfwd on the stacked
system. This is the one-primitive unification behind everything bordered:
pseudo-arclength continuation (aux = the control parameter, constraint = the
arclength condition), Moore-Spence fold refinement (constraint = null-vector
normalisation), canonical-ensemble mass constraints (aux = a Lagrange
multiplier / chemical potential, constraint = fixed total mass), phase
conditions. Dense linear algebra (suited to N up to a few thousand); the
matrix-free bordered solve is the planned scale-up path.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def bordered_newton(residual: Callable, constraints: Callable,
                    x0: jnp.ndarray, aux0, tol: float = 1e-10,
                    max_iter: int = 25):
    """Newton on ``residual(x, aux) = 0`` stacked with ``constraints(x, aux) = 0``.

    Both callables receive (x, aux) with aux a (k,) array (a scalar ``aux0``
    is promoted to k = 1) and may return arrays or scalars — outputs are
    flattened and stacked, and must total N + k rows. Returns (x, aux, res)
    with aux a (k,) array and res the max-abs stacked residual.
    """
    x0 = jnp.asarray(x0)
    aux0 = jnp.atleast_1d(jnp.asarray(aux0, dtype=x0.dtype))
    N = int(x0.shape[0])

    def G(z):
        x, aux = z[:N], z[N:]
        return jnp.concatenate([jnp.ravel(residual(x, aux)),
                                jnp.ravel(constraints(x, aux))])

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

    z, res = run(jnp.concatenate([x0, aux0]))
    return z[:N], z[N:], res
