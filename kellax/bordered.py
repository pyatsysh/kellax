"""Generic bordered Newton in JAX — one solver, k border rows.

Solve the square (N+k) system

    residual(x, aux)    = 0        (N rows)
    constraints(x, aux) = 0        (k rows)

for (x in R^N, aux in R^k) by dense Newton with jax.jacfwd on the stacked
system. This is the one-primitive unification behind everything bordered:
(1) pseudo-arclength continuation — aux = the control parameter,
constraint = the arclength condition; (2) Moore-Spence fold refinement —
constraint = the null-vector normalisation; (3) canonical-ensemble mass
constraints — aux = a Lagrange multiplier / chemical potential, constraint
= fixed total mass; (4) phase conditions. Dense linear algebra (N up to a
few thousand); the matrix-free engine (matrixfree.py) covers the arclength
specialisation at large N.
"""
import jax
import jax.numpy as np

# 2Do: generic matrix-free k-row bordered solve (GMRES on the stacked JVP),
# so mass and phase constraints scale the way the arclength row already does


def bordered_newton(residual, constraints, x0, aux0, tol = 1e-10, max_iter = 25):
    """Newton on ``residual(x, aux) = 0`` stacked with ``constraints(x, aux) = 0``.

    Both callables receive (x, aux) with aux a (k,) array (a scalar ``aux0``
    is promoted to k = 1) and may return arrays or scalars — outputs are
    flattened and stacked, and must total N + k rows. Returns (x, aux, res)
    with aux a (k,) array and res the max-abs stacked residual.
    """
    x0 = np.asarray(x0)
    aux0 = np.atleast_1d(np.asarray(aux0, dtype = x0.dtype))
    N = int(x0.shape[0])

    def G(z):
        x, aux = z[:N], z[N:]
        return np.concatenate([np.ravel(residual(x, aux)),
                               np.ravel(constraints(x, aux))])

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

    z, res = run(np.concatenate([x0, aux0]))
    return z[:N], z[N:], res
