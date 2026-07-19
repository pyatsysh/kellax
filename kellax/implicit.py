"""Differentiable solutions: the implicit-function-theorem injection.

The seam that lets gradients flow from a loss back to model parameters
*through a converged solve*, without unrolling the solver (wrong, and not
reverse-mode friendly at depth). Given a residual R(x, theta) and a
converged x* (obtained however you like, under stop_gradient), attach the
exact gradient with a single injection step

    x = x* - J^{-1} R(x*, theta),     J = dR/dx |_{x*, theta frozen}.

Because R(x*) ~ 0 the *value* is unchanged, but autodiff through this line
yields dx*/dtheta = -J^{-1} dR/dtheta — gradients on the solution manifold,
independent of the solver path. J is frozen w.r.t. theta (stop_gradient) so
third-order terms that R ~ 0 kills are never paid for.

Two linear-solve routes: dense (default; jacobian + linalg.solve, right for
small x) and matrix-free (pass ``linear_solve``, e.g. a GMRES closure; the
injection becomes a ``lax.custom_linear_solve`` whose cotangent solve uses
the transposed operator — no dense Jacobian anywhere, forward or adjoint;
jit- and vmap-compatible). Consolidated from the cdft learn layer, where it
carried training gradients through canonical DFT solves; ``fold_sensitivity``
applies the same principle to a converged Moore-Spence system.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def ift_injection(residual, x_star, params, linear_solve=None,
                  transpose_solve=None):
    """Return x* with the exact d x*/d params attached (value unchanged).

    ``residual(x, params)`` must vanish at the converged ``x_star``.
    ``linear_solve(matvec, b)`` enables the matrix-free route;
    ``transpose_solve`` defaults to ``linear_solve``.
    """
    p_sg = jax.lax.stop_gradient(params)
    x = jax.lax.stop_gradient(x_star)
    R = residual(x, params)                      # carries dR/dtheta

    if linear_solve is None:
        J = jax.jacobian(lambda r: residual(r, p_sg))(x)
        delta = jnp.linalg.solve(jax.lax.stop_gradient(J), R)
    else:
        def Jv(v):
            return jax.jvp(lambda q: residual(q, p_sg), (x,), (v,))[1]

        delta = jax.lax.custom_linear_solve(
            Jv, R, solve=linear_solve,
            transpose_solve=(transpose_solve or linear_solve))
    return x - delta
