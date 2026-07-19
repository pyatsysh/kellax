"""Autodiff-Hessian spectra: stability analysis of any scalar objective.

Consolidated from the cdft toolbox (where F was a free-energy functional and
the spectrum classified stability / spinodals / transition states — but
nothing here knows that): given a scalar F(x), the Hessian-vector product is
one autodiff line, H v = jvp(grad(F))(x, v), and a matrix-free Lanczos
(scipy ``eigsh``) extracts the extreme eigenvalues without ever forming H —
so it scales to large fields.

scipy is imported lazily by the spectrum functions only: the HVP itself is
pure JAX, and kellax's hard dependencies stay numpy + jax (install scipy to
use ``hessian_spectrum`` / ``smallest_eigenvalue`` / ``morse_index``).
"""
from __future__ import annotations

import numpy as onp
import jax
import jax.numpy as jnp


def hessian_vector_product(F, x):
    """Return a jitted HVP  v -> (d^2F/dx^2) v  at the point x."""
    grad = jax.grad(F)
    x = jnp.asarray(x, dtype=float)

    @jax.jit
    def hvp(v):
        return jax.jvp(grad, (x,), (v,))[1]

    return hvp


def hessian_spectrum(F, x, k: int = 6, which: str = "SA"):
    """The ``k`` extreme eigenvalues of the Hessian of F at ``x``
    (which='SA' = smallest algebraic; 'LA' = largest). Matrix-free Lanczos."""
    from scipy.sparse.linalg import LinearOperator, eigsh   # lazy: optional dep
    hvp = hessian_vector_product(F, x)
    N = int(onp.asarray(x).shape[0])
    A = LinearOperator((N, N),
                       matvec=lambda v: onp.asarray(hvp(jnp.asarray(v))), dtype=float)
    vals = eigsh(A, k=min(k, N - 2), which=which, return_eigenvectors=False)
    return onp.sort(onp.asarray(vals))


def smallest_eigenvalue(F, x) -> float:
    """The stability indicator: min eigenvalue of the Hessian (0 = marginal)."""
    return float(hessian_spectrum(F, x, k=1, which="SA")[0])


def morse_index(F, x, k: int = 8) -> int:
    """Number of negative Hessian eigenvalues (0 = minimum, 1 = saddle /
    transition state)."""
    return int((hessian_spectrum(F, x, k=k, which="SA") < 0).sum())
