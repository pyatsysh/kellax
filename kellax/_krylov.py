"""Shared Krylov plumbing: one statement of the GMRES forcing.

Every matrix-free solve in kellax stops on the same rule, and this module is
the single place that says what it is. jax's ``gmres`` tests the
PRECONDITIONED residual against ``tol * |b|`` of the RAW right-hand side, so
a well-normalised Minv (|Minv b| << |b|) reports convergence at iterate 0
and hands back a zero step; the caller then sees a stalled line search and a
false ``converged = False``. Stating the forcing as ``tol = 0`` and
``atol = factor * |Minv b|`` is invariant to scaling Minv and reduces to the
old ``tol = factor`` wherever there is no preconditioner.

That defect was repaired twice before this module existed — once in
``matrixfree`` and once in ``newton_krylov``, a week apart — because the
forcing had been written out by hand at four call sites and the second copy
arrived from the cdft toolbox carrying the original flaw. Four hand-written
copies of a stopping rule is three too many: it is written once here, and
the bordered and unbordered engines both call it.
"""
import jax
import jax.numpy as np
from jax.scipy.sparse.linalg import gmres

ARMIJO = 1e-4
TRUST_MAX = 64.0


def trust_update(trust, ok, t, capped, dx_max):
    """Adaptive trust radius: a clean full step that hit the cap doubles it
    (an exactly-linear far field is walked in O(log) steps instead of
    crawling), a backtrack halves it (never below the configured dx_max)."""
    return np.where(ok & (t >= 1.0) & capped,
                    np.minimum(trust * 2.0, TRUST_MAX),
                    np.where(t < 1.0, np.maximum(trust * 0.5, dx_max), trust))


def backtrack(merit_state, merit_norm, m0, ls_max):
    """Armijo backtracking: halve t until ||merit|| <= (1 - c t) m0 or ls_max
    halvings are spent. Returns (t, state, k).

    The test is written negated so that it is NaN-safe: an overflow in a trial
    step gives a NaN norm, ~(NaN <= ...) is True, and the step is rejected
    rather than accepted. ``merit_state(t)`` returns whatever the caller's
    residual bundle is (an array, or an (R, constraint) pair); ``merit_norm``
    reduces that bundle to a scalar."""

    def cond(c):
        t, s, k = c
        return (k < ls_max) & ~(merit_norm(s) <= (1.0 - ARMIJO * t) * m0)

    def body(c):
        t, _, k = c
        t = 0.5 * t
        return t, merit_state(t), k + 1

    return jax.lax.while_loop(cond, body, (1.0, merit_state(1.0), 0))


def tree_l2(t):
    """L2 norm over a pytree, so one norm serves a flat state and a bordered
    ``(state, scalar)`` pair alike. ``vdot`` conjugates, so a complex leaf
    contributes |v|^2 rather than v^2; on the real fields kellax traces the
    two agree exactly."""
    return np.sqrt(sum(np.vdot(v, v) for v in jax.tree_util.tree_leaves(t)))


def forced_gmres(op, b, M, atol_factor, x0, restart, maxiter):
    """Preconditioned GMRES on ``op``, forcing stated in the preconditioned
    norm: ``tol = 0`` and ``atol = atol_factor * |M b|``.

    ``atol_factor`` is the Eisenstat-Walker eta at a Newton step and a fixed
    tight constant for a tangent solve. ``b``, ``x0`` and the result share
    one pytree structure — a flat array for an unbordered system, an
    ``(N-block, scalar)`` tuple for a bordered one. Non-finite entries of the
    solution are zeroed: a Krylov breakdown then costs the step rather than
    poisoning the branch with NaN.
    """
    bM = b if M is None else M(b)
    d, _ = gmres(op, b, x0 = x0, M = M, tol = 0.0,
                 atol = atol_factor * tree_l2(bM), restart = restart,
                 maxiter = maxiter, solve_method = "batched")
    return jax.tree_util.tree_map(lambda v: np.where(np.isfinite(v), v, 0.0), d)
