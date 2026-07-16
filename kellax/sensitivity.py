"""Differentiable continuation — gradients of fold locations.

The fold of R(x, p; theta) = 0 sits at the solution z* = (x*, p*, v*) of the
Moore-Spence system G(z; theta) = 0 (folds.py). G is non-singular in z at a
generic fold, so the implicit function theorem gives the exact sensitivity
of the WHOLE fold state to the model parameters:

    dz*/dtheta = - G_z^{-1} G_theta,        dp*/dtheta = row N of it.

No unrolling of the Newton iterations, no finite differences: one linear
solve against the same Moore-Spence Jacobian that Newton already used, with
G_theta supplied by jax.jacfwd. This is what lets a model be optimised or
learned AGAINST its bifurcation diagram — move a fold to a prescribed
parameter value, set the width of a hysteresis loop — and it is the reason
kellax is written in JAX.

E.G. the cubic with a tilt, R = x^3 - theta x + p: the fold law is
p*(theta) = 2 (theta/3)^{3/2}, so dp*/dtheta = sqrt(theta/3) — and
fold_sensitivity returns exactly that, to Newton precision:

    R3 = lambda x, p, th: np.array([x[0]**3 - th[0]*x[0] + p])
    x, p, v, dp, res = fold_sensitivity(R3, np.array([0.6]), 0.4, np.array([1.0]))
    # p -> 0.3849, dp -> [0.57735] = sqrt(1/3)
"""
import jax
import jax.numpy as np

from .folds import refine_fold

# 2Do: a jax.custom_jvp wrapper theta -> p_fold with this rule, so that
# jax.grad composes through the fold location directly (needs the refinement
# loop in pure traced form — the while_loop is already there, the float()
# casts at the boundary are not)


def fold_sensitivity(residual, x0, p0, theta0, v0 = None,
                     tol = 1e-10, max_iter = 25):
    """Fold location AND its gradient with respect to the model parameters.

    ``residual(x, p, theta)`` with theta a (T,) array of model parameters.
    The fold near (x0, p0) at theta = theta0 is refined by Moore-Spence
    Newton (folds.refine_fold), then differentiated implicitly. Returns
    (x, p, v, dp_dtheta, res): the fold state, its location, the null
    vector, the (T,) gradient of the location, and the refinement residual.
    """
    theta0 = np.atleast_1d(np.asarray(theta0, dtype = float))

    # (1) converge the fold at theta0
    x, p, v, res = refine_fold(lambda x_, p_: residual(x_, p_, theta0),
                               x0, p0, v0 = v0, tol = tol, max_iter = max_iter)
    x = np.asarray(x)
    N = int(x.shape[0])
    c = v / np.linalg.norm(v)             # same normalisation as the refinement

    # (2) the Moore-Spence map, now explicitly a function of theta
    def G(z, th):
        x_, p_, v_ = z[:N], z[N], z[N + 1:]
        Rx = jax.jacfwd(residual, argnums = 0)
        return np.concatenate([residual(x_, p_, th), Rx(x_, p_, th) @ v_,
                               np.reshape(c @ v_ - 1.0, (1,))])

    z = np.concatenate([x, np.reshape(np.asarray(p, dtype = x.dtype), (1,)), v])

    # (3) implicit function theorem: dz/dtheta = -G_z^{-1} G_theta
    Gz = jax.jacfwd(G, argnums = 0)(z, theta0)           # (2N+1, 2N+1)
    Gth = jax.jacfwd(G, argnums = 1)(z, theta0)          # (2N+1, T)
    dz = -np.linalg.solve(Gz, Gth)
    dp_dtheta = dz[N]                                    # (T,)

    return x, float(p), v, dp_dtheta, res
