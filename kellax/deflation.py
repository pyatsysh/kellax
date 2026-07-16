"""Deflated Newton — find the solutions you have NOT already found.

Given known solutions x_1..x_k of R(x, p) = 0, Newton on the deflated
residual

    G(x) = M(x) R(x, p),        M(x) = prod_i ( 1/||x - x_i||^power + shift )

cannot converge back to any x_i (the deflation factor blows up there), so a
plain initial guess is pushed toward a DIFFERENT solution (Farrell, Birkisson
& Funke 2015). This is how disconnected branches are found without a
hand-crafted seed, and how a branch point is jumped (branching.branch_off
deflates against the branch you are already on).

The Jacobian of the deflated residual involves derivatives of M(x) times
R(x) — jax.jacfwd differentiates the whole product, so none of that operator
calculus is written by hand here.
"""
import jax
import jax.numpy as np

# 2Do: deflated continuation (run the arclength trace under a deflation
# operator that steps along KNOWN branches) — the DeflatedContinuation of
# BifurcationKit; needs the engine to accept a residual factory


def deflated_newton(residual, x0, p, known, power = 2.0, shift = 1.0,
                    tol = 1e-10, max_iter = 50):
    """Newton on the deflated residual: converge to a solution NOT in ``known``.

    ``known`` is a list/array of solutions to deflate against, shape (k, N).
    Returns (x, res) with res the max-abs of the ORIGINAL residual at x —
    check res < tol AND the distance to ``known`` before trusting the root
    (a deflated solve that stalls returns wherever it stopped).
    """
    x0 = np.asarray(x0, dtype = float)
    K = np.atleast_2d(np.asarray(known, dtype = float))   # (k, N)

    def G(x):
        d2 = np.sum((x[None, :] - K) ** 2, axis = 1)      # ||x - x_i||^2
        M = np.prod(d2 ** (-power / 2.0) + shift)
        return M * residual(x, p)

    Gf = jax.jit(G)
    Gx = jax.jit(jax.jacfwd(G))
    Rf = jax.jit(residual)

    @jax.jit
    def run(x):
        def cond(s):
            _, it, res = s
            return (it < max_iter) & (res > tol)

        def body(s):
            x, it, _ = s
            g = Gf(x)
            x = x - np.linalg.solve(Gx(x), g)
            return x, it + 1, np.max(np.abs(Rf(x, p)))

        x, _, _ = jax.lax.while_loop(cond, body, (x, 0, np.inf))
        return x, np.max(np.abs(Rf(x, p)))

    return run(x0)


def deflated_search(residual, x0, p, max_solutions = 8, seed_scale = 0.0,
                    power = 2.0, shift = 1.0, tol = 1e-9, max_iter = 50,
                    dist_tol = 1e-6):
    """Collect distinct solutions of R(x, p) = 0 from ONE initial guess.

    (1) plain Newton finds the first root; (2) each further root is found by
    deflating against everything already known; (3) the search stops when a
    deflated solve fails to produce a fresh solution, or at
    ``max_solutions``. ``seed_scale`` optionally perturbs the guess between
    attempts (0 = reuse x0, which deflation alone usually moves off).
    Returns the list of solutions found.
    """
    from .keller import newton

    x0 = np.asarray(x0, dtype = float)
    x, r = newton(residual, x0, p, tol = tol, max_iter = max_iter)
    if float(r) > tol:
        return []
    found = [x]
    for k in range(1, max_solutions):
        seed = x0 if seed_scale == 0.0 else x0 + seed_scale * np.sin(np.arange(x0.shape[0]) + k)
        xn, rn = deflated_newton(residual, seed, p, found, power = power,
                                 shift = shift, tol = tol, max_iter = max_iter)
        fresh = min(float(np.linalg.norm(xn - f)) for f in found) > dist_tol
        if float(rn) > tol or not fresh:
            break
        found.append(xn)
    return found
