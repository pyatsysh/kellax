"""v0.5 inner solvers: fixed point, Newton-Krylov (plain + bordered),
Hessian spectra, and the IFT injection.

Run: pytest tests/test_solvers.py -q
"""
import numpy as onp
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from kellax import (fixed_point_solve, newton_krylov, make_step_bordered,
                    hessian_spectrum, smallest_eigenvalue, morse_index,
                    ift_injection)


def test_fixed_point_and_anderson_speedup():
    """x = cos(x) elementwise; Anderson reaches the same point in fewer
    iterations than damped Picard, and the while_loop stops at tolerance."""
    gmap = jnp.cos
    x0 = jnp.full(64, 0.5)
    xp, rp, kp = fixed_point_solve(gmap, x0, tol=1e-12, max_steps=20000,
                                   damping=0.5, m=0)
    xa, ra, ka = fixed_point_solve(gmap, x0, tol=1e-12, max_steps=20000,
                                   damping=0.5, m=5, warmup=5)
    assert float(rp) < 1e-12 and float(ra) < 1e-12
    assert float(jnp.max(jnp.abs(xa - xp))) < 1e-10
    assert int(ka) < int(kp)
    assert int(kp) < 20000                          # early exit, not budget burn


def test_clamp_none_admits_signed_states():
    """A signed fixed point (x* < 0) is reachable with clamp=None; the
    density-style floor would pin it at the clamp instead."""
    gmap = lambda x: -0.5 + 0.1 * jnp.tanh(x)
    x0 = jnp.full(8, 0.2)
    x, res, _ = fixed_point_solve(gmap, x0, tol=1e-13, max_steps=5000,
                                  damping=0.5, m=0, clamp=None)
    assert float(res) < 1e-13
    assert float(x[0]) < -0.5                       # genuinely negative
    xc, resc, _ = fixed_point_solve(gmap, x0, tol=1e-13, max_steps=200,
                                    damping=0.5, m=0)   # default floor
    assert float(jnp.min(xc)) >= 1e-30              # floored, wrong for this map


def test_newton_krylov_plain():
    """Solve x^3 = b elementwise, far from the solution (trust + Armijo)."""
    b = jnp.linspace(1.0, 8.0, 32)
    residual = lambda x: x ** 3 - b
    x, res, k, ok = newton_krylov(residual, jnp.full(32, 3.0), tol=1e-12)
    assert ok and res < 1e-12
    assert float(jnp.max(jnp.abs(x - b ** (1.0 / 3.0)))) < 1e-10


def test_newton_krylov_bordered():
    """Bordered system: R(x, lam) = A x - lam * c = 0 with the mass
    constraint sum(x)/K = 1 — the (x, lam) pytree solved in one GMRES."""
    K = 24
    a = jnp.linspace(1.0, 2.0, K)
    c = jnp.ones(K)
    residual = lambda x, lam: a * x - lam * c
    constraint = lambda x: jnp.sum(x) / K - 1.0
    # analytic: x_i = lam / a_i, lam = K / sum(1/a)
    lam_exact = K / float(jnp.sum(1.0 / a))
    step = make_step_bordered(residual, constraint, None, dx_max=4.0,
                              restart=40, maxiter=25, eta_min=1e-6,
                              eta_max=1e-1, ls_max=25)
    x, lam, trust = jnp.full(K, 0.5), jnp.asarray(0.5), jnp.asarray(4.0)
    for _ in range(30):
        x, lam, trust, res, t, ok = step(x, lam, trust)
        if float(res) < 1e-12:
            break
    assert float(res) < 1e-12
    assert abs(float(lam) - lam_exact) < 1e-10
    assert float(jnp.max(jnp.abs(x - lam_exact / a))) < 1e-10


def test_hessian_spectrum_and_morse():
    """Quadratic with known spectrum; one negative direction flips Morse."""
    a = jnp.linspace(1.0, 3.0, 40)
    F = lambda x: 0.5 * jnp.sum(a * x * x)
    x0 = jnp.zeros(40)
    lam = hessian_spectrum(F, x0, k=3, which="SA")
    assert onp.allclose(lam, onp.linspace(1.0, 3.0, 40)[:3], atol=1e-8)
    assert abs(smallest_eigenvalue(F, x0) - 1.0) < 1e-8
    assert morse_index(F, x0) == 0
    G = lambda x: F(x) - 0.75 * x[0] * x[0]          # H_00: 1.0 - 1.5 = -0.5
    assert morse_index(G, x0) == 1
    assert abs(smallest_eigenvalue(G, x0) + 0.5) < 1e-8


def test_ift_injection_dense_and_matrixfree():
    """x*(theta) solves x - theta^2 * c = 0; d x*/d theta = 2 theta c,
    recovered through the injection on a solver-produced (noisy-path) x*."""
    from jax.scipy.sparse.linalg import gmres
    c = jnp.linspace(0.5, 1.5, 16)
    theta = 1.3

    def xstar_of(theta, linear_solve=None):
        residual = lambda x, th: x - th ** 2 * c
        x0 = jax.lax.stop_gradient(theta ** 2 * c)   # "converged" solution
        return ift_injection(residual, x0, theta, linear_solve=linear_solve)

    g_dense = jax.grad(lambda th: jnp.sum(xstar_of(th)))(theta)
    assert abs(float(g_dense) - float(2 * theta * jnp.sum(c))) < 1e-10

    solve = lambda mv, b: gmres(mv, b, tol=1e-13, atol=1e-13)[0]
    g_mf = jax.grad(lambda th: jnp.sum(xstar_of(th, solve)))(theta)
    assert abs(float(g_mf) - float(2 * theta * jnp.sum(c))) < 1e-8


def test_newton_krylov_preconditioned():
    """An exactly normalised inverse Laplacian used to silence gmres at the
    first iterate (zero step, stalled line search, converged=False): the
    forcing now lives in the preconditioned norm, so the solve must converge
    and match the unpreconditioned solution."""
    N = 200
    h = 1.0 / (N + 1)

    def residual(u):
        up = jnp.concatenate([u[1:], jnp.zeros(1)])
        um = jnp.concatenate([jnp.zeros(1), u[:-1]])
        return (um - 2.0 * u + up) / h ** 2 + jnp.exp(u)

    eig = jnp.asarray((2.0 * onp.cos(onp.arange(1, N + 1) * onp.pi / (N + 1))
                       - 2.0) / h ** 2)

    def dst1(v):
        ext = jnp.concatenate([jnp.zeros(1), v, jnp.zeros(1), -v[::-1]])
        return -jnp.fft.rfft(ext).imag[1:N + 1]

    precond = lambda v: dst1(dst1(v) / eig) / (2.0 * (N + 1))
    x, res, k, ok = newton_krylov(residual, jnp.zeros(N), precond=precond,
                                  tol=1e-10)
    assert ok and res < 1e-10
    x0, res0, k0, ok0 = newton_krylov(residual, jnp.zeros(N), tol=1e-10)
    assert ok0
    assert float(jnp.max(jnp.abs(x - x0))) < 1e-8
    assert k <= k0                                  # the preconditioner helps


def test_bordered_preconditioned():
    """make_step_bordered with the exact block inverse of a stiff diagonal
    system: |Minv b| ~ |b|/100, the regime where the old raw-norm tolerance
    silenced gmres. The step must move and the solve must converge."""
    K = 24
    a = jnp.linspace(1.0, 2.0, K)
    residual = lambda x, lam: 100.0 * (a * x - lam)
    constraint = lambda x: jnp.sum(x) / K - 1.0
    lam_exact = K / float(jnp.sum(1.0 / a))
    step = make_step_bordered(residual, constraint, lambda v: v / (100.0 * a),
                              dx_max=4.0, restart=40, maxiter=25,
                              eta_min=1e-6, eta_max=1e-1, ls_max=25)
    x, lam, trust = jnp.full(K, 0.5), jnp.asarray(0.5), jnp.asarray(4.0)
    for _ in range(30):
        x, lam, trust, res, t, ok = step(x, lam, trust)
        if float(res) < 1e-10:
            break
    assert float(res) < 1e-10
    assert abs(float(lam) - lam_exact) < 1e-8
    assert float(jnp.max(jnp.abs(x - lam_exact / a))) < 1e-8


def test_hessian_small_systems():
    """Dense fallback where Lanczos cannot run (eigsh crashed for N <= 2),
    and a Morse index that stays exact when more than k directions are
    negative (it was silently floored at k)."""
    F2 = lambda x: 0.5 * (x[0] ** 2 - x[1] ** 2)
    x2 = jnp.zeros(2)
    assert abs(smallest_eigenvalue(F2, x2) + 1.0) < 1e-12
    assert morse_index(F2, x2) == 1
    F1 = lambda x: 1.5 * x[0] ** 2
    assert abs(smallest_eigenvalue(F1, jnp.zeros(1)) - 3.0) < 1e-12
    a = jnp.concatenate([-jnp.linspace(1.0, 2.0, 10), jnp.asarray([0.5, 1.0])])
    F12 = lambda x: 0.5 * jnp.sum(a * x * x)
    assert morse_index(F12, jnp.zeros(12), k=8) == 10
