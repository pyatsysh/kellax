"""kellax standalone validation — no DFT anywhere.

Run: .venv/bin/python tests/test_kellax.py  (or pytest -q)
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as np
import numpy as onp

from kellax import (arclength_continuation, bordered_newton, newton,
                    refine_fold, track_fold, mf_arclength_continuation)


def test_newton_scalar_root():
    R = lambda x, p: np.array([x[0] ** 3 - 2.0 * x[0] + p])
    x, res = newton(R, np.array([2.0]), p = 1.0)
    assert float(res) < 1e-12
    assert abs(float(x[0] ** 3 - 2 * x[0] + 1.0)) < 1e-12


def test_cubic_fold_curve():
    """R(x,p) = x^3 - x + p: the canonical S-curve with folds at
    p = +-2/(3 sqrt 3). Trace through both folds, locate them, and span all
    three solution regimes."""
    R = lambda x, p: np.array([x[0] ** 3 - x[0] + p])
    x0 = np.array([-1.2])                           # lower branch at p ~ 0.7
    br = arclength_continuation(R, x0, p0 = 0.7, ds = 0.05, ds_max = 0.1,
                                n_steps = 400, p_min = -1.5, p_max = 1.5,
                                direction = -1.0)    # toward the folds

    p_fold = 2.0 / (3.0 * onp.sqrt(3.0))
    assert len(br.turning_points) >= 2, br.turning_points
    # the reported fold is the nearest ACCEPTED point -> step-limited
    # accuracy here; refine_fold pins it exactly (tested below)
    folds = sorted(br.p[br.turning_points[:2]])
    assert abs(abs(folds[0]) - p_fold) < 1e-2, folds
    assert abs(abs(folds[1]) - p_fold) < 1e-2, folds
    xs = br.x[:, 0]
    assert xs.min() < -1.0 and xs.max() > 1.0        # spans lower -> upper branch
    # every accepted point satisfies the residual
    worst = max(abs(float(x ** 3 - x + p)) for x, p in zip(xs, br.p))
    assert worst < 1e-8, worst


def test_multidim_fold():
    """A 2D system with a fold: x1 follows a cubic in p, x2 relaxes to x1^2.
    Continuation handles the coupled system and finds the folds."""
    def R(x, p):
        return np.array([x[0] ** 3 - x[0] + p,
                         x[1] - x[0] ** 2])

    br = arclength_continuation(R, np.array([-1.2, 1.44]), p0 = 0.7,
                                ds = 0.05, ds_max = 0.1, n_steps = 400,
                                p_min = -1.5, p_max = 1.5, direction = -1.0)
    assert len(br.turning_points) >= 2
    assert onp.max(onp.abs(br.x[:, 1] - br.x[:, 0] ** 2)) < 1e-8


def test_refine_fold_cubic():
    """Moore-Spence refinement: from a coarse continuation's turning points
    (step-limited, ~1e-2) to the exact cubic folds at p = -+2/(3 sqrt 3),
    x = -+1/sqrt(3), to 1e-10."""
    R = lambda x, p: np.array([x[0] ** 3 - x[0] + p])
    br = arclength_continuation(R, np.array([-1.2]), p0 = 0.7, ds = 0.05,
                                ds_max = 0.1, n_steps = 400, p_min = -1.5,
                                p_max = 1.5, direction = -1.0)
    p_fold = 2.0 / (3.0 * onp.sqrt(3.0))
    assert len(br.turning_points) >= 2

    i = br.turning_points[0]                        # lower-branch fold: p < 0
    x, p, v, res = refine_fold(R, np.array(br.x[i]), float(br.p[i]))
    assert float(res) < 1e-10
    assert abs(p + p_fold) < 1e-10, p
    assert abs(float(x[0]) + 1.0 / onp.sqrt(3.0)) < 1e-10
    assert abs(float(3.0 * x[0] ** 2 - 1.0)) < 1e-10     # dR/dx singular
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-10   # c . v = 1, c = v0/|v0|

    j = br.turning_points[1]                        # middle-branch fold: p > 0
    _, p2, _, res2 = refine_fold(R, np.array(br.x[j]), float(br.p[j]))
    assert float(res2) < 1e-10
    assert abs(p2 - p_fold) < 1e-10, p2


def test_track_fold_cubic_in_q():
    """R(x,p,q) = x^3 - q*x + p folds at p = +-2 (q/3)^{3/2}. Track the
    positive fold from q=1 to q=2: every accepted point is a converged
    Moore-Spence solution, so the whole p_fold(q) curve matches the analytic
    law to Newton precision (not step-limited)."""
    R2 = lambda x, p, q: np.array([x[0] ** 3 - q * x[0] + p])
    fb = track_fold(R2, np.array([0.6]), p0 = 0.4, q0 = 1.0, ds = 0.1,
                    q_min = 0.5, q_max = 2.0)

    assert fb.q[0] == 1.0 and fb.q.max() >= 2.0      # reached q = 2
    err_p = onp.max(onp.abs(fb.p - 2.0 * (fb.q / 3.0) ** 1.5))
    assert err_p < 1e-8, err_p
    err_x = onp.max(onp.abs(fb.x[:, 0] - onp.sqrt(fb.q / 3.0)))
    assert err_x < 1e-8, err_x
    # null vectors stay normalised against the frozen c ( = initial v)
    assert onp.max(onp.abs(onp.abs(fb.v[:, 0]) - 1.0)) < 1e-8
    assert len(fb.branch.turning_points) == 0        # no cusps on this curve


def test_bordered_newton_mass_constraint():
    """Canonical-ensemble structure: minimise the toy free energy
    f(x) = sum((x - a)^2) + eps * sum_{i<j} x_i x_j under the mass constraint
    sum(x) = M, with the Lagrange multiplier (chemical potential) as the
    bordered auxiliary. Analytic: x_i (2 - eps) = 2 a_i - eps*M + lam."""
    a = np.array([0.3, -0.1, 0.7, 0.2])
    eps, M = 0.25, 2.0
    n = int(a.shape[0])

    def f(x):                                        # pair term via (sum^2 - sum of squares)/2
        return np.sum((x - a) ** 2) + 0.5 * eps * (np.sum(x) ** 2 - np.sum(x ** 2))

    grad_f = jax.grad(f)
    residual = lambda x, aux: grad_f(x) - aux[0]     # grad f - lam * 1 = 0
    constraints = lambda x, aux: np.sum(x) - M       # sum(x) - M = 0

    x, aux, res = bordered_newton(residual, constraints, np.zeros(n), 0.0)
    assert float(res) < 1e-10

    lam = ((2.0 - eps) * M + n * eps * M - 2.0 * float(np.sum(a))) / n
    x_exact = (2.0 * a - eps * M + lam) / (2.0 - eps)
    assert abs(float(np.sum(x)) - M) < 1e-12
    assert float(np.max(np.abs(x - x_exact))) < 1e-10
    assert abs(float(aux[0]) - lam) < 1e-10


def test_bordered_newton_moore_spence():
    """The cubic fold re-derived as a bordered problem — state (x, v), aux p,
    Moore-Spence rows as the residual, normalisation as the border — and
    cross-checked against refine_fold from the same seed."""
    R = lambda x, p: np.array([x[0] ** 3 - x[0] + p])
    Rx = jax.jacfwd(R, argnums = 0)
    c = np.array([1.0])

    def residual(xv, aux):
        x, v = xv[:1], xv[1:]
        return np.concatenate([R(x, aux[0]), Rx(x, aux[0]) @ v])

    constraints = lambda xv, aux: c @ xv[1:] - 1.0

    xv, aux, res = bordered_newton(residual, constraints,
                                   np.array([0.9, 1.0]), 0.5)
    p_fold = 2.0 / (3.0 * onp.sqrt(3.0))
    assert float(res) < 1e-10
    assert abs(float(aux[0]) - p_fold) < 1e-10
    assert abs(float(xv[0]) - 1.0 / onp.sqrt(3.0)) < 1e-10

    xr, pr, vr, rr = refine_fold(R, np.array([0.9]), 0.5, v0 = np.array([1.0]))
    assert float(rr) < 1e-10
    assert abs(float(aux[0]) - pr) < 1e-12           # same fold, two formulations
    assert abs(float(xv[0] - xr[0])) < 1e-12
    assert abs(float(xv[1] - vr[0])) < 1e-12


def _bratu(N):
    """Discretised -u'' = lambda e^u on N interior points, with a spectral
    (DST) preconditioner ~ (D2)^{-1} for the matrix-free solve."""
    h = 1.0 / (N + 1)
    off = np.ones(N - 1)
    D2 = (np.diag(-2.0 * np.ones(N)) + np.diag(off, 1) + np.diag(off, -1)) / h ** 2
    R = lambda u, lam: D2 @ u + lam * np.exp(u)
    lap_eig = np.asarray((2.0 * onp.cos(onp.arange(1, N + 1) * onp.pi / (N + 1)) - 2.0) / h ** 2)

    def dst1(v):
        v = np.asarray(v)
        ext = np.concatenate([np.zeros(1), v, np.zeros(1), -v[::-1]])
        return -np.fft.rfft(ext).imag[1:N + 1]

    precond = lambda v: dst1(dst1(v) * (2.0 / (N + 1)) / lap_eig)
    return R, precond


def test_mf_cubic_fold():
    """Matrix-free engine on the scalar cubic: same S-curve, same two folds as
    the dense engine, every accepted point satisfying the residual."""
    R = lambda x, p: np.array([x[0] ** 3 - x[0] + p])
    br = mf_arclength_continuation(R, np.array([-1.2]), p0 = 0.7, ds = 0.05,
                                   ds_max = 0.1, n_steps = 300, p_min = -1.2,
                                   p_max = 1.2, direction = -1.0)
    assert len(br.turning_points) >= 2, br.turning_points
    p_fold = 2.0 / (3.0 * onp.sqrt(3.0))
    folds = sorted(br.p[br.turning_points[:2]])
    assert abs(abs(folds[0]) - p_fold) < 2e-2, folds
    assert abs(abs(folds[1]) - p_fold) < 2e-2, folds
    worst = max(abs(float(x ** 3 - x + p)) for x, p in zip(br.x[:, 0], br.p))
    assert worst < 1e-6, worst


def test_mf_matches_dense_bratu():
    """Matrix-free == dense on a discretised BVP. With a spectral preconditioner
    the matrix-free trace finds the same Bratu fold as the dense engine, and
    Moore-Spence refinement of both agrees to Newton precision."""
    N = 80
    R, precond = _bratu(N)
    kw = dict(p0 = 0.3, ds = 0.05, ds_max = 0.2, n_steps = 400,
              p_min = 0.25, p_max = 6.0, direction = 1.0)
    bd = arclength_continuation(R, np.zeros(N), **kw)
    bm = mf_arclength_continuation(R, np.zeros(N), precond = precond,
                                   gmres_restart = 60, gmres_maxiter = 60, **kw)
    assert bd.turning_points and bm.turning_points
    _, lam_d, _, rd = refine_fold(R, np.array(bd.x[bd.turning_points[0]]),
                                  float(bd.p[bd.turning_points[0]]))
    _, lam_m, _, rm = refine_fold(R, np.array(bm.x[bm.turning_points[0]]),
                                  float(bm.p[bm.turning_points[0]]))
    assert float(rd) < 1e-8 and float(rm) < 1e-8, (rd, rm)
    assert abs(lam_d - lam_m) < 1e-6, (lam_d, lam_m)   # same discrete fold, two engines


def _bratu_2d(N):
    """Discretised Delta u + lambda e^u on the N x N unit-square grid, with the
    exactly normalised inverse-Laplacian (2-D DST) preconditioner. The exact
    normalisation is the point: it is the case that used to silence gmres."""
    h = 1.0 / (N + 1)
    k = onp.arange(1, N + 1)
    eig1 = (2.0 * onp.cos(k * onp.pi / (N + 1)) - 2.0) / h ** 2
    lam2 = np.asarray(eig1[:, None] + eig1[None, :])

    def R(u, lam):
        U = u.reshape(N, N)
        up = np.concatenate([U[1:], np.zeros((1, N))], 0)
        dn = np.concatenate([np.zeros((1, N)), U[:-1]], 0)
        lt = np.concatenate([U[:, 1:], np.zeros((N, 1))], 1)
        rt = np.concatenate([np.zeros((N, 1)), U[:, :-1]], 1)
        return ((up + dn + lt + rt - 4.0 * U) / h ** 2 + lam * np.exp(U)).reshape(-1)

    def dst1(v, axis):
        n = v.shape[axis]
        zeros = np.zeros(v.shape[:axis] + (1,) + v.shape[axis + 1:])
        ext = np.concatenate([zeros, v, zeros, -np.flip(v, axis)], axis)
        return -np.take(np.fft.rfft(ext, axis = axis).imag, np.arange(1, n + 1), axis)

    dst2d = lambda V: dst1(dst1(V, 0), 1)
    precond = lambda v: (dst2d(dst2d(v.reshape(N, N)) / lam2)
                         / (2.0 * (N + 1)) ** 2).reshape(-1)
    return R, precond


def test_mf_precond_2d_bratu():
    """Preconditioned matrix-free on a 2-D field. An exactly normalised
    inverse-Laplacian shrinks |Minv b| far below |b|, and gmres's stopping
    rule (preconditioned residual against tol * |b|) then returned a zero
    step at the seed, so the initial corrector stalled and the trace never
    started. The engine now states its forcing in the preconditioned norm;
    the trace must leave the seed, pass the fold, and agree with dense."""
    N = 16
    R, precond = _bratu_2d(N)
    kw = dict(p0 = 0.3, ds = 0.1, ds_max = 0.6, n_steps = 100,
              p_min = 0.25, p_max = 12.0, direction = 1.0)
    bd = arclength_continuation(R, np.zeros(N * N), **kw)
    bm = mf_arclength_continuation(R, np.zeros(N * N), precond = precond, **kw)
    assert bd.turning_points and bm.turning_points
    _, lam_d, _, rd = refine_fold(R, np.array(bd.x[bd.turning_points[0]]),
                                  float(bd.p[bd.turning_points[0]]))
    _, lam_m, _, rm = refine_fold(R, np.array(bm.x[bm.turning_points[0]]),
                                  float(bm.p[bm.turning_points[0]]))
    assert float(rd) < 1e-8 and float(rm) < 1e-8, (rd, rm)
    assert abs(lam_d - lam_m) < 1e-6, (lam_d, lam_m)   # same discrete fold


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"running {name} ...", flush = True)
            fn()
            print(f"  OK  {name}")
    print("all tests passed")
