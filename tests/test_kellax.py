"""kellax standalone validation — no DFT anywhere.

Run: .venv/bin/python tests/test_kellax.py
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from kellax import (arclength_continuation, bordered_newton, newton,  # noqa: E402
                    refine_fold, track_fold)


def test_newton_scalar_root():
    R = lambda x, p: jnp.array([x[0] ** 3 - 2.0 * x[0] + p])
    x, res = newton(R, jnp.array([2.0]), p=1.0)
    assert float(res) < 1e-12
    assert abs(float(x[0] ** 3 - 2 * x[0] + 1.0)) < 1e-12


def test_cubic_fold_curve():
    """R(x,p) = x^3 - x + p: the canonical S-curve with folds at
    p = +-2/(3 sqrt 3). Trace through both folds, locate them, and span all
    three solution regimes."""
    R = lambda x, p: jnp.array([x[0] ** 3 - x[0] + p])
    x0 = jnp.array([-1.2])                          # lower branch at p ~ 0.7
    br = arclength_continuation(R, x0, p0=0.7, ds=0.05, ds_max=0.1,
                                n_steps=400, p_min=-1.5, p_max=1.5,
                                direction=-1.0)      # toward the folds

    p_fold = 2.0 / (3.0 * np.sqrt(3.0))
    assert len(br.turning_points) >= 2, br.turning_points
    # the reported fold is the nearest ACCEPTED point -> step-limited accuracy
    # (exact fold refinement = the Moore-Spence roadmap item)
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
        return jnp.array([x[0] ** 3 - x[0] + p,
                          x[1] - x[0] ** 2])

    br = arclength_continuation(R, jnp.array([-1.2, 1.44]), p0=0.7,
                                ds=0.05, ds_max=0.1, n_steps=400,
                                p_min=-1.5, p_max=1.5, direction=-1.0)
    assert len(br.turning_points) >= 2
    assert np.max(np.abs(br.x[:, 1] - br.x[:, 0] ** 2)) < 1e-8


def test_refine_fold_cubic():
    """Moore-Spence refinement: from a coarse continuation's turning points
    (step-limited, ~1e-2) to the exact cubic folds at p = -+2/(3 sqrt 3),
    x = -+1/sqrt(3), to 1e-10."""
    R = lambda x, p: jnp.array([x[0] ** 3 - x[0] + p])
    br = arclength_continuation(R, jnp.array([-1.2]), p0=0.7, ds=0.05,
                                ds_max=0.1, n_steps=400, p_min=-1.5,
                                p_max=1.5, direction=-1.0)
    p_fold = 2.0 / (3.0 * np.sqrt(3.0))
    assert len(br.turning_points) >= 2

    i = br.turning_points[0]                        # lower-branch fold: p < 0
    x, p, v, res = refine_fold(R, jnp.array(br.x[i]), float(br.p[i]))
    assert float(res) < 1e-10
    assert abs(p + p_fold) < 1e-10, p
    assert abs(float(x[0]) + 1.0 / np.sqrt(3.0)) < 1e-10
    assert abs(float(3.0 * x[0] ** 2 - 1.0)) < 1e-10  # dR/dx singular
    assert abs(float(jnp.linalg.norm(v)) - 1.0) < 1e-10  # c . v = 1, c = v0/|v0|

    j = br.turning_points[1]                        # middle-branch fold: p > 0
    _, p2, _, res2 = refine_fold(R, jnp.array(br.x[j]), float(br.p[j]))
    assert float(res2) < 1e-10
    assert abs(p2 - p_fold) < 1e-10, p2


def test_track_fold_cubic_in_q():
    """R(x,p,q) = x^3 - q*x + p folds at p = +-2 (q/3)^{3/2}. Track the
    positive fold from q=1 to q=2: every accepted point is a converged
    Moore-Spence solution, so the whole p_fold(q) curve matches the analytic
    law to Newton precision (not step-limited)."""
    R2 = lambda x, p, q: jnp.array([x[0] ** 3 - q * x[0] + p])
    fb = track_fold(R2, jnp.array([0.6]), p0=0.4, q0=1.0, ds=0.1,
                    q_min=0.5, q_max=2.0)

    assert fb.q[0] == 1.0 and fb.q.max() >= 2.0      # reached q = 2
    err_p = np.max(np.abs(fb.p - 2.0 * (fb.q / 3.0) ** 1.5))
    assert err_p < 1e-8, err_p
    err_x = np.max(np.abs(fb.x[:, 0] - np.sqrt(fb.q / 3.0)))
    assert err_x < 1e-8, err_x
    # null vectors stay normalised against the frozen c ( = initial v)
    assert np.max(np.abs(np.abs(fb.v[:, 0]) - 1.0)) < 1e-8
    assert len(fb.branch.turning_points) == 0        # no cusps on this curve


def test_bordered_newton_mass_constraint():
    """Canonical-ensemble structure: minimise the toy free energy
    f(x) = sum((x - a)^2) + eps * sum_{i<j} x_i x_j under the mass constraint
    sum(x) = M, with the Lagrange multiplier (chemical potential) as the
    bordered auxiliary. Analytic: x_i (2 - eps) = 2 a_i - eps*M + lam."""
    a = jnp.array([0.3, -0.1, 0.7, 0.2])
    eps, M = 0.25, 2.0
    n = int(a.shape[0])

    def f(x):                                        # pair term via (sum^2 - sum of squares)/2
        return jnp.sum((x - a) ** 2) + 0.5 * eps * (jnp.sum(x) ** 2 - jnp.sum(x ** 2))

    grad_f = jax.grad(f)
    residual = lambda x, aux: grad_f(x) - aux[0]     # grad f - lam * 1 = 0
    constraints = lambda x, aux: jnp.sum(x) - M      # sum(x) - M = 0

    x, aux, res = bordered_newton(residual, constraints, jnp.zeros(n), 0.0)
    assert float(res) < 1e-10

    lam = ((2.0 - eps) * M + n * eps * M - 2.0 * float(jnp.sum(a))) / n
    x_exact = (2.0 * a - eps * M + lam) / (2.0 - eps)
    assert abs(float(jnp.sum(x)) - M) < 1e-12
    assert float(jnp.max(jnp.abs(x - x_exact))) < 1e-10
    assert abs(float(aux[0]) - lam) < 1e-10


def test_bordered_newton_moore_spence():
    """The cubic fold re-derived as a bordered problem — state (x, v), aux p,
    Moore-Spence rows as the residual, normalisation as the border — and
    cross-checked against refine_fold from the same seed."""
    R = lambda x, p: jnp.array([x[0] ** 3 - x[0] + p])
    Rx = jax.jacfwd(R, argnums=0)
    c = jnp.array([1.0])

    def residual(xv, aux):
        x, v = xv[:1], xv[1:]
        return jnp.concatenate([R(x, aux[0]), Rx(x, aux[0]) @ v])

    constraints = lambda xv, aux: c @ xv[1:] - 1.0

    xv, aux, res = bordered_newton(residual, constraints,
                                   jnp.array([0.9, 1.0]), 0.5)
    p_fold = 2.0 / (3.0 * np.sqrt(3.0))
    assert float(res) < 1e-10
    assert abs(float(aux[0]) - p_fold) < 1e-10
    assert abs(float(xv[0]) - 1.0 / np.sqrt(3.0)) < 1e-10

    xr, pr, vr, rr = refine_fold(R, jnp.array([0.9]), 0.5, v0=jnp.array([1.0]))
    assert float(rr) < 1e-10
    assert abs(float(aux[0]) - pr) < 1e-12           # same fold, two formulations
    assert abs(float(xv[0] - xr[0])) < 1e-12
    assert abs(float(xv[1] - vr[0])) < 1e-12


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"running {name} ...", flush=True)
            fn()
            print(f"  OK  {name}")
    print("all tests passed")
