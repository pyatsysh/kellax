"""Validation of the v0.4 bifurcation layer — every feature against an
analytic anchor.

Run: .venv/bin/python tests/test_bifurcations.py  (or pytest -q)
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as np
import numpy as onp

from kellax import (arclength_continuation, analyze_branch, branch_eigenvalues,
                    deflated_newton, deflated_search, refine_hopf,
                    fold_sensitivity, branch_off, bifurcation_diagram)


# ----------------------------------------------------------------------
# ANALYSIS: stability + classification along a branch
# ----------------------------------------------------------------------

def test_analysis_cubic_stability():
    """S-curve x^3 - x + p with x_dot = -R: outer arms stable, middle arm
    unstable; the only special points are the two folds — no branch points,
    no Hopf candidates."""
    R = lambda x, p: np.array([x[0] ** 3 - x[0] + p])
    br = arclength_continuation(R, np.array([-1.2]), p0 = 0.7, ds = 0.05,
                                ds_max = 0.1, n_steps = 400, p_min = -1.5,
                                p_max = 1.5, direction = -1.0)
    an = analyze_branch(R, br, sign = -1.0)
    assert an.eigenvalues.shape == (len(br.p), 1)
    assert an.branch_points == [] and an.hopf_points == []
    assert an.folds == list(br.turning_points)
    # stability flips exactly at the folds: stable -> unstable -> stable
    i, j = br.turning_points[:2]
    assert an.stable[0] and an.stable[-1]
    assert not an.stable[(i + j) // 2]                   # middle arm
    # n_unstable is 0/1/0 along the three arms
    assert set(onp.unique(an.n_unstable)) == {0, 1}


def test_analysis_detects_branch_point():
    """Pitchfork p*x - x^3 along the trivial branch x = 0: the eigenvalue is
    exactly p, so a branch point (not a fold) is detected at p ~ 0."""
    R = lambda x, p: np.array([p * x[0] - x[0] ** 3])
    br = arclength_continuation(R, np.array([0.0]), p0 = -1.0, ds = 0.05,
                                ds_max = 0.1, n_steps = 100, p_min = -1.2,
                                p_max = 1.0, direction = 1.0)
    an = analyze_branch(R, br, sign = 1.0)
    assert len(br.turning_points) == 0                   # the branch never folds
    assert len(an.branch_points) == 1
    _, p_est = an.branch_points[0]
    assert abs(p_est) < 5e-2, p_est                      # interpolated crossing


# ----------------------------------------------------------------------
# DEFLATION: all three roots of the cubic from one guess
# ----------------------------------------------------------------------

def test_deflated_search_cubic_roots():
    """x^3 - x = 0 has roots -1, 0, +1. One guess + deflation finds all
    three, and a fourth attempt fails cleanly (there is nothing left)."""
    R = lambda x, p: np.array([x[0] ** 3 - x[0] + p])
    sols = deflated_search(R, np.array([0.2]), p = 0.0, max_solutions = 5)
    roots = sorted(float(s[0]) for s in sols)
    assert len(roots) == 3, roots
    assert max(abs(r - e) for r, e in zip(roots, [-1.0, 0.0, 1.0])) < 1e-9


def test_deflated_newton_escapes_known_root():
    """Deflating against the root the plain guess would find forces Newton
    to a DIFFERENT root of the same residual."""
    R = lambda x, p: np.array([x[0] ** 3 - x[0] + p])
    x, res = deflated_newton(R, np.array([0.2]), 0.0, [np.array([0.0])])
    assert float(res) < 1e-10
    assert abs(abs(float(x[0])) - 1.0) < 1e-9            # landed on -1 or +1


# ----------------------------------------------------------------------
# HOPF: normal form — crossing at exactly p = 0, omega = 1
# ----------------------------------------------------------------------

def _hopf_normal_form(x, p):
    r2 = x[0] ** 2 + x[1] ** 2
    return np.array([p * x[0] - x[1] - x[0] * r2,
                     x[0] + p * x[1] - x[1] * r2])


def test_hopf_detection_and_refinement():
    """Along the trivial branch of the Hopf normal form the eigenvalues are
    p +- i: analyze_branch reports a Hopf candidate near p = 0 with
    omega ~ 1, and refine_hopf pins it to p = 0, omega = 1 exactly."""
    br = arclength_continuation(_hopf_normal_form, np.zeros(2), p0 = -0.5,
                                ds = 0.05, ds_max = 0.1, n_steps = 100,
                                p_min = -0.6, p_max = 0.6, direction = 1.0)
    an = analyze_branch(_hopf_normal_form, br, sign = 1.0)
    assert len(an.hopf_points) == 1, an.hopf_points
    i, p_est, om_est = an.hopf_points[0]
    assert abs(p_est) < 5e-2 and abs(om_est - 1.0) < 1e-6

    x, p, om, q, res = refine_hopf(_hopf_normal_form,
                                   np.array(br.x[i]), float(br.p[i]))
    assert float(res) < 1e-10
    assert abs(p) < 1e-10, p                             # Hopf at p = 0 exactly
    assert abs(om - 1.0) < 1e-10, om                     # crossing frequency 1
    assert float(np.max(np.abs(x))) < 1e-8               # at the origin
    # the eigenvector satisfies J q = i omega q
    J = onp.array([[p, -1.0], [1.0, p]])
    assert onp.max(onp.abs(J @ q - 1j * om * q)) < 1e-9


def test_hopf_brusselator():
    """The Brusselator x' = a - (b+1)x + x^2 y, y' = bx - x^2 y has its
    equilibrium at (a, b/a) and a Hopf at exactly b = 1 + a^2, with crossing
    frequency omega = a. Trace the equilibrium in b, detect, refine."""
    a = 1.5
    def R(z, b):
        x, y = z[0], z[1]
        return np.array([a - (b + 1.0) * x + x ** 2 * y,
                         b * x - x ** 2 * y])

    b_star = 1.0 + a ** 2                                # = 3.25
    br = arclength_continuation(R, np.array([a, 2.0 / a]), p0 = 2.0,
                                ds = 0.05, ds_max = 0.15, n_steps = 200,
                                p_min = 1.5, p_max = 4.5, direction = 1.0)
    an = analyze_branch(R, br, sign = 1.0)
    assert len(an.hopf_points) == 1, an.hopf_points
    i, p_est, om_est = an.hopf_points[0]
    x, p, om, q, res = refine_hopf(R, np.array(br.x[i]), float(br.p[i]))
    assert float(res) < 1e-10
    assert abs(p - b_star) < 1e-10, p                    # b* = 1 + a^2
    assert abs(om - a) < 1e-10, om                       # omega = a
    assert abs(float(x[0]) - a) < 1e-10                  # equilibrium (a, b*/a)
    assert abs(float(x[1]) - b_star / a) < 1e-10


# ----------------------------------------------------------------------
# SENSITIVITY: dp*/dtheta against the closed-form fold law
# ----------------------------------------------------------------------

def test_fold_sensitivity_cubic_law():
    """R = x^3 - theta*x + p folds at p*(theta) = 2 (theta/3)^{3/2}, so
    dp*/dtheta = sqrt(theta/3). At theta = 1 that is 1/sqrt(3)."""
    R3 = lambda x, p, th: np.array([x[0] ** 3 - th[0] * x[0] + p])
    x, p, v, dp, res = fold_sensitivity(R3, np.array([0.6]), 0.4, np.array([1.0]))
    assert float(res) < 1e-10
    assert abs(p - 2.0 / (3.0 * onp.sqrt(3.0))) < 1e-10, p
    assert dp.shape == (1,)
    assert abs(float(dp[0]) - 1.0 / onp.sqrt(3.0)) < 1e-9, dp
    # cross-check the implicit gradient against a finite difference
    eps = 1e-6
    _, p_hi, _, _, _ = fold_sensitivity(R3, np.array([0.6]), 0.4, np.array([1.0 + eps]))
    fd = (p_hi - p) / eps
    assert abs(float(dp[0]) - fd) < 1e-5, (float(dp[0]), fd)


# ----------------------------------------------------------------------
# BRANCHING: pitchfork — switch onto x = +-sqrt(p) and recurse
# ----------------------------------------------------------------------

def test_branch_off_pitchfork():
    R = lambda x, p: np.array([p * x[0] - x[0] ** 3])
    seeds = branch_off(R, onp.array([0.0]), 0.0, dp = 1e-3, eps = 1e-2)
    assert len(seeds) == 2, [s[0] for s in seeds]        # both pitchfork arms
    for xs, ps in seeds:
        assert abs(abs(float(xs[0])) - onp.sqrt(ps)) < 1e-8   # on x^2 = p


def test_bifurcation_diagram_pitchfork():
    """The full driver on the pitchfork: the trivial branch plus both
    bifurcating arms, each child satisfying x^2 = p everywhere and carrying
    no further branch points."""
    R = lambda x, p: np.array([p * x[0] - x[0] ** 3])
    kw = dict(ds = 0.05, ds_max = 0.1, n_steps = 100, p_min = -1.2,
              p_max = 1.0, direction = 1.0)
    diag = bifurcation_diagram(R, np.array([0.0]), -1.0, max_depth = 1,
                               sign = 1.0, branch_kwargs = kw)
    assert len(diag) == 3, len(diag)                     # root + two arms
    assert diag[0].depth == 0 and diag[0].parent == -1
    signs = set()
    for child in diag[1:]:
        assert child.depth == 1 and child.parent == 0
        xs, ps = child.branch.x[:, 0], child.branch.p
        assert onp.max(onp.abs(xs ** 2 - ps)) < 1e-8     # on the parabola
        assert child.analysis.branch_points == []        # arms are BP-free
        assert bool(child.analysis.stable.all())         # arms are stable
        signs.add(onp.sign(xs[-1]))
    assert signs == {-1.0, 1.0}                          # one arm each side


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"running {name} ...", flush = True)
            fn()
            print(f"  OK  {name}")
    print("all tests passed")
