"""kellax standalone validation — no DFT anywhere.

Run: .venv/bin/python tests/test_kellax.py
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from kellax import arclength_continuation, newton  # noqa: E402


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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"running {name} ...", flush=True)
            fn()
            print(f"  OK  {name}")
    print("all tests passed")
