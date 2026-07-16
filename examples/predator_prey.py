r"""Example 7 — a predator--prey fold pair.

A 2-D ecological model (MatCont's "EcoMod" tutorial): a Rosenzweig--MacArthur
predator--prey system with a predator that also suffers density-dependent loss,

    x' = r x (1 - x) - x y / (x + a)
    y' = -c y + x y / (x + a) - d y^2 / (y^2 + b^2)

with r = 2, a = 0.6, b = c = 0.25 and the predator-mortality coefficient d as the
control. Over a window of d the system has **three** coexistence equilibria; the
outer two are born/destroyed at a pair of folds. Tracing the coexistence branch
locates them.

Validation (MatCont): folds at d = 0.256805 (x = 0.619532, y = 0.927986) and
d = 0.176927 (x = 0.911266, y = 0.268200).

The branch is an S in d, so we seed one coexistence state *inside* the window and
continue **both** directions to sweep the whole curve. (A Hopf bifurcation also
sits on this branch — that is out of kellax's current scope, so it appears only as
a change of stability, not a marked point.)

Run:  python examples/predator_prey.py
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt

from kellax import arclength_continuation, refine_fold, newton
from _style import (apply_style, savefig, branch_stability, plot_branch,
                    mark_folds, stability_legend, ACCENT, FILL_C)

R_, A, B, C = 2.0, 0.6, 0.25, 0.25


def R(z, d):
    x, y = z[0], z[1]
    return np.array([R_ * x * (1 - x) - x * y / (x + A),
                     -C * y + x * y / (x + A) - d * y ** 2 / (y ** 2 + B ** 2)])


def coexistence_branch(d0 = 0.22):
    """Seed a coexistence equilibrium inside the window and trace both ways."""
    z0, res = newton(R, np.array([0.79, 0.58]), d0, tol = 1e-11)
    assert float(res) < 1e-9, res
    segs = []
    for direction in (-1.0, +1.0):
        segs.append(arclength_continuation(R, z0, p0 = d0, ds = 0.008,
                                           ds_max = 0.02, n_steps = 1500,
                                           p_min = 0.12, p_max = 0.30,
                                           direction = direction))
    d = onp.concatenate([segs[0].p[::-1], segs[1].p])
    Z = onp.concatenate([segs[0].x[::-1], segs[1].x])
    return d, Z, segs


def main():
    apply_style()
    d, Z, segs = coexistence_branch()

    folds = {}
    for br in segs:
        for i in br.turning_points:
            zf, df, _, res = refine_fold(R, np.array(br.x[i]), float(br.p[i]))
            if float(res) < 1e-8:
                folds[round(float(df), 5)] = (float(zf[0]), float(zf[1]))
    print(f"coexistence branch: {len(d)} pts, {len(folds)} folds")
    for df in sorted(folds):
        print(f"  fold: d={df:.6f}, x={folds[df][0]:.5f}, y={folds[df][1]:.5f}")
    print("  ref: d=0.256805 (0.619532, 0.927986); d=0.176927 (0.911266, 0.268200)")

    stable = branch_stability(R, list(Z), list(d), sign = +1.0)     # z_dot = R
    d_lo, d_hi = min(folds), max(folds)

    fig, axes = plt.subplots(1, 2, figsize = (10.6, 4.6))

    # -- left: predator y vs d (the fold pair) ---------------------------
    ax = axes[0]
    ax.axvspan(d_lo, d_hi, color=FILL_C, alpha=0.12, lw=0, zorder=0)
    plot_branch(ax, d, Z[:, 1], stable = stable)
    mark_folds(ax, [df for df in folds], [folds[df][1] for df in folds])
    ax.set_xlabel(r"predator mortality $d$")
    ax.set_ylabel(r"predator $y$")
    ax.set_title(r"Predator--prey fold pair")
    ax.text((d_lo + d_hi) / 2, 0.4, "3 coexistence\nstates", ha="center",
            fontsize=9, color="#456")
    stability_legend(ax, loc="upper left")

    # -- right: the coexistence locus in the phase plane -----------------
    ax = axes[1]
    plot_branch(ax, Z[:, 0], Z[:, 1], stable = stable)
    mark_folds(ax, [folds[df][0] for df in folds], [folds[df][1] for df in folds])
    ax.set_xlabel(r"prey $x$")
    ax.set_ylabel(r"predator $y$")
    ax.set_title("Coexistence equilibria in phase space")

    fig.tight_layout()
    savefig(fig, "predator_prey.png")


if __name__ == "__main__":
    main()
