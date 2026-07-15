r"""Example 1 — the fold, from first principles.

The canonical "hello world" of numerical continuation: the cubic normal form of
a saddle-node (fold) bifurcation,

    R(x, p) = x^3 - x + p = 0,

an S-shaped curve with two folds at  p = -+2/(3 sqrt 3) ~ -+0.3849,  x = +-1/sqrt 3.
Naive p-stepping cannot get past a fold; pseudo-arclength continuation walks
straight through both. We then sharpen each detected turning point to Newton
precision with Moore--Spence refinement.

Every serious continuation tool opens with this picture (AUTO's normal-form
demos, MatCont's LP tutorial, BifurcationKit's "temperature" getting-started).

Run:  python examples/cubic_fold.py
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

from kellax import arclength_continuation, refine_fold
from _style import (apply_style, savefig, branch_stability, plot_branch,
                    mark_folds, stability_legend, ACCENT)


def R(x, p):
    return jnp.array([x[0] ** 3 - x[0] + p])


def main():
    apply_style()

    # trace from the lower branch, heading toward (and through) both folds
    br = arclength_continuation(R, jnp.array([-1.2]), p0=0.7, ds=0.03, ds_max=0.06,
                                n_steps=600, p_min=-1.2, p_max=1.2, direction=-1.0)
    xs, ps = br.x[:, 0], br.p
    print(f"traced {len(ps)} points; {len(br.turning_points)} turning points detected")

    # refine each detected fold to Newton precision
    p_star = 2.0 / (3.0 * np.sqrt(3.0))
    folds = []
    for i in br.turning_points[:2]:
        xf, pf, vf, res = refine_fold(R, jnp.array(br.x[i]), float(br.p[i]))
        folds.append((float(pf), float(xf[0])))
        print(f"  refined fold: p = {pf:+.10f}  (exact {np.sign(pf) * p_star:+.10f}), "
              f"x = {float(xf[0]):+.6f}, residual {float(res):.1e}")

    # stability: gradient/S-curve convention  x_dot = -R  (outer branches stable)
    stable = branch_stability(R, list(br.x), list(ps), sign=-1.0)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    plot_branch(ax, ps, xs, stable=stable)
    mark_folds(ax, [f[0] for f in folds], [f[1] for f in folds])
    for pf, xf in folds:
        ax.annotate(rf"$p^\ast={pf:+.3f}$", (pf, xf),
                    textcoords="offset points", xytext=(12, -4 if xf < 0 else 8),
                    fontsize=10, color="#444")
    ax.axvspan(-p_star, p_star, color=ACCENT, alpha=0.07, lw=0, zorder=0)
    ax.text(0.0, -1.35, "3 solutions", ha="center", fontsize=9.5, color=ACCENT)
    ax.set_xlabel(r"control parameter $p$")
    ax.set_ylabel(r"state $x$")
    ax.set_title(r"Fold normal form  $x^3 - x + p = 0$")
    stability_legend(ax, loc="upper right")
    fig.tight_layout()
    savefig(fig, "cubic_fold.png")


if __name__ == "__main__":
    main()
