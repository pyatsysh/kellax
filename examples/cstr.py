r"""Example 6 — CSTR ignition/extinction: hysteresis and a physical cusp.

The exothermic continuous stirred-tank reactor, the textbook example of
multiplicity in chemical engineering. In dimensionless adiabatic form the steady
conversion x solves

    G(x, Da) = -x + Da (1 - x) exp(B x) = 0,

with Da the Damkohler number (residence time) and B the exothermicity. For B > 4
the reactor has three steady states over a window of Da: a cold (low-conversion)
branch, a hot branch, and an unstable one between — the classic **ignition /
extinction hysteresis** loop.

This one problem ties chapters 1 and 2 together on real physics:
  * at B = 6, trace the S-curve and locate the ignition and extinction folds
    (`refine_fold`);
  * then track a fold in the *second* parameter B; the two folds merge at the
    **cusp** at B = 4, x = 1/2, Da = e^-2 -- below B = 4 the hysteresis is gone.

Reference values (Uppal--Ray--Poore): ignition Da = 0.075403, extinction
Da = 0.032873 at B = 6; cusp at B = 4, Da = e^-2 = 0.135335.

Run:  python examples/cstr.py
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

from kellax import arclength_continuation, refine_fold, track_fold
from _style import (apply_style, savefig, branch_stability, plot_branch,
                    mark_folds, stability_legend, ACCENT, FILL_C, FOLD_C)


def G(x, Da, B):
    return jnp.array([-x[0] + Da * (1.0 - x[0]) * jnp.exp(B * x[0])])


def fold_curve(B0=6.0, x0=0.21132, Da0=0.075403):
    """Track the ignition fold in B: down through the cusp and up the other arm.
    Returns (B, Da) sorted by conversion x (monotone along the whole fold set)."""
    thru = track_fold(G, jnp.array([x0]), p0=Da0, q0=B0, ds=0.1, ds_max=0.3,
                      q_min=3.9, q_max=14.0, direction=-1.0)     # -> cusp -> other arm
    up = track_fold(G, jnp.array([x0]), p0=Da0, q0=B0, ds=0.1, ds_max=0.5,
                    q_min=3.9, q_max=14.0, direction=+1.0)       # B up, x -> 0
    x = np.concatenate([thru.x[:, 0], up.x[:, 0]])
    B = np.concatenate([thru.q, up.q])
    Da = np.concatenate([thru.p, up.p])
    order = np.argsort(x)
    return B[order], Da[order], x[order]


def main():
    apply_style()
    B = 6.0
    R = lambda x, Da: G(x, Da, B)

    # -- Part 1: the hysteresis S-curve at B = 6 -------------------------
    br = arclength_continuation(R, jnp.array([0.03]), p0=0.02, ds=0.008, ds_max=0.02,
                                n_steps=900, p_min=0.0, p_max=0.20, direction=1.0)
    folds = []
    for i in br.turning_points[:2]:
        xf, Daf, _, res = refine_fold(R, jnp.array(br.x[i]), float(br.p[i]))
        folds.append((float(Daf), float(xf[0])))
    (Da_ig, _), (Da_ext, _) = sorted(folds, reverse=True)
    print(f"S-curve at B=6: ignition Da={Da_ig:.6f} (ref 0.075403), "
          f"extinction Da={Da_ext:.6f} (ref 0.032873)")

    stable = branch_stability(R, list(br.x), list(np.asarray(br.p)), sign=+1.0)  # x_dot = G

    # -- Part 2: the cusp in the (B, Da) plane ---------------------------
    Bc, Dac, xc = fold_curve()
    i_cusp = int(np.argmin(Bc))
    print(f"cusp: min B={Bc[i_cusp]:.4f} (ref 4), Da={Dac[i_cusp]:.5f} (ref e^-2={np.exp(-2):.5f})")

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))

    ax = axes[0]
    ax.axvspan(Da_ext, Da_ig, color=FILL_C, alpha=0.12, lw=0, zorder=0)
    plot_branch(ax, br.p, br.x[:, 0], stable=stable)
    mark_folds(ax, [f[0] for f in folds], [f[1] for f in folds])
    ax.annotate("ignition", (Da_ig, 0.21), textcoords="offset points", xytext=(6, -2),
                fontsize=9.5, color="#444")
    ax.annotate("extinction", (Da_ext, 0.79), textcoords="offset points", xytext=(-64, 2),
                fontsize=9.5, color="#444")
    ax.set_xlabel("Damköhler number $Da$")
    ax.set_ylabel(r"conversion $x$")
    ax.set_title(r"CSTR hysteresis  ($B=6$)")
    ax.set_xlim(0, 0.16)
    stability_legend(ax, loc="lower right")

    ax = axes[1]
    ax.fill(Bc, Dac, color=FILL_C, alpha=0.14, lw=0, zorder=0)
    ax.plot(Bc, Dac, "-", color=FOLD_C, lw=2.2, zorder=3)
    ax.plot(4.0, np.exp(-2), "o", color=ACCENT, ms=9, zorder=5)
    ax.annotate(r"cusp $(4,\,e^{-2})$", (4.0, np.exp(-2)), textcoords="offset points",
                xytext=(10, 6), fontsize=10, color=ACCENT)
    ax.axvline(6.0, color="#bbb", lw=1, ls=":", zorder=1)
    ax.text(6.05, 0.02, "$B=6$ slice", fontsize=9, color="#888")
    ax.text(9.0, 0.09, "3 steady states", ha="center", color="#444", fontsize=10)
    ax.set_xlabel(r"exothermicity $B$")
    ax.set_ylabel(r"fold $Da^\ast$")
    ax.set_title(r"Fold set: hysteresis is born at the cusp")
    ax.set_xlim(3.5, 13)

    fig.tight_layout()
    savefig(fig, "cstr.png")


if __name__ == "__main__":
    main()
