r"""Example 3 — Bratu--Gelfand: a fold in a boundary-value problem.

The classic solid-fuel-ignition / thermal-runaway model, and the canonical
continuation BVP (AUTO's demo, MatCont, BifurcationKit all ship it):

    -u''(x) = lambda * exp(u(x)),   x in (0, 1),   u(0) = u(1) = 0.

Two solution branches (a "cool" and a "hot" one) meet at a fold at the critical
ignition parameter  lambda* ~ 3.5138  (below it two steady states exist, above it
none). We discretise with second-order finite differences and continue in lambda;
the Jacobians are autodiff, so the discretisation is the only modelling we do.

Physically u_t = u_xx + lambda e^u, so the lower branch is stable and the upper
branch unstable — the fold is the ignition threshold.

Run:  python examples/bratu.py
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt

from kellax import arclength_continuation, refine_fold
from _style import (apply_style, savefig, branch_stability, plot_branch,
                    mark_folds, stability_legend, ACCENT, STABLE_C, UNSTABLE_C)

N = 200                                   # interior grid points
h = 1.0 / (N + 1)
_off = np.ones(N - 1)
D2 = (np.diag(-2.0 * np.ones(N)) + np.diag(_off, 1) + np.diag(_off, -1)) / h ** 2
XS = np.linspace(0.0, 1.0, N + 2)         # full grid incl. boundaries


def R(u, lam):
    return D2 @ u + lam * np.exp(u)       # = 0  <=>  -u'' = lambda e^u


def main():
    apply_style()
    LAMBDA_STAR = 3.51383071912516        # known 1D Bratu turning point

    # continue from the cool branch (small lambda) up through the fold and back
    br = arclength_continuation(R, np.zeros(N), p0 = 0.3, ds = 0.03, ds_max = 0.15,
                                n_steps = 900, p_min = 0.25, p_max = 6.0,
                                direction = 1.0)
    lam = onp.asarray(br.p)
    norm = onp.max(onp.abs(br.x), axis = 1)   # ||u||_inf
    print(f"traced {len(lam)} points; {len(br.turning_points)} fold(s) detected")

    i = br.turning_points[0]
    _, lam_f, _, res = refine_fold(R, np.array(br.x[i]), float(br.p[i]))
    print(f"  refined fold: lambda* = {lam_f:.6f}  (reference {LAMBDA_STAR:.6f}, "
          f"diff {abs(lam_f - LAMBDA_STAR):.1e} from O(h^2) discretisation), res {float(res):.1e}")

    stable = branch_stability(R, list(br.x), list(lam), sign = +1.0)  # u_t = R

    fig, axes = plt.subplots(1, 2, figsize = (10.6, 4.6))

    # -- left: the bifurcation diagram ------------------------------------
    ax = axes[0]
    plot_branch(ax, lam, norm, stable = stable)
    mark_folds(ax, lam_f, onp.max(onp.abs(br.x[i])))
    ax.annotate(rf"$\lambda^\ast\approx{lam_f:.3f}$", (lam_f, onp.max(onp.abs(br.x[i]))),
                textcoords="offset points", xytext=(-96, 6), fontsize=10.5, color="#444")
    ax.set_xlabel(r"ignition parameter $\lambda$")
    ax.set_ylabel(r"$\|u\|_\infty$")
    ax.set_title(r"Bratu--Gelfand  $-u''=\lambda e^{u}$")
    ax.set_xlim(0, 3.9)
    stability_legend(ax, loc="upper left")

    # -- right: solution profiles on each branch --------------------------
    ax = axes[1]
    picks = [(onp.argmin(onp.abs(lam[:i] - 1.0)), STABLE_C, "cool, $\\lambda=1$"),
             (i, ACCENT, rf"fold, $\lambda^\ast$"),
             (len(lam) - 1 - onp.argmin(onp.abs(lam[::-1] - 1.0)), UNSTABLE_C,
              "hot, $\\lambda=1$")]
    for idx, col, lab in picks:
        u_full = onp.concatenate([[0.0], onp.asarray(br.x[idx]), [0.0]])
        ax.plot(onp.asarray(XS), u_full, "-", color=col, lw=2.2, label=lab)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x)$")
    ax.set_title("Solution profiles")
    ax.legend(loc="upper right")

    fig.tight_layout()
    savefig(fig, "bratu.png")


if __name__ == "__main__":
    main()
