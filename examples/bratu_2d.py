r"""Example 8 — Bratu--Gelfand in 2D: a fold of a genuine PDE field.

The Bratu problem on the unit square (BifurcationKit's `mittelmann` is a close
relative):

    Delta u + lambda e^u = 0   on (0,1)^2,   u = 0 on the boundary.

As in 1D there are two solution branches meeting at a fold, but now u(x, y) is a
2-D thermal field. The turning point of the unit square is lambda* ~ 6.808 (higher
than the 1-D value 3.5138 because the square loses heat through more boundary).

We discretise with the 5-point Laplacian on an N x N grid and continue in lambda;
autodiff gives every Jacobian. This is the dense engine on ~10^3 dof — comfortable
here, and the matrix-free engine of [chapter 4](../book/04-matrix-free.md) is the
route to finer 2-D/3-D grids.

Run:  python examples/bratu_2d.py
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

N = 20                                   # N x N interior grid -> N^2 dof (dense-engine friendly)
h = 1.0 / (N + 1)
REF = 6.808                              # unit-square 2-D Bratu turning point


def R(u, lam):
    """5-point Laplacian (Dirichlet) + lambda e^u, on the flattened N x N field."""
    U = u.reshape(N, N)
    up = jnp.concatenate([U[1:], jnp.zeros((1, N))], 0)
    dn = jnp.concatenate([jnp.zeros((1, N)), U[:-1]], 0)
    lt = jnp.concatenate([U[:, 1:], jnp.zeros((N, 1))], 1)
    rt = jnp.concatenate([jnp.zeros((N, 1)), U[:, :-1]], 1)
    return ((up + dn + lt + rt - 4.0 * U) / h ** 2 + lam * jnp.exp(U)).reshape(-1)


def main():
    apply_style()

    br = arclength_continuation(R, jnp.zeros(N * N), p0=0.3, ds=0.1, ds_max=0.6,
                                n_steps=400, p_min=0.25, p_max=12.0, direction=1.0)
    lam = np.asarray(br.p)
    umax = np.max(np.abs(br.x), axis=1)
    i = br.turning_points[0]
    _, lam_f, _, res = refine_fold(R, jnp.array(br.x[i]), float(br.p[i]))
    print(f"2-D Bratu (N^2 = {N * N} dof): {len(br.turning_points)} fold(s)")
    print(f"  refined fold lambda* = {lam_f:.5f}  (unit-square ref ~{REF}; "
          f"O(h^2) below it at this grid), res {float(res):.1e}")

    stable = branch_stability(R, list(br.x), list(lam), sign=+1.0)   # u_t = R

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6),
                             gridspec_kw={"width_ratios": [1.1, 1]})

    # -- left: the bifurcation diagram -----------------------------------
    ax = axes[0]
    plot_branch(ax, lam, umax, stable=stable)
    mark_folds(ax, lam_f, umax[i])
    ax.annotate(rf"$\lambda^\ast\approx{lam_f:.3f}$", (lam_f, umax[i]),
                textcoords="offset points", xytext=(-92, 4), fontsize=10.5, color="#444")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\|u\|_\infty$")
    ax.set_title(r"2-D Bratu  $\Delta u + \lambda e^{u}=0$")
    ax.set_xlim(0, 7.4)
    stability_legend(ax, loc="upper left")

    # -- right: the solution field at the fold ---------------------------
    ax = axes[1]
    U = np.zeros((N + 2, N + 2))
    U[1:-1, 1:-1] = np.asarray(br.x[i]).reshape(N, N)
    im = ax.imshow(U, origin="lower", extent=[0, 1, 0, 1], cmap="magma", aspect="equal")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(rf"$u(x,y)$ at the fold  ($\|u\|_\infty={umax[i]:.2f}$)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85, label=r"$u$")

    fig.tight_layout()
    savefig(fig, "bratu_2d.png")


if __name__ == "__main__":
    main()
