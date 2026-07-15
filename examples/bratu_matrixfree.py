r"""Example 4 — scaling up: Bratu, matrix-free.

The dense engine forms the Jacobian and factorises it — fine for the N = 200
Bratu of the previous example, ruinous once N reaches the 10^4-10^6 of a 2-D/3-D
field. `mf_arclength_continuation` runs the *same* Keller continuation but never
forms a matrix: each bordered solve is preconditioned GMRES on Jacobian-vector
products (from `jax.linearize`).

Two things to take away:

  * **Matrix-free reproduces dense exactly** — same branch, same fold — so it is a
    drop-in once the Jacobian stops fitting in memory. We show the two overlaid at
    a moderate N where the dense engine is still quick.
  * **A preconditioner is essential.** Unpreconditioned GMRES on the stiff
    Laplacian stalls before the fold; the Dirichlet Laplacian is diagonalised by a
    sine transform, so a spectral (DST) preconditioner makes the solve trivial —
    and then the matrix-free engine sails on to N in the thousands.

Run:  python examples/bratu_matrixfree.py
"""
from __future__ import annotations

import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

from kellax import (arclength_continuation, mf_arclength_continuation,
                    refine_fold)
from _style import (apply_style, savefig, branch_stability, plot_branch,
                    mark_folds, stability_legend, ACCENT)

REF = 3.51383071912516        # 1-D Bratu turning point (continuum limit)


def bratu_problem(N):
    """Return (residual, spectral preconditioner) for -u''=lambda e^u on N points."""
    h = 1.0 / (N + 1)

    def R(u, lam):                                    # no N x N matrix is ever formed
        up = jnp.concatenate([u[1:], jnp.zeros(1)])
        um = jnp.concatenate([jnp.zeros(1), u[:-1]])
        return (um - 2.0 * u + up) / h ** 2 + lam * jnp.exp(u)

    # M^{-1} ~ (D2)^{-1}: the Dirichlet Laplacian is diagonalised by the DST-I,
    # so its inverse is one sine transform, a divide by eigenvalues, and one back.
    lap_eig = jnp.asarray((2.0 * np.cos(np.arange(1, N + 1) * np.pi / (N + 1)) - 2.0) / h ** 2)

    def dst1(v):                                      # DST-I via FFT of the odd extension
        v = jnp.asarray(v)
        ext = jnp.concatenate([jnp.zeros(1), v, jnp.zeros(1), -v[::-1]])
        return -jnp.fft.rfft(ext).imag[1:N + 1]

    def precond(v):
        return dst1(dst1(v) * (2.0 / (N + 1)) / lap_eig)

    return R, precond


def trace(engine, R, refine=True, **kw):
    t = time.time()
    br = engine(R, jnp.zeros(kw.pop("N")), p0=0.3, ds=0.03, ds_max=0.15, n_steps=700,
                p_min=0.25, p_max=6.0, direction=1.0, **kw)
    i = br.turning_points[0]
    if refine:                            # dense Moore-Spence: O(N^3), fine at moderate N
        _, lam_f, _, _ = refine_fold(R, jnp.array(br.x[i]), float(br.p[i]))
    else:                                 # step-limited bracket only (matrix-free fold
        lam_f = float(br.p[i])            # refinement is still a roadmap item)
    return br, float(lam_f), time.time() - t


def main():
    apply_style()

    # -- correctness: dense vs matrix-free at a moderate N ----------------
    Nc = 300
    R, precond = bratu_problem(Nc)
    br_d, lam_d, t_d = trace(arclength_continuation, R, N=Nc)
    br_m, lam_m, t_m = trace(mf_arclength_continuation, R, N=Nc, precond=precond,
                             gmres_restart=80, gmres_maxiter=80)
    print(f"correctness check at N = {Nc}:")
    print(f"  dense       fold lambda* = {lam_d:.6f}  [{t_d:.1f}s]")
    print(f"  matrix-free fold lambda* = {lam_m:.6f}  [{t_m:.1f}s]")
    print(f"  |dense - matrix-free| = {abs(lam_d - lam_m):.2e}  (both ~ ref {REF:.4f})")

    # -- scaling: matrix-free only, large N (dense would need an N^2 matrix)
    Nb = 4000
    Rb, precond_b = bratu_problem(Nb)
    _, lam_b, t_b = trace(mf_arclength_continuation, Rb, N=Nb, refine=False,
                          precond=precond_b, gmres_restart=100, gmres_maxiter=100)
    print(f"scaling: matrix-free at N = {Nb} -> fold bracket lambda ~ {lam_b:.4f}  [{t_b:.1f}s]"
          f"  (dense here needs a {Nb}x{Nb} Jacobian; fold refinement is dense -> roadmap)")

    # -- figure: the two engines agree -----------------------------------
    stable = branch_stability(R, list(br_d.x), list(np.asarray(br_d.p)), sign=+1.0)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    plot_branch(ax, br_d.p, np.max(np.abs(br_d.x), axis=1), stable=stable)
    ax.plot(br_m.p, np.max(np.abs(br_m.x), axis=1), "o", ms=4.5, color=ACCENT,
            mfc="none", mew=1.3, zorder=4)
    mark_folds(ax, lam_d, np.max(np.abs(br_d.x[br_d.turning_points[0]])))
    ax.set_xlabel(r"ignition parameter $\lambda$")
    ax.set_ylabel(r"$\|u\|_\infty$")
    ax.set_title(rf"Bratu at $N={Nc}$: dense (lines) vs matrix-free (markers)")
    ax.set_xlim(0, 3.9)
    ax.text(0.05, 5.4, f"matrix-free also traces $N={Nb}$ in {t_b:.0f}s\n"
            "(no Jacobian ever formed)", fontsize=9.5, color="#555")
    stability_legend(ax, loc="center left",
                     extra=[plt.Line2D([0], [0], marker="o", color=ACCENT, mfc="none",
                                       mew=1.3, ls="none", ms=6.5,
                                       label="matrix-free")])
    fig.tight_layout()
    savefig(fig, "bratu_matrixfree.png")


if __name__ == "__main__":
    main()
