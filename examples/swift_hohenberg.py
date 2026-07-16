r"""Example 5 — homoclinic snaking in the Swift--Hohenberg equation.

The pattern-formation showcase, and the case BifurcationKit runs matrix-free on a
GPU at ~10^6 dof. The Swift--Hohenberg equation (quadratic-cubic, "SH23")

    u_t = -(1 + d_xx^2)^2 u + r u + nu u^2 - u^3

has, for nu large enough, a subcritical Turing bifurcation off u = 0 at r = 0.
Below it, spatially **localized** patterns (a patch of rolls in a flat background)
live on a branch that **snakes**: it wiggles back and forth across a narrow
"pinning" interval of r, and at every fold the localized state grows by one pair
of rolls. Dozens of folds on one connected branch — exactly what pseudo-arclength
continuation is for, and no branch-switching is needed once you are on it.

kellax has no automatic branch-switching, so we reach the localized branch the
standard way: **seed Newton with a localized envelope** sech(x) cos(x) near the
Maxwell point and continue. We use a Fourier (spectral) discretisation, so the
stiff operator (1 + d_xx^2)^2 is diagonal in k-space — which is also the natural
preconditioner for the matrix-free solve.

Run:  python examples/swift_hohenberg.py
"""
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt

from kellax import arclength_continuation, mf_arclength_continuation, newton
from _style import (apply_style, savefig, STABLE_C, UNSTABLE_C, ACCENT, FILL_C,
                    mark_folds)

N = 256
L = 20.0 * onp.pi                    # ~10 wavelengths (Turing wavenumber k = 1)
X = onp.arange(N) * (L / N)
K = np.asarray(2.0 * onp.pi * onp.fft.fftfreq(N, d = L / N))
SYM = (1.0 - K ** 2) ** 2            # symbol of (1 + d_xx^2)^2
NU = 1.6
R0 = -0.2                            # near the Maxwell point


def R(u, r):
    lin = np.real(np.fft.ifft(-SYM * np.fft.fft(u)))    # spectral, matrix-free
    return lin + r * u + NU * u ** 2 - u ** 3


def precond(v):                     # M^{-1} ~ inverse of the linear operator
    return np.real(np.fft.ifft(np.fft.fft(v) / (-SYM + R0)))


def main():
    apply_style()

    # localized seed: an envelope of rolls in a flat background
    xc = L / 2
    u0 = np.asarray(1.5 / onp.cosh(0.6 * (X - xc)) * onp.cos(X - xc))
    u, res = newton(R, u0, R0, tol = 1e-9, max_iter = 100)
    print(f"seed converged to a localized state: res={float(res):.1e}, "
          f"max|u|={float(np.max(np.abs(u))):.2f}")

    # walk the snake (dense here — the 1-D field is small; the identical spectral
    # setup is what the matrix-free engine takes to 2-D/3-D at 10^4-10^6 dof)
    t = time.time()
    br = arclength_continuation(R, u, p0 = R0, ds = 0.02, ds_max = 0.06,
                                ds_min = 1e-4, n_steps = 700, p_min = -0.30,
                                p_max = -0.10, direction = 1.0)
    L2 = onp.linalg.norm(br.x, axis = 1) * onp.sqrt(L / N)
    fr = br.p[br.turning_points]
    print(f"snake: {len(br.turning_points)} folds, pinning region "
          f"r in [{fr.min():.3f}, {fr.max():.3f}], ||u||_2 {L2.min():.2f} -> {L2.max():.2f}"
          f"  [{time.time() - t:.0f}s, dense]")

    # confirm the matrix-free engine snakes on the same problem (spectral precond)
    t = time.time()
    brm = mf_arclength_continuation(R, u, p0 = R0, ds = 0.02, ds_max = 0.06,
                                    ds_min = 1e-4, n_steps = 250, p_min = -0.30,
                                    p_max = -0.10, direction = 1.0,
                                    precond = precond, gmres_restart = 40,
                                    gmres_maxiter = 40)
    print(f"matrix-free (GMRES on JVPs, Fourier precond): {len(brm.turning_points)} folds "
          f"in {time.time() - t:.0f}s")

    fig, axes = plt.subplots(1, 2, figsize = (10.6, 4.8),
                             gridspec_kw = {"width_ratios": [1.15, 1]})

    # -- left: the snaking bifurcation diagram ---------------------------
    ax = axes[0]
    ax.axvspan(fr.min(), fr.max(), color=FILL_C, alpha=0.12, lw=0, zorder=0)
    ax.plot(br.p, L2, "-", color=STABLE_C, lw=1.7, zorder=2)
    ii = onp.array(br.turning_points)
    mark_folds(ax, br.p[ii], L2[ii], ms = 5.0)
    ax.set_xlabel(r"parameter $r$")
    ax.set_ylabel(r"$\|u\|_2$")
    ax.set_title(rf"Swift--Hohenberg snaking ({len(br.turning_points)} folds)")
    ax.text(fr.min() - 0.002, L2.max(), "pinning\nregion", ha="right", va="top",
            fontsize=9, color="#5a7", zorder=5)

    # -- right: localized states climbing the snake ----------------------
    ax = axes[1]
    picks = [onp.argmin(onp.abs(L2 - v)) for v in
             [L2.min() + 0.5, onp.median(L2), L2.max() - 0.2]]
    for j, (idx, col) in enumerate(zip(picks, [STABLE_C, ACCENT, UNSTABLE_C])):
        ax.plot(X, onp.asarray(br.x[idx]) + j * 3.2, "-", color=col, lw=1.5)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x)$  (offset)")
    ax.set_title("Localized states up the snake")
    ax.set_yticks([])

    fig.tight_layout()
    savefig(fig, "swift_hohenberg.png")


if __name__ == "__main__":
    main()
