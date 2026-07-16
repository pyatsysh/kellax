r"""Example 9 — differentiable continuation: the gradient of an ignition threshold.

The CSTR of example 6 ignites at a fold Da*(B). How strongly does that
threshold move when the exothermicity B changes? kellax answers with a
DERIVATIVE, not a difference: the fold is the solution of the Moore--Spence
system G(z; B) = 0, so the implicit function theorem gives

    dz/dB = -G_z^{-1} G_B,    d(Da*)/dB = the Da-row of it,

one linear solve against the Jacobian Newton already used, every block by
autodiff (`fold_sensitivity`). No finite differences, no unrolling.

The cross-validation writes itself: `track_fold` traces the very same fold
curve Da*(B) point by point (example 6), so the implicit gradient must match
the slope of the tracked curve. It does, everywhere we test — the tangent
segments drawn from `fold_sensitivity` lie on the `track_fold` curve.

This is the capability the JAX substrate buys: a bifurcation diagram you can
differentiate, and therefore optimise against — pick a target ignition
threshold and let a gradient method move the model to it.

Run:  python examples/cstr_sensitivity.py
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt

from kellax import track_fold, fold_sensitivity
from _style import apply_style, savefig, ACCENT, FOLD_C, STABLE_C

# the CSTR steady state, with the exothermicity as a differentiable parameter
def G3(x, Da, th):
    B = th[0]
    return np.array([-x[0] + Da * (1.0 - x[0]) * np.exp(B * x[0])])


def main():
    apply_style()

    # ------------------------------------------------------------------
    # THE FOLD CURVE, TRACED (example 6's route: ignition fold in B)
    # ------------------------------------------------------------------
    G2 = lambda x, Da, B: G3(x, Da, np.array([B]))
    thru = track_fold(G2, np.array([0.21132]), p0 = 0.075403, q0 = 6.0,
                      ds = 0.1, ds_max = 0.3, q_min = 4.2, q_max = 12.0,
                      direction = -1.0)
    up = track_fold(G2, np.array([0.21132]), p0 = 0.075403, q0 = 6.0,
                    ds = 0.1, ds_max = 0.5, q_min = 4.2, q_max = 12.0,
                    direction = +1.0)
    xc = onp.concatenate([thru.x[:, 0], up.x[:, 0]])
    Bc = onp.concatenate([thru.q, up.q])
    Dac = onp.concatenate([thru.p, up.p])
    order = onp.argsort(xc)
    Bc, Dac, xc = Bc[order], Dac[order], xc[order]
    # keep the ignition arm (low conversion), where the seeds below live
    arm = xc < 0.45
    B_arm, Da_arm, x_arm = Bc[arm], Dac[arm], xc[arm]

    # ------------------------------------------------------------------
    # THE GRADIENT AT SELECTED B — implicit, then checked against the curve
    # ------------------------------------------------------------------
    print("ignition fold and its exact B-gradient (implicit function theorem):")
    print(f"  {'B':>5} {'Da*':>10} {'d(Da*)/dB':>12} {'curve slope':>12} {'diff':>9}")
    picks = []
    for B0 in (5.0, 6.0, 8.0, 10.0):
        j = int(onp.argmin(onp.abs(B_arm - B0)))         # seed from the curve
        x, Da, v, dDa, res = fold_sensitivity(G3, np.array([x_arm[j]]),
                                              float(Da_arm[j]), np.array([B0]))
        # slope of the TRACKED curve at B0, by central difference on the arm
        k = int(onp.argmin(onp.abs(B_arm - B0)))
        slope = ((Da_arm[k + 1] - Da_arm[k - 1]) / (B_arm[k + 1] - B_arm[k - 1])
                 if 0 < k < len(B_arm) - 1 else onp.nan)
        print(f"  {B0:5.1f} {Da:10.6f} {float(dDa[0]):+12.3e} {slope:+12.3e}"
              f" {abs(float(dDa[0]) - slope):9.1e}")
        picks.append((B0, Da, float(dDa[0])))

    # ------------------------------------------------------------------
    # FIGURE: the tracked curve, with implicit-gradient tangents on it
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize = (6.8, 4.8))
    ax.plot(B_arm, Da_arm, "-", color=FOLD_C, lw=2.2, zorder=3,
            label="fold curve (track_fold)")
    w = 0.9
    for B0, Da, g in picks:
        ax.plot([B0 - w, B0 + w], [Da - w * g, Da + w * g], "-",
                color=ACCENT, lw=2.6, zorder=4)
        ax.plot(B0, Da, "o", color=ACCENT, ms=6.5, mfc="white", mew=1.8, zorder=5)
    ax.plot([], [], "-", color=ACCENT, lw=2.6,
            label=r"tangents from fold_sensitivity")
    ax.set_xlabel(r"exothermicity $B$")
    ax.set_ylabel(r"ignition threshold $Da^\ast$")
    ax.set_title("The fold curve and its autodiff tangents")
    ax.legend(loc="upper right")
    fig.tight_layout()
    savefig(fig, "cstr_sensitivity.png")


if __name__ == "__main__":
    main()
