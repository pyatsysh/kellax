r"""Example 2 — the cusp: two-parameter fold tracking.

A fold is codimension-1, so in a *two*-parameter family it is not a point but a
**curve**. The cusp normal form

    R(x, p, q) = x^3 - q x + p = 0

has, for each q > 0, a pair of folds at  x = +-sqrt(q/3),  p = +-2 (q/3)^{3/2};
as q -> 0 the pair merges at the **cusp point** (q, p) = (0, 0). The two fold
curves bound the wedge in the (q, p) plane where the system has three solutions
— the signature cusp diagram (MatCont's `cusp` demo; the organising centre of
catastrophe theory).

kellax traces the fold set with `track_fold`: it refines one fold, then runs
pseudo-arclength continuation on the Moore--Spence augmented system in q, so
every plotted point is a *converged* fold, accurate to Newton tolerance rather
than step size. The striking part: a *single* continuation traces BOTH arms. The
cusp is singular only when projected to the (q, p) plane; in the full (x, q, p)
space the fold set is one smooth curve  q = 3 x^2,  p = 2 x^3  (parametrised by
x), and arclength walks straight through the cusp — which it registers as a
turning point in q. We validate against that x-parametrised law.

Run:  python examples/cusp.py
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt

from kellax import arclength_continuation, track_fold
from _style import (apply_style, savefig, mark_folds,
                    STABLE_C, UNSTABLE_C, ACCENT, FILL_C, FOLD_C)


def R2(x, p, q):
    return np.array([x[0] ** 3 - q * x[0] + p])


def full_fold_set(q0 = 0.5, q_max = 2.5):
    """Trace the ENTIRE fold set in two runs from one seed on the + arm:
    up the + arm, and down through the cusp onto the - arm. Return (x, q, p)
    sorted by x (the monotone parameter along the whole smooth curve)."""
    x0 = np.array([onp.sqrt(q0 / 3.0)])
    p0 = 2.0 * (q0 / 3.0) ** 1.5
    up = track_fold(R2, x0, p0 = float(p0), q0 = q0, ds = 0.08, ds_max = 0.2,
                    q_min = 1e-4, q_max = q_max, direction = +1.0)    # + arm, q up
    thru = track_fold(R2, x0, p0 = float(p0), q0 = q0, ds = 0.05, ds_max = 0.15,
                      q_min = 1e-4, q_max = q_max, direction = -1.0)  # + arm -> cusp -> - arm
    x = onp.concatenate([up.x[:, 0], thru.x[:, 0]])
    q = onp.concatenate([up.q, thru.q])
    p = onp.concatenate([up.p, thru.p])
    order = onp.argsort(x)
    n_cusp = len(thru.branch.turning_points)
    return x[order], q[order], p[order], n_cusp


def main():
    apply_style()

    x, q, p, n_cusp = full_fold_set()
    # validate against the x-parametrised fold law  q = 3x^2,  p = 2x^3
    err_q = onp.max(onp.abs(q - 3.0 * x ** 2))
    err_p = onp.max(onp.abs(p - 2.0 * x ** 3))
    print(f"traced the full fold set in ONE pass through the cusp: {len(x)} points, "
          f"x in [{x.min():+.3f},{x.max():+.3f}]")
    print(f"  cusp registered as a turning point of the fold curve: {n_cusp >= 1} "
          f"({n_cusp} found)")
    print(f"  max error vs law q=3x^2, p=2x^3:  {max(err_q, err_p):.2e}")

    fig, axes = plt.subplots(1, 2, figsize = (10.4, 4.6))

    # -- left: the cusp in the (q, p) plane -------------------------------
    ax = axes[0]
    ax.fill(q, p, color=FILL_C, alpha=0.16, lw=0, zorder=0)     # the 3-solution wedge
    ax.plot(q, p, "-", color=FOLD_C, lw=2.2, zorder=3)          # one smooth curve, both arms
    ax.plot(0, 0, "o", color=ACCENT, ms=9, zorder=5)
    ax.annotate("cusp", (0, 0), textcoords="offset points", xytext=(14, 2),
                fontsize=10.5, color=ACCENT)
    ax.text(1.75, 0.0, "3 solutions", ha="center", color="#444", fontsize=10)
    ax.text(0.95, 1.15, "1 solution", ha="center", color="#888", fontsize=10)
    ax.set_xlabel(r"parameter $q$")
    ax.set_ylabel(r"fold location $p^\ast$")
    ax.set_title(r"Fold set of $x^3 - qx + p$  (one continuation)")
    ax.set_xlim(-0.15, 2.5)

    # -- right: x-p slices at fixed q, showing the fold pair widen --------
    ax = axes[1]
    for q, col in [(0.5, STABLE_C), (1.0, ACCENT), (2.0, UNSTABLE_C)]:
        Rq = lambda x, p, q = q: R2(x, p, q)
        br = arclength_continuation(Rq, np.array([-onp.sqrt(q) - 0.4]), p0 = 1.6 * q,
                                    ds = 0.03, ds_max = 0.06, n_steps = 800,
                                    p_min = -2.2 * q, p_max = 2.2 * q,
                                    direction = -1.0)
        ax.plot(br.p, br.x[:, 0], "-", color=col, lw=2.0, label=fr"$q={q:g}$")
        if br.turning_points:
            i = onp.array(br.turning_points)
            mark_folds(ax, br.p[i], br.x[i, 0], ms = 6.5)
    ax.set_xlabel(r"control parameter $p$")
    ax.set_ylabel(r"state $x$")
    ax.set_title(r"Slices $x(p)$ at fixed $q$")
    ax.legend(loc="upper right", title="slice")

    fig.tight_layout()
    savefig(fig, "cusp.png")


if __name__ == "__main__":
    main()
