r"""Example 10 — the Lorenz equilibria: a pitchfork, and the Hopf that ends it.

The Lorenz system, taken not as a chaotic attractor but as a one-parameter
family of *equilibria*,

    x' = sigma (y - x)
    y' = x (r - z) - y
    z' = x y - b z

with sigma = 10, b = 8/3 and the Rayleigh number r as the control. The whole
bifurcation layer of kellax appears on this one problem:

  * the origin is an equilibrium for every r, and loses stability at a
    **pitchfork** at r = 1, where the convection pair C(+/-) is born;
  * the convection branch loses stability at a subcritical **Hopf** at
    r_H = sigma (sigma + b + 3) / (sigma - b - 1) = 470/19.

At C(+/-) the characteristic polynomial is lam^3 + a2 lam^2 + a1 lam + a0 with
a1 = b (sigma + r). A cubic has a pure-imaginary pair exactly when a2 a1 = a0,
and then omega^2 = a1, so the Hopf frequency is omega_H = sqrt(b (sigma + r_H)).

Validation (classical, e.g. Strogatz section 9.2): the branch point at r = 1
exactly, C(+/-) = (+/-sqrt(b(r-1)), +/-sqrt(b(r-1)), r-1), the Hopf at
r_H = 24.7368421... and omega_H = 9.6245301...

Run:  python examples/lorenz.py
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt

from kellax import (arclength_continuation, analyze_branch, branch_off,
                    refine_hopf, deflated_search)
from _style import (apply_style, savefig, plot_branch, stability_legend,
                    ACCENT, FOLD_C)

SIGMA, B = 10.0, 8.0 / 3.0

R_HOPF = SIGMA * (SIGMA + B + 3.0) / (SIGMA - B - 1.0)          # 470/19
OMEGA_HOPF = onp.sqrt(B * (SIGMA + R_HOPF))       # a1 at the Hopf


def R(u, r):
    x, y, z = u[0], u[1], u[2]
    return np.array([SIGMA * (y - x),
                     x * (r - z) - y,
                     x * y - B * z])


def main():
    apply_style()

    # -- 1. the trivial branch, and the pitchfork on it -------------------
    br0 = arclength_continuation(R, np.zeros(3), 0.2, ds = 0.02, ds_max = 0.12,
                                 n_steps = 900, p_max = 32.0, direction = 1.0)
    an0 = analyze_branch(R, br0, sign = 1.0)
    print(f"trivial branch: {len(br0.p)} points, r in "
          f"[{br0.p.min():.3f}, {br0.p.max():.3f}]")
    print(f"  branch points detected: {len(an0.branch_points)}"
          f"  -> r = {[round(float(p), 6) for _, p in an0.branch_points]}"
          f"   (exact 1.0)")

    i_bp, r_bp = an0.branch_points[0]

    # -- 2. switch onto the convection pair -------------------------------
    seeds = branch_off(R, np.asarray(br0.x[i_bp]), float(r_bp))
    print(f"branch_off returned {len(seeds)} seed(s) at the pitchfork")

    branches = []
    for x_s, p_s in seeds:
        br = arclength_continuation(R, np.asarray(x_s), float(p_s), ds = 0.15,
                                    ds_max = 0.9, n_steps = 900, p_max = 32.0,
                                    direction = 1.0)
        branches.append(br)

    # the convection states are known in closed form; check the traced one
    br_c = branches[0]
    r_end = float(br_c.p[-1])
    x_exact = onp.sqrt(B * (r_end - 1.0))
    err = abs(abs(float(br_c.x[-1][0])) - x_exact)
    print(f"convection branch: traced to r = {r_end:.3f}; |x| = "
          f"{abs(float(br_c.x[-1][0])):.6f} vs exact {x_exact:.6f}  (err {err:.1e})")

    # -- 3. the Hopf that ends the convection branch ----------------------
    an_c = analyze_branch(R, br_c, sign = 1.0)
    print(f"  Hopf candidates on it: {len(an_c.hopf_points)}"
          f"  -> r ~ {[round(float(p), 4) for _, p, _ in an_c.hopf_points]}")

    i_h, r_h_est, om_est = an_c.hopf_points[0]
    x_h, r_h, omega, _, res_h = refine_hopf(R, np.asarray(br_c.x[i_h]),
                                            float(r_h_est), omega0 = float(om_est))
    print(f"refine_hopf: r_H = {float(r_h):.9f}  (exact {R_HOPF:.9f}, "
          f"err {abs(float(r_h) - R_HOPF):.2e})")
    print(f"             omega = {float(omega):.9f}  (exact {OMEGA_HOPF:.9f}, "
          f"err {abs(float(omega) - OMEGA_HOPF):.2e})")
    print(f"             residual {float(res_h):.2e}")

    # -- 4. all three equilibria at one r, without knowing they exist -----
    r_probe = 28.0
    found = deflated_search(R, np.array([1.0, 1.0, 1.0]), r_probe,
                            max_solutions = 6, seed_scale = 10.0)
    xc = onp.sqrt(B * (r_probe - 1.0))
    print(f"deflated_search at r = {r_probe}: {len(found)} distinct equilibria")
    for s in sorted(found, key = lambda v: float(v[0])):
        print(f"    x = {float(s[0]):+9.6f}  y = {float(s[1]):+9.6f}  "
              f"z = {float(s[2]):+9.6f}")
    print(f"    exact: x = 0 and x = +/-{xc:.6f}, z = {r_probe - 1.0:.1f}")

    # -- figure -----------------------------------------------------------
    fig, ax = plt.subplots(figsize = (6.8, 4.8))
    plot_branch(ax, br0.p, onp.asarray(br0.x)[:, 0], stable = an0.stable)
    for br, an in zip(branches, [analyze_branch(R, b, sign = 1.0) for b in branches]):
        plot_branch(ax, br.p, onp.asarray(br.x)[:, 0], stable = an.stable)

    ax.plot([1.0], [0.0], "s", color = FOLD_C, ms = 7.5, zorder = 6,
            label = "branch point")
    ax.plot([float(r_h), float(r_h)], [float(x_h[0]), -float(x_h[0])], "^",
            color = ACCENT, ms = 8.5, zorder = 6, label = "Hopf")
    ax.set_xlabel(r"Rayleigh number $r$")
    ax.set_ylabel(r"$x$")
    ax.set_title("Lorenz equilibria: pitchfork at $r=1$, Hopf at $r=470/19$")
    ax.set_xlim(0, 32)
    stability_legend(ax, loc = "upper left", folds = False,
                     extra = [plt.Line2D([0], [0], marker = "s", color = FOLD_C,
                                         ls = "none", ms = 6.5,
                                         label = "branch point"),
                              plt.Line2D([0], [0], marker = "^", color = ACCENT,
                                         ls = "none", ms = 7.0, label = "Hopf")])
    fig.tight_layout()
    savefig(fig, "lorenz.png")


if __name__ == "__main__":
    main()
