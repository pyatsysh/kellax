r"""Example 11 — the inner solvers: a phase field, its transition state, and
the Morse index.

Not a continuation problem. This chapter exercises the layer *underneath* the
continuation engines, the one kellax inherited from the classical-DFT toolbox
it was extracted from: a fixed-point globaliser, a matrix-free Newton-Krylov
endgame, autodiff Hessian spectra, and the implicit-function-theorem seam.

The problem is Allen-Cahn on [0, 1] with Neumann ends. Its energy

    F[u] = int_0^1 (eps^2 / 2) u'^2 + (1/4) (u^2 - 1)^2 dx

has two uniform minima u = +1 and u = -1, and between them sits a single
interface, the state a nucleating system must pass through. The critical
points of F solve

    R(u) = eps^2 u'' + u - u^3 = 0.

Three of them are known exactly or asymptotically, which is what makes the
problem worth solving twice:

  * u = +1 and u = -1 are exact critical points, and minima: Morse index 0.
  * u = 0 is an exact critical point, and a maximum in every long-wavelength
    mode: its index counts the modes with eps^2 (k pi)^2 < 1.
  * the single interface is the transition state between the two minima, so
    its index is exactly 1, and as eps -> 0 its energy tends to the classical
    surface tension (2 sqrt 2 / 3) eps = 0.942809... eps.

Run:  python examples/allen_cahn.py
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt

from kellax import (fixed_point_solve, newton_krylov, make_step_bordered,
                    hessian_spectrum, smallest_eigenvalue, morse_index,
                    ift_injection)
from _style import apply_style, savefig, ACCENT, STABLE_C, UNSTABLE_C, FOLD_C

N = 201
EPS = 0.05
H = 1.0 / N
X = (onp.arange(N) + 0.5) * H              # cell centres, Neumann at both ends

SIGMA_ASYMPT = 2.0 * onp.sqrt(2.0) / 3.0   # interface energy / eps as eps -> 0


def laplacian(u):
    """Neumann Laplacian: ghost cells mirror the end values, so a constant
    field has zero curvature and u = +/-1 are exact critical points."""
    up = np.concatenate([u[1:], u[-1:]])
    um = np.concatenate([u[:1], u[:-1]])
    return (um - 2.0 * u + up) / H ** 2


def energy(u, eps = EPS):
    grad = (u[1:] - u[:-1]) / H
    return H * (0.5 * eps ** 2 * np.sum(grad ** 2)
                + 0.25 * np.sum((u ** 2 - 1.0) ** 2))


def R(u, eps = EPS):
    return eps ** 2 * laplacian(u) + u - u ** 3


def main():
    apply_style()

    # -- 0. the residual IS minus the energy gradient ----------------------
    u_probe = np.asarray(onp.tanh((X - 0.5) / (onp.sqrt(2.0) * EPS)))
    mismatch = float(np.max(np.abs(jax.grad(energy)(u_probe) + H * R(u_probe))))
    print(f"grad F  vs  -h R : max mismatch {mismatch:.2e}   "
          "(so Morse indices below classify critical points of F)")

    # -- 1. the ladder: Picard, then Anderson, then Newton-Krylov ----------
    # semi-implicit step  (I - dt eps^2 D2) u_new = u + dt (u - u^3);
    # its fixed point is R(u) = 0. The linear operator is constant, so it is
    # inverted once and applied as a matvec.
    dt = 0.5
    D2 = onp.zeros((N, N))
    for i in range(N):
        D2[i, i] = -2.0
        D2[i, max(i - 1, 0)] += 1.0
        D2[i, min(i + 1, N - 1)] += 1.0
    D2 /= H ** 2
    MINV = np.asarray(onp.linalg.inv(onp.eye(N) - dt * EPS ** 2 * D2))

    def gmap(u):
        return MINV @ (u + dt * (u - u ** 3))

    u0 = np.asarray(onp.tanh((X - 0.5) / (onp.sqrt(2.0) * EPS)) * 0.6)

    # clamp = None matters: the cdft default floors every iterate at 1e-30,
    # which is right for a density and fatal for a signed order parameter.
    u_p, res_p, k_p = fixed_point_solve(gmap, u0, 1e-8, 20000, 0.8,
                                        m = 0, clamp = None)
    u_a, res_a, k_a = fixed_point_solve(gmap, u0, 1e-8, 20000, 0.8,
                                        m = 5, warmup = 10, clamp = None)
    print(f"Picard  (m=0): {int(k_p):5d} iterations, residual {float(res_p):.2e}")
    print(f"Anderson(m=5): {int(k_a):5d} iterations, residual {float(res_a):.2e}"
          f"   ({float(k_p) / max(float(k_a), 1.0):.1f}x fewer)")

    u_i, res_i, k_i, ok = newton_krylov(R, u_a, tol = 1e-12, max_newton = 20)
    print(f"newton_krylov finishes: {k_i} Newton steps, residual "
          f"{res_i:.2e}, converged={bool(ok)}")

    # -- 2. the three critical points, by Morse index ----------------------
    u_plus = np.ones(N)
    u_zero = np.zeros(N)
    n_unstable_exact = int(onp.sum(EPS ** 2 * (onp.pi * onp.arange(N)) ** 2 < 1.0))
    print("critical point        R residual     index   smallest eigenvalue")
    for name, u in (("uniform u=+1", u_plus), ("interface", u_i),
                    ("uniform u=0", u_zero)):
        print(f"  {name:18s} {float(np.max(np.abs(R(u)))):.2e}"
              f"    {morse_index(energy, u, k = 8):3d}"
              f"     {smallest_eigenvalue(energy, u):+.6e}")
    print(f"  (u=0 index counts modes with eps^2 (k pi)^2 < 1: exact "
          f"{n_unstable_exact})")

    # -- 2b. is that index-1 direction real, or is it roundoff? ------------
    # At eps = 0.05 the interface's smallest eigenvalue is ~ -1e-13, and no
    # sign should be trusted at that magnitude. It is the translation mode:
    # a centred interface is pulled to the nearer wall, exponentially weakly
    # in 1/eps. Sweeping eps shows the eigenvalue emerging from the noise
    # with a consistent sign, which is what makes the index-1 claim safe.
    print("   eps    smallest eigenvalue of the interface   index")
    for eps in (0.05, 0.10, 0.15, 0.20, 0.25):
        u_e, _, _, _ = newton_krylov(
            lambda u: R(u, eps),
            np.asarray(onp.tanh((X - 0.5) / (onp.sqrt(2.0) * eps))), tol = 1e-13)
        Fe = lambda u: energy(u, eps)
        print(f"  {eps:.2f}          {smallest_eigenvalue(Fe, u_e):+.3e}"
              f"                {morse_index(Fe, u_e, k = 4)}")

    # -- 3. the interface energy against the surface-tension law -----------
    e_i = float(energy(u_i)) - float(energy(u_plus))
    print(f"interface excess energy {e_i:.6f}  vs  (2 sqrt2/3) eps = "
          f"{SIGMA_ASYMPT * EPS:.6f}   (ratio {e_i / (SIGMA_ASYMPT * EPS):.4f})")

    # -- 4. a constrained critical point: prescribed mean ------------------
    # R(u) + lam = 0 with mean(u) = m is the conserved (Cahn-Hilliard)
    # equilibrium, and lam is the chemical potential holding the mass in
    # place. The border row is what makes this ONE GMRES rather than a Schur
    # complement against a nearly-singular block.
    #
    # Seeding: place the interface where the constraint already wants it,
    # x0 = (1 + m)/2. Seeded instead from the centred interface, every solve
    # with |m| >~ 0.8 stalls near mean -0.80: a bordered Newton step corrects
    # an interface, it does not transport one across the domain.
    print("prescribed mean    lambda (chemical potential)    residual")
    u_c = None
    for m_target in (-0.9, -0.5, 0.0, 0.5, 0.9):
        seed = np.asarray(onp.tanh((X - 0.5 * (1.0 + m_target))
                                   / (onp.sqrt(2.0) * EPS)))
        step = make_step_bordered(lambda u, lam, mm: R(u) + lam,
                                  lambda u, mm: np.mean(u) - mm,
                                  None, dx_max = 1.0, restart = 40,
                                  maxiter = 40, eta_min = 1e-6,
                                  eta_max = 1e-1, ls_max = 25)
        u_m, lam_m, trust = seed, np.asarray(0.0), np.asarray(1.0)
        for _ in range(40):
            u_m, lam_m, trust, res_m, t_m, ok_m = step(u_m, lam_m, trust,
                                                       m_target)
        print(f"     {m_target:+.2f}            {float(lam_m):+.4e}"
              f"              {float(res_m):.1e}")
        if abs(m_target + 0.5) < 1e-12:
            u_c = u_m
    print("  lambda ~ 0 across the coexistence window (a symmetric double")
    print("  well has its Maxwell plateau at zero) and lifts once the")
    print("  interface is close enough to a wall to feel it.")

    # -- 5. d(energy)/d(eps) through the solve, by the IFT -----------------
    def dF_deps(eps):
        u_star = ift_injection(lambda uu, e: R(uu, e), u_i, eps)
        return energy(u_star, eps)

    g_ift = float(jax.grad(dF_deps)(EPS))
    de = 1e-5
    ups, _, _, _ = newton_krylov(lambda u: R(u, EPS + de), u_i, tol = 1e-13)
    ums, _, _, _ = newton_krylov(lambda u: R(u, EPS - de), u_i, tol = 1e-13)
    g_fd = (float(energy(ups, EPS + de)) - float(energy(ums, EPS - de))) / (2 * de)
    print(f"dF/d(eps) by IFT {g_ift:+.8f}   by central difference {g_fd:+.8f}"
          f"   (agree to {abs(g_ift - g_fd):.1e})")

    # -- figure ------------------------------------------------------------
    fig, (ax, axs) = plt.subplots(1, 2, figsize = (9.4, 4.0))
    ax.plot(X, onp.asarray(u_i), "-", color = STABLE_C, lw = 2.4,
            label = "interface (index 1)")
    ax.plot(X, onp.tanh((X - 0.5) / (onp.sqrt(2.0) * EPS)), "--",
            color = FOLD_C, lw = 1.4, label = r"$\tanh$ asymptotic")
    ax.plot(X, onp.asarray(u_c), "-", color = ACCENT, lw = 2.0,
            label = r"constrained, $\langle u\rangle=-0.5$")
    ax.axhline(1.0, color = UNSTABLE_C, ls = ":", lw = 1.4)
    ax.axhline(-1.0, color = UNSTABLE_C, ls = ":", lw = 1.4)
    ax.set_xlabel("$x$"); ax.set_ylabel("$u$")
    ax.set_title(rf"Allen–Cahn critical points, $\varepsilon={EPS}$")
    ax.legend(loc = "lower right", fontsize = 8.5)

    for name, u, c in (("$u=+1$", u_plus, UNSTABLE_C),
                       ("interface", u_i, STABLE_C),
                       ("$u=0$", u_zero, ACCENT)):
        vals = hessian_spectrum(energy, u, k = 8)
        axs.plot(onp.arange(len(vals)), vals, "o-", color = c, ms = 4.5,
                 lw = 1.6, label = name)
    axs.axhline(0.0, color = FOLD_C, lw = 1.0)
    axs.set_xlabel("eigenvalue index"); axs.set_ylabel(r"$\lambda$ of $\nabla^2 F$")
    axs.set_title("eight smallest Hessian eigenvalues")
    axs.legend(loc = "upper left", fontsize = 8.5)
    fig.tight_layout()
    savefig(fig, "allen_cahn.png")


if __name__ == "__main__":
    main()
