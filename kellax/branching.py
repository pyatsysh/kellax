"""Branch switching and a minimal bifurcation-diagram driver.

At a simple branch point (pitchfork, transcritical) a second solution branch
crosses the one being traced; arclength continuation sails straight past it.
``branch_off`` jumps onto the other branch by the deflation route: (1) sharpen
the primary solution just past the detected crossing, (2) perturb it along
the null vector of dR/dx, (3) run deflated Newton against the primary branch,
which CANNOT converge back to it, so any convergence is the new branch.
Robust, derivative-light, and honest about what it is — the normal-form
branch switching of the Julia toolchain is more surgical; this one simply
works for simple branch points.

``bifurcation_diagram`` composes the pieces into a bounded-depth recursive
driver: trace, classify (analysis.analyze_branch), branch off at every
detected branch point, recurse on the children. Equilibria only; Hopf
candidates are reported, not followed (periodic orbits are the roadmap).
"""
from dataclasses import dataclass

import jax
import jax.numpy as np
import numpy as onp

from .keller import newton, arclength_continuation
from .analysis import analyze_branch
from .deflation import deflated_newton

# 2Do: transcritical points give one seed on each side of the BP in p —
# branch_off currently looks on one side only; probe p_bp - dp as well


def branch_off(residual, x_bp, p_bp, dp = 1e-3, eps = 1e-2,
               tol = 1e-9, max_iter = 50):
    """Seeds on the bifurcating branch near a simple branch point.

    (x_bp, p_bp) is the branch-point estimate (from analyze_branch). Probes
    x_bp +- eps * phi at p = p_bp + dp, with phi the null vector of dR/dx,
    deflating against the primary-branch solution there. Returns a list of
    (x, p) seeds on the OTHER branch — pass each to arclength_continuation.
    """
    x_bp = np.asarray(x_bp, dtype = float)
    Rx = jax.jacfwd(residual, argnums = 0)

    # null direction at the branch point
    _, _, Vt = np.linalg.svd(np.atleast_2d(Rx(x_bp, np.asarray(float(p_bp)))))
    phi = Vt[-1]

    # the primary branch, sharpened just past the crossing — the thing to escape
    p1 = float(p_bp) + dp
    x_prim, r = newton(residual, x_bp, p1, tol = tol, max_iter = max_iter)
    if float(r) > 1e-6:
        return []

    seeds = []
    for s in (+1.0, -1.0):
        xn, rn = deflated_newton(residual, x_prim + s * eps * phi, p1,
                                 [x_prim], tol = tol, max_iter = max_iter)
        if float(rn) < 1e-6 and float(np.linalg.norm(xn - x_prim)) > 10 * tol:
            if all(float(np.linalg.norm(xn - xs)) > 1e-6 for xs, _ in seeds):
                seeds.append((xn, p1))
    return seeds


@dataclass
class DiagramBranch:
    """One traced branch of a bifurcation diagram."""
    branch: object             # the Branch
    analysis: object           # its BranchAnalysis
    depth: int                 # 0 = the branch traced from the user's seed
    parent: int                # index of the parent DiagramBranch (-1 for root)


def bifurcation_diagram(residual, x0, p0, max_depth = 1, sign = 1.0,
                        branch_kwargs = None):
    """Trace a branch, switch at every simple branch point, recurse.

    A minimal automatic-diagram driver for equilibria: each traced branch is
    classified by ``analyze_branch`` (with stability convention ``sign``),
    every detected branch point is jumped with ``branch_off``, and each child
    is traced with the same continuation settings (``branch_kwargs`` is passed
    to ``arclength_continuation``). Recursion stops at ``max_depth``. Returns
    a list of DiagramBranch — index 0 is the root. Hopf candidates appear in
    each branch's analysis but are not followed.
    """
    kw = dict(branch_kwargs or {})
    out = []

    def explore(x_seed, p_seed, depth, parent):
        br = arclength_continuation(residual, np.asarray(x_seed, dtype = float),
                                    float(p_seed), **kw)
        an = analyze_branch(residual, br, sign = sign)
        out.append(DiagramBranch(branch = br, analysis = an,
                                 depth = depth, parent = parent))
        me = len(out) - 1
        if depth >= max_depth:
            return
        for (i, p_est) in an.branch_points:
            for xs, ps in branch_off(residual, onp.asarray(br.x[i]), p_est):
                explore(xs, ps, depth + 1, me)

    explore(x0, p0, 0, -1)
    return out
