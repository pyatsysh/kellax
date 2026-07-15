"""Shared plotting style and branch-diagram helpers for the kellax examples.

One import gives every example the same look: the Okabe--Ito colourblind-safe
qualitative palette, thin recessive axes, and — the important part — **stable and
unstable branches drawn as solid vs dashed**, a non-colour cue so the diagrams
stay readable in greyscale and under colour-vision deficiency. Folds are marked
with ringed dots. Stability is computed honestly from the eigenvalues of the
state Jacobian dR/dx along the branch.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

# Okabe & Ito (2008) colourblind-safe qualitative palette.
OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7",
}
STABLE_C = OKABE_ITO["blue"]        # stable branch (solid)
UNSTABLE_C = OKABE_ITO["vermillion"]  # unstable branch (dashed)
FOLD_C = OKABE_ITO["black"]         # fold markers
ACCENT = OKABE_ITO["orange"]        # highlights, profiles
FILL_C = OKABE_ITO["skyblue"]       # shaded regions

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))


def apply_style():
    """Clean, publication-ish rcParams (light background, recessive chrome)."""
    mpl.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e8e8e8", "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.edgecolor": "#5a5a5a",
        "xtick.color": "#5a5a5a", "ytick.color": "#5a5a5a",
        "axes.labelcolor": "#1a1a1a", "text.color": "#1a1a1a",
        "legend.frameon": False, "lines.linewidth": 2.0,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def savefig(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path)
    print(f"  wrote {os.path.relpath(path)}")
    return path


def branch_stability(residual, xs, ps, sign=-1.0):
    """Boolean mask: is each (x, p) a stable equilibrium of  x_dot = sign * R ?

    Stable iff every eigenvalue of ``sign * dR/dx`` has negative real part.
    ``sign=-1`` suits residuals written as a rate you drive to zero (gradient /
    S-curve convention, outer branches stable); ``sign=+1`` suits a residual that
    *is* the vector field (e.g. a parabolic PDE R = u_xx + f(u), lower branch
    stable). Dense eigensolve — for the small/medium N of these examples.
    """
    Rx = jax.jacfwd(residual, argnums=0)
    mask = []
    for x, p in zip(xs, ps):
        J = sign * np.atleast_2d(np.asarray(Rx(jnp.asarray(x, dtype=float), float(p))))
        mask.append(bool(np.max(np.linalg.eigvals(J).real) < 0.0))
    return np.asarray(mask)


def _runs(mask):
    """Yield (start, stop) slices of maximal constant runs in a boolean array."""
    n = len(mask)
    start = 0
    for i in range(1, n + 1):
        if i == n or mask[i] != mask[start]:
            yield start, i
            start = i


def plot_branch(ax, xvals, yvals, stable=None, lw=2.2, zorder=2,
                c_stable=STABLE_C, c_unstable=UNSTABLE_C):
    """Plot ``y`` vs ``x`` along a branch: solid where stable, dashed where not.

    ``stable`` is a boolean array (same length) or None (all solid, one colour).
    Runs are bridged by one point so solid/dashed segments meet without a gap.
    """
    xvals = np.asarray(xvals, float)
    yvals = np.asarray(yvals, float)
    if stable is None:
        ax.plot(xvals, yvals, "-", color=c_stable, lw=lw, zorder=zorder)
        return
    stable = np.asarray(stable, bool)
    for a, b in _runs(stable):
        b2 = min(b + 1, len(xvals))     # bridge into the next run
        s = bool(stable[a])
        ax.plot(xvals[a:b2], yvals[a:b2], "-" if s else "--",
                color=c_stable if s else c_unstable, lw=lw, zorder=zorder)


def mark_folds(ax, xf, yf, ms=8.5, zorder=6):
    """Ringed dots at fold locations (white fill, dark ring — reads in greyscale)."""
    ax.plot(np.atleast_1d(xf), np.atleast_1d(yf), "o", color=FOLD_C, ms=ms,
            mfc="white", mew=1.9, zorder=zorder, label="_nolegend_")


def stability_legend(ax, loc="best", extra=None):
    """A stable(solid)/unstable(dashed)/fold legend built from proxy artists."""
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], color=STABLE_C, lw=2.2, ls="-", label="stable"),
         Line2D([0], [0], color=UNSTABLE_C, lw=2.2, ls="--", label="unstable"),
         Line2D([0], [0], color=FOLD_C, marker="o", mfc="white", mew=1.9,
                ls="none", ms=8.5, label="fold")]
    if extra:
        h.extend(extra)
    ax.legend(handles=h, loc=loc)
