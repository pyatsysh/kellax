"""Stability and bifurcation detection along a traced branch.

A Branch stores every accepted state, so the spectrum of the state Jacobian
dR/dx can be computed post-hoc, point by point, and the branch read as a
dynamical object: x_dot = sign * R(x, p) is stable where every eigenvalue of
sign * dR/dx has negative real part. Watching the eigenvalues cross the
imaginary axis between consecutive accepted points classifies the special
points of the branch:

    (1) a REAL eigenvalue crosses zero inside an interval that contains a
        detected turning point  -> the fold's own zero crossing;
    (2) a REAL eigenvalue crosses zero with NO turning point nearby -> a
        branch-point candidate (pitchfork/transcritical) — hand it to
        branching.branch_off to jump onto the bifurcating branch;
    (3) a COMPLEX pair crosses with nonzero imaginary part -> a Hopf
        candidate, with the crossing frequency estimated from |Im| — hand it
        to hopf.refine_hopf to pin it to Newton precision.

Dense eigensolves (onp.linalg.eigvals per accepted point): meant for the
dense-engine regime, N up to a few thousand. Crossing locations are
estimated by linear interpolation of the crossing eigenvalue's real part.
"""
from dataclasses import dataclass

import jax
import jax.numpy as np
import numpy as onp

# 2Do: matrix-free stability — a few leading eigenvalues by Arnoldi on the
# jax.linearize JVP (scipy.sparse.linalg.eigs accepts a LinearOperator), so
# the mf engine gets the same classification at 10^4+ dof


@dataclass
class BranchAnalysis:
    """Spectrum and special points of a traced branch."""
    eigenvalues: onp.ndarray   # (M, N) complex — spectrum of sign * dR/dx per point
    n_unstable: onp.ndarray    # (M,) int — eigenvalues with positive real part
    stable: onp.ndarray        # (M,) bool — n_unstable == 0
    folds: list                # indices (as in Branch.turning_points)
    branch_points: list        # (i, p_est) — real crossing, no fold nearby
    hopf_points: list          # (i, p_est, omega_est) — complex pair crossing


def branch_eigenvalues(residual, br, sign = 1.0):
    """Spectrum of ``sign * dR/dx`` at every accepted point. Returns (M, N) complex.

    ``sign = +1`` when the residual IS the vector field (x_dot = R, e.g. a
    parabolic PDE); ``sign = -1`` for a residual you drive to zero along a
    gradient flow (x_dot = -R, the S-curve convention).
    """
    Rx = jax.jit(jax.jacfwd(residual, argnums = 0))
    lams = []
    for x, p in zip(br.x, br.p):
        J = sign * onp.atleast_2d(onp.asarray(Rx(np.asarray(x, dtype = float), float(p))))
        lams.append(onp.linalg.eigvals(J))
    return onp.array(lams)


def analyze_branch(residual, br, sign = 1.0, hopf_im_tol = 1e-6):
    """Classify the special points of a Branch from its spectrum.

    Between consecutive accepted points, eigenvalues crossing the imaginary
    axis are counted and classified: a fold (real crossing at a detected
    turning point), a branch-point candidate (real crossing, no turning
    point), or a Hopf candidate (complex pair, |Im| > hopf_im_tol at the
    crossing). Returns a BranchAnalysis; candidate locations are linear
    interpolations, meant as seeds for refine_fold / branch_off / refine_hopf.
    """
    lam = branch_eigenvalues(residual, br, sign = sign)
    n_uns = onp.sum(lam.real > 0.0, axis = 1)
    turning = set(br.turning_points)

    bps, hopfs = [], []
    M = lam.shape[0]
    for i in range(M - 1):
        d = int(n_uns[i + 1]) - int(n_uns[i])
        if d == 0:
            continue
        # eigenvalues that changed the sign of their real part across the step
        # (matched greedily by proximity — adequate away from collisions)
        a, b = lam[i], lam[i + 1]
        used = onp.zeros(len(b), dtype = bool)
        crossings = []
        for la in a:
            j = int(onp.argmin(onp.where(used, onp.inf, onp.abs(b - la))))
            used[j] = True
            if (la.real > 0.0) != (b[j].real > 0.0):
                # linear interpolation of the zero of Re(lambda) in p
                t = la.real / (la.real - b[j].real)
                p_est = float(br.p[i] + t * (br.p[i + 1] - br.p[i]))
                om = 0.5 * (abs(la.imag) + abs(b[j].imag))
                crossings.append((p_est, om))
        if not crossings:
            continue
        is_fold = (i in turning) or ((i + 1) in turning)
        complex_pairs = [c for c in crossings if c[1] > hopf_im_tol]
        if len(complex_pairs) >= 2:                      # a pair crossed together
            p_est = float(onp.mean([c[0] for c in complex_pairs[:2]]))
            om = float(onp.mean([c[1] for c in complex_pairs[:2]]))
            hopfs.append((i, p_est, om))
        elif not is_fold:
            bps.append((i, float(crossings[0][0])))

    return BranchAnalysis(eigenvalues = lam, n_unstable = n_uns,
                          stable = (n_uns == 0), folds = list(br.turning_points),
                          branch_points = bps, hopf_points = hopfs)
