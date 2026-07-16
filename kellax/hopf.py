"""Hopf refinement — pin an oscillatory instability to Newton precision.

A Hopf point of x_dot = R(x, p) is a solution where a complex pair of
eigenvalues of J = dR/dx sits exactly on the imaginary axis: J q = i omega q
with omega > 0. Splitting q = qr + i qi into real arithmetic, the augmented
system for the unknowns (x, p, qr, qi, omega) reads

    (1) R(x, p)          = 0        (N rows)
    (2) J qr + omega qi  = 0        (N rows)
    (3) J qi - omega qr  = 0        (N rows)
    (4) Re(c . q) - 1    = 0        (1 row: scale of the eigenvector)
    (5) Im(c . q)        = 0        (1 row: phase of the eigenvector)

a square (3N+2) system, non-singular at a generic Hopf point, so plain
Newton converges quadratically from a nearby seed (Griewank & Reddien; the
standard-form Hopf system, e.g. Govaerts 2000). Every block — including the
derivative of J q with respect to x, a second derivative of R — comes from
jax.jacfwd of the stacked map. Seed (x0, p0) from analysis.analyze_branch's
hopf_points; the eigenpair seed defaults to the eigenvalue of J nearest the
imaginary axis with positive imaginary part.
"""
import jax
import jax.numpy as np
import numpy as onp

# 2Do: continue the Hopf point in a second parameter (arclength on this
# augmented system — folds.track_fold does exactly this for the fold), and
# from there periodic-orbit continuation, the big one


def refine_hopf(residual, x0, p0, omega0 = None, q0 = None,
                tol = 1e-10, max_iter = 25):
    """Sharpen a point near a Hopf bifurcation of ``residual(x, p)`` exactly.

    Newton on the (3N+2) standard Hopf system in (x, p, qr, qi, omega).
    ``omega0``/``q0`` seed the crossing frequency and eigenvector; by default
    they are taken from the eigenpair of dR/dx(x0, p0) with the real part
    closest to zero and Im > 0. Returns (x, p, omega, q, res) with q the
    complex eigenvector and res the max-abs augmented residual.
    """
    x0 = np.asarray(x0, dtype = float)
    N = int(x0.shape[0])
    Rx = jax.jacfwd(residual, argnums = 0)

    # ------------------------------------------------------------------
    # SEED THE EIGENPAIR
    # ------------------------------------------------------------------
    if q0 is None or omega0 is None:
        J0 = onp.asarray(Rx(x0, np.asarray(float(p0), dtype = x0.dtype)))
        w, V = onp.linalg.eig(J0)
        cand = [k for k in range(len(w)) if w[k].imag > 1e-12]
        if not cand:
            raise RuntimeError("no complex eigenpair at the seed — not near a Hopf point")
        k = min(cand, key = lambda k: abs(w[k].real))
        omega0 = float(abs(w[k].imag)) if omega0 is None else float(omega0)
        q0 = V[:, k] if q0 is None else onp.asarray(q0)
    q0 = onp.asarray(q0, dtype = complex)
    c = onp.conj(q0) / float(onp.vdot(q0, q0).real)      # so that c . q0 = 1
    cr = np.asarray(c.real, dtype = x0.dtype)
    ci = np.asarray(c.imag, dtype = x0.dtype)

    # ------------------------------------------------------------------
    # THE STACKED REAL SYSTEM  z = (x, p, qr, qi, omega)
    # ------------------------------------------------------------------
    def G(z):
        x, p = z[:N], z[N]
        qr, qi = z[N + 1:2 * N + 1], z[2 * N + 1:3 * N + 1]
        om = z[3 * N + 1]
        J_qr = Rx(x, p) @ qr
        J_qi = Rx(x, p) @ qi
        return np.concatenate([
            residual(x, p),
            J_qr + om * qi,
            J_qi - om * qr,
            np.reshape(cr @ qr - ci @ qi - 1.0, (1,)),   # Re(c . q) - 1
            np.reshape(cr @ qi + ci @ qr, (1,)),         # Im(c . q)
        ])

    Gf = jax.jit(G)
    Gz = jax.jit(jax.jacfwd(G))

    @jax.jit
    def run(z):
        def cond(s):
            _, it, res = s
            return (it < max_iter) & (res > tol)

        def body(s):
            z, it, _ = s
            g = Gf(z)
            z = z - np.linalg.solve(Gz(z), g)
            return z, it + 1, np.max(np.abs(g))

        z, _, _ = jax.lax.while_loop(cond, body, (z, 0, np.inf))
        return z, np.max(np.abs(Gf(z)))

    z0 = np.concatenate([x0, np.reshape(np.asarray(float(p0), dtype = x0.dtype), (1,)),
                         np.asarray(q0.real, dtype = x0.dtype),
                         np.asarray(q0.imag, dtype = x0.dtype),
                         np.reshape(np.asarray(float(omega0), dtype = x0.dtype), (1,))])
    z, res = run(z0)
    x, p = z[:N], float(z[N])
    q = onp.asarray(z[N + 1:2 * N + 1]) + 1j * onp.asarray(z[2 * N + 1:3 * N + 1])
    return x, p, float(z[3 * N + 1]), q, res
