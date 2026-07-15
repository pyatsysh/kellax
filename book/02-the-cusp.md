# 2 — The cusp

> Script: [`examples/cusp.py`](../examples/cusp.py) · run it to regenerate the figure.

A fold is **codimension-1**: it needs one condition ($\partial R/\partial x$
singular), so it survives under one control parameter as an isolated point. Add a
*second* parameter and the fold is no longer a point but a **curve** — the locus of
turning points as conditions vary (a spinodal, an ignition boundary, the edge of a
hysteresis region). Take the cusp normal form:

$$R(x, p, q) = x^3 - q x + p = 0.$$

For each $q>0$ there are two folds, at $x=\pm\sqrt{q/3}$, $p=\pm 2(q/3)^{3/2}$;
as $q\to0$ they merge at the **cusp point** $(q,p)=(0,0)$. The two fold curves
bound the wedge where the system has three solutions.

![The cusp](../figures/cusp.png)

## Tracking the fold set

`track_fold` refines one fold, then runs pseudo-arclength continuation *on the
Moore–Spence augmented system* in $q$ — so every point it returns is a converged
fold, accurate to Newton tolerance, not to step size.

```python
R2 = lambda x, p, q: jnp.array([x[0]**3 - q*x[0] + p])
up   = track_fold(R2, jnp.array([+0.408]), p0=+0.136, q0=0.5, q_max=2.5, direction=+1.0)
thru = track_fold(R2, jnp.array([+0.408]), p0=+0.136, q0=0.5, q_max=2.5, direction=-1.0)
```

## The nice part: one continuation, both arms

Tracking *down* in $q$ from the $+$ fold does **not** stop at the cusp — it passes
smoothly through onto the $-$ arm. The cusp is singular only when you project onto
the $(q,p)$ plane; in the full $(x, q, p)$ space the fold set is a single smooth
curve

$$q = 3x^2, \qquad p = 2x^3,$$

parametrised by $x$, and arclength walks straight along it. The cusp is merely a
**turning point in $q$** (where $\mathrm{d}q/\mathrm{d}s = 0$), which kellax detects
and reports:

```
traced the full fold set in ONE pass through the cusp: 45 points, x in [-0.920, +0.938]
  cusp registered as a turning point of the fold curve: True (1 found)
  max error vs law q = 3x^2, p = 2x^3:  4.44e-16
```

Machine precision along the entire curve, cusp included. The left panel plots that
one curve (both arms); the right panel shows $x(p)$ slices at fixed $q$, the fold
pair visibly widening as $q$ grows.

## What to notice

- **Sorting by $x$ untangles the projection.** Because $x$ is monotone along the
  whole fold set, sorting the tracked points by $x$ draws the cusp in a single
  clean stroke through the origin — no special-casing the tip.
- **The second derivative is free.** The augmented residual contains
  $\partial R/\partial x\, v$; continuing it needs derivatives of *that*, supplied by
  `jax.jacfwd` with no hand-coded Hessian.

Background: the vault note *Folds & Moore–Spence*.

Next: [Bratu–Gelfand](03-bratu.md) — the same tools on a discretised PDE.
