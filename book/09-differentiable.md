# 9 — Differentiable continuation: the gradient of a fold

> Script: [`examples/cstr_sensitivity.py`](../examples/cstr_sensitivity.py) · run it to regenerate the figure.

Every chapter so far has *located* structure — folds, cusps, branch points,
Hopf candidates. This one differentiates it. The question is concrete: the
CSTR of [chapter 6](06-cstr.md) ignites at a fold $Da^\ast(B)$; how strongly
does that threshold move when the exothermicity $B$ changes?

## One linear solve, not a difference quotient

The fold is the solution $z^\ast = (x^\ast, Da^\ast, v^\ast)$ of the
Moore–Spence system of [chapter 1](01-the-fold.md), now read as a function of
the model parameter:

$$G(z;\,B) = 0.$$

$G_z$ is non-singular at a generic fold — that is the whole reason
Moore–Spence Newton converges — so the implicit function theorem applies
directly:

$$\frac{dz^\ast}{dB} = -\,G_z^{-1}\,G_B,$$

and $d(Da^\ast)/dB$ is one row of it. One linear solve against the same
Jacobian Newton already used; $G_B$ comes from `jax.jacfwd`. No unrolling of
the Newton iterations, no finite differences. In kellax this is
`fold_sensitivity`, and it returns the fold state, its location, and the
exact gradient together.

## Validation, twice

Against the closed-form law first: $x^3 - \theta x + p$ folds at
$p^\ast(\theta) = 2(\theta/3)^{3/2}$, so $dp^\ast/d\theta = \sqrt{\theta/3}$.
`fold_sensitivity` returns it to $10^{-9}$, and a finite difference of two
*refined* folds agrees to $10^{-5}$ (the test suite carries both checks).

Then on the reactor. `track_fold` traces the ignition curve $Da^\ast(B)$
point by point; `fold_sensitivity` differentiates it point-wise:

```
    B        Da*    d(Da*)/dB  curve slope
  5.0   0.095906   -2.651e-02   -2.533e-02
  6.0   0.075403   -1.593e-02   -1.559e-02
  8.0   0.053167   -7.786e-03   -8.408e-03
 10.0   0.041153   -4.638e-03   -4.923e-03
```

The residual disagreement (~$10^{-3}$) is the *curve's* sampling error — a
central difference over the tracker's adaptive steps — not the gradient's.
The tangent segments drawn from the implicit gradient lie on the tracked
curve:

![The fold curve and its autodiff tangents](../figures/cstr_sensitivity.png)

## What to notice

- **The gradient is a by-product of convergence.** Everything it needs —
  $G_z$, its factorisation, the converged $z^\ast$ — already exists when
  Newton finishes. The marginal cost of the exact sensitivity is one
  Jacobian evaluation in $B$ and one back-solve.
- **This is why kellax is written in JAX.** $G$ contains second derivatives
  of the residual (the $d(R_x v)/dx$ block); $G_B$ needs mixed derivatives in
  the model parameters. All of it is `jax.jacfwd` applied to code that was
  written once, for the residual alone.
- **The payoff is optimisation against the diagram.** A differentiable
  $Da^\ast(B)$ means a gradient method can *place* an ignition threshold —
  fit a model so its fold sits at a prescribed value, or set the width of a
  hysteresis loop. That inverse use of bifurcation analysis is kellax's
  research motivation, and this chapter is its first working piece.

This is the last written chapter. The book grows with the example suite —
periodic-orbit continuation from Hopf points is the next major front (see
the roadmap in the [README](../README.md)).
