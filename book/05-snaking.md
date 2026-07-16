# 5 — Homoclinic snaking in Swift–Hohenberg

> Script: [`examples/swift_hohenberg.py`](../examples/swift_hohenberg.py) · run it to regenerate the figure.

The pattern-formation showcase, and the problem BifurcationKit runs matrix-free on
a GPU at $\sim\!10^6$ degrees of freedom. The Swift–Hohenberg equation (the
quadratic–cubic "SH23" form)

$$u_t = -(1+\partial_x^2)^2 u + r\,u + \nu\,u^2 - u^3$$

has, for $\nu$ large enough, a *subcritical* Turing instability off $u=0$ at $r=0$.
Just below it, spatially **localized** patterns — a patch of rolls in an otherwise
flat background — live on a branch that **snakes**: it oscillates back and forth
across a narrow *pinning* interval of $r$, and at every fold the localized state
grows by one more pair of rolls.

![Swift–Hohenberg snaking](../figures/swift_hohenberg.png)

## Reaching the snake without branch-switching

The localized branch is disconnected from $u=0$; tools like BifurcationKit reach it
by branch-switching or deflation, which kellax does not have. But you do not need
them — the snake is **one connected branch of folds**, and that is precisely what
pseudo-arclength continuation walks. You only have to get *onto* it, by seeding
Newton with a localized envelope:

```python
u0 = 1.5 / np.cosh(0.6*(x - L/2)) * np.cos(x - L/2)     # sech envelope of rolls
u, _ = newton(R, u0, r0 = -0.2, tol = 1e-9)             # converge to a localized state
br = arclength_continuation(R, u, p0 = -0.2, ds = 0.02, ds_max = 0.06,
                            n_steps = 700, p_min = -0.30, p_max = -0.10, direction = 1.0)
```

```
snake: 38 folds, pinning region r in [-0.243, -0.183], ||u||_2 0.94 -> 4.19
```

Thirty-eight turning points on a single continuation — the left panel plots
$\lVert u\rVert_2$ against $r$ (the snake, with the pinning region shaded), the
right panel three localized states climbing it, each fold having added a roll pair.

## Spectral discretisation, and the matrix-free scale-up

The operator $(1+\partial_x^2)^2$ is diagonal in Fourier space (symbol
$(1-k^2)^2$), so the residual is a couple of FFTs — no matrix — and its diagonal
inverse is the natural preconditioner. The example confirms the **matrix-free**
engine snakes on the same problem:

```
matrix-free (GMRES on JVPs, Fourier precond): 12 folds in 2s
```

At $N=256$ the 1-D field is small enough for the dense engine (this is also what
BifurcationKit's 1-D tutorial uses); the point is that the *identical* spectral
formulation is what the matrix-free engine takes to 2-D/3-D at $10^4$–$10^6$ dof,
where the Jacobian cannot be formed — the regime of
[chapter 4](04-matrix-free.md).

## What to notice

- **Snaking is folds, all the way up.** No new machinery: the same
  `arclength_continuation` from [chapter 1](01-the-fold.md), just seeded on a
  localized state. Arclength sails through all 38 turning points because they lie
  on one branch.
- **Seeding is the whole trick.** kellax's lack of branch-switching is not a
  barrier here — a physically-motivated initial guess (an envelope of rolls) puts
  you on the branch, and continuation does the rest.
- **This problem drove an engine fix.** Seeding the matrix-free corrector with an
  *already-converged* localized state used to be rejected (its line search cannot
  improve an exact point); the engine now accepts a seed that already meets
  tolerance — see the `matrixfree` corrector.

Background: Burke & Knobloch on homoclinic snaking; Knoll & Keyes (2004) on
Jacobian-free Newton–Krylov.
