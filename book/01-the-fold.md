# 1. The fold

> Script: [`examples/cubic_fold.py`](../examples/cubic_fold.py) · run it to regenerate the figure.

Consider the simplest problem that breaks naive continuation:

$$R(x, p) = x^3 - x + p = 0.$$

For $|p| < 2/(3\sqrt3) \approx 0.385$ this cubic has **three** real roots. Outside of that window only one of them remains. The two roots that annihilate at each end of the window form a **fold**, which is also known as a saddle-node or as a turning point. Thus, the branch of solutions in the $(p, x)$ plane is an S lying on its side.

![The fold normal form](../figures/cubic_fold.png)

## Why naive stepping fails

Suppose we fix $p$ and solve for $x$ by Newton iterations. We then nudge $p$ and solve again. At the fold the branch is vertical: $\mathrm{d}x/\mathrm{d}p = \infty$ and $\partial R/\partial x = 3x^2-1 = 0$ is singular. Just past the fold there is no nearby root at all. Therefore, the next solve either jumps to the far branch or diverges. Moreover, folds are generic. Thus, failures of this kind are common in practice.

## The fix, in code

Pseudo-arclength continuation is the standard cure for this failure. In this approach we reparametrise the branch by its own arclength and solve a bordered system of equations, which remains non-singular through the turning point. In kellax the whole construction reduces to a single call, with the bordering hidden from view:

```python
R = lambda x, p: np.array([x[0]**3 - x[0] + p])
br = arclength_continuation(R, np.array([-1.2]), p0 = 0.7, ds = 0.03, ds_max = 0.06,
                            n_steps = 600, p_min = -1.2, p_max = 1.2, direction = -1.0)
# traced 80 points; 2 turning points detected
```

`br.turning_points` holds the indices at which the $p$-component of the tangent changes sign. These are the folds. Notice that the detection only brackets each fold to within a single continuation step. The call `refine_fold` then pins each of them to Newton precision by solving the Moore–Spence augmented system:

```python
for i in br.turning_points[:2]:
    xf, pf, vf, res = refine_fold(R, np.array(br.x[i]), float(br.p[i]))
# refined fold: p = -0.3849001795  (exact -2/(3√3)),  x = -0.577350,  residual 5.6e-17
# refined fold: p = +0.3849001795  (exact +2/(3√3)),  x = +0.577350,  residual 5.6e-17
```

We obtain ten correct digits and a residual at machine epsilon.

## What to notice

Notice that the stability of a branch can be read off the Jacobian. With the gradient-flow reading $\dot x = -R$, a point of the branch is stable when every eigenvalue of $-\partial R/\partial x$ is negative. Here the stable points lie on the two outer arms of the S. The middle arm between the folds is unstable, and we recover the classic picture of hysteresis. In the figure we draw stable states with solid lines and unstable states with dashed lines. Such a cue does not rely on colour and survives greyscale.

Moreover, all of the calculus is supplied by automatic differentiation. We wrote only the residual `R`. The Jacobian $\partial R/\partial x$, the parameter derivative $\partial R/\partial p$ and the second-derivative term $\partial(R_x v)/\partial x$ of the fold refinement all come from `jax.jacfwd`.

Background: Keller (1977); Seydel, *Practical Bifurcation and Stability
Analysis*; Govaerts (2000).

Next: in [the cusp](02-the-cusp.md) we follow these two folds as a second parameter turns them into a curve.
