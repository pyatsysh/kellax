# 7. A predator–prey fold pair

> Script: [`examples/predator_prey.py`](../examples/predator_prey.py) · run it to regenerate the figure.

Folds are not just a chemistry story. Consider the Rosenzweig–MacArthur predator–prey
model with density-dependent predator loss, taken from the "EcoMod" tutorial of MatCont,

$$
\begin{aligned}
\dot x &= r\,x(1-x) - \frac{x y}{x+a},\\
\dot y &= -c\,y + \frac{x y}{x+a} - d\,\frac{y^2}{y^2+b^2},
\end{aligned}
$$

For the parameters $r=2,\ a=0.6,\ b=c=0.25$ this model carries **three** coexistence
equilibria over a window of the predator-mortality parameter $d$. Notice that the outer
two of them appear and disappear together, and that both do so at a fold. The two folds
bound the window from below and from above.

![Predator–prey fold pair](../figures/predator_prey.png)

## Sweeping an S-curve in two directions

The coexistence branch traces an S in the parameter $d$ and thus needs no
branch-switching. We seed one of the coexistence states *inside* the window and we
continue in **both** directions to trace it:

```python
z0, _ = newton(R, np.array([0.79, 0.58]), d0 = 0.22)    # a coexistence equilibrium
for direction in (-1.0, +1.0):
    arclength_continuation(R, z0, p0 = 0.22, direction = direction, ...)
```
```
fold: d=0.176930, x=0.91127, y=0.26820
fold: d=0.256800, x=0.61953, y=0.92799
   ref: d=0.256805 (0.619532, 0.927986); d=0.176927 (0.911266, 0.268200)
```

Both folds match MatCont to five digits. The left panel plots the predator density
$y$ against $d$ and shows the fold pair with the three coexistence states shaded. The
right panel draws the same branch in the $(x, y)$ phase plane. It is the locus of the
coexistence equilibria as they fold twice.

## What to notice

- **A 2-D state, the same machinery.** Nothing changed from the scalar chapters.
  The state $x$ is now the pair $(x, y)$ and the stability is set by the sign of the
  leading eigenvalue of a $2\times2$ Jacobian. Both `arclength_continuation` and
  `refine_fold` do not care. The move to two dimensions is invisible to them.
- **Every crossing here is a fold: verified, not assumed.** `analyze_branch`
  computes the spectrum along the branch, and from it we can classify each of the
  stability changes. Over this window of $d$ both of the changes turn out to be real
  crossings of the two folds, as we expect for a fold, and no complex pair crosses
  at all. An earlier draft asserted a Hopf on this branch. The spectrum says
  otherwise and so the claim went. Both Hopf detection and Newton-exact refinement
  are already in the library.
  Indeed, `refine_hopf` is validated both on the Hopf normal form and on the
  Brusselator at $b = 1+a^2$. The continuation of periodic orbits from a Hopf point
  remains the one item on the roadmap, to be tackled next.

Background: Kuznetsov, *Elements of Applied Bifurcation Theory* (the model is
MatCont's EcoMod tutorial).

Next: [Bratu in 2D](08-bratu-2d.md), a fold of a genuine 2-D field.
