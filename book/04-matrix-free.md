# 4. Scaling up: matrix-free continuation

> Script: [`examples/bratu_matrixfree.py`](../examples/bratu_matrixfree.py) · run it to regenerate the figure.

So far we have been using the **dense** engine: form the Jacobian $\partial R/\partial x$,
factorise it and solve. This is perfectly adequate for the $N=200$ Bratu of
[chapter 3](03-bratu.md). However, a 2-D or 3-D field discretises to
$N = 10^4\text{–}10^6$ unknowns, and for such sizes the Jacobian of the residual
can no longer be stored. The **matrix-free** engine runs the same Keller
continuation and never forms a matrix. Each bordered solve consists of
preconditioned GMRES on Jacobian–vector products, with JAX supplying each $Jv$
from one `jax.linearize` of the residual.

![Bratu, dense vs matrix-free](../figures/bratu_matrixfree.png)

## Same branch, two engines

In what follows, we take the Bratu residual in matrix-free form and trace the
branch with the two engines at $N=300$, as a check of correctness. We apply the
Laplacian by roll-and-subtract, so that no $N\times N$ matrix appears:

```python
br_d = arclength_continuation(R, np.zeros(N), ...)                    # dense
br_m = mf_arclength_continuation(R, np.zeros(N), precond = precond)   # matrix-free
```
```
correctness check at N = 300:
  dense       fold lambda* = 3.513811
  matrix-free fold lambda* = 3.513811
  |dense - matrix-free| = 2.40e-14   (both ~ ref 3.5138)
```

The two engines agree to machine precision in the location of the fold. Indeed,
the markers of the matrix-free branch in the figure sit exactly on the dense
curve. The matrix-free engine then
goes where the dense one cannot and traces the branch at $N=4000$. The
$4000\times4000$ Jacobian of the residual is never formed.

## A preconditioner is not optional

Unpreconditioned GMRES on the stiff Laplacian stalls before reaching the fold.
Notice that the discrete sine transform diagonalises the Dirichlet Laplacian.
Thus one DST, a division by the eigenvalues and one DST back provide an
approximation of $(\partial R/\partial x)^{-1}$ at a negligible cost, and the
solve of the bordered system becomes trivial:

```python
def precond(v):                       # M^{-1} ~ (D2)^{-1} via the sine transform
    return dst1(dst1(v) * (2/(N+1)) / lap_eig)
```

This is precisely the purpose of the `precond` hook. In [chapter 5](05-snaking.md)
the same role falls to a Fourier preconditioner, natural to the spectral
Swift–Hohenberg operator.

## What to notice

- **The algorithm did not change.** The API and the `Branch`/fold semantics of
  the dense engine are mirrored exactly by `mf_arclength_continuation`. Only the
  linear solve is different. Moreover, the border row of the arclength system
  stays *inside* the Krylov space, which regularises the near-singular fold mode
  without ever forming the inverse of that soft mode.
- **Fold *refinement* is still dense.** `refine_fold` forms the Moore–Spence
  Jacobian, with a cost of $O(N^3)$. Therefore at $N=4000$ we report the
  step-limited bracket of the fold and not a refined value. A matrix-free
  bordered solve for the fold remains a roadmap item.

Background: Knoll & Keyes (2004) on Jacobian-free Newton–Krylov; Eisenstat &
Walker (1996) for the forcing schedule.

Next: [homoclinic snaking](05-snaking.md), where the matrix-free-capable engine
meets a pattern-forming PDE.
