# 8. Bratu in 2D: a fold of a PDE field

> Script: [`examples/bratu_2d.py`](../examples/bratu_2d.py) · run it to regenerate the figure.

Back to ignition, now on a surface. Consider the Bratu–Gelfand problem on the unit
square, of which BifurcationKit's `mittelmann` is a close relative:

$$\Delta u + \lambda\, e^{u} = 0 \quad\text{on } (0,1)^2, \qquad u = 0 \text{ on the boundary}.$$

As in 1D, a cool branch and a hot branch meet at a fold. The field $u(x,y)$ is now a
genuine 2-D field. It is a bump of temperature, pinned to zero on the boundary.

![2-D Bratu](../figures/bratu_2d.png)

## The fold of a 2-D field

We discretise with the 5-point Laplacian on a $20\times20$ interior grid of 400 dof.
We then continue in $\lambda$, and we read off the fold:

```python
def R(u, lam):                                  # u is the flattened N x N field
    U = u.reshape(N, N)
    lap = (roll_up + roll_dn + roll_lt + roll_rt - 4*U) / h**2
    return (lap + lam*np.exp(U)).reshape(-1)

br = arclength_continuation(R, np.zeros(N*N), p0 = 0.3, ...)
_, lam_f, _, _ = refine_fold(R, np.array(br.x[i]), float(br.p[i]))
```
```
2-D Bratu (N^2 = 400 dof): refined fold lambda* = 6.80469   (unit-square ref ~6.808)
```

The turning point sits at $\lambda^\ast \approx 6.808$. This is higher than the 1-D
value of $3.5138$. The reason is simple: the square sheds heat through more of its
boundary. The right panel shows the solution *at the fold*. It is a smooth thermal bump
with $\lVert u\rVert_\infty \approx 1.4$.

## What to notice

- **From a line to a field, unchanged.** The call is identical to the one in
  [chapter 3](03-bratu.md), and only the residual differs. It now returns a
  flattened 2-D Laplacian. Notice that autodiff differentiates through the `reshape`
  and through the 5-point stencil without any comment.
- **The scaling path.** 400 dof is comfortable for the dense engine. A finer 2-D or
  3-D grid is matrix-free territory. That is the GMRES-on-JVP engine of
  [chapter 4](04-matrix-free.md), which reaches $10^4$–$10^6$ dof. *(One caveat is
  logged honestly here. The **preconditioned** solve of the matrix-free engine
  currently stumbles on this 2-D residual at the initial step. This is a known rough
  edge, and the 1-D matrix-free path of chapter 4 is solid.)*

Next: [differentiable continuation](09-differentiable.md), the gradient of
the folds this book has been locating.
