# 3 — Bratu–Gelfand: a fold in a boundary-value problem

> Script: [`examples/bratu.py`](../examples/bratu.py) · run it to regenerate the figure.

The same fold, now in a PDE. The Bratu–Gelfand problem models solid-fuel ignition
(thermal runaway):

$$-u''(x) = \lambda\, e^{u(x)}, \qquad x\in(0,1), \qquad u(0)=u(1)=0.$$

It is *the* canonical continuation boundary-value problem — AUTO, MatCont, and
BifurcationKit all ship it. Two steady states, a **cool** one and a **hot** one,
exist for $\lambda$ below a critical ignition value and collide at a fold; above it
there is no steady state (runaway).

![Bratu–Gelfand](../figures/bratu.png)

## Discretise, then continue

Second-order finite differences on $N=200$ interior points turn the BVP into
$R(u,\lambda)=D_2 u + \lambda e^{u}=0$, a system of $N$ equations. The
discretisation is the *only* modelling — kellax differentiates $R$ for every
Jacobian it needs:

```python
D2 = (np.diag(-2*np.ones(N)) + np.diag(off, 1) + np.diag(off, -1)) / h**2
R  = lambda u, lam: D2 @ u + lam*np.exp(u)

br = arclength_continuation(R, np.zeros(N), p0 = 0.3, ds = 0.03, ds_max = 0.15,
                            n_steps = 900, p_min = 0.25, p_max = 6.0, direction = 1.0)
i = br.turning_points[0]
_, lam_f, _, res = refine_fold(R, np.array(br.x[i]), float(br.p[i]))
# refined fold: lambda* = 3.513785  (reference 3.513831, diff 4.5e-5),  res 2.9e-11
```

The refined turning point matches the known 1-D Bratu value
$\lambda^\ast = 3.513831\ldots$ to the $O(h^2)$ discretisation error — halve $h$ and
the gap quarters.

## Stability comes from the spectrum

Physically $u_t = u_{xx} + \lambda e^{u}$, so a steady state is stable iff every
eigenvalue of $\partial R/\partial u = D_2 + \lambda\,\mathrm{diag}(e^{u})$ is
negative. That flips exactly at the fold: the cool (lower) branch is stable, the hot
(upper) branch unstable. The left panel plots $\lVert u\rVert_\infty$ against
$\lambda$ with that solid/dashed styling; the right panel shows the three profiles —
cool, at-fold, and hot — the hot one a tall thermal spike.

## What to notice

- **Nothing about the method changed.** The exact calls from
  [chapter 1](01-the-fold.md) work on a 200-dimensional discretised field; only the
  residual is different.
- **This is the crossover point.** At $N=200$ the dense engine (forming
  $\partial R/\partial u$ and factorising it) is instant. Push $N$ into the tens of
  thousands — a 2-D or 3-D field — and you can no longer form that matrix; that is
  what the matrix-free engine and the *snaking* chapter are for.

Background: Seydel, *Practical Bifurcation and Stability Analysis* (BVP
continuation); Allgower & Georg, *Numerical Continuation Methods*.

Next: [scaling up](04-matrix-free.md) — the same trace, matrix-free — then
[homoclinic snaking](05-snaking.md) in the Swift–Hohenberg equation.
