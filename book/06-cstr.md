# 6. CSTR: hysteresis, and where it is born

> Script: [`examples/cstr.py`](../examples/cstr.py) · run it to regenerate the figure.

In chemical engineering, an exothermic continuous stirred-tank reactor serves as
a textbook case of steady-state multiplicity. The reactor also gives us a
chance to use the tools of chapters 1 and 2 on real physics. In the
dimensionless adiabatic form of the model the steady conversion $x$ solves

$$G(x, Da) = -x + Da\,(1-x)\,e^{Bx} = 0,$$

where $Da$ denotes the Damköhler number (residence time) and $B$ measures the
exothermicity of the reaction.

![CSTR hysteresis and its cusp](../figures/cstr.png)

## The hysteresis loop (a fold at each end)

For $B = 6$ the reactor has three steady states in a window of $Da$: a cold
branch, a hot branch and an unstable branch between them. We trace the S-curve
and refine the two folds which mark **ignition** and **extinction**:

```python
R = lambda x, Da: G(x, Da, B = 6.0)
br = arclength_continuation(R, np.array([0.03]), p0 = 0.02, ds = 0.008, ...)
for i in br.turning_points[:2]:
    xf, Daf, _, _ = refine_fold(R, np.array(br.x[i]), float(br.p[i]))
```
```
S-curve at B=6: ignition Da=0.075403 (ref 0.075403), extinction Da=0.032873 (ref 0.032873)
```

The agreement with the Uppal–Ray–Poore values is exact. The left panel of the
figure shows the classic loop of ignition and extinction. On raising $Da$ past
ignition the cold reactor jumps to the hot branch, and on lowering $Da$ past
extinction the reactor drops back to the cold branch. Thus the reactor exhibits
a hysteresis.

## Where the hysteresis comes from: the cusp

Notice that the hysteresis is by no means inevitable and exists only for $B > 4$.
To see this, we track a fold in the *second* parameter $B$ by pointing the
`track_fold` of [chapter 2](02-the-cusp.md) at the reactor. As $B$ falls, the two
folds sweep together and **annihilate at a cusp**,

$$B = 4, \quad x = \tfrac12, \quad Da = e^{-2} = 0.135335.$$

```
cusp: min B=4.0024 (ref 4), Da=0.13518 (ref e^-2=0.13534)
```

The set of folds in the $(B, Da)$ plane can be seen in the right panel of the
figure. Inside the wedge three states are available to the reactor. The tip of
the wedge is the cusp, where multiplicity is born. This is the same object as
the abstract cusp of chapter 2, now with an engineering meaning. Below $B = 4$ the reactor can only be
monostable at any value of the residence time.

## What to notice

- **One problem, both tools.** We compute the S-curve with
  `arclength_continuation` and `refine_fold`, and the cusp with `track_fold`.
  None of the machinery belongs to the reactor. Indeed, we simply point the fold
  toolkit of chapters 1–2 at $G(x, Da)$.
- **Stability is the reactor's.** With $\dot x = G$, the cold and the hot
  branches of the reactor are stable, while the middle branch is not. That is
  exactly why the reactor jumps.

Background: Uppal, Ray & Poore (1974) for the CSTR multiplicity diagram.

Next: [a predator–prey fold pair](07-predator-prey.md), where the same folds
appear in ecology.
