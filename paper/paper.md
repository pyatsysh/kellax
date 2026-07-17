---
title: 'kellax: autodiff-native pseudo-arclength continuation in JAX'
tags:
  - Python
  - JAX
  - numerical continuation
  - bifurcation analysis
  - pseudo-arclength
  - fold tracking
  - automatic differentiation
authors:
  - name: Peter Yatsyshin
    orcid: 0000-0002-8844-281X
    affiliation: 1
affiliations:
  - name: "TODO: affiliation"   # TODO: add affiliation
    index: 1
date: 16 July 2026
bibliography: paper.bib
---

# Summary

Numerical continuation traces how the solution of a nonlinear system
$R(x, p) = 0$, with state $x \in \mathbb{R}^N$, deforms as the scalar control
parameter $p$ varies. The defining difficulty is the fold, or turning point.
There the branch reverses in $p$ and the state Jacobian $\partial R/\partial x$
becomes singular, so the naive scheme of re-solving at successive values of $p$
jumps to a distant branch or stalls. Folds are generic, so this is the common
case. In the present contribution we introduce `kellax`, a JAX library
implementing the pseudo-arclength continuation of @keller1977, which
reparametrises the branch by arclength and solves a bordered linear system that
stays non-singular through the folds.

We build `kellax` on JAX [@jax2018], so that every derivative of the
algorithm comes from automatic differentiation of the user-supplied residual:
the dense engine takes the Jacobians $R_x$ and $R_p$ from `jax.jacfwd`,
and the matrix-free engine takes the Jacobian--vector products from
`jax.linearize`. No analytic or finite-difference Jacobian is ever
required. Thus, the physics lives in the residual, and the machinery stays
fixed. The whole predictor-corrector is jit-compiled and runs unchanged on CPU
or GPU. The matrix-free engine solves each bordered system by preconditioned
GMRES on the Jacobian--vector products, which scales the continuation to
discretised fields with $10^4$ to $10^6$ unknowns, where the Jacobian cannot
even be formed. We additionally refine the detected folds to Newton precision
through the Moore--Spence augmented system [@mooreSpence1980], continue
the folds in a second parameter, and expose the generic bordered-Newton
primitive that underlies all of it.

# Statement of need

Continuation and bifurcation analysis are basic tools wherever a nonlinear
model has multiple states or hysteresis. Steady states of discretised PDEs,
reaction networks, phase equilibria and the fixed points of learned dynamical
systems all present the same fold-crossing problem. Yet the mature software
lives outside of the modern Python ecosystem: AUTO-07p in Fortran
[@auto07p]; MatCont, COCO, pde2path and DDE-BIFTOOL in MATLAB
[@matcont2003; @coco2013; @pde2path2014; @ddebiftool2002]; LOCA/Trilinos in
C++ [@loca2002]. The one modern and feature-rich outlier is
BifurcationKit.jl in Julia [@bifurcationkit]. It already unifies automatic
differentiation, GPU execution and matrix-free Newton--Krylov continuation. It
is also considerably more complete than `kellax`: periodic orbits, Hopf and
higher-codimension bifurcations, branch switching. We do not claim to displace
it.

The gap we fill is in the Python/JAX ecosystem, where no comparable tool
exists. The available options in Python are either finite-difference and small
(bice, PyCont-Lite), stale wrappers around AUTO [@pydstool2012], or
matrix-free but autodiff-less and latterly proprietary-licensed
[@pacopy]. It is noteworthy that the one prior artifact of the
JAX-with-pseudo-arclength kind, FContin [@fcontin], is a dormant wrapper
over pacopy. It performs no fold or bifurcation tracking of its own. Indeed,
the classical continuation codes share a practical friction that automatic
differentiation removes: the user must supply analytic Jacobians or accept
finite differences. We ask only for the residual $R(x, p)$ and differentiate
through it, inheriting the jit compilation, the GPU support and the float64
numerics of JAX throughout. In the matrix-free path we keep the arclength
border row inside the Krylov space, which regularises the near-singular fold
mode without ever forming the inverse of that mode. The engine also carries
the standard inexact-Newton machinery of Jacobian-free Newton--Krylov practice
[@knollKeyes2004]: the forcing schedule of Eisenstat and Walker
[@eisenstatWalker1996]; a NaN-safe Armijo line search with a trust
region; and an optional preconditioner of the user's choice. This is what
makes the bordered solves converge on stiff discretised operators.

Moreover, `kellax` is a deliberate step toward differentiable continuation.
The entire solve is a JAX program, so in principle the branch it produces ---
including the locations of the folds --- is a differentiable function of the
model parameters. The right route is implicit differentiation of the converged
bordered or Moore--Spence system [@blondel2022], not an unrolling of the
Newton iterations. This would allow us to optimise or learn against a
bifurcation diagram: fitting a model so that a fold sits at a prescribed value
of the parameter, or so that a hysteresis loop has a target width. The idea is
not unprecedented, and BifurcationInference.jl in Julia differentiates through
bifurcation diagrams for parameter inference [@szep2021]. Yet it is
unoccupied in the end-to-end-differentiable Python/JAX setting that makes it
most natural, and it is the primary research motivation of `kellax`.

We extracted `kellax` from a toolbox of classical density functional theory,
where it served as the isotherm engine, tracing capillary-condensation loops
and locating the spinodals --- the folds of the adsorption isotherm --- as the
external conditions vary. It is model-agnostic: any smooth residual is a valid
input.

# Functionality

The dense trace of $R(x, p) = 0$ is `arclength_continuation`, which
adapts its steps and detects the folds by the sign changes of the parameter
component of the tangent. It returns a `Branch` of the states, the
parameter values, the tangents, the step sizes and the turning-point indices.
The matrix-free twin for large $N$ is `mf_arclength_continuation`. It
runs GMRES on the JVPs, takes an optional preconditioner, and can land on an
exact target value of the parameter. We pin a detected fold to Newton
precision with `refine_fold`, which solves the Moore--Spence augmented
system with the directional second derivative supplied by autodiff. We
continue a fold in a second parameter with `track_fold`, which runs
the arclength trace on the augmented system, so that the cusps of the fold
curve are traversed and reported. The generic $(N{+}k)$ bordered-Newton
primitive is `bordered_newton`, covering the arclength, mass and phase
constraints. Finally, `newton` is the plain dense Newton at fixed $p$,
for branch seeding.

We test the package against problems with analytic folds: the cubic S-curve
and a coupled two-dimensional system. The two-parameter fold-tracking curve
matches its closed-form law to $10^{-8}$, and we cross-validate the
Moore--Spence fold against the generic bordered solver. The worked examples
validate against the literature: the Bratu--Gelfand ignition folds in one and
two dimensions; the ignition-extinction pair of the exothermic CSTR at the
values of Uppal, Ray and Poore; a predator-prey fold pair of MatCont to five
digits; and a homoclinic-snaking branch of the Swift--Hohenberg equation with
38 folds, traced in a single continuation. Standard references for the
underlying theory are @govaerts2000, @allgowerGeorg1990,
@kuznetsov2004 and @seydel2010.

# Acknowledgements

`kellax` began as the continuation engine of a codebase of classical density
functional theory, and we factored it out as a standalone, model-agnostic
library.

# References
