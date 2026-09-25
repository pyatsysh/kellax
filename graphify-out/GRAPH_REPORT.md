# Graph Report - kellax  (2026-09-06)

## Corpus Check
- 47 files · ~72,429 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 329 nodes · 709 edges · 13 communities (12 shown, 1 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 21 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `2293f0a8`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- arclength_continuation
- __init__.py
- book/README.md
- test_solvers.py
- test_kellax.py
- arclength_continuation
- kellax: pseudo-arclength continuation and bifurcation analysis in JAX
- kellax tutorial
- 11. The inner solvers: a phase field, its transition state, and the Morse index
- Changelog
- paper.md
- 10. The Lorenz equilibria: a pitchfork, and the Hopf that ends it
- kellax

## God Nodes (most connected - your core abstractions)
1. `arclength_continuation()` - 37 edges
2. `apply_style()` - 24 edges
3. `refine_fold()` - 24 edges
4. `savefig()` - 23 edges
5. `mark_folds()` - 18 edges
6. `plot_branch()` - 17 edges
7. `stability_legend()` - 16 edges
8. `branch_stability()` - 14 edges
9. `analyze_branch()` - 14 edges
10. `track_fold()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `deflated_newton()` --indirect_call--> `G()`  [INFERRED]
  kellax/deflation.py → examples/cstr.py
- `refine_fold()` --indirect_call--> `G()`  [INFERRED]
  kellax/folds.py → examples/cstr.py
- `refine_hopf()` --indirect_call--> `G()`  [INFERRED]
  kellax/hopf.py → examples/cstr.py
- `test_newton_scalar_root()` --calls--> `newton()`  [EXTRACTED]
  tests/test_kellax.py → kellax/keller.py
- `main()` --calls--> `newton_krylov()`  [EXTRACTED]
  examples/allen_cahn.py → kellax/newton_krylov.py

## Import Cycles
- None detected.

## Communities (13 total, 1 thin omitted)

### Community 0 - "arclength_continuation"
Cohesion: 0.11
Nodes (48): main(), R(), r"""Example 8 — Bratu--Gelfand in 2D: a fold of a genuine PDE field. The Bratu…, 5-point Laplacian (Dirichlet) + lambda e^u, on the flattened N x N field., main(), bratu_problem(), main(), r"""Example 4 — scaling up: Bratu, matrix-free. The dense engine forms the… (+40 more)

### Community 1 - "__init__.py"
Cohesion: 0.06
Nodes (47): analyze_branch(), branch_eigenvalues(), BranchAnalysis, Stability and bifurcation detection along a traced branch. A Branch stores…, Spectrum and special points of a traced branch., Spectrum of ``sign * dR/dx`` at every accepted point. Returns (M, N) complex.…, Classify the special points of a Branch from its spectrum. Between consecutive…, bifurcation_diagram() (+39 more)

### Community 2 - "book/README.md"
Cohesion: 0.07
Nodes (39): 1. The fold, The fix, in code, What to notice, Why naive stepping fails, 2. The cusp, The nice part: one continuation, both arms, Tracking the fold set, What to notice (+31 more)

### Community 3 - "test_solvers.py"
Cohesion: 0.08
Nodes (38): energy(), laplacian(), main(), R(), r"""Example 11 — the inner solvers: a phase field, its transition state, and…, Neumann Laplacian: ghost cells mirror the end values, so a constant field has…, fixed_point_solve(), Accelerated fixed-point driver: damped Picard warm-up -> Anderson (DIIS). The… (+30 more)

### Community 4 - "test_kellax.py"
Cohesion: 0.06
Nodes (39): fold_curve(), G(), Track the ignition fold in B: down through the cusp and up the other arm.…, G3(), main(), r"""Example 9 — differentiable continuation: the gradient of an ignition…, bordered_newton(), Generic bordered Newton in JAX — one solver, k border rows. Solve the square… (+31 more)

### Community 5 - "arclength_continuation"
Cohesion: 0.10
Nodes (31): main(), precond(), R(), r"""Example 5 — homoclinic snaking in the Swift--Hohenberg equation. The…, Branch, A traced branch: states, parameter values, tangents, fold indices., backtrack(), forced_gmres() (+23 more)

### Community 6 - "kellax: pseudo-arclength continuation and bifurcation analysis in JAX"
Cohesion: 0.25
Nodes (8): Citing, Install / test, kellax: pseudo-arclength continuation and bifurcation analysis in JAX, Motivation, Roadmap, The book, The toolbox (v0.5.0), Why JAX

### Community 7 - "kellax tutorial"
Cohesion: 0.25
Nodes (8): 60 seconds: trace a fold, Install, kellax tutorial, Scale up: matrix-free, Sharpen a fold, The API, in one screen, Track a fold in a second parameter, Where to go next

### Community 8 - "11. The inner solvers: a phase field, its transition state, and the Morse index"
Cohesion: 0.29
Nodes (7): 11. The inner solvers: a phase field, its transition state, and the Morse index, Classifying by index, Climbing the ladder, The gradient through the solve, The interface energy, and a constraint, The residual and the energy must agree, What to notice

### Community 9 - "Changelog"
Cohesion: 0.29
Nodes (6): 0.1.0 — 2026-07-08, 0.2.0 — 2026-07-08, 0.3.0 — 2026-07-15, 0.4.0 — 2026-07-18, 0.5.0 — unreleased, Changelog

### Community 10 - "paper.md"
Cohesion: 0.33
Nodes (5): Acknowledgements, Functionality, References, Statement of need, Summary

### Community 11 - "10. The Lorenz equilibria: a pitchfork, and the Hopf that ends it"
Cohesion: 0.40
Nodes (5): 10. The Lorenz equilibria: a pitchfork, and the Hopf that ends it, Continuing the pair, and pinning the Hopf, The structure a fold detector cannot see, Three solutions, from one guess, What to notice

## Knowledge Gaps
- **64 isolated node(s):** `kellax`, `0.5.0 — unreleased`, `0.4.0 — 2026-07-18`, `0.3.0 — 2026-07-15`, `0.2.0 — 2026-07-08` (+59 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 169 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `arclength_continuation()` connect `arclength_continuation` to `__init__.py`, `test_kellax.py`, `arclength_continuation`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
- **Why does `refine_fold()` connect `arclength_continuation` to `__init__.py`, `test_kellax.py`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._
- **Why does `arclength_continuation()` connect `arclength_continuation` to `arclength_continuation`, `__init__.py`, `test_kellax.py`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **What connects `kellax`, `0.5.0 — unreleased`, `0.4.0 — 2026-07-18` to the rest of the system?**
  _64 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `arclength_continuation` be split into smaller, more focused modules?**
  _Cohesion score 0.11278195488721804 - nodes in this community are weakly interconnected._
- **Should `__init__.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06428571428571428 - nodes in this community are weakly interconnected._
- **Should `book/README.md` be split into smaller, more focused modules?**
  _Cohesion score 0.06748911465892599 - nodes in this community are weakly interconnected._