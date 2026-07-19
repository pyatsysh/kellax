"""kellax — pseudo-arclength continuation and bifurcation analysis in JAX.

Trace solution branches of R(x, p) = 0 through folds with autodiff
Jacobians: supply the residual, get the branch, its tangents, and its
turning points. Folds are then exact, not step-limited: Moore-Spence
refinement (``refine_fold``) pins a detected turning point to Newton
precision and ``track_fold`` continues the fold itself in a second
parameter; ``bordered_newton`` is the generic k-constraint primitive
(arclength, mass, phase conditions) underlying all of it.
``mf_arclength_continuation`` scales the trace to 10^4-10^6-dof fields
(2D/3D) with GMRES-on-JVP bordered solves where the dense engine cannot
form the Jacobian.

The branch is then read as a dynamical object: ``analyze_branch`` computes
the spectrum along it and classifies folds, branch points and Hopf
candidates; ``refine_hopf`` pins a Hopf point exactly (3N+2 augmented
system, second derivatives by autodiff); ``branch_off`` jumps onto a
bifurcating branch through deflation (``deflated_newton`` /
``deflated_search`` find the solutions you have not already found); and
``bifurcation_diagram`` composes trace -> classify -> switch -> recurse.
``fold_sensitivity`` is the differentiable-continuation primitive: the
exact gradient of a fold location with respect to model parameters, by
implicit differentiation of the converged Moore-Spence system. v0.5 (consolidation from the cdft toolbox): the ladder's inner solvers —
``fixed_point_solve`` (damped Picard -> Anderson/DIIS globaliser),
``newton_krylov`` with the generic bordered step factory
(``make_step_bordered``: mass constraints, phase conditions), autodiff-
Hessian spectra (``hessian_spectrum``, matrix-free Lanczos; scipy lazily),
and ``ift_injection`` (differentiable solutions by the implicit-function
theorem, dense or Krylov-adjoint). Roadmap: periodic-orbit continuation
from Hopf points; delegate matrixfree's arclength step through
``make_step_bordered`` (internal dedup).
"""
from .keller import Branch, arclength_continuation, newton
from .fixedpoint import fixed_point_solve
from .newton_krylov import newton_krylov, make_step, make_step_bordered
from .hessian import (hessian_vector_product, hessian_spectrum,
                      smallest_eigenvalue, morse_index)
from .implicit import ift_injection
from .folds import FoldBranch, refine_fold, track_fold
from .bordered import bordered_newton
from .matrixfree import arclength_continuation as mf_arclength_continuation
from .analysis import BranchAnalysis, branch_eigenvalues, analyze_branch
from .deflation import deflated_newton, deflated_search
from .hopf import refine_hopf
from .sensitivity import fold_sensitivity
from .branching import DiagramBranch, branch_off, bifurcation_diagram

__version__ = "0.5.0.dev0"
