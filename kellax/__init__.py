"""kellax — pseudo-arclength continuation and bordered solvers in JAX.

Trace solution branches of R(x, p) = 0 through folds with autodiff Jacobians:
supply the residual, get the branch, its tangents, and its turning points.
Folds are then exact, not step-limited: Moore-Spence refinement
(``refine_fold``) pins a detected turning point to Newton precision and
``track_fold`` continues the fold itself in a second parameter;
``bordered_newton`` is the generic k-constraint primitive (arclength, mass,
phase conditions) underlying all of it. Roadmap: matrix-free bordered solves
and differentiable continuation (fold locations as differentiable functions
of model parameters).
"""
from .keller import Branch, arclength_continuation, newton
from .folds import FoldBranch, refine_fold, track_fold
from .bordered import bordered_newton

__version__ = "0.2.0"
__all__ = ["Branch", "FoldBranch", "arclength_continuation", "newton",
           "refine_fold", "track_fold", "bordered_newton"]
