"""kellax — pseudo-arclength continuation and bordered solvers in JAX.

Trace solution branches of R(x, p) = 0 through folds with autodiff
Jacobians: supply the residual, get the branch, its tangents, and its
turning points. Folds are then exact, not step-limited: Moore-Spence
refinement (``refine_fold``) pins a detected turning point to Newton
precision and ``track_fold`` continues the fold itself in a second
parameter; ``bordered_newton`` is the generic k-constraint primitive
(arclength, mass, phase conditions) underlying all of it.
``mf_arclength_continuation`` scales the trace to 10^4-10^6-dof fields
(2D/3D) with GMRES-on-JVP bordered solves where the dense engine cannot
form the Jacobian. Roadmap: differentiable continuation — fold locations
as differentiable functions of model parameters.
"""
from .keller import Branch, arclength_continuation, newton
from .folds import FoldBranch, refine_fold, track_fold
from .bordered import bordered_newton
from .matrixfree import arclength_continuation as mf_arclength_continuation

__version__ = "0.3.0"
