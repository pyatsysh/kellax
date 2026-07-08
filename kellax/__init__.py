"""kellax — pseudo-arclength continuation and bordered solvers in JAX.

Trace solution branches of R(x, p) = 0 through folds with autodiff Jacobians:
supply the residual, get the branch, its tangents, and its turning points.
Roadmap: Moore-Spence fold tracking, generic bordered constraints, matrix-free
bordered solves, and differentiable continuation (fold locations as
differentiable functions of model parameters).
"""
from .keller import Branch, arclength_continuation, newton

__version__ = "0.1.0"
__all__ = ["Branch", "arclength_continuation", "newton"]
