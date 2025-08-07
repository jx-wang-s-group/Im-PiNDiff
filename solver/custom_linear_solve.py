"""Linear system solvers."""

from functools import partial
import operator

import numpy as np

import jax
import jax.numpy as jnp
from jax import device_put
from jax import lax
from jax import scipy as jsp
from jax.tree_util import (tree_leaves, tree_map, tree_structure,
                           tree_reduce, Partial)

from jax._src import dtypes
from jax._src.lax import lax as lax_internal
from jax._src.util import safe_map as map


_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)
_vdot = partial(jnp.vdot, precision=lax.Precision.HIGHEST)
_einsum = partial(jnp.einsum, precision=lax.Precision.HIGHEST)


from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from jaxopt._src.tree_util import tree_add_scalar_mul

from jax._src.scipy.sparse.linalg import (_normalize_matvec, _identity, _shapes, _vdot_real_tree, _vdot_tree, 
                                          _add, _mul, _sub)


def _make_ridge_matvec(matvec: Callable, ridge: float = 0.0):
  def ridge_matvec(v: Any) -> Any:
    return tree_add_scalar_mul(matvec(v), ridge, v)
  return ridge_matvec


def _bicgstab_solve(A, b, x0=None, *, maxiter, tol=1e-5, atol=0.0, M=_identity):

  # tolerance handling uses the "non-legacy" behavior of scipy.sparse.linalg.bicgstab
  bs = _vdot_real_tree(b, b)
  atol2 = jnp.maximum(jnp.square(tol) * bs, jnp.square(atol))

  # https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method#Preconditioned_BiCGSTAB

  def cond_fun(value):
    x, r, *_, k = value
    rs = _vdot_real_tree(r, r)
    # the last condition checks breakdown
    return (rs > atol2) & (k < maxiter) & (k >= 0)

  def body_fun(value):
    x, r, rhat, alpha, omega, rho, p, q, k = value
    rho_ = _vdot_tree(rhat, r)
    beta = rho_ / rho * alpha / omega
    p_ = _add(r, _mul(beta, _sub(p, _mul(omega, q))))
    phat = M(p_)
    q_ = A(phat)
    alpha_ = rho_ / _vdot_tree(rhat, q_)
    s = _sub(r, _mul(alpha_, q_))
    exit_early = _vdot_real_tree(s, s) < atol2
    shat = M(s)
    t = A(shat)
    omega_ = _vdot_tree(t, s) / _vdot_tree(t, t)  # make cases?
    x_ = tree_map(partial(jnp.where, exit_early),
                  _add(x, _mul(alpha_, phat)),
                  _add(x, _add(_mul(alpha_, phat), _mul(omega_, shat)))
                  )
    r_ = tree_map(partial(jnp.where, exit_early),
                  s, _sub(s, _mul(omega_, t)))
    k_ = jnp.where((omega_ == 0) | (alpha_ == 0), -11, k + 1)
    k_ = jnp.where((rho_ == 0), -10, k_)
    return x_, r_, rhat, alpha_, omega_, rho_, p_, q_, k_

  r0 = _sub(b, A(x0))
  rho0 = alpha0 = omega0 = lax_internal._convert_element_type(
      1, *dtypes._lattice_result_type(*tree_leaves(b)))
  initial_value = (x0, r0, r0, alpha0, omega0, rho0, r0, r0, 0)

  x_final, *_ = lax.while_loop(cond_fun, body_fun, initial_value)
  x_final, r,_, gamma, _,_,_,_, k = lax.while_loop(cond_fun, body_fun, initial_value)
  rs = gamma if M is _identity else _vdot_real_tree(r, r)
  converged = rs <= atol2
  # info = {'error': jnp.sqrt(rs/bs), 'niter': k} 
  info = {'error': jnp.sqrt(_vdot_real_tree(r, r)/(bs+1e-6)), 'niter': k} 
  return x_final, info

def _isolve(_isolve_solve, A, b, x0=None, *, tol=1e-5, atol=0.0,
            maxiter=None, M=None, check_symmetric=False, has_aux=False):
  if x0 is None:
    x0 = tree_map(jnp.zeros_like, b)

  b, x0 = device_put((b, x0))

  if maxiter is None:
    size = sum(bi.size for bi in tree_leaves(b))
    maxiter = 10 * size  # copied from scipy

  if M is None:
    M = _identity
  A = _normalize_matvec(A)
  M = _normalize_matvec(M)

  if tree_structure(x0) != tree_structure(b):
    raise ValueError(
        'x0 and b must have matching tree structure: '
        f'{tree_structure(x0)} vs {tree_structure(b)}')

  if _shapes(x0) != _shapes(b):
    raise ValueError(
        'arrays in x0 and b must have matching shapes: '
        f'{_shapes(x0)} vs {_shapes(b)}')

  isolve_solve = partial(
      _isolve_solve, x0=x0, tol=tol, atol=atol, maxiter=maxiter, M=M)

  # real-valued positive-definite linear operators are symmetric
  def real_valued(x):
    return not issubclass(x.dtype.type, np.complexfloating)
  symmetric = all(map(real_valued, tree_leaves(b))) \
    if check_symmetric else False

  x,info= lax.custom_linear_solve(
      A, b, solve=isolve_solve, transpose_solve=isolve_solve,
      symmetric=symmetric,has_aux=True)
  # info = None
  return x, info


def bicgstab(A, b, x0=None, *, tol=1e-5, atol=0.0, maxiter=None, M=None):
  """Use Bi-Conjugate Gradient Stable iteration to solve ``Ax = b``.

  The numerics of JAX's ``bicgstab`` should exact match SciPy's
  ``bicgstab`` (up to numerical precision), but note that the interface
  is slightly different: you need to supply the linear operator ``A`` as
  a function instead of a sparse matrix or ``LinearOperator``.

  As with ``cg``, derivatives of ``bicgstab`` are implemented via implicit
  differentiation with another ``bicgstab`` solve, rather than by
  differentiating *through* the solver. They will be accurate only if
  both solves converge.

  Parameters
  ----------
  A: ndarray, function, or matmul-compatible object
      2D array or function that calculates the linear map (matrix-vector
      product) ``Ax`` when called like ``A(x)`` or ``A @ x``. ``A`` can represent
      any general (nonsymmetric) linear operator, and function must return array(s)
      with the same structure and shape as its argument.
  b : array or tree of arrays
      Right hand side of the linear system representing a single vector. Can be
      stored as an array or Python container of array(s) with any shape.

  Returns
  -------
  x : array or tree of arrays
      The converged solution. Has the same structure as ``b``.
  info : None
      Placeholder for convergence information. In the future, JAX will report
      the number of iterations when convergence is not achieved, like SciPy.

  Other Parameters
  ----------------
  x0 : array or tree of arrays
      Starting guess for the solution. Must have the same structure as ``b``.
  tol, atol : float, optional
      Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
      We do not implement SciPy's "legacy" behavior, so JAX's tolerance will
      differ from SciPy unless you explicitly pass ``atol`` to SciPy's ``cg``.
  maxiter : integer
      Maximum number of iterations.  Iteration will stop after maxiter
      steps even if the specified tolerance has not been achieved.
  M : ndarray, function, or matmul-compatible object
      Preconditioner for A.  The preconditioner should approximate the
      inverse of A.  Effective preconditioning dramatically improves the
      rate of convergence, which implies that fewer iterations are needed
      to reach a given error tolerance.

  See also
  --------
  scipy.sparse.linalg.bicgstab
  jax.lax.custom_linear_solve
  """

  return _isolve(_bicgstab_solve,
                 A=A, b=b, x0=x0, tol=tol, atol=atol,
                 maxiter=maxiter, M=M)

def solve_bicgstab(matvec: Callable,
                   b: Any,
                   ridge: Optional[float] = None,
                   init: Optional[Any] = None,
                   **kwargs) -> Any:
  """Solves ``A x = b`` using bicgstab.

  Args:
    matvec: product between ``A`` and a vector.
    b: pytree.
    ridge: optional ridge regularization.
    init: optional initialization to be used by bicgstab.
    **kwargs: additional keyword arguments for solver.

  Returns:
    pytree with same structure as ``b``.
  """
  if ridge is not None:
    matvec = _make_ridge_matvec(matvec, ridge=ridge)
#   return jsp.sparse.linalg.bicgstab(matvec, b, x0=init, **kwargs)[0]
  return bicgstab(matvec, b, x0=init, **kwargs)