from typing import Any
from typing import Callable
from typing import Optional
from jaxopt._src.linear_solve import _make_rmatvec, _make_ridge_matvec, _normal_matvec

import jax


def solve_cg_custom(matvec: Callable,
             b: Any,
             ridge: Optional[float] = None,
             init: Optional[Any] = None,
             **kwargs) -> Any:
  """Solves ``A x = b`` using conjugate gradient.

  It assumes that ``A`` is  a Hermitian, positive definite matrix.

  Args:
    matvec: product between ``A`` and a vector.
    b: pytree.
    ridge: optional ridge regularization.
    init: optional initialization to be used by conjugate gradient.
    **kwargs: additional keyword arguments for solver.

  Returns:
    pytree with same structure as ``b``.
  """
  if ridge is not None:
    matvec = _make_ridge_matvec(matvec, ridge=ridge)
  return jax.scipy.sparse.linalg.cg(matvec, b, x0=init, **kwargs)


def solve_normal_cg_custom(matvec: Callable,
                    b: Any,
                    ridge: Optional[float] = None,
                    init: Optional[Any] = None,
                    **kwargs) -> Any:
  """Solves the normal equation ``A^T A x = A^T b`` using conjugate gradient.

  This can be used to solve Ax=b using conjugate gradient when A is not
  hermitian, positive definite.

  Args:
    matvec: product between ``A`` and a vector.
    b: pytree.
    ridge: optional ridge regularization.
    init: optional initialization to be used by normal conjugate gradient.
    **kwargs: additional keyword arguments for solver.

  Returns:
    pytree with same structure as ``b``.
  """
  if init is None:
    example_x = b  # This assumes that matvec is a square linear operator.
  else:
    example_x = init

  try:
    rmatvec = _make_rmatvec(matvec, example_x)
  except TypeError:
    raise TypeError("The initialization `init` of solve_normal_cg is "
                    "compulsory when `matvec` is nonsquare. It should "
                    "have the same pytree structure as a solution. "
                    "Typically, a pytree filled with zeros should work.")

  def normal_matvec(x):
    return _normal_matvec(matvec, x)

  if ridge is not None:
    normal_matvec = _make_ridge_matvec(normal_matvec, ridge=ridge)

  Ab = rmatvec(b)  # A.T b

  return jax.scipy.sparse.linalg.cg(normal_matvec, Ab, x0=init, **kwargs)


def solve_bicgstab_custom(matvec: Callable,
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
  return jax.scipy.sparse.linalg.bicgstab(matvec, b, x0=init, **kwargs)

