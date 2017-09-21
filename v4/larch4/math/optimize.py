
import numpy


def debug_array(name, arr):
	print(f"<{name} shape={arr.shape}>")
	print(arr)
	print(f"</{name}>")


def _approx_fprime_helper(xk, f, epsilon, args=(), f0=None):
	"""
	See ``approx_fprime``.  An optional initial function value arg is added.

	"""
	if f0 is None:
		f0_ = f(*((xk,) + args))
		try:
			f0 = f0_.copy()
		except AttributeError:
			f0 = f0_
	numpy.nan_to_num(f0, copy=False)
	try:
		f_shape = f0.shape
	except AttributeError:
		f_shape = ()
	grad = numpy.zeros((len(xk),)+f_shape, float)
	ei = numpy.zeros((len(xk),), float)
	for k in range(len(xk)):
		ei[k] = 1.0
		d = epsilon * ei
		grad[k] = ( numpy.nan_to_num(f(*((xk + d,) + args))) - f0) / d[k]
		ei[k] = 0.0
	return grad


def approx_fprime(xk, f, epsilon, *args):
	"""Finite-difference approximation of the gradient of a scalar function.

	Parameters
	----------
	xk : array_like
	    The coordinate vector at which to determine the gradient of `f`.
	f : callable
	    The function of which to determine the gradient (partial derivatives).
	    Should take `xk` as first argument, other arguments to `f` can be
	    supplied in ``*args``.  Should return a scalar, the value of the
	    function at `xk`.
	epsilon : array_like
	    Increment to `xk` to use for determining the function gradient.
	    If a scalar, uses the same finite difference delta for all partial
	    derivatives.  If an array, should contain one value per element of
	    `xk`.
	\\*args : args, optional
	    Any other arguments that are to be passed to `f`.

	Returns
	-------
	grad : ndarray
	    The partial derivatives of `f` to `xk`.

	See Also
	--------
	check_grad : Check correctness of gradient function against approx_fprime.

	Notes
	-----
	The function gradient is determined by the forward finite difference
	formula::

	             f(xk[i] + epsilon[i]) - f(xk[i])
	    f'[i] = ---------------------------------
	                        epsilon[i]

	The main use of `approx_fprime` is in scalar function optimizers like
	`fmin_bfgs`, to determine numerically the Jacobian of a function.

	"""
	return _approx_fprime_helper(xk, f, epsilon, args=args)
