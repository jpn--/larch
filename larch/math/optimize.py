
import numpy, pandas
from collections import OrderedDict
from ..warning import ignore_warnings

import logging
from ..log import logger_name
logger = logging.getLogger(logger_name)

try:
	from ipywidgets import FloatProgress as ipywidgets_FloatProgress
except ImportError:
	ipywidgets_FloatProgress = ()

def debug_array(name, arr):
	print(f"<{name} shape={arr.shape}>")
	print(arr)
	print(f"</{name}>")


def _approx_fprime_helper(xk, f, epsilon, args=(), f0=None, *, status_widget=None):
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
		if status_widget:
			status_widget("{} / {}".format(k, len(xk)))
	return grad


def _approx_fprime_helper_trailing(xk, f, epsilon, args=(), f0=None):
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
	grad = numpy.zeros(f_shape+(len(xk),), float)
	ei = numpy.zeros((len(xk),), float)
	for k in range(len(xk)):
		ei[k] = 1.0
		d = epsilon * ei
		grad[...,k] = ( numpy.nan_to_num(f(*((xk + d,) + args))) - f0) / d[k]
		ei[k] = 0.0
	return grad


def approx_fprime(xk, f, epsilon=None, trailing=False, *args, status_widget=None):
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
	if epsilon is None:
		epsilon = numpy.sqrt(numpy.finfo(float).eps)
	if trailing:
		return _approx_fprime_helper_trailing(xk, f, epsilon, args=args)
	else:
		return _approx_fprime_helper(xk, f, epsilon, args=args, status_widget=status_widget)

def similarity(a,b, to_zero=None):
	"""
	The similarity between two values or arrays.

	Returns a similarity measure indicating approximately the number of
	significant figures in common between the two values.  Returns 100
	if the values are exactly equal.

	Parameters
	----------
	a,b : numeric or arrays of same size
		Scalar values are converted to numpy arrays.
	to_zero : numeric, optional
		A threshold level for values that can be
		considered to match zero, where if one value is
		exactly zero and the other is not, the similarity
		is equal to the inverse absolute value of the other value
		multiplied by `to_zero`

	Returns
	-------
	ndarray

	"""
	a = numpy.asarray(a)
	b = numpy.asarray(b)
	magnitude = numpy.fmax(numpy.fabs(a),numpy.fabs(b))
	difference = numpy.abs(a-b)
	with numpy.errstate(divide='ignore',invalid='ignore'):
		similar = -numpy.log10(difference/magnitude)
	similar[a==b] = 100
	if to_zero:
		with ignore_warnings():
			similar[a == 0] = (to_zero / difference)[a == 0]
			similar[b == 0] = (to_zero / difference)[b == 0]
	return similar


def _color_poor_similarity(val):
	"""
	Takes a scalar similarity and returns a string with
	the css property `'color: red'` for negative
	strings, black otherwise.
	"""
	if val < 3:
		return 'color: #FF0000'
	if val < 4.5:
		return 'color: #BB0000'
	if val < 6:
		return 'color: #880000'
	return 'color: black'



def check_gradient(func, grad, x0, epsilon=None, names=None, require_sim=None, stylize=True, skip_zeros=True, trailing=False):
	"""Check the gradient.

	Check the correctness of a gradient function by comparing it against a
	(forward) finite-difference approximation of the gradient.

	Parameters
	----------
	func : callable ``func(x0, *args)``
	    Function whose derivative is to be checked.
	grad : callable ``grad(x0, *args)``
	    Gradient of `func`.
	x0 : ndarray
	    Points to check `grad` against forward difference approximation of grad
	    using `func`.
	args : \\*args, optional
	    Extra arguments passed to `func` and `grad`.
	epsilon : float, optional
	    Step size used for the finite difference approximation. It defaults to
	    ``sqrt(numpy.finfo(float).eps)``, which is approximately 1.49e-08.
	names : list or pandas.Index
		A list of names by position in the returned gradient.
	require_sim : float, optional
		If any item has similarity less than this, a GradientCheckError is raise

	Returns
	-------
	err : float
	    The square root of the sum of squares (i.e. the 2-norm) of the
	    difference between ``grad(x0, *args)`` and the finite difference
	    approximation of `grad` using func at the points `x0`.
	"""
	if epsilon is None:
		epsilon = numpy.sqrt(numpy.finfo(float).eps)
	g_f = approx_fprime(x0, func, epsilon, trailing=trailing)
	g_a = grad(x0)

	g_f_ = g_f.reshape(-1)
	if isinstance(g_a, pandas.Series):
		g_a = g_a.values
	g_a_ = g_a.reshape(-1)
	lister = [
			# ('x', x0),
			('finite_diff', g_f_),
			('analytic', g_a_),
			('diff', g_f_ - g_a_),
			('similarity', similarity(g_f_, g_a_, to_zero=0.01)),
		]
	try:
		f = pandas.DataFrame.from_dict(OrderedDict(lister))
	except:
		print(lister)
		raise
	if names is not None and len(names)==g_a_.size:
		f.index = names
	elif names is not None and len(names)**2 == g_a_.size:
		from itertools import product
		f.index = ["{}, {}".format(n1,n2) for n1,n2 in product(names, repeat=2)]
	elif names is not None:
		try:
			f.index = numpy.broadcast_to(names, g_f.shape).reshape(-1)
		except ValueError:
			f.index = numpy.broadcast_to(names, g_f.T.shape).T.reshape(-1)

	if skip_zeros:
		f = f[(f['finite_diff']!=0) | (f['analytic']!=0)]

	if not stylize:
		return f
	try:
		return f.style.applymap(_color_poor_similarity, subset='similarity')
	except ValueError:
		return f


def assert_gradient_check(*args, min_similarity=4, **kwargs):
	kwargs['stylize'] = False
	df = check_gradient(*args, **kwargs)
	assert df['similarity'].min() >= min_similarity