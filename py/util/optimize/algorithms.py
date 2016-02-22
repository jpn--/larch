# Much of this code is borrowed from scipy.optimize, with minor modifications

import warnings
import sys
import numpy
from numpy import (atleast_1d, eye, mgrid, argmin, zeros, shape, squeeze,
				   vectorize, asarray, sqrt, Inf, asfarray, isinf)
import numpy.linalg

from scipy.optimize import OptimizeResult
from scipy.optimize import OptimizeWarning

_status_message = {'success': 'Optimization terminated successfully.',
				   'maxfev': 'Maximum number of function evaluations has '
							  'been exceeded.',
				   'maxiter': 'Maximum number of iterations has been '
							  'exceeded.',
				   'pr_loss': 'Desired error not necessarily achieved due '
							  'to precision loss.'}


class WantsToViolateBounds(Exception):
	def __init__(self, x, nit=None):
		self.x = x
		self.success = False
		self.message = "bounds violation"
		self.nit = nit
	def result(self):
		return OptimizeResult(x=self.x, message=self.message, nit=self.nit, status=-5, success=self.success)



# Borrowed from scipy.optimize, but ignore bogus keys with value None
def _check_unknown_options(unknown_options):
	if unknown_options:
		bogus_keys = set()
		for k in unknown_options.keys():
			if unknown_options[k] is not None and unknown_options[k] != ():
				bogus_keys.add(k)
		if len(bogus_keys) > 0:
			msg = ", ".join(map(str, bogus_keys))
			# Stack level 4: this is called from _minimize_*, which is
			# called from another function in Scipy. Level 4 is the first
			# level in user code.
			warnings.warn("Unknown solver options: %s" % msg, OptimizeWarning, 4)

_epsilon = sqrt(numpy.finfo(float).eps)


# Borrowed from scipy.optimize wholesale
def wrap_function(function, args):
	ncalls = [0]
	if function is None:
		return ncalls, None

	def function_wrapper(*wrapper_args):
		ncalls[0] += 1
		return function(*(wrapper_args + args))

	return ncalls, function_wrapper

# Borrowed from scipy.optimize wholesale
def vecnorm(x, ord=2):
	if ord == Inf:
		return numpy.amax(numpy.abs(x))
	elif ord == -Inf:
		return numpy.amin(numpy.abs(x))
	else:
		return numpy.sum(numpy.abs(x)**ord, axis=0)**(1.0 / ord)

# Borrowed from scipy.optimize wholesale
class _LineSearchError(RuntimeError):
	pass

from scipy.optimize.linesearch import line_search_wolfe1, line_search_wolfe2, LineSearchWarning

# Borrowed from scipy.optimize wholesale
def _line_search_wolfe12(f, fprime, xk, pk, gfk, old_fval, old_old_fval,
						 **kwargs):
	"""
	Same as line_search_wolfe1, but fall back to line_search_wolfe2 if
	suitable step length is not found, and raise an exception if a
	suitable step length is not found.
	Raises
	------
	_LineSearchError
		If no suitable step size is found
	"""
	ret = line_search_wolfe1(f, fprime, xk, pk, gfk,
							 old_fval, old_old_fval,
							 **kwargs)

	if ret[0] is None:
		# line search failed: try different one.
		with warnings.catch_warnings():
			warnings.simplefilter('ignore', LineSearchWarning)
			ret = line_search_wolfe2(f, fprime, xk, pk, gfk,
									 old_fval, old_old_fval)

	if ret[0] is None:
		raise _LineSearchError()

	return ret





# Borrowed from scipy.optimize, edited to take hess as an intialization
#  array for Hk. Note usually hess is callable, but here it can also just be
#  array.
def _minimize_bfgs_1(fun, x0, args=(), jac=None, callback=None,
				   gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None,
				   disp=False, return_all=False, hess=None,
				   **unknown_options):
	"""
	Minimization of scalar function of one or more variables using the
	BFGS algorithm.
	Options
	-------
	disp : bool
		Set to True to print convergence messages.
	maxiter : int
		Maximum number of iterations to perform.
	gtol : float
		Gradient norm must be less than `gtol` before successful
		termination.
	norm : float
		Order of norm (Inf is max, -Inf is min).
	eps : float or ndarray
		If `jac` is approximated, use this value for the step size.
	hess : ndarray
		The initial inverse hessian approximation. If not given, 
		uses eye (like standard BFGS)
	"""
	_check_unknown_options(unknown_options)
	f = fun
	fprime = jac
	epsilon = eps
	retall = return_all

	x0 = asarray(x0).flatten()
	if x0.ndim == 0:
		x0.shape = (1,)
	if maxiter is None:
		maxiter = len(x0) * 200
	func_calls, f = wrap_function(f, args)
	if fprime is None:
		grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
	else:
		grad_calls, myfprime = wrap_function(fprime, args)
	gfk = myfprime(x0)
	k = 0
	N = len(x0)
	I = numpy.eye(N, dtype=int)
	if hess is None:
		Hk = I
	else:
		if callable(hess):
			Hk = hess(x0)
		else:
			Hk = hess
	old_fval = f(x0)
	old_old_fval = None
	xk = x0
	if retall:
		allvecs = [x0]
	sk = [2 * gtol]
	warnflag = 0
	gnorm = vecnorm(gfk, ord=norm)
	while (gnorm > gtol) and (k < maxiter):
		pk = -numpy.dot(Hk, gfk)
		try:
			alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
					 _line_search_wolfe12(f, myfprime, xk, pk, gfk,
										  old_fval, old_old_fval)
		except _LineSearchError:
			# Line search failed to find a better solution.
			warnflag = 2
			break

		xkp1 = xk + alpha_k * pk
		if retall:
			allvecs.append(xkp1)
		sk = xkp1 - xk
		xk = xkp1
		if gfkp1 is None:
			gfkp1 = myfprime(xkp1)

		yk = gfkp1 - gfk
		gfk = gfkp1
		if callback is not None:
			callback(xk)
		k += 1
		gnorm = vecnorm(gfk, ord=norm)
		if (gnorm <= gtol):
			break

		if not numpy.isfinite(old_fval):
			# We correctly found +-Inf as optimal value, or something went
			# wrong.
			warnflag = 2
			break

		try:  # this was handled in numeric, let it remaines for more safety
			rhok = 1.0 / (numpy.dot(yk, sk))
		except ZeroDivisionError:
			rhok = 1000.0
			if disp:
				print("Divide-by-zero encountered: rhok assumed large")
		if isinf(rhok):	 # this is patch for numpy
			rhok = 1000.0
			if disp:
				print("Divide-by-zero encountered: rhok assumed large")
		A1 = I - sk[:, numpy.newaxis] * yk[numpy.newaxis, :] * rhok
		A2 = I - yk[:, numpy.newaxis] * sk[numpy.newaxis, :] * rhok
		Hk = numpy.dot(A1, numpy.dot(Hk, A2)) + (rhok * sk[:, numpy.newaxis] *
												 sk[numpy.newaxis, :])

	fval = old_fval
	if numpy.isnan(fval):
		# This can happen if the first call to f returned NaN;
		# the loop is then never entered.
		warnflag = 2

	if warnflag == 2:
		msg = _status_message['pr_loss']
		if disp:
			print("Warning: " + msg)
			print("			Current function value: %f" % fval)
			print("			Iterations: %d" % k)
			print("			Function evaluations: %d" % func_calls[0])
			print("			Gradient evaluations: %d" % grad_calls[0])

	elif k >= maxiter:
		warnflag = 1
		msg = _status_message['maxiter']
		if disp:
			print("Warning: " + msg)
			print("			Current function value: %f" % fval)
			print("			Iterations: %d" % k)
			print("			Function evaluations: %d" % func_calls[0])
			print("			Gradient evaluations: %d" % grad_calls[0])
	else:
		msg = _status_message['success']
		if disp:
			print(msg)
			print("			Current function value: %f" % fval)
			print("			Iterations: %d" % k)
			print("			Function evaluations: %d" % func_calls[0])
			print("			Gradient evaluations: %d" % grad_calls[0])

	result = OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=func_calls[0],
							njev=grad_calls[0], status=warnflag,
							success=(warnflag == 0), message=msg, x=xk,
							nit=k)
	if retall:
		result['allvecs'] = allvecs
	return result





# Borrowed from scipy.optimize, edited to take bhhh in place of hess for Hk.
def _minimize_bhhh_wolfe(fun, x0, args=(), jac=None, callback=None,
				   gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None,
				   disp=False, return_all=False, hess=None,
				   **unknown_options):
	"""
	Minimization of scalar function of one or more variables using the
	BFGS algorithm.
	Options
	-------
	disp : bool
		Set to True to print convergence messages.
	maxiter : int
		Maximum number of iterations to perform.
	gtol : float
		Gradient norm must be less than `gtol` before successful
		termination.
	norm : float
		Order of norm (Inf is max, -Inf is min).
	eps : float or ndarray
		If `jac` is approximated, use this value for the step size.
	hess : ndarray
		The BHHH array
	"""
	_check_unknown_options(unknown_options)
	f = fun
	fprime = jac
	epsilon = eps
	retall = return_all

	x0 = asarray(x0).flatten()
	if x0.ndim == 0:
		x0.shape = (1,)
	if maxiter is None:
		maxiter = len(x0) * 200
	func_calls, f = wrap_function(f, args)
	if fprime is None:
		grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
	else:
		grad_calls, myfprime = wrap_function(fprime, args)
	gfk = myfprime(x0)
	k = 0
	N = len(x0)
	I = numpy.eye(N, dtype=int)
	if hess is None:
		raise LarchError('BHHH algorithm requires a BHHH function!')
	else:
		Hk = hess(x0)
	old_fval = f(x0)
	old_old_fval = None
	xk = x0
	if retall:
		allvecs = [x0]
	sk = [2 * gtol]
	warnflag = 0
	gnorm = vecnorm(gfk, ord=norm)
	while (gnorm > gtol) and (k < maxiter):
		pk = -numpy.dot(Hk, gfk)
		try:
			alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
					 _line_search_wolfe12(f, myfprime, xk, pk, gfk,
										  old_fval, old_old_fval)
		except _LineSearchError:
			# Line search failed to find a better solution.
			warnflag = 2
			break

		xkp1 = xk + alpha_k * pk
		if retall:
			allvecs.append(xkp1)
		sk = xkp1 - xk
		xk = xkp1
		if gfkp1 is None:
			gfkp1 = myfprime(xkp1)

		yk = gfkp1 - gfk
		gfk = gfkp1
		if callback is not None:
			callback(xk)
		k += 1
		gnorm = vecnorm(gfk, ord=norm)
		if (gnorm <= gtol):
			break

		if not numpy.isfinite(old_fval):
			# We correctly found +-Inf as optimal value, or something went
			# wrong.
			warnflag = 2
			break

		Hk = hess(xk)

	fval = old_fval
	if numpy.isnan(fval):
		# This can happen if the first call to f returned NaN;
		# the loop is then never entered.
		warnflag = 2

	if warnflag == 2:
		msg = _status_message['pr_loss']
		if disp:
			print("Warning: " + msg)
			print("			Current function value: %f" % fval)
			print("			Iterations: %d" % k)
			print("			Function evaluations: %d" % func_calls[0])
			print("			Gradient evaluations: %d" % grad_calls[0])

	elif k >= maxiter:
		warnflag = 1
		msg = _status_message['maxiter']
		if disp:
			print("Warning: " + msg)
			print("			Current function value: %f" % fval)
			print("			Iterations: %d" % k)
			print("			Function evaluations: %d" % func_calls[0])
			print("			Gradient evaluations: %d" % grad_calls[0])
	else:
		msg = _status_message['success']
		if disp:
			print(msg)
			print("			Current function value: %f" % fval)
			print("			Iterations: %d" % k)
			print("			Function evaluations: %d" % func_calls[0])
			print("			Gradient evaluations: %d" % grad_calls[0])

	result = OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=func_calls[0],
							njev=grad_calls[0], status=warnflag,
							success=(warnflag == 0), message=msg, x=xk,
							nit=k)
	if retall:
		result['allvecs'] = allvecs
	return result







def _line_search_evaluation(fun, xk, direction, steplength, previous_best, max_bound=None, min_bound=None):
	
	if max_bound is not None or min_bound is not None:
		shifted_bounds = numpy.stack((max_bound-xk,min_bound-xk))
		directed_bounds = shifted_bounds[numpy.signbit(direction).astype(int),range(shifted_bounds.shape[1])]
		if (numpy.abs(directed_bounds) / numpy.abs(xk)).min() < 1e-10:
			raise WantsToViolateBounds(xk)
		bounded_max_step = (directed_bounds/direction).min()
		use_step = min(bounded_max_step, steplength)
	else:
		use_step = steplength

	trial_point = xk + direction*use_step
	new_value = fun(trial_point)
	if numpy.isfinite(new_value):
		improvement =  previous_best - new_value
	else:
		improvement = 0
	return new_value, improvement, trial_point, use_step

_LINE_SEARCH_SUCCESS_BIG    = 2
_LINE_SEARCH_SUCCESS_SMALL	= 1
_LINE_SEARCH_NO_IMPROVEMENT	= 0
_LINE_SEARCH_FAIL         	=-1
_LINE_SEARCH_ERROR_NAN   	=-2


def _simple_line_search(fun, xk, f_at_xk, direction, init_step, max_step, min_step, extend_step, reduce_step, max_bound=None, min_bound=None, logger=None):
	step = init_step
	best_value = f_at_xk
	new_value, improvement, trial_point, used_step = _line_search_evaluation(fun, xk, direction, step, best_value, max_bound, min_bound)
	total_improvement = improvement
	if numpy.isnan(new_value):
		status = _LINE_SEARCH_ERROR_NAN
	if improvement > 0:
		if step == used_step:
			status = _LINE_SEARCH_SUCCESS_BIG
		else:
			status = _LINE_SEARCH_SUCCESS_SMALL
		best_value = new_value
	else:
		status = _LINE_SEARCH_NO_IMPROVEMENT
	step = used_step

	# When First Step was an improvement and did not hit the bound, potentially try some more
	if status==_LINE_SEARCH_SUCCESS_BIG and step < max_step:
		while improvement>0 and step < max_step:
			step *= extend_step;
			new_value, improvement, trial_point, used_step = _line_search_evaluation(fun, xk, direction, step, best_value, max_bound, min_bound)
			if improvement>0:
				total_improvement += improvement
				best_value = new_value

		if best_value != new_value:
			# we extended too far, need to back up a step
			step /= extend_step;
			new_value, improvement, trial_point, used_step = _line_search_evaluation(fun, xk, direction, step, best_value, max_bound, min_bound)

	# When First Step was NOT an improvement
	while status in (_LINE_SEARCH_NO_IMPROVEMENT, _LINE_SEARCH_ERROR_NAN):
		step *= reduce_step;
		if step < min_step:
			status = _LINE_SEARCH_FAIL
			break
		
		new_value, improvement, trial_point, used_step = _line_search_evaluation(fun, xk, direction, step, best_value, max_bound, min_bound)
		total_improvement = improvement

		if numpy.isnan(new_value):
			status = _LINE_SEARCH_ERROR_NAN
		if step < min_step:
			status = _LINE_SEARCH_FAIL
		if improvement > 0:
			status = _LINE_SEARCH_SUCCESS_SMALL
			best_value = new_value
	
	if numpy.isnan(new_value):
		status = _LINE_SEARCH_ERROR_NAN;

	if status in (_LINE_SEARCH_SUCCESS_SMALL, _LINE_SEARCH_SUCCESS_BIG):
		if logger:
			logger.log(25, "line search found improvement to {} ({:+0.5g}) using stepsize {}".format(new_value, total_improvement, used_step))
		return status, trial_point, step, best_value
	else:
		return status, xk, 0, best_value








def _minimize_bhhh_simple(fun, x0, args=(), jac=None, callback=None,
				   gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None,
				   disp=False, return_all=False, hess=None,
				   logger=None, bounds=None,
				   **unknown_options):
	"""
	Minimization of scalar function of one or more variables using the
	BFGS algorithm.
	Options
	-------
	disp : bool
		Set to True to print convergence messages.
	maxiter : int
		Maximum number of iterations to perform.
	gtol : float
		Gradient norm must be less than `gtol` before successful
		termination.
	norm : float
		Order of norm (Inf is max, -Inf is min).
	eps : float or ndarray
		If `jac` is approximated, use this value for the step size.
	hess : ndarray
		The BHHH array
	"""
	_check_unknown_options(unknown_options)
	f = fun
	fprime = jac
	epsilon = eps
	retall = return_all

	if bounds:
		max_bound = numpy.empty_like(x0)
		min_bound = numpy.empty_like(x0)
		for bn,(bmin,bmax) in enumerate(bounds):
			min_bound[bn] = -numpy.inf if bmin is None else bmin
			max_bound[bn] =  numpy.inf if bmax is None else bmax
	else:
		max_bound, min_bound = None, None

	x0 = asarray(x0).flatten()
	if x0.ndim == 0:
		x0.shape = (1,)
	if maxiter is None:
		maxiter = len(x0) * 200
	func_calls, f = wrap_function(f, args)
	if fprime is None:
		grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
	else:
		grad_calls, myfprime = wrap_function(fprime, args)
	gfk = myfprime(x0)
	k = 0
	N = len(x0)
	I = numpy.eye(N, dtype=int)
	if hess is None:
		raise LarchError('BHHH algorithm requires a BHHH function!')
	else:
		Hk = hess(x0)
	old_fval = f(x0)
	old_old_fval = None
	xk = x0
	if retall:
		allvecs = [x0]
	warnflag = 0
	gnorm = vecnorm(gfk, ord=norm)
	while (gnorm > gtol) and (k < maxiter):
		pk = -numpy.dot(Hk, gfk)
		
#		print("\n pk=\n",pk,"\n")
#		print("\n gfk=\n",gfk,"\n")
#		print("\n Hk=\n",Hk,"\n")

		try:
			status, xkp1, alpha_k, old_fval = _simple_line_search(f, xk, old_fval, pk, 1, 4, 1e-6, 2, 0.5, max_bound=max_bound, min_bound=min_bound, logger=logger)
		except WantsToViolateBounds as violate:
			warnflag = 3
			break
		if status==_LINE_SEARCH_FAIL:
			warnflag = 4
			break
		if retall:
			allvecs.append(xkp1)
		xk = xkp1
		gfkp1 = myfprime(xkp1)

		gfk = gfkp1
		if callback is not None:
			callback(xk)
		k += 1
		gnorm = vecnorm(gfk, ord=norm)
		if (gnorm <= gtol):
			break

		if not numpy.isfinite(old_fval):
			# We correctly found +-Inf as optimal value, or something went
			# wrong.
			warnflag = 2
			break

		Hk = hess(xk)

	fval = old_fval
	if numpy.isnan(fval):
		# This can happen if the first call to f returned NaN;
		# the loop is then never entered.
		warnflag = 2

	if warnflag == 2:
		msg = _status_message['pr_loss']
		if disp:
			print("Warning: " + msg)
			print("			Current function value: %f" % fval)
			print("			Iterations: %d" % k)
			print("			Function evaluations: %d" % func_calls[0])
			print("			Gradient evaluations: %d" % grad_calls[0])

	elif warnflag == 3:
		msg = "unable to make reasonable improvements due to bounds"
		if disp:
			print("Warning: " + msg)
			print("			Current function value: %f" % fval)
			print("			Iterations: %d" % k)
			print("			Function evaluations: %d" % func_calls[0])
			print("			Gradient evaluations: %d" % grad_calls[0])

	elif warnflag == 4:
		msg = "unable to make reasonable improvements due to some kind of problem"
		if disp:
			print("Warning: " + msg)
			print("			Current function value: %f" % fval)
			print("			Iterations: %d" % k)
			print("			Function evaluations: %d" % func_calls[0])
			print("			Gradient evaluations: %d" % grad_calls[0])

	elif k >= maxiter:
		warnflag = 1
		msg = _status_message['maxiter']
		if disp:
			print("Warning: " + msg)
			print("			Current function value: %f" % fval)
			print("			Iterations: %d" % k)
			print("			Function evaluations: %d" % func_calls[0])
			print("			Gradient evaluations: %d" % grad_calls[0])
	else:
		msg = _status_message['success']
		if disp:
			print(msg)
			print("			Current function value: %f" % fval)
			print("			Iterations: %d" % k)
			print("			Function evaluations: %d" % func_calls[0])
			print("			Gradient evaluations: %d" % grad_calls[0])

	result = OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=func_calls[0],
							njev=grad_calls[0], status=warnflag,
							success=(warnflag == 0), message=msg, x=xk,
							nit=k)
	if retall:
		result['allvecs'] = allvecs
	return result


#def _minimize_bhhh_wrap(model, *ignore_args, **ignore_kwargs):
#	model._minimize_bhhh() TODO


