
import numpy
import pandas

include "../general_precision.pxi"
from ..general_precision import l4_float_dtype
from ..general_precision cimport l4_float_t

from .parameter_frame cimport ParameterFrame
from .persist_flags cimport *
from ..exceptions import MissingDataError

import logging
from ..log import logger_name
logger = logging.getLogger(logger_name+'.model')


cdef class AbstractChoiceModel(ParameterFrame):

	def __init__(
			self, *,
			parameters=None,
			frame=None,
			title=None,
	):

		self._cached_loglike_nil = 0
		self._cached_loglike_null = 0
		self._cached_loglike_constants_only = 0
		self._cached_loglike_best = -numpy.inf
		self._most_recent_estimation_result = None

		self._dashboard = None

		super().__init__(
			parameters=parameters,
			frame=frame,
			title=title,
		)

	def _get_cached_loglike_values(self):
		return {
			'nil': self._cached_loglike_nil,
			'null': self._cached_loglike_null,
			'constants_only': self._cached_loglike_constants_only,
			'best': self._cached_loglike_best,
		}

	def mangle(self, *args, **kwargs):
		self.clear_best_loglike()
		super().mangle(*args, **kwargs)

	def clear_best_loglike(self):
		if self._frame is not None and 'best' in self._frame.columns:
			del self._frame['best']
		self._cached_loglike_best = -numpy.inf

	def loglike2(
			self,
			x=None,
			*,
			start_case=0,
			stop_case=-1,
			step_case=1,
			persist=0,
			leave_out=-1,
			keep_only=-1,
			subsample=-1,
			return_series=True,
			probability_only=False,
	):
		"""
		Compute a log likelihood value and it first derivative.

		Parameters
		----------
		x : {'null', 'init', 'best', array-like, dict, scalar}, optional
			Values for the parameters.  See :ref:`set_values` for details.
		start_case : int, default 0
			The first case to include in the log likelihood computation.
			To include all cases, start from 0 (the default).
		stop_case : int, default -1
			One past the last case to include in the log likelihood
			computation.  This is processed as usual for Python slicing
			and iterating, and negative values count backward from the
			end.  To include all cases, end at -1 (the default).
		step_case : int, default 1
			The step size of the case iterator to use in likelihood
			calculation.  This is processed as usual for Python slicing
			and iterating.  To include all cases, step by 1 (the default).
		persist : int, default 0
			Whether to return a variety of internal and intermediate
			arrays in the result dictionary. If set to 0, only the
			final `ll` value is included.
		leave_out, keep_only, subsample : int, optional
			Settings for cross validation calculations.
			If `leave_out` and `subsample` are set, then case rows
			where rownumber % subsample == leave_out are dropped.
			If `keep_only` and `subsample` are set, then only case
			rows where rownumber % subsample == keep_only are used.
		return_series : bool
			Deprecated, no effect.  Derivatives are always returned
			as a Series.

		Returns
		-------
		dictx
			The log likelihood is given by key 'll' and the first
			derivative by key 'dll'. Other arrays are also included
			if `persist` is set to True.

		"""
		raise NotImplementedError("abstract base class, use a derived class instead")

	def _loglike2_tuple(self, *args, **kwargs):
		"""
		Compute a log likelihood value and it first derivative.

		This is a convenience function that returns these values
		in a 2-tuple instead of a dictx, for compatibility with
		scipy.optimize.  It accepts all the same input arguments
		as :ref:`loglike2`.

		Returns
		-------
		Tuple[float, array-like]
			The log likelihood is given by key 'll' and the first
			derivative by key 'dll'. Other arrays are also included
			if `persist` is set to True.

		"""
		result = self.loglike2(*args, **kwargs)
		return result.ll, result.dll

	def loglike2_bhhh(
			self,
			x=None,
			*,
			return_series=False,
			start_case=0, stop_case=-1, step_case=1,
			persist=0,
			leave_out=-1, keep_only=-1, subsample=-1,
	):
		"""
		Compute a log like, it first deriv, and the BHHH approx of the Hessian.

		The `BHHH algorithm <https://en.wikipedia.org/wiki/Berndt–Hall–Hall–Hausman_algorithm>`
		employs a matrix computated as the sum of the casewise
		outer product of the gradient, to approximate the
		hessian matrix.

		Parameters
		----------
		x : {'null', 'init', 'best', array-like, dict, scalar}, optional
			Values for the parameters.  See :ref:`set_values` for details.
		start_case : int, default 0
			The first case to include in the log likelihood computation.
			To include all cases, start from 0 (the default).
		stop_case : int, default -1
			One past the last case to include in the log likelihood
			computation.  This is processed as usual for Python slicing
			and iterating, and negative values count backward from the
			end.  To include all cases, end at -1 (the default).
		step_case : int, default 1
			The step size of the case iterator to use in likelihood
			calculation.  This is processed as usual for Python slicing
			and iterating.  To include all cases, step by 1 (the default).
		persist : int, default False
			Whether to return a variety of internal and intermediate
			arrays in the result dictionary. If set to 0, only the final
			`ll` value is included.
		leave_out, keep_only, subsample : int, optional
			Settings for cross validation calculations.
			If `leave_out` and `subsample` are set, then case rows where
			rownumber % subsample == leave_out are dropped. If `keep_only`
			and `subsample` are set, then only case rows where
			rownumber % subsample == keep_only are used.
		return_series : bool
			Deprecated, no effect.  Derivatives are always returned
			as a Series.

		Returns
		-------
		dictx
			The log likelihood is given by key 'll', the first derivative
			by key 'dll', and the BHHH matrix by 'bhhh'. Other arrays are
			also included if `persist` is set to a non-zero value.
		"""
		raise NotImplementedError("abstract base class, use a derived class instead")

	def _loglike2_bhhh_tuple(self, *args, **kwargs):
		"""
		Compute a log likelihood value, its first derivative, and the BHHH approximation of the Hessian.

		This is a convenience function that returns these values in a 3-tuple instead of a Dict,
		for compatibility with scipy.optimize.  It accepts all the same input arguments as :ref:`loglike2_bhhh`.

		"""
		result = self.loglike2_bhhh(*args, **kwargs)
		return result.ll, result.dll, result.bhhh

	def bhhh(self, x=None, *, return_series=False):
		return self.loglike2_bhhh(x=x,return_series=return_series).bhhh

	def d_probability(
			self,
			x=None,
			start_case=0, stop_case=-1, step_case=1,
			leave_out=-1, keep_only=-1, subsample=-1,
	):
		"""
		Compute the partial derivative of probability w.r.t. the parameters.

		Parameters
		----------
		x
		start_case
		stop_case
		step_case
		leave_out
		keep_only
		subsample

		Returns
		-------
		ndarray
		"""
		return self.loglike2(
			x=x,
			start_case=start_case, stop_case=stop_case, step_case=step_case,
			leave_out=leave_out, keep_only=keep_only, subsample=subsample,
			persist=PERSIST_D_PROBABILITY,
		).dprobability


	def _free_slots_inverse_matrix(self, matrix):
		try:
			take = numpy.full_like(matrix, True, dtype=bool)
			dense_s = len(self.pvals)
			for i in range(dense_s):
				ii = self.pnames[i]
				#if self.pf.loc[ii, 'holdfast'] or (self.pf.loc[ii, 'value'] >= self.pf.loc[ii, 'maximum']) or (self.pf.loc[ii, 'value'] <= self.pf.loc[ii, 'minimum']):
				if self.pf.loc[ii, 'holdfast']:
					take[i, :] = False
					take[:, i] = False
					dense_s -= 1
			hess_taken = matrix[take].reshape(dense_s, dense_s)
			from ..linalg import general_inverse
			try:
				invhess = general_inverse(hess_taken)
			except numpy.linalg.linalg.LinAlgError:
				invhess = numpy.full_like(hess_taken, numpy.nan, dtype=numpy.float64)
			result = numpy.full_like(matrix, 0, dtype=numpy.float64)
			result[take] = invhess.reshape(-1)
			return result
		except:
			logger.exception("error in AbstractChoiceModel._free_slots_inverse_matrix")
			raise

	def _bhhh_simple_direction(self, *args, **kwargs):
		bhhh_inv = self._free_slots_inverse_matrix(self.bhhh(*args))
		return numpy.dot(self.d_loglike(*args), bhhh_inv)

	def simple_step_bhhh(self, steplen=1.0, printer=None, leave_out=-1, keep_only=-1, subsample=-1):
		"""
		Makes one step using the BHHH algorithm.

		Parameters
		----------
		steplen: float
		printer: callable

		Returns
		-------
		loglike, convergence_tolerance
		"""
		current = self.pvals.copy()
		current_ll, current_dll, current_bhhh = self._loglike2_bhhh_tuple(leave_out=leave_out,
																		  keep_only=keep_only,
																		  subsample=subsample,
																		  )
		bhhh_inv = self._free_slots_inverse_matrix(current_bhhh)
		direction = numpy.dot(current_dll, bhhh_inv)
		tolerance = numpy.dot(direction, current_dll)
		while True:
			self.set_values(current + direction * steplen)
			val = self.loglike()
			if val > current_ll: break
			steplen *= 0.5
			if steplen < 0.01: break
		if val <= current_ll:
			self.set_values(current)
			raise RuntimeError("simple step bhhh failed")
		if printer is not None:
			printer("simple step bhhh {} to gain {}".format(steplen, val - current_ll))
		return val, tolerance

	def jumpstart_bhhh(self, steplen=0.5, jumpstart=0, jumpstart_split=5, leave_out=-1, keep_only=-1, subsample=-1):
		"""
		Jump start optimization

		Parameters
		----------
		steplen
		jumpstart
		jumpstart_split

		"""
		for jump in range(jumpstart):
			j_pvals = self.pvals.copy()
			n_cases = self._dataframes._n_cases()
			jump_breaks = list(range(0,n_cases, n_cases//jumpstart_split+(1 if n_cases%jumpstart_split else 0))) + [-1]
			for j0,j1 in zip(jump_breaks[:-1],jump_breaks[1:]):
				j_ll, current_dll, current_bhhh = self._loglike2_bhhh_tuple(start_case=j0, stop_case=j1,
																			leave_out=leave_out, keep_only=keep_only,
																			subsample=subsample)
				bhhh_inv = self._free_slots_inverse_matrix(current_bhhh)
				direction = numpy.dot(current_dll, bhhh_inv)
				self.set_values(j_pvals + direction * steplen)



	def simple_fit_bhhh(
			self,
			steplen=1.0,
			printer=None,
			ctol=1e-4,
			maxiter=100,
			callback=None,
			jumpstart=0,
			jumpstart_split=5,
			minimum_steplen=0.0001,
			leave_out=-1,
			keep_only=-1,
			subsample=-1,
	):
		"""
		Makes a series of steps using the BHHH algorithm.

		Parameters
		----------
		steplen: float
		printer: callable

		Returns
		-------
		loglike, convergence_tolerance, n_iters, steps
		"""
		current_pvals = self.pvals.copy()
		iter = 0
		steps = []

		if jumpstart:
			self.jumpstart_bhhh(jumpstart=jumpstart, jumpstart_split=jumpstart_split)
			iter += jumpstart

		current_ll, current_dll, current_bhhh = self._loglike2_bhhh_tuple(leave_out=leave_out,
																		  keep_only=keep_only,
																		  subsample=subsample,
																		  )
		bhhh_inv = self._free_slots_inverse_matrix(current_bhhh)
		direction = numpy.dot(current_dll, bhhh_inv)
		tolerance = numpy.dot(direction, current_dll)

		while abs(tolerance) > ctol and iter < maxiter:
			iter += 1
			while True:
				self.set_values(current_pvals + direction * steplen)
				proposed_ll, proposed_dll, proposed_bhhh = self._loglike2_bhhh_tuple(leave_out=leave_out,
																					 keep_only=keep_only,
																					 subsample=subsample,
																					 )
				if proposed_ll > current_ll: break
				steplen *= 0.5
				if steplen < minimum_steplen: break
			if proposed_ll <= current_ll:
				self.set_values(current_pvals)
				raise RuntimeError(f"simple step bhhh failed\ndirection = {str(direction)}")
			if printer is not None:
				printer("simple step bhhh {} to gain {}".format(steplen, proposed_ll - current_ll))
			steps.append(steplen)

			current_ll, current_dll, current_bhhh = proposed_ll, proposed_dll, proposed_bhhh
			current_pvals = self.pvals.copy()
			if callback is not None:
				callback(current_pvals)
			bhhh_inv = self._free_slots_inverse_matrix(current_bhhh)
			direction = numpy.dot(current_dll, bhhh_inv)
			tolerance = numpy.dot(direction, current_dll)

		if abs(tolerance) <= ctol:
			message = "Optimization terminated successfully."
		else:
			message = f"Optimization terminated after {iter} iterations."

		return current_ll, tolerance, iter, numpy.asarray(steps), message

	def _bhhh_direction_and_convergence_tolerance(self, *args):
		bhhh_inv = self._free_slots_inverse_matrix(self.bhhh(*args))
		_1 = self.d_loglike(*args)
		direction = numpy.dot(_1, bhhh_inv)
		return direction, numpy.dot(direction, _1)

	def loglike3(self, x=None, **kwargs):
		"""
		Compute a log likelihood value, it first derivative, and the finite-difference approximation of the Hessian.

		See :ref:`loglike2` for a description of arguments.

		Returns
		-------
		Dict
			The log likelihood is given by key 'll', the first derivative by key 'dll', and the second derivative by 'd2ll'.
			Other arrays are also included if `persist` is set to True.

		"""
		part = self.loglike2(x=x, **kwargs)
		from ..math.optimize import approx_fprime
		part['d2ll'] = approx_fprime(self.pvals, lambda y: self.d_loglike(y, **kwargs))
		return part

	def neg_loglike2(self, x=None, start_case=0, stop_case=-1, step_case=1, leave_out=-1, keep_only=-1, subsample=-1):
		result = self.loglike2(
			x=x,
			start_case=start_case, stop_case=stop_case, step_case=step_case,
			leave_out=leave_out, keep_only=keep_only, subsample=subsample
		)
		return (-result.ll, -result.dll)

	def neg_loglike3(self, *args, **kwargs):
		from ..util import dictx
		result = dictx()
		part = self.loglike3(*args, **kwargs)
		return (-result.ll, -result.dll, -result.d2ll)

	def _check_if_best(self, computed_ll):
		if computed_ll > self._cached_loglike_best:
			self._cached_loglike_best = computed_ll
			self._frame['best'] = self._frame['value']

	def loglike(
			self,
			x=None,
			*,
			start_case=0, stop_case=-1, step_case=1,
			persist=0,
			leave_out=-1, keep_only=-1, subsample=-1,
			probability_only=False,
	):
		"""
		Compute a log likelihood value.

		Parameters
		----------
		x : {'null', 'init', 'best', array-like, dict, scalar}, optional
			Values for the parameters.  See :ref:`set_values` for details.
		start_case : int, default 0
			The first case to include in the log likelihood computation.  To include all
			cases, start from 0 (the default).
		stop_case : int, default -1
			One past the last case to include in the log likelihood computation.  This is processed as usual for
			Python slicing and iterating, and negative values count backward from the end.  To include all cases,
			end at -1 (the default).
		step_case : int, default 1
			The step size of the case iterator to use in likelihood calculation.  This is processed as usual for
			Python slicing and iterating.  To include all cases, step by 1 (the default).
		persist : int, default 0
			Whether to return a variety of internal and intermediate arrays in the result dictionary.
			If set to 0, only the final `ll` value is included.
		leave_out, keep_only, subsample : int, optional
			Settings for cross validation calculations.
			If `leave_out` and `subsample` are set, then case rows where rownumber % subsample == leave_out are dropped.
			If `keep_only` and `subsample` are set, then only case rows where rownumber % subsample == keep_only are used.
		probability_only : bool, default False
			Compute only the probability and ignore the likelihood.  If this setting is active, the
			dataframes need not include the "actual" choice data.

		Returns
		-------
		dictx or float or array
			The log likelihood or the probability.  Other arrays are also included if `persist` is set to True.
		"""
		raise NotImplementedError("abstract base class, use a derived class instead")

	def d_loglike(self, x=None, *, start_case=0, stop_case=-1, step_case=1, leave_out=-1, keep_only=-1, subsample=-1,):
		"""
		Compute the first derivative of log likelihood with respect to the parameters.

		Parameters
		----------
		x : {'null', 'init', 'best', array-like, dict, scalar}, optional
			Values for the parameters.  See :ref:`set_values` for details.
		start_case : int, default 0
			The first case to include in the log likelihood computation.  To include all
			cases, start from 0 (the default).
		stop_case : int, default -1
			One past the last case to include in the log likelihood computation.  This is processed as usual for
			Python slicing and iterating, and negative values count backward from the end.  To include all cases,
			end at -1 (the default).
		step_case : int, default 1
			The step size of the case iterator to use in likelihood calculation.  This is processed as usual for
			Python slicing and iterating.  To include all cases, step by 1 (the default).
		leave_out, keep_only, subsample : int, optional
			Settings for cross validation calculations.
			If `leave_out` and `subsample` are set, then case rows where rownumber % subsample == leave_out are dropped.
			If `keep_only` and `subsample` are set, then only case rows where rownumber % subsample == keep_only are used.

		Returns
		-------
		pandas.Series
			First derivatives of log likelihood with respect to the parameters (given as the index for the Series).

		"""
		return self.loglike2(x,start_case=start_case,stop_case=stop_case,step_case=step_case,
							 leave_out=leave_out, keep_only=keep_only, subsample=subsample,).dll

	def d2_loglike(self, x=None, *, start_case=0, stop_case=-1, step_case=1, leave_out=-1, keep_only=-1, subsample=-1,):
		"""
		Compute the (approximate) second derivative of log likelihood with respect to the parameters.

		Parameters
		----------
		x : {'null', 'init', 'best', array-like, dict, scalar}, optional
			Values for the parameters.  See :ref:`set_values` for details.
		start_case : int, default 0
			The first case to include in the log likelihood computation.  To include all
			cases, start from 0 (the default).
		stop_case : int, default -1
			One past the last case to include in the log likelihood computation.  This is processed as usual for
			Python slicing and iterating, and negative values count backward from the end.  To include all cases,
			end at -1 (the default).
		step_case : int, default 1
			The step size of the case iterator to use in likelihood calculation.  This is processed as usual for
			Python slicing and iterating.  To include all cases, step by 1 (the default).
		leave_out, keep_only, subsample : int, optional
			Settings for cross validation calculations.
			If `leave_out` and `subsample` are set, then case rows where rownumber % subsample == leave_out are dropped.
			If `keep_only` and `subsample` are set, then only case rows where rownumber % subsample == keep_only are used.

		Returns
		-------
		pandas.DataFrame
			First derivatives of log likelihood with respect to the parameters (given as both the columns and
			index for the DataFrame).

		"""
		return self.loglike3(x,start_case=start_case,stop_case=stop_case,step_case=step_case,
							 leave_out=leave_out, keep_only=keep_only, subsample=subsample,).d2ll

	def check_d_loglike(self, stylize=True, skip_zeros=False):
		"""
		Check that the analytic and finite-difference gradients are approximately equal.

		Primarily used for debugging.

		Parameters
		----------
		stylize : bool, default True
			See :ref:`check_gradient` for details.
		skip_zeros : bool, default False
			See :ref:`check_gradient` for details.

		Returns
		-------
		pandas.DataFrame or Stylized DataFrame
		"""
		from ..math.optimize import check_gradient
		IF DOUBLE_PRECISION:
			epsilon=numpy.sqrt(numpy.finfo(float).eps)
		ELSE:
			epsilon=0.0001
		return check_gradient(self.loglike, self.d_loglike, self.pvals.copy(), names=self.pnames, stylize=stylize, skip_zeros=skip_zeros, epsilon=epsilon)

	@property
	def n_cases(self):
		"""int : The number of cases in the attached dataframes."""
		raise NotImplementedError("abstract base class, use a derived class instead")

	def total_weight(self):
		"""float : The total weight of cases in the attached dataframes."""
		raise NotImplementedError("abstract base class, use a derived class instead")

	def maximize_loglike(
			self,
			method='bhhh',
			quiet=False,
			screen_update_throttle=2,
			final_screen_update=True,
			check_for_overspecification=True,
			return_tags=False,
			reuse_tags=None,
			iteration_number=0,
			iteration_number_tail="",
			options=None,
			maxiter=None,
			jumpstart=0,
			jumpstart_split=5,
			leave_out=-1,
			keep_only=-1,
			subsample=-1,
			**kwargs,
	):
		from ..util.timesize import Timer
		from scipy.optimize import minimize
		from .. import _doctest_mode_
		from ..util.rate_limiter import NonBlockingRateLimiter
		from ..util.display import display_head, display_p

		if self.dataframes is None:
			raise ValueError("you must load data first -- try Model.load_data()")

		if _doctest_mode_:
			from ..model import Model
			if type(self) == Model:
				self.unmangle()
				self._frame.sort_index(inplace=True)
				self.unmangle(True)

		if options is None:
			options = {}
		if maxiter is not None:
			options['maxiter'] = maxiter

		timer = Timer()
		if isinstance(screen_update_throttle, NonBlockingRateLimiter):
			throttle_gate = screen_update_throttle
		else:
			throttle_gate = NonBlockingRateLimiter(screen_update_throttle)

		if throttle_gate and not quiet and not _doctest_mode_:
			if reuse_tags is None:
				tag1 = display_head(f'Iteration 000 {iteration_number_tail}', level=3)
				tag2 = display_p(f'LL = {self.loglike()}')
				tag3 = display_p('...')
			else:
				tag1, tag2, tag3 = reuse_tags
		else:
			tag1 = lambda *ax, **kx: None
			tag2 = lambda *ax, **kx: None
			tag3 = lambda *ax, **kx: None

		def callback(x, status=None):
			nonlocal iteration_number, throttle_gate
			iteration_number += 1
			if throttle_gate:
				#clear_output(wait=True)
				tag1.update(f'Iteration {iteration_number:03} {iteration_number_tail}')
				tag2.update(f'LL = {self._cached_loglike_best}')
				tag3.update(self.pf)
			return False

		if quiet or _doctest_mode_:
			callback = None

		try:
			if method=='bhhh':
				max_iter = options.get('maxiter',100)
				stopping_tol = options.get('ctol',1e-5)

				current_ll, tolerance, iter_bhhh, steps_bhhh, message = self.simple_fit_bhhh(
					ctol=stopping_tol,
					maxiter=max_iter,
					callback=callback,
					jumpstart=jumpstart,
					jumpstart_split=jumpstart_split,
					leave_out=leave_out,
					keep_only=keep_only,
					subsample=subsample,
				)
				raw_result = {
					'loglike':current_ll,
					'x': self.pvals,
					'tolerance':tolerance,
					'steps':steps_bhhh,
					'message':message,
				}
			else:

				bounds = None
				if method in ('SLSQP', 'L-BFGS-B', 'TNC', 'trust-constr'):
					bounds = self.pbounds

				raw_result = minimize(
					self.neg_loglike2,
					self.pvals,
					args=(0, -1, 1, leave_out, keep_only, subsample), # start_case, stop_case, step_case, leave_out, keep_only, subsample
					method=method,
					jac=True,
					bounds=bounds,
					callback=callback,
					options=options,
					**kwargs
				)
		except:
			tag1.update(f'Iteration {iteration_number:03} [Exception] {iteration_number_tail}', force=True)
			tag3.update(self.pf, force=True)
			raise

		timer.stop()

		if final_screen_update and not quiet and not _doctest_mode_:
			tag1.update(f'Iteration {iteration_number:03} [Converged] {iteration_number_tail}', force=True)
			tag2.update(f'LL = {self.loglike()}', force=True)
			tag3.update(self.pf, force=True)

		#if check_for_overspecification:
		#	self.check_for_possible_overspecification()

		from ..util import dictx
		result = dictx()
		for k,v in raw_result.items():
			if k == 'fun':
				result['loglike'] = -v
			elif k == 'jac':
				result['d_loglike'] = pandas.Series(-v, index=self.pnames)
			elif k == 'x':
				result['x'] = pandas.Series(v, index=self.pnames)
			else:
				result[k] = v
		result['elapsed_time'] = timer.elapsed()
		result['method'] = method
		result['n_cases'] = self.n_cases
		result['iteration_number'] = iteration_number

		if 'loglike' in result:
			result['logloss'] = -result['loglike'] / self.total_weight()

		if _doctest_mode_:
			result['__verbose_repr__'] = True

		self._most_recent_estimation_result = result

		if return_tags:
			return result, tag1, tag2, tag3

		return result

	def estimate(self, dataservice=None, autoscale_weights=True, **kwargs):
		"""
		A convenience method to load data, maximize loglike, and get covariance.

		This runs the following methods in order:
		- load_data
		- maximize_loglike
		- calculate_parameter_covariance

		Parameters
		----------
		dataservice : DataService, optional
			A dataservice from which to load data.  If a dataservice
			has not been previously defined for this model, this
			argument is not optional.
		autoscale_weights : bool, default True
			If True and data_wt is not None, the loaded dataframes will
			have the weights automatically scaled such that the average
			value for data_wt is 1.0.  See `autoscale_weights` for more
			information.
		**kwargs
			All other keyword arguments are passed through to
			`maximize_loglike`.

		Returns
		-------
		dictx
		"""
		self.load_data(
			dataservice=dataservice,
			autoscale_weights=autoscale_weights
		)
		result = self.maximize_loglike(**kwargs)
		self.calculate_parameter_covariance()
		return result

	def cross_validate(self, cv=5, *args, **kwargs):
		"""
		A simple but well optimized cross-validated log likelihood.

		This method assumes that cases are already ordered randomly.
		It is optimized to avoid shuffling or copying the source data.  As such, it is not
		directly compatible with the cross-validation tools in scikit-learn.  If the larch
		discrete choice model is stacked or ensembled with other machine learning tools,
		the cross-validation tools in scikit-learn are preferred, even though they are potentially not
		as memory efficient.

		Parameters
		----------
		cv : int
			The number of folds in k-fold cross-validation.

		Returns
		-------
		float
			The log likelihood as computed from the holdout folds.
		"""
		ll_cv = 0
		for fold in range(cv):
			i = self.maximize_loglike(leave_out=fold, subsample=cv, quiet=True, *args, **kwargs)
			ll_cv += self.loglike(keep_only=fold, subsample=cv)
			self._frame[f'cv_{fold:03d}'] = self._frame['value']
		return ll_cv

	def noop(self):
		print("No op!")

	def loglike_null(self, use_cache=True):
		"""
		Compute the log likelihood at null values.

		Set all parameter values to the value indicated in the
		"nullvalue" column of the parameter frame, and compute
		the log likelihood with the currently loaded data.  Note
		that the null value for each parameter may not be zero
		(for example, the default null value for logsum parameters
		in a nested logit model is 1).

		Parameters
		----------
		use_cache : bool, default True
			Use the cached value if available.  Set to -1 to
			raise an exception if there is no cached value.

		Returns
		-------
		float
		"""
		if self._cached_loglike_null != 0 and use_cache:
			return self._cached_loglike_null
		elif use_cache == -1:
			raise ValueError("no cached value")
		else:
			current_parameters = self.get_values()
			self.set_values('null')
			self._cached_loglike_null = self.loglike()
			self.set_values(current_parameters)
			return self._cached_loglike_null

	def rho_sq_null(self, x=None, use_cache=True, adj=False):
		"""
		Compute the rho squared value w.r.t. null values.

		Parameters
		----------
		x : {'null', 'init', 'best', array-like, dict, scalar}, optional
			Values for the parameters.  See :ref:`set_values` for details.
		use_cache : bool, default True
			Use the cached value for `loglike_null` if available.
			Set to -1 to raise an exception if there is no cached value.
		adj : bool, default False
			Compute adjusted rho squared, which accounts for the
			degrees of freedom.

		Returns
		-------
		float
		"""
		if adj:
			k = (self._frame.holdfast == 0).sum()
		else:
			k = 0
		return 1 - ((self.loglike(x=x)-k) / self.loglike_null(use_cache=use_cache))

	def loglike_nil(self, use_cache=True):
		"""
		Compute the log likelihood with no model at all.

		This is different from the null model, in that
		no explanatory data and no model structure is used to compute the log
		likelihood.  For simple MNL models, this is my be equivalent to
		the Null model, but for more complex models this may be
		different.

		Parameters
		----------
		use_cache : bool, default True
			Use the cached value if available.  Set to -1 to
			raise an exception if there is no cached value.

		Returns
		-------
		float
		"""
		if self._cached_loglike_nil != 0 and use_cache:
			return self._cached_loglike_nil
		elif use_cache == -1:
			raise ValueError("no cached loglike_nil")
		else:
			from .model import Model
			nil = Model()
			try:
				nil.dataservice = self.dataframes
			except AttributeError:
				raise ValueError('cannot access model.dataframes')
			nil.load_data(log_warnings=False)
			self._cached_loglike_nil = nil.loglike()
			return self._cached_loglike_nil

	def rho_sq_nil(self, x=None, use_cache=True, adj=False):
		"""
		Compute the rho squared value w.r.t. no model at all.

		Parameters
		----------
		x : {'null', 'init', 'best', array-like, dict, scalar}, optional
			Values for the parameters.  See :ref:`set_values` for details.
		use_cache : bool, default True
			Use the cached value for `loglike_nil` if available.
			Set to -1 to raise an exception if there is no cached value.
		adj : bool, default False
			Compute adjusted rho squared, which accounts for the
			degrees of freedom.

		Returns
		-------
		float
		"""
		if adj:
			k = (self._frame.holdfast == 0).sum()
		else:
			k = 0
		return 1 - ((self.loglike(x=x)-k) / self.loglike_nil(use_cache=use_cache))


	def loglike_constants_only(self):
		raise NotImplementedError
		# try:
		# 	return self._cached_loglike_constants_only
		# except AttributeError:
		# 	mm = self.constants_only_model()
		# 	from ..warning import ignore_warnings
		# 	with ignore_warnings():
		# 		result = mm.maximize_loglike(quiet=True, final_screen_update=False, check_for_overspecification=False)
		# 	self._cached_loglike_constants_only = result.loglike
		# 	return self._cached_loglike_constants_only

	def estimation_statistics(self, compute_loglike_null=True):
		"""
		Create an XHTML summary of estimation statistics.

		This will generate a small table of estimation statistics,
		containing:

		*	Log Likelihood at Convergence
		*	Log Likelihood at Null Parameters (if known)
		*	Log Likelihood with No Model (if known)
		*	Log Likelihood at Constants Only (if known)

		Additionally, for each included reference value (i.e.
		everything except log likelihood at convergence) the
		rho squared with respect to that value is also given.

		Each statistic is reported in aggregate, as well as
		per case.

		Parameters
		----------
		compute_loglike_null : bool, default True
			If the log likelihood at null values has not already
			been computed (i.e., if it is not cached) then compute
			it, cache its value, and include it in the output.

		Returns
		-------
		xmle.Elem

		"""

		from xmle import Elem
		div = Elem('div')
		table = div.put('table')

		thead = table.put('thead')
		tr = thead.put('tr')
		tr.put('th', text='Statistic')
		tr.put('th', text='Aggregate')
		tr.put('th', text='Per Case')


		tbody = table.put('tbody')

		try:
			ncases = self.n_cases
		except MissingDataError:
			ncases = None

		tr = thead.put('tr')
		tr.put('td', text='Number of Cases')
		if ncases:
			tr.put('td', text=str(ncases), colspan='2')
		else:
			tr.put('td', text="not available", colspan='2')

		mostrecent = self._most_recent_estimation_result
		if mostrecent is not None:
			tr = thead.put('tr')
			tr.put('td', text='Log Likelihood at Convergence')
			tr.put('td', text="{:.2f}".format(mostrecent.loglike))
			if ncases:
				tr.put('td', text="{:.2f}".format(mostrecent.loglike / ncases))
			else:
				tr.put('td', text="na")


		ll_z = self._cached_loglike_null
		if ll_z == 0:
			if compute_loglike_null:
				try:
					ll_z = self.loglike_null()
				except MissingDataError:
					pass
				else:
					self.loglike()
			else:
				ll_z = 0
		if ll_z != 0:
			tr = thead.put('tr')
			tr.put('td', text='Log Likelihood at Null Parameters')
			tr.put('td', text="{:.2f}".format(ll_z))
			if ncases:
				tr.put('td', text="{:.2f}".format(ll_z / ncases))
			else:
				tr.put('td', text="na")
			if mostrecent is not None:
				tr = thead.put('tr')
				tr.put('td', text='Rho Squared w.r.t. Null Parameters')
				rsz = 1.0 - (mostrecent.loglike / ll_z)
				tr.put('td', text="{:.3f}".format(rsz), colspan='2')

		ll_nil = self._cached_loglike_nil
		if ll_nil != 0:
			tr = thead.put('tr')
			tr.put('td', text='Log Likelihood with No Model')
			tr.put('td', text="{:.2f}".format(ll_nil))
			if ncases:
				tr.put('td', text="{:.2f}".format(ll_nil / ncases))
			else:
				tr.put('td', text="na")
			if mostrecent is not None:
				tr = thead.put('tr')
				tr.put('td', text='Rho Squared w.r.t. No Model')
				rsz = 1.0 - (mostrecent.loglike / ll_nil)
				tr.put('td', text="{:.3f}".format(rsz), colspan='2')


		ll_c = self._cached_loglike_constants_only
		if ll_c != 0:
			tr = thead.put('tr')
			tr.put('td', text='Log Likelihood at Constants Only')
			tr.put('td', text="{:.2f}".format(ll_c))
			if ncases:
				tr.put('td', text="{:.2f}".format(ll_c / ncases))
			else:
				tr.put('td', text="na")
			if mostrecent is not None:
				tr = thead.put('tr')
				tr.put('td', text='Rho Squared w.r.t. Constants Only')
				rsc = 1.0 - (mostrecent.loglike / ll_c)
				tr.put('td', text="{:.3f}".format(rsc), colspan='2')

		# for parallel_title, parallel_loglike in self._parallel_model_results.items():
		# 	tr = thead.put('tr')
		# 	tr.put('td', text=f'Log Likelihood {parallel_title}')
		# 	tr.put('td', text="{:.2f}".format(parallel_loglike))
		# 	tr.put('td', text="{:.2f}".format(parallel_loglike / self.n_cases))
		# 	if mostrecent is not None:
		# 		tr = thead.put('tr')
		# 		tr.put('td', text=f'Rho Squared w.r.t. {parallel_title}')
		# 		rsc = 1.0 - (mostrecent.loglike / parallel_loglike)
		# 		tr.put('td', text="{:.3f}".format(rsc), colspan='2')

		return div

	@property
	def most_recent_estimation_result(self):
		return self._most_recent_estimation_result



	def calculate_parameter_covariance(self, status_widget=None, preserve_hessian=False):
		hess = -self.d2_loglike()

		from ..model.possible_overspec import compute_possible_overspecification, PossibleOverspecification
		overspec = compute_possible_overspecification(hess, self.pf.loc[:,'holdfast'])
		if overspec:
			import warnings
			warnings.warn("WARNING: Model is possibly over-specified (hessian is nearly singular).", category=PossibleOverspecification)
			possible_overspecification = []
			for eigval, ox, eigenvec in overspec:
				if eigval == 'LinAlgError':
					possible_overspecification.append((eigval, [ox, ], ["", ]))
				else:
					paramset = list(numpy.asarray(self.pf.index)[ox])
					possible_overspecification.append((eigval, paramset, eigenvec[ox]))
			self._possible_overspecification = possible_overspecification

		take = numpy.full_like(hess, True, dtype=bool)
		dense_s = len(self.pvals)
		for i in range(dense_s):
			ii = self.pnames[i]
			if self.pf.loc[ii, 'holdfast'] or (self.pf.loc[ii, 'value'] >= self.pf.loc[ii, 'maximum']) or (
				self.pf.loc[ii, 'value'] <= self.pf.loc[ii, 'minimum']):
				take[i, :] = False
				take[:, i] = False
				dense_s -= 1
		if dense_s > 0:
			hess_taken = hess[take].reshape(dense_s, dense_s)
			from ..linalg import general_inverse
			try:
				invhess = general_inverse(hess_taken)
			except numpy.linalg.linalg.LinAlgError:
				invhess = numpy.full_like(hess_taken, numpy.nan, dtype=numpy.float64)
			covariance_matrix = numpy.full_like(hess, 0, dtype=numpy.float64)
			covariance_matrix[take] = invhess.reshape(-1)
			# robust...
			try:
				bhhh_taken = self.bhhh()[take].reshape(dense_s, dense_s)
			except NotImplementedError:
				pass
			else:
				# import scipy.linalg.blas
				# temp_b_times_h = scipy.linalg.blas.dsymm(float(1), invhess, bhhh_taken)
				# robusto = scipy.linalg.blas.dsymm(float(1), invhess, temp_b_times_h, side=1)
				robusto = numpy.dot(numpy.dot(invhess, bhhh_taken), invhess)
				robust_covariance_matrix = numpy.full_like(hess, 0, dtype=numpy.float64)
				robust_covariance_matrix[take] = robusto.reshape(-1)
		else:
			covariance_matrix = numpy.full_like(hess, 0, dtype=numpy.float64)
			robust_covariance_matrix = numpy.full_like(hess, 0, dtype=numpy.float64)
		self.pf['std err'] = numpy.sqrt(covariance_matrix.diagonal())
		self.pf['t stat'] = (self.pf['value'] - self.pf['nullvalue']) / self.pf['std err']
		self.pf['robust std err'] = numpy.sqrt(robust_covariance_matrix.diagonal())
		self.pf['robust t stat'] = (self.pf['value'] - self.pf['nullvalue']) / self.pf['robust std err']

		if preserve_hessian:
			self._matrixes['hessian_matrix'] = hess

		self._matrixes['covariance_matrix'] = covariance_matrix
		self._matrixes['robust_covariance_matrix'] = robust_covariance_matrix

	@property
	def possible_overspecification(self):
		return self._possible_overspecification

