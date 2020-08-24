
import numpy
import pandas

include "../general_precision.pxi"
from ..general_precision import l4_float_dtype
from ..general_precision cimport l4_float_t

from .parameter_frame cimport ParameterFrame
from .persist_flags cimport *
from ..exceptions import MissingDataError, BHHHSimpleStepFailure

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
			raise BHHHSimpleStepFailure("simple step bhhh failed")
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
				raise BHHHSimpleStepFailure(f"simple step bhhh failed\ndirection = {str(direction)}")
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
		return (-result.ll, -numpy.asarray(result.dll))

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
		"""
		The total weight of cases in the attached dataframes.

		Returns
		-------
		float
		"""
		raise NotImplementedError("abstract base class, use a derived class instead")

	def maximize_loglike(
			self,
			method=None,
			method2=None,
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
		"""
		Maximize the log likelihood.

		The `dataframes` for this model should previously have been
		prepared using the `load_data` method.

		Parameters
		----------
		method : str, optional
			The optimization method to use.  See scipy.optimize for
			most possibilities, or use 'BHHH'. Defaults to SLSQP if
			there are any constraints or finite parameter bounds,
			otherwise defaults to BHHH.
		quiet : bool, default False
			Whether to suppress the dashboard.

		Returns
		-------
		dictx
			A dictionary of results, including final log likelihood,
			elapsed time, and other statistics.  The exact items
			included in output will vary by estimation method.

		Raises
		------
		ValueError
			If the `dataframes` are not already loaded.

		"""
		try:
			from ..util.timesize import Timer
			from scipy.optimize import minimize
			from .. import _doctest_mode_
			from ..util.rate_limiter import NonBlockingRateLimiter
			from ..util.display import display_head, display_p, display_nothing

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
				tag1 = display_nothing()
				tag2 = display_nothing()
				tag3 = display_nothing()

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

			if method is None:
				if self.constraints or numpy.isfinite(self.pf['minimum'].max()) or numpy.isfinite(self.pf['maximum'].min()):
					method = 'slsqp'
				else:
					method = 'bhhh'

			if method2 is None and method.lower() == 'bhhh':
				method2 = 'slsqp'

			method_used = method
			raw_result = None

			if method.lower()=='bhhh':
				try:
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
				except NotImplementedError:
					tag1.update(f'Iteration {iteration_number:03} [BHHH Not Available] {iteration_number_tail}', force=True)
					tag3.update(self.pf, force=True)
					if method2 is not None:
						method_used = f"{method2}"
						method = method2
				except BHHHSimpleStepFailure:
					tag1.update(f'Iteration {iteration_number:03} [Exception Recovery] {iteration_number_tail}', force=True)
					tag3.update(self.pf, force=True)
					if method2 is not None:
						method_used = f"{method_used}|{method2}"
						method = method2
				except:
					tag1.update(f'Iteration {iteration_number:03} [Exception] {iteration_number_tail}', force=True)
					tag3.update(self.pf, force=True)
					raise

			if method.lower() != 'bhhh':
				try:
					bounds = None
					if isinstance(method,str) and method.lower() in ('slsqp', 'l-bfgs-b', 'tnc', 'trust-constr'):
						bounds = self.pbounds

					try:
						constraints = self._get_constraints(method)
					except:
						constraints = ()

					raw_result = minimize(
						self.neg_loglike2,
						self.pvals,
						args=(0, -1, 1, leave_out, keep_only, subsample), # start_case, stop_case, step_case, leave_out, keep_only, subsample
						method=method,
						jac=True,
						bounds=bounds,
						callback=callback,
						options=options,
						constraints=constraints,
						**kwargs
					)
				except:
					tag1.update(f'Iteration {iteration_number:03} [Exception] {iteration_number_tail}', force=True)
					tag3.update(self.pf, force=True)
					raise

			timer.stop()

			if final_screen_update and not quiet and not _doctest_mode_ and raw_result is not None:
				tag1.update(f'Iteration {iteration_number:03} [Converged] {iteration_number_tail}', force=True)
				tag2.update(f'LL = {self.loglike()}', force=True)
				tag3.update(self.pf, force=True)

			if raw_result is None:
				raw_result = {}
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
			result['method'] = method_used
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
		except:
			logger.exception("error in maximize_loglike")
			raise

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
		try:
			if self._cached_loglike_constants_only == 0:
				raise AttributeError
			return self._cached_loglike_constants_only
		except AttributeError:
			from . import Model
			constants_only_model = Model(dataservice=self.dataframes)
			altcodes = constants_only_model.dataservice.alternative_codes()
			from ..roles import P
			for j in altcodes[1:]:
				constants_only_model.utility_co[j] = P(str(j))
			from ..warning import ignore_warnings
			with ignore_warnings():
				constants_only_model.load_data()
				result = constants_only_model.maximize_loglike(
					quiet=True, final_screen_update=False,
					check_for_overspecification=False,
				)
			self._cached_loglike_constants_only = result.loglike
			return self._cached_loglike_constants_only

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

	def _estimation_statistics_excel(
			self,
			xlsxwriter,
			sheetname,
			start_row=0,
			start_col=0,
			buffer_cols=0,
			compute_loglike_null=True,
	):
		"""
		Write a tabular summary of estimation statistics to excel.

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
		xlsxwriter : ExcelWriter
		sheetname : str
		start_row, start_col : int
			Zero-based index of upper left cell
		buffer_cols : int
			Number of extra columns between statistic label and
			values.
		compute_loglike_null : bool, default True
			If the log likelihood at null values has not already
			been computed (i.e., if it is not cached) then compute
			it, cache its value, and include it in the output.

		"""
		try:
			if start_row is None:
				start_row = xlsxwriter.sheet_startrow.get(sheetname, 0)
			worksheet = xlsxwriter.add_worksheet(sheetname)

			row = start_row

			fixed_2 = xlsxwriter.book.add_format({'num_format': '#,##0.00'})
			fixed_4 = xlsxwriter.book.add_format({'num_format': '0.0000'})
			comma_0 = xlsxwriter.book.add_format({'num_format': '#,##0'})
			bold = xlsxwriter.book.add_format({'bold': True})
			bold_centered = xlsxwriter.book.add_format({'bold': True})

			fixed_2.set_align('center')
			fixed_4.set_align('center')
			comma_0.set_align('center')
			bold_centered.set_align('center')
			bold.set_border(1)
			bold_centered.set_border(1)

			datum_col = start_col+buffer_cols

			def catname(j):
				nonlocal row, start_col, buffer_cols
				if buffer_cols:
					worksheet.merge_range(row, start_col,row, start_col+buffer_cols, j, bold)
				else:
					worksheet.write(row, start_col, j, bold)

			catname('Statistic')
			worksheet.write(row, datum_col+1, 'Aggregate', bold_centered)
			worksheet.write(row, datum_col+2, 'Per Case', bold_centered)
			row += 1

			try:
				ncases = self.n_cases
			except MissingDataError:
				ncases = None

			catname('Number of Cases')
			if ncases:
				worksheet.merge_range(row, datum_col+1, row, datum_col+2, ncases, cell_format=comma_0)
			else:
				worksheet.merge_range(row, datum_col+1, row, datum_col+2, "not available")
			row += 1

			mostrecent = self._most_recent_estimation_result
			if mostrecent is not None:
				catname('Log Likelihood at Convergence')
				worksheet.write(row, datum_col+1, mostrecent.loglike, fixed_2) # "{:.2f}".format(mostrecent.loglike)
				if ncases:
					worksheet.write(row, datum_col+2, mostrecent.loglike/ ncases, fixed_4) # "{:.2f}".format(mostrecent.loglike/ ncases)
				else:
					worksheet.write(row, datum_col+2, "na")
				row += 1

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
				catname('Log Likelihood at Null Parameters')
				worksheet.write(row, datum_col+1, ll_z, fixed_2) # "{:.2f}".format(ll_z)
				if ncases:
					worksheet.write(row, datum_col+2, ll_z/ ncases, fixed_4) # "{:.2f}".format(ll_z/ ncases)
				else:
					worksheet.write(row, datum_col+2, "na")
				if mostrecent is not None:
					row += 1
					catname('Rho Squared w.r.t. Null Parameters')
					rsz = 1.0 - (mostrecent.loglike / ll_z)
					worksheet.merge_range(row, datum_col+1, row, datum_col+2, rsz, cell_format=fixed_4) # "{:.3f}".format(rsz)
				row += 1

			ll_nil = self._cached_loglike_nil
			if ll_nil != 0:
				catname('Log Likelihood with No Model')
				worksheet.write(row, datum_col+1, ll_nil, fixed_2) # "{:.2f}".format(ll_nil)
				if ncases:
					worksheet.write(row, datum_col+2, ll_nil/ ncases, fixed_4) # "{:.2f}".format(ll_nil/ ncases)
				else:
					worksheet.write(row, datum_col+2, "na")
				if mostrecent is not None:
					row += 1
					catname('Rho Squared w.r.t. No Model')
					rsz = 1.0 - (mostrecent.loglike / ll_nil)
					worksheet.merge_range(row, datum_col+1, row, datum_col+2, rsz, cell_format=fixed_4) # "{:.3f}".format(rsz)
				row += 1

			ll_c = self._cached_loglike_constants_only
			if ll_c != 0:
				catname('Log Likelihood at Constants Only')
				worksheet.write(row, datum_col+1, ll_c, fixed_2) # "{:.2f}".format(ll_c)
				if ncases:
					worksheet.write(row, datum_col+2, ll_c/ ncases, fixed_4) # "{:.2f}".format(ll_c/ ncases)
				else:
					worksheet.write(row, datum_col+2, "na")
				if mostrecent is not None:
					row += 1
					catname('Rho Squared w.r.t. Constants Only')
					rsc = 1.0 - (mostrecent.loglike / ll_c)
					worksheet.merge_range(row, datum_col+1, row, datum_col+2, rsc, cell_format=fixed_4) # "{:.3f}".format(rsc)
				row += 1

			if mostrecent is not None:
				if 'message' in mostrecent:
					catname('Optimization Message')
					worksheet.write(row, datum_col+1, mostrecent.message)
					row += 1

			if sheetname not in xlsxwriter._col_widths:
				xlsxwriter._col_widths[sheetname] = {}
			current_width = xlsxwriter._col_widths[sheetname].get(start_col, 8)
			proposed_width = 28
			if buffer_cols:
				for b in range(buffer_cols):
					proposed_width -= xlsxwriter._col_widths[sheetname].get(start_col+1+b, 8)
			new_width = max(current_width, proposed_width)
			xlsxwriter._col_widths[sheetname][start_col] = new_width
			worksheet.set_column(start_col,start_col,new_width)

			row += 2 # gap
			xlsxwriter.sheet_startrow[worksheet.name] = row
		except:
			logger.exception("error in _estimation_statistics_excel")
			raise

	@property
	def most_recent_estimation_result(self):
		return self._most_recent_estimation_result



	def calculate_parameter_covariance(self, status_widget=None, preserve_hessian=False, like_ratio=True):
		"""
		Compute the parameter covariance matrix.

		This function computes the parameter covariance by
		taking the inverse of the hessian (2nd derivative
		of log likelihood with respect to the parameters.)

		It does not return values directly, but rather stores
		the result in `covariance_matrix`, and computes the
		standard error of the estimators (i.e. the square root
		of the diagonal) and stores those values in `pf['std_err']`.

		Parameters
		----------
		like_ratio : bool, default True
			For parameters where the
		"""
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
			if self.pf.loc[ii, 'holdfast']:
				# or (self.pf.loc[ii, 'value'] >= self.pf.loc[ii, 'maximum'])
				# or (self.pf.loc[ii, 'value'] <= self.pf.loc[ii, 'minimum']):
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
				robust_covariance_matrix = numpy.full_like(hess, 0, dtype=numpy.float64)
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
		self.pf['std_err'] = numpy.sqrt(covariance_matrix.diagonal())
		self.pf['t_stat'] = (self.pf['value'] - self.pf['nullvalue']) / self.pf['std_err']
		self.pf['robust_std_err'] = numpy.sqrt(robust_covariance_matrix.diagonal())
		self.pf['robust_t_stat'] = (self.pf['value'] - self.pf['nullvalue']) / self.pf['robust_std_err']

		if preserve_hessian:
			self._matrixes['hessian_matrix'] = hess

		self._matrixes['covariance_matrix'] = covariance_matrix
		self._matrixes['robust_covariance_matrix'] = robust_covariance_matrix

		# constrained covariance
		constraints = list(self.constraints)
		try:
			constraints.extend(self._get_bounds_constraints())
		except:
			pass

		if constraints:

			binding_constraints = list()

			self.pf['unconstrained_std_err'] = self.pf['std_err'].copy()
			self.pf['unconstrained_t_stat'] = self.pf['t_stat'].copy()

			self._matrixes['unconstrained_covariance_matrix'] = self.covariance_matrix.values.copy()
			s = self.covariance_matrix.values
			for c in constraints:
				if numpy.absolute(c.fun(self.pf.value)) < c.binding_tol:
					binding_constraints.append(c)
					b = c.jac(self.pf.value)
					den = b@s@b
					if den != 0:
						s = s-(1/den)*s@b.reshape(-1,1)@b.reshape(1,-1)@s
			self._matrixes['covariance_matrix'] = s
			self.pf['std_err'] = numpy.sqrt(s.diagonal())
			self.pf['t_stat'] = (self.pf['value'] - self.pf['nullvalue']) / self.pf['std_err']

			# Fix numerical issues on some constraints, add constrained notes
			if binding_constraints or any(self.pf['holdfast']!=0):
				notes = {}
				for c in binding_constraints:
					pa = c.get_parameters()
					for p in pa:
						if self.pf.loc[p,'t_stat'] > 1e5:
							self.pf.loc[p,'t_stat'] = numpy.inf
							self.pf.loc[p,'std_err'] = numpy.nan
						if self.pf.loc[p,'t_stat'] < -1e5:
							self.pf.loc[p,'t_stat'] = -numpy.inf
							self.pf.loc[p,'std_err'] = numpy.nan
						n = notes.get(p,[])
						n.append(c.get_binding_note(self.pvals))
						notes[p] = n
				self.pf['constrained'] = pandas.Series({k:'\n'.join(v) for k,v in notes.items()}, dtype=object)
				self.pf['constrained'].fillna('', inplace=True)
				self.pf.loc[self.pf['holdfast']!=0, 'constrained'] = 'fixed value'
				self.pf.loc[self.pf['holdfast']!=0, 'std_err'] = numpy.nan
				self.pf.loc[self.pf['holdfast']!=0, 't_stat'] = numpy.nan

		if like_ratio:
			non_finite_t = ~numpy.isfinite(self.pf.t_stat)
			if numpy.any(non_finite_t):
				self.likelihood_ratio(self.pf.index[non_finite_t])

	@property
	def possible_overspecification(self):
		from ..util.overspec_viewer import OverspecView
		if self._possible_overspecification:
			return OverspecView(self._possible_overspecification)
		return self._possible_overspecification

	def likelihood_ratio(self, param=None, ref_value=None, include_holdfast=False, *, _current_ll=None):
		"""
		Compute a likelihood ratio for changing a parameter to its null value.

		Parameters
		----------
		param: str or Iterable[str], optional
			The name of the parameter[s] to vary.  If not given, the
			likelihood ratios will be computed for all parameters
			(except those with `holdfast` set to True).
		ref_value: numeric, optional
			The alternative value to use for the named parameter. If
			not given, the alternative value used is the parameter's
			null value.
		include_holdfast: bool, default False
			Whether to compute likelihood ratios for holdfast parameters.
			This argument is ignored if only a single parameter name
			is given as a str in `param`.

		Returns
		-------
		float or pandas.Series:
			The likelihood ratio (i.e. the difference in the log likelihoods).
		"""
		try:
			ref_value_orig = ref_value
			if _current_ll is None:
				_current_ll = self.loglike()
			if param is None:
				result = pandas.Series(data=numpy.nan, index=self.pf.index)
				for p in self.pf.index:
					if not self.pf.loc[p, 'holdfast'] or include_holdfast:
						result[p] = self.likelihood_ratio(p, ref_value=ref_value, _current_ll=_current_ll)
				return result
			elif isinstance(param, str):
				if ref_value is None:
					ref_value = self.pf.loc[param, 'nullvalue']
				current_value = self.pf.loc[param, 'value']
				if ref_value == current_value:
					if ref_value_orig is None:
						self.pf.loc[param, 'likelihood_ratio'] = 0
					return 0
				else:
					self.set_value(param, ref_value)
					ll_alt = self.loglike()
					self.set_value(param, current_value)
					like_ratio = _current_ll - ll_alt
					if ref_value_orig is None:
						self.pf.loc[param, 'likelihood_ratio'] = like_ratio
				return like_ratio
			else:
				result = pandas.Series(data=numpy.nan, index=list(param))
				for p in result.index:
					if not self.pf.loc[p, 'holdfast'] or include_holdfast:
						result[p] = self.likelihood_ratio(p, ref_value=ref_value, _current_ll=_current_ll)
				return result
		except:
			logger.exception("error in likelihood_ratio")
			raise
