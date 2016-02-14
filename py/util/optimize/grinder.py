

from ...linalg import general_inverse as _general_inverse
from scipy.optimize import minimize as _minimize
from enum import Enum
from scipy.optimize import OptimizeResult
from collections import deque
import numpy
from ...core import LarchError, LarchCacheError, runstats
import time as _time

class outcomes(Enum):
	success = 1
	fail = -1
	slow = 0





class ProgressTooSlow(Exception):
	def __init__(self, message, x, nit=None, slowness=None):
		self.x = x
		self.success = False
		self.message = message
		self.nit = nit
		self.slowness = slowness
	def result(self):
		return OptimizeResult(x=self.x, message=self.message, nit=self.nit, status=-1, success=self.success, slow=self.slowness)


class ComputedToleranceMet(Exception):
	def __init__(self, ctol, ctol_thresh, x, nit=None):
		self.x = x
		self.nit = nit
		self.ctol = ctol
		self.ctol_thresh = ctol_thresh
	def result(self):
		return OptimizeResult(x=self.x, nit=self.nit, status=0, success=True, ctol=self.ctol)
	def describe(self):
		return "Computed tolerance is {:.4g}, lower than {:.2g} threshold".format(self.ctol, self.ctol_thresh)


class Watcher():
	def __init__(self, obj_fun, x0, *, slow_len=(), slow_thresh=(), logger=None, ctol_fun=None, ctol=1e-6):
		self.count = 0
		self.maxlen = 1
		self.slowness = []
		self.ctol = ctol
		self.ctol_fun = ctol_fun
		self.fun = obj_fun
		self.logger = logger
		try:
			for s_len, s_thresh in zip(slow_len, slow_thresh):
				self.maxlen = max(self.maxlen,s_len)
				self.slowness.append((s_len, s_thresh))
		except TypeError:
			self.maxlen = max(self.maxlen,slow_len)
			self.slowness.append((slow_len, slow_thresh))
		self.loglike_memory = deque([],self.maxlen+1)
		self.loglike_memory.append(self.fun(x0))
	
	def __call__(self, x):
		self.count += 1
		ll = self.fun(x)
		self.loglike_memory.append(ll)
		if self.ctol and self.ctol_fun:
			try:
				tol = self.ctol_fun(x)
			except LarchError:
				tol = numpy.inf
			except LarchCacheError:
				tol = numpy.inf
			if numpy.abs(tol) < self.ctol:
				if self.logger:
					self.logger.log(10, "Computed tolerance of %g below threshold of %g",tol,self.ctol)
				raise ComputedToleranceMet(tol, self.ctol, x, self.count)
			elif self.logger:
				self.logger.log(10, "Computed tolerance of %g not below threshold of %g",tol,self.ctol)
		for length, threshold in self.slowness:
			if len(self.loglike_memory) > length:
				avg = (self.loglike_memory[-1]-self.loglike_memory[-length-1])/length
				if numpy.abs(avg) < numpy.abs(threshold):
					if self.logger:
						self.logger.log(10, "LL Trace: %s",str(self.loglike_memory))
						mem = numpy.asarray(self.loglike_memory)
						self.logger.log(10, "delta LL Trace: %s",str(mem[:-1]-mem[1:]))
					raise ProgressTooSlow('average improvement only {:.4g} over {} iterations'.format(-avg,length), x, self.count, slowness=(-avg,length))


def minimize_with_watcher(fun, x0, args=(), *, slow_len=(), slow_thresh=(), ctol_fun=None, ctol=1e-6, logger=None, callback=None, method_str="default method", options={}, **k):
	watcher = Watcher(fun, x0, ctol_fun=ctol_fun, ctol=ctol, logger=logger, slow_len=slow_len, slow_thresh=slow_thresh)
	if callback:
		new_callback = lambda z: (callback(z), watcher(z), )
	else:
		new_callback = watcher
	if method_str in ("bhhh","bhhh-wolfe",):
		options['logger'] = logger
	try:
		r = _minimize(fun, x0, args, callback=new_callback, options=options, **k)
	except ProgressTooSlow as err:
		r = err.result()
		if logger:
			logger.log(30,"Progressing too slowly [{}], {}".format(method_str,r.message))
		r.slow = err.slowness
	except ComputedToleranceMet as suc:
		r = suc.result()
		r.message = "Optimization terminated successfully per computed tolerance"
		if logger:
			logger.log(30,"{} [{}]".format(suc.describe(),method_str))
	else:
		if logger:
			logger.log(30,"{} [{}]".format(r.message,method_str))
	return r




class OptimizeTechnique():
	def __str__(self):
		watcher_args = ""
		if self.slow_len:
			try:
				for s_len, s_thresh in zip(self.slow_len, self.slow_thresh):
					watcher_args += " [slow {1} on {0}]".format(s_len, s_thresh)
			except TypeError:
				watcher_args += " [slow {1} on {0}]".format(self.slow_len, self.slow_thresh)
		if self.ctol:
			watcher_args += " [ctol {0}]".format(self.ctol)
		return "<larch.util.optimize.OptimizeTechnique {}{!s}>".format(self.method, watcher_args)
	def __repr__(self):
		return self.__str__()
	def __init__(self, method, fun, *, slow_len=(), slow_thresh=(), ctol=None, ctol_fun=None, options=None,
					bhhh=None, init_inv_hess=None, jac=None, logger=None, hess=None):
		self.fun = fun
		self.bhhh = bhhh
		self.init_inv_hess = init_inv_hess
		self.hess = hess
		self.method_str = str(method)
		self.options = {} if options is None else options
		if isinstance(method,str) and method.lower()=='bfgs-init':
			from .algorithms import _minimize_bfgs_1
			method = _minimize_bfgs_1
			if init_inv_hess is not None:
				self.hess = self.init_inv_hess
			else:
				self.hess = lambda z: _general_inverse(numpy.asarray(self.bhhh(z)))
		elif isinstance(method,str) and method.lower()=='bhhh':
			from .algorithms import _minimize_bhhh_simple
			method = _minimize_bhhh_simple
			self.hess = lambda z: _general_inverse(numpy.asarray(self.bhhh(z)))
			self.options['logger'] = logger
		elif isinstance(method,str) and method.lower()=='bhhh-wolfe':
			from .algorithms import _minimize_bhhh_wolfe
			method = _minimize_bhhh_wolfe
			self.hess = lambda z: _general_inverse(numpy.asarray(self.bhhh(z)))
		elif method in ['Newton-CG-bhhh', 'dogleg-bhhh', 'trust-ncg-bhhh']:
			self.hess = self.bhhh
			method = method[:-5]
		self.jac = jac
		self.method = method
		self.slow_len = slow_len
		self.slow_thresh = slow_thresh
		self.last_metaiter = None
		self.last_result = None
		self.ctol = ctol
		self.ctol_fun = ctol_fun
		self.count_slow = 0
		self.flag_fail = False
		self.count_fail = 0
		self.flag_success = 0
		self.logger = logger
		self.flag_last_best_hope = False
	def __call__(self, ignored_fun, x0, args=(), options={}, **kwargs):
		if not self.logger and 'logger' in kwargs:
			self.logger = kwargs['logger']
		if self.logger:
			self.logger.log(20,"Beginning leg using {!s} from {!s}".format(self.method_str, x0))
		local_kwargs = dict(
			ctol_fun = self.ctol_fun,
			ctol = self.ctol,
			slow_len = () if self.flag_last_best_hope else self.slow_len,
			slow_thresh = () if self.flag_last_best_hope else self.slow_thresh,
			method = self.method,
			hess = self.hess,
			jac = self.jac,
			method_str = self.method_str,
			logger = self.logger,
		)
		if 'ctol' in kwargs and kwargs['ctol'] is not None:
			if local_kwargs['ctol'] is None:
				local_kwargs['ctol'] = kwargs['ctol']
			elif kwargs['ctol'] is None:
				pass
			else:
				local_kwargs['ctol'] = max( numpy.abs(local_kwargs['ctol']), numpy.abs(kwargs['ctol']) )
		if 'ctol_fun' in kwargs and local_kwargs['ctol_fun'] is None:
			local_kwargs['ctol_fun'] = kwargs['ctol_fun']
		use_kwargs = {k:v for k,v in kwargs.items() if k not in local_kwargs}
		local_kwargs.update(use_kwargs)
		r = self.last_result = minimize_with_watcher(
			self.fun, x0, args,
			options=options,
			**local_kwargs
		)
		if hasattr(r,'slow'):
			self.count_slow += 1
			r.outcome = outcomes.slow
			if self.logger:
				self.logger.log(30,"Ending leg (slow) using {!s} at {!s}".format(self.method_str, r.x))
			return r
		elif r.success:
			self.flag_success = True
			r.outcome = outcomes.success
			if self.logger:
				self.logger.log(30,"Ending leg (success) using {!s} at {!s}".format(self.method_str, r.x))
			return r
		else:
			self.count_fail += 1
			self.flag_fail = True
			r.outcome = outcomes.fail
			if self.logger:
				self.logger.log(30,"Ending leg (fail) using {!s} at {!s}".format(self.method_str, r.x))
			return r
#
#
#
#	def go(self, model, constraints=()):
#		r = self.last_result = model.maximize_loglike(
#			method      = self.method,
#			constraints = constraints,
#			ctol        = self.ctol,
#			options     = self.options,
#			callback    = Watcher(model.negative_loglike, model.parameter_values(), *self.watcher_args, tol_fun=model.bhhh_tolerance),
#			)
#		if hasattr(r,'slow'):
#			self.count_slow += 1
#			return r, outcomes.slow
#		elif r.success:
#			self.flag_success = True
#			return r, outcomes.success
#		else:
#			self.count_fail += 1
#			self.flag_fail = True
#			return r, outcomes.fail
#



class OptimizeResults(OptimizeResult):
    def __repr__(self):
        if self.keys():
            kys = list(self.keys())
            kys.remove('intermediate')
            m = max(map(len, kys)) + 1
            kys = sorted(kys)
            return '\n'.join([k.rjust(m) + ': ' + repr(self[k]).replace("\n","\n"+(' '*m)+'| ') for k in kys])
            #return '\n'.join([k.rjust(m) + ': ' + repr(v)
            #                  for k, v in self.items() if k!='intermediate'])
        else:
            return self.__class__.__name__ + "()"





class OptimizeTechniques():
	def __init__(self, techniques=None, ctol_fun=None, ctol=1e-6, logger=None, fun=None, jac=None, hess=None, bhhh=None, start_timer=None, end_timer=None):
		self._techniques = [] if techniques is None else list(techniques)
		self.meta_iteration = 0
		self.ctol_fun = ctol_fun
		self.ctol = ctol
		self.logger = logger
		self._fun = fun
		self._jac = jac
		self._hess = hess
		self._bhhh = bhhh
		self._start_timer = start_timer
		self._end_timer = end_timer
	def unfail_all(self, but=None):
		for i in self._techniques:
			if i is not but:
				i.flag_fail = False
	def add(self, *arg, **kwarg):
		if self._fun is not None and 'fun' not in kwarg:
			kwarg['fun'] = self._fun
		if self._jac is not None and 'jac' not in kwarg:
			kwarg['jac'] = self._jac
		if self._bhhh is not None and 'bhhh' not in kwarg:
			kwarg['bhhh'] = self._bhhh
		if self._hess is not None and 'hess' not in kwarg:
			kwarg['hess'] = self._hess
		self._techniques.append(OptimizeTechnique(*arg, **kwarg))
	def __len__(self):
		return len(self._techniques)
	def __iter__(self):
		return self
	def __next__(self):
		"""Get the next technique to try.
		
		Return the first unflagged technique, and if none
		return the first non-failed technique."""
		self.meta_iteration += 1
		# SCAN for first technique that is neither flagged as failed or success
		for i in self._techniques:
			candidate = i
			if not i.flag_fail and not i.flag_success:
				break
			candidate = None
		# IF None, abort
		if candidate is None:
			raise StopIteration
		# SCAN for any technique that is neither flagged as failed or success, and has fewer slow attempts
		number_remaining = 0
		for i in self._techniques:
			if not i.flag_success:
				number_remaining += 1
				if i.count_slow < candidate.count_slow and not i.flag_fail:
					candidate = i
		candidate.last_metaiter = self.meta_iteration
		if number_remaining <= 1:
			candidate.flag_last_best_hope = True
		else:
			candidate.flag_last_best_hope = False
		return candidate

	def __call__(self, fun, x0, args=(), **kwrds):
		metaresult = OptimizeResults(x=x0, message="incomplete run",
									niter=[], status=-9, success=False, stats=runstats(), intermediate=[])
		metaresult.stats.timestamp = _time.strftime("%A, %B %d %Y, %I:%M:%S %p")
		x_n = x0
		
		def complete_metaresult(from_result, successful):
			metaresult.stats.end_process()
			for part in ['x', 'fun','jac','hess','hess_inv']:
				if part in from_result:
					metaresult[part] = from_result[part]
			metaresult.stats.iteration = sum(i[1] for i in metaresult.niter)
			metaresult.nit = metaresult.stats.iteration
			if successful:
				metaresult.status = 0
				metaresult.success = True
				metaresult.stats.results = 'success'
		
		for technique in self:
			metaresult.stats.start_process('optimize:{}'.format(technique.method_str))
			if self.logger:
				self.logger.log(30, "Using {!s}".format(technique))
			result = technique(fun, x_n, args, logger=self.logger, ctol=self.ctol, ctol_fun=self.ctol_fun, **kwrds)
			if not numpy.isnan(result.x).any():
				x_n = result.x
			try:
				metaresult.niter.append( (technique.method_str, result.nit) )
			except AttributeError:
				pass
			metaresult.intermediate.append(result)
			try:
				metaresult.intermediate[-1].method = technique.method_str
			except AttributeError:
				pass
			if result.outcome in (outcomes.success, outcomes.slow):
				self.unfail_all(but=technique)
			if self.ctol_fun and self.ctol:
				actual_utol = self.ctol_fun(x_n)
				if numpy.abs(actual_utol) < self.ctol:
					metaresult.ctol = actual_utol
					metaresult.message = "Optimization terminated successfully per computed tolerance. [{}]".format(technique.method_str)
					complete_metaresult(result, True)
					return metaresult
			else:
				if result.outcome == outcomes.success:
					metaresult.message = "Optimization terminated successfully. [{}]".format(technique.method_str)
					complete_metaresult(result, True)
					return metaresult
		metaresult.stats.results = 'failed'
		complete_metaresult(result, False)
		return metaresult


#
#
#
#
#
#
#
#		method_str = str(method)
#		if isinstance(method,str) and method.lower()=='bfgs-init':
#			from .algorithms import _minimize_bfgs_1
#			method = _minimize_bfgs_1
#			if 'init_inv_hess' in kwrds:
#				hess = kwrds['init_inv_hess']
#			else:
#				hess = _general_inverse(numpy.asarray(hessp(x0)))
#		elif method in ['Newton-CG-bhhh', 'dogleg-bhhh', 'trust-ncg-bhhh']:
#			hess = hessp
#			method = method[:-5]
#		else:
#			hess = None
#		try:
#			r = _minimize(fun, x0, method=method, jac=jac,
#							 hess=hess, bounds=bounds, constraints=constraints,
#							 tol=tol, callback=callback, options=options)
#		except ProgressTooSlow as err:
#			r = err.result()
#			r.slow = err.slowness
#		except ComputedToleranceMet as suc:
#			r = suc.result()
#			r.message = "Optimization terminated successfully per BHHH tolerance using {}.".format(method_str)
#		return r
#
#
#def grind():
#		method_str = str(method)
#		if x0 is None:
#			x0 = model.parameter_values()
#		if not model.is_provisioned() and model._ref_to_db is not None:
#			model.provision()
#		if isinstance(method,str) and method.lower()=='bfgs-init':
#			from .algorithms import _minimize_bfgs_1
#			method = _minimize_bfgs_1
#			if 'init_inv_hess' in kwrds:
#				hess = kwrds['init_inv_hess']
#			else:
#				hess = numpy.linalg.inv(numpy.asarray(model.bhhh(x0)))
#		elif method in ['Newton-CG-bhhh', 'dogleg-bhhh', 'trust-ncg-bhhh']:
#			hess = model.bhhh
#			method = method[:-5]
#		else:
#			hess = None
#		try:
#			r = _minimize(model.negative_loglike, x0, method=method, jac=model.negative_d_loglike,
#							 hess=hess, bounds=model.parameter_bounds(), constraints=constraints,
#							 tol=tol, callback=callback, options=options)
#		except ProgressTooSlow as err:
#			r = err.result()
#			r.slow = err.slowness
#		except ComputedToleranceMet as suc:
#			r = suc.result()
#			r.message = "Optimization terminated successfully per BHHH tolerance using {}.".format(method_str)
#		return r

