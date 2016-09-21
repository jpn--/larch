#
#
#  larch.util.optimize.__init__
#
#  Copyright 2007-2016 Jeffrey Newman
#
#  Larch is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  
#  Larch is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with Larch.  If not, see <http://www.gnu.org/licenses/>.
#  
#

from scipy.optimize import minimize as _minimize
import numpy
from scipy.optimize import OptimizeResult, OptimizeWarning
from enum import Enum
from ...core import LarchError, runstats
import warnings
import os
import ast

if os.environ.get('READTHEDOCS', None) != 'True':
	warnings.filterwarnings(action="ignore", message='.*Unknown solver options.*', category=OptimizeWarning, module='', lineno=0)



from .grinder import *




def optimizers(model, *arg, ctol=1e-7):
	ot = OptimizeTechniques(ctol=ctol, ctol_fun=model.bhhh_tolerance, logger=model.logger(),
							fun = model.negative_loglike, bhhh = model.bhhh,
							jac = model.negative_d_loglike,
							)
	for a in arg:
		if isinstance(a,dict):
			ot.add(**a)
		elif isinstance(a,str):
			ot.add(a)
		else:
			raise LarchError("optimizer designees must be a dict or method name string")
	return ot


def _default_optimizers(model):
	if hasattr(model, 'constraints') and model.constraints:
		if hasattr(model, 'use_cobyla'):
			if model.use_cobyla == 'only':
				return (
					dict(method='COBYLA', options={'maxiter':max(len(model)*100, 1000), 'rhobeg':0.1}),
				)
			elif model.use_cobyla:
				return (
					dict(method='SLSQP', options={'ftol':1e-7, 'maxiter':max(len(model)*10, 100)}),
					dict(method='COBYLA', options={'maxiter':min(len(model)*5, 100), 'rhobeg':0.1, 'tol':0.00000001, 'catol':0.00000001}), #likely too tight to actually converge, but will hand off a different point to SLSQP to restart
					dict(method='SLSQP', options={'ftol':1e-7, 'maxiter':max(len(model)*10, 100)}),
					#dict(method='COBYLA', options={'maxiter':max(len(model)*10, 1000), 'rhobeg':0.1, 'tol':0.00001, 'catol':0.00001}),
				)
		else:
			return (
				dict(method='SLSQP', options={'ftol':1e-6, 'maxiter':max(len(model)*10, 100)}),
			)
	if model.parameter_bounds() is not None:
		return (
			dict(method='SLSQP', options={'ftol':1e-9, }),
			dict(method='bhhh', slow_len=(5,2), slow_thresh=(0.001,1e-10)),
		)
	else:
		return (
			dict(method='bhhh', slow_len=(5,2), slow_thresh=(0.001,1e-10)),
			dict(method='SLSQP', options={'ftol':1e-9, }),
		)

def weight_choice_rebalance(model):
	ch = model.DataEdit("Choice")
	ch_tot = ch.sum(1)
	wg = model.DataEdit("Weight")
	if not numpy.allclose(ch_tot, 1.0):
		wg *= ch_tot
		ch /= (ch_tot[:,numpy.newaxis,:] + (ch_tot[:,numpy.newaxis,:]==0))
		return True
	return False



class _NameOrNameTimesNumberOrNumber:
	def __init__(self, a, text_of_a="input"):
		if not isinstance(a,ast.AST):
			try:
				a = ast.parse(a, mode='eval').body
			except TypeError:
				print('type(a)=',type(a))
				print('str(a)=',str(a))
				print('repr(a)=',repr(a))
				raise
		if isinstance(a, ast.Name):
			self.id = a.id
			self.n = 1
		elif isinstance(a, ast.Num):
			self.id = None
			self.n = a.n
		elif isinstance(a, ast.UnaryOp):
			if isinstance(a.op, ast.USub) and isinstance(a.operand, ast.Num):
				self.id = None
				self.n = -(a.operand.n)
			elif isinstance(a.op, ast.UAdd) and isinstance(a.operand, ast.Num):
				self.id = None
				self.n = a.operand.n
		else:
			if not isinstance(a, ast.BinOp) or not isinstance(a.op, ast.Mult):
				raise TypeError("must give a name or a name times a number, in {}".format(text_of_a))
			
			lefty = _NameOrNameTimesNumberOrNumber(a.left)
			righty = _NameOrNameTimesNumberOrNumber(a.right)
			
			if lefty.isName() and righty.isNumber():
				self.id = lefty.id
				self.n = righty.n
			elif righty.isName() and lefty.isNumber():
				self.id = righty.id
				self.n = lefty.n
			else:
				raise TypeError("must give a name or a name times a number, in {}".format(text_of_a))
	def nom(self):
		if self.id is None:
			return "{}".format(self.n)
		elif self.n==1:
			return self.id
		return "{}*{}".format(self.id,self.n)
	def isNumber(self):
		if self.id is None:
			return True
		return False
	def getNumber(self):
		if self.id is None:
			raise TypeError("not a number")
		return self.n
	def isName(self):
		if self.id is not None and self.n==1:
			return True
		return False

def _build_constraints(model, ignore_problems=False, include_bounds=True):
	if model.option.enforce_constraints:
		if include_bounds:
			b_constraints = model._bounds_as_constraints()
		else:
			b_constraints = ()
		try:
			model.constraints
		except AttributeError:
			constraints = ()
		else:
			constraints = {}
			for eachconstrain in model.constraints+b_constraints:
				if eachconstrain is None or eachconstrain=="":
					continue
				tree = ast.parse(eachconstrain, mode='eval')
				if not isinstance(tree.body, ast.Compare):
					raise TypeError("incompatible constraint, must be a comparison: "+eachconstrain)
				left = tree.body.left
				i = 0
				right = tree.body.comparators[i]
				op = tree.body.ops[i]
				while i<len(tree.body.comparators):
					
					try:
						lefty = _NameOrNameTimesNumberOrNumber(left)
						righty = _NameOrNameTimesNumberOrNumber(right)
						
						if lefty.id is not None and lefty.id not in model.parameter_names():
							if lefty.id in model.alias_names():
								lefty_old = lefty.id
								lefty.id = model.shadow_parameter[lefty_old].refers_to
								lefty.n *= model.shadow_parameter[lefty_old].multiplier
							else:
								raise KeyError("parameter not found: {} in constraint {}".format(lefty.id,eachconstrain))

						if righty.id is not None and righty.id not in model.parameter_names():
							if righty.id in model.alias_names():
								righty_old = righty.id
								righty.id = model.shadow_parameter[righty_old].refers_to
								righty.n *= model.shadow_parameter[righty_old].multiplier
							else:
								raise KeyError("parameter not found: {} in constraint {}".format(righty.id,eachconstrain))

						if isinstance(op, (ast.GtE, ast.Gt )):
							if lefty.id is None and righty.id is None:
								raise TypeError("incompatible constraint: "+eachconstrain)
							elif lefty.id is not None and righty.id is not None and (righty.n<0 or lefty.n<0):
								raise TypeError("incompatible constraint, negative multipliers not yet implemented: "+eachconstrain)
							elif lefty.id is None and righty.n>0:
								c = _build_single_lte_constraint(model[righty.id].index, lefty.n/righty.n, '{}>={}'.format(lefty.nom(),righty.nom()))
							elif lefty.id is None and righty.n<0:
								c = _build_single_gte_constraint(model[righty.id].index, lefty.n/righty.n, '{}>={}'.format(lefty.nom(),righty.nom()))
							elif righty.id is None and lefty.n>0:
								c = _build_single_gte_constraint(model[lefty.id].index, righty.n/lefty.n, '{}>={}'.format(lefty.nom(),righty.nom()))
							elif righty.id is None and lefty.n<0:
								c = _build_single_lte_constraint(model[lefty.id].index, righty.n/lefty.n, '{}>={}'.format(lefty.nom(),righty.nom()))
							else:
								c = _build_ineq_constraint_scaled(model[lefty.id].index, lefty.n, model[righty.id].index, righty.n, '{}>={}'.format(lefty.nom(),righty.nom()))
						elif isinstance(op, (ast.LtE, ast.Lt )):
							if lefty.id is None and righty.id is None:
								raise TypeError("incompatible constraint: "+eachconstrain)
							elif lefty.id is None and righty.n>0:
								c = _build_single_gte_constraint(model[righty.id].index, lefty.n/righty.n, '{}<={}'.format(lefty.nom(),righty.nom()))
							elif lefty.id is None and righty.n<0:
								c = _build_single_lte_constraint(model[righty.id].index, lefty.n/righty.n, '{}<={}'.format(lefty.nom(),righty.nom()))
							elif righty.id is None and lefty.n>0:
								c = _build_single_lte_constraint(model[lefty.id].index, righty.n/lefty.n, '{}<={}'.format(lefty.nom(),righty.nom()))
							elif righty.id is None and lefty.n<0:
								c = _build_single_gte_constraint(model[lefty.id].index, righty.n/lefty.n, '{}<={}'.format(lefty.nom(),righty.nom()))
							else:
								c = _build_ineq_constraint_scaled(model[righty.id].index, righty.n, model[lefty.id].index, lefty.n, '{}<={}'.format(lefty.nom(),righty.nom()))
						else:
							raise TypeError("incompatible constraint: "+eachconstrain)

						constraints[c['description']] = c
					except:
						if not model.option.ignore_bad_constraints:
							raise
					left = right
					i += 1
					if i>=len(tree.body.comparators): break
					right = tree.body.comparators[i]
					op = tree.body.ops[i]
			constraints = tuple(constraints.values())
	else:
		constraints = ()
	if model.option.enforce_network_constraints:
		constraints = constraints + tuple(model.network_based_constraints())
	return constraints

def maximize_loglike(model, *arg, ctol=1e-6, options={}, metaoptions=None, two_stage_constraints=False, pre_bhhh=0):
	"""
	Maximize the log likelihood of the model.
	
	Parameters
	----------
	arg : optimizers
	ctol : float
		The global convergence tolerance
	options : dict
		Options to pass to the outer minimizer
	metaoptions :
		Options to pass to the inner minimizers
	two_stage_constraints : bool
	pre_bhhh : int
		How many BHHH steps should be attempted before switching to the other optimizers.
		No convergence checks are made nor bounds or constraints enforced on these
		simple pre-steps, but it can be useful to help warm-start other algorithms (esp. SLSQP)
		that can enforce these things but perform badly with poor starting points.
	"""
	if metaoptions is not None:
		options['options'] = metaoptions
	stat = runstats()
	if not model.Data_UtilityCE_manual.active():
		stat.start_process('setup')
		model.tearDown()
		if not model.is_provisioned() and model._ref_to_db is not None:
			model.provision(idca_avail_ratio_floor = model.option.idca_avail_ratio_floor)
		model.setUp(False)

	if pre_bhhh:
		stat.start_process('pre_bhhh')
		while pre_bhhh>0:
			try:
				model._bhhh_simple_step()
			except RuntimeError:
				pre_bhhh = 0
			pre_bhhh -= 1

	from ...metamodel import MetaModel
	if isinstance(model, MetaModel):
		stat.start_process('setup_meta')
		model.setUp()
	
	try:
		self.df.cache_alternatives()
	except:
		pass

	x0 = model.parameter_values()
	if model.option.calc_null_likelihood:
		stat.start_process('null_likelihood')
		llnull = model.loglike_null()
		model._LL_null = float(llnull)

	if model.option.weight_choice_rebalance:
		stat.start_process("weight choice rebalance")
		if model.weight_choice_rebalance():
			stat.write("rebalanced weights and choices")

	if model.option.weight_autorescale:
		stat.start_process("weight autorescale")
		stat.write(model.auto_rescale_weights())
	
	stat.end_process()
	try:
		use_cobyla = model.use_cobyla
	except AttributeError:
		use_cobyla = False
	if use_cobyla:
		constraints = model._build_constraints(include_bounds=True)
	else:
		constraints = model._build_constraints()

	model._built_constraints_cache = constraints

	bounds=None
	if model.option.enforce_bounds and not use_cobyla:
		bounds=model.parameter_bounds()

	if two_stage_constraints:
		if len(arg):
			ot = model.optimizers(*arg, ctol=ctol)
		else:
			ot = model.optimizers(*_default_optimizers(model), ctol=ctol)
		r0 = _minimize(lambda z: 0.123999, x0, method=ot, options=options, bounds=None, constraints=() )
		print(model)
		if bounds or constraints:
			ctol = None
		if len(arg):
			ot = model.optimizers(*arg, ctol=ctol)
		else:
			ot = model.optimizers(*_default_optimizers(model), ctol=ctol)
		r = _minimize(lambda z: 0.123999, x0, method=ot, options=options, bounds=bounds, constraints=constraints )
		r.prepend(r0)
		r.stats.prepend_timing(stat)
	else:
		if bounds or constraints:
			if "BHHH" not in arg:
				ctol = None
		if len(arg):
			ot = model.optimizers(*arg, ctol=ctol)
		else:
			ot = model.optimizers(*_default_optimizers(model), ctol=ctol)
		r = _minimize(lambda z: 0.123999, x0, method=ot, options=options, bounds=bounds, constraints=constraints )
		r.stats.prepend_timing(stat)
	
#	try:
#		r_message = r.intermediate[-1].message
#	except:
#		r_message = ""
#	if r_message == 'Positive directional derivative for linesearch' and len(constraints) and model.option.enforce_constraints:
#		model.option.enforce_constraints = False
#		r0 = r
#		r1 = _minimize(lambda z: 0.123999, x0, method=ot, options=options, bounds=bounds, constraints=() )
#		model.option.enforce_constraints = True
#		r = _minimize(lambda z: 0.123999, x0, method=ot, options=options, bounds=bounds, constraints=constraints )

	if model.logger():
		model.logger().log(30,"Preliminary Results\n{!s}".format(model.art_params().ascii()))

	ll = model.loglike()

	if model.option.weight_autorescale and model.get_weight_scale_factor() != 1.0:
		r.stats.start_process("weight unrescale")
		model.restore_scale_weights()
		model.clear_cache()
		ll = model.loglike(cached=False)

	if model.option.calc_std_errors:
		r.stats.start_process("parameter covariance")
		if len(constraints) == 0:
			model.calculate_parameter_covariance()
			holdfasts = model.parameter_holdfast_array
		else:
			holdfasts = model._compute_constrained_covariance(constraints=constraints)
		from ...linalg import possible_overspecification
		overspec = possible_overspecification(model.hessian_matrix, holdfasts)
		if overspec:
			r.stats.write("WARNING: Model is possibly over-specified (hessian is nearly singular).")
			r.possible_overspecification = []
			for eigval, ox, eigenvec in overspec:
				if eigval=='LinAlgError':
					r.possible_overspecification.append( (eigval, [ox,], ["",]) )
				else:
					paramset = list(numpy.asarray(model.parameter_names())[ox])
					r.possible_overspecification.append( (eigval, paramset, eigenvec[ox]) )
			model.possible_overspecification = r.possible_overspecification

	r.stats.start_process("cleanup")
	r.stats.number_threads = model.option.threads
	ll = float(ll)
	model._LL_best = ll
	model._LL_current = ll
	r.loglike = ll
	if model.option.calc_null_likelihood:
		r.loglike_null = llnull
	r.stats.end_process()
	# peak memory usage
	from ..sysinfo import get_peak_memory_usage
	r.peak_memory_usage = get_peak_memory_usage()
	# installed memory
	try:
		import psutil
	except ImportError:
		pass
	else:
		mem = psutil.virtual_memory().total
		if mem >= 2.0*2**30:
			mem_size = str(mem/2**30) + " GiB"
		else:
			mem_size = str(mem/2**20) + " MiB"
		r.installed_memory = mem_size
	# save
	model._set_estimation_run_statistics_pickle(r.stats.pickled_dictionary())
	model.maximize_loglike_results = r
	del model._built_constraints_cache
	if model.logger():
		model.logger().log(30,"Final Results\n{!s}".format(model.art_params().ascii()))

	try:
		self.df.uncache_alternatives()
	except:
		pass

	return r



#def maximize_with(model, techniques, btol=1e-5):
#	if not model.is_provisioned() and model._ref_to_db is not None:
#		model.provision()
#	metaresult = OptimizeResult(x=model.parameter_values(), message="incomplete run",
#	                            niter=[], status=-9, success=False)
#	for technique in techniques:
#		if model.logger():
#			model.logger().log(30, "Using {!s}".format(technique))
#		result, rtype = technique.go(model)
#		try:
#			metaresult.niter.append( (technique.method, result.nit) )
#		except AttributeError:
#			pass
#		if rtype in (outcomes.success, outcomes.slow):
#			techniques.unfail_all(but=technique)
#		actual_btol = model.bhhh_tolerance()
#		if numpy.abs(actual_btol) < btol:
#			metaresult.btol = actual_btol
#			metaresult.x = result.x
#			metaresult.message = "Optimization terminated successfully using {}.".format(technique.method)
#			metaresult.status = 0
#			metaresult.success = True
#			try:
#				metaresult.jac = result.jac
#			except AttributeError:
#				pass
#			return metaresult
#	return metaresult

#
#def _maximize_loglike_using_technique(model, technique):
#	technique.last_result = model.maximize_loglike(
#		method=technique.method,
#		constraints=technique.constraints,
#		tol=technique.tol,
#		options=technique.options,
#		callback=Watcher(model, model.parameter_values(), technique.watcher_args),
#		)
#
#
#def maximize_loglike(model, x0=None, method=None, constraints=(), tol=None, callback=None, options=None, **kwrds):
#	method_str = str(method)
#	if x0 is None:
#		x0 = model.parameter_values()
#	if not model.is_provisioned() and model._ref_to_db is not None:
#		model.provision()
#	if isinstance(method,str) and method.lower()=='bfgs-init':
#		from .algorithms import _minimize_bfgs_1
#		method = _minimize_bfgs_1
#		if 'init_inv_hess' in kwrds:
#			hess = kwrds['init_inv_hess']
#		else:
#			hess = numpy.linalg.inv(numpy.asarray(model.bhhh(x0)))
#	elif method in ['Newton-CG-bhhh', 'dogleg-bhhh', 'trust-ncg-bhhh']:
#		hess = model.bhhh
#		method = method[:-5]
#	else:
#		hess = None
##	if callback is None:
##		callback = Watcher(model, x0, 'slow',2,10000.0)
#	try:
#		r = _minimize(model.negative_loglike, x0, method=method, jac=model.negative_d_loglike,
#						 hess=hess, bounds=model.parameter_bounds(), constraints=constraints,
#						 tol=tol, callback=callback, options=options)
#	except ProgressTooSlow as err:
#		r = err.result()
#		r.slow = err.slowness
#	except UserToleranceMet as suc:
#		r = suc.result()
#		r.message = "Optimization terminated successfully per BHHH tolerance using {}.".format(method_str)
#	return r

_LTE = "\u2266"
_GTE = "\u2267"
_NoBreakSpace = "\u00A0"

class _abandoned():
	def __init__(self, to_slot=None, to_name=None, mult=None, fixed_value=None, this_slot=None, this_mult=None, symbol="\u2266"):
		self.first_slot = this_slot
		self.first_mult = this_mult
		self.to_slot = to_slot
		self.to_name = to_name
		self.mult = mult
		self.fixed_value = None
		if fixed_value is not None:
			self.fixed_value = float(fixed_value)
		self.symbol = symbol
	def get_fixie(self):
		if self.mult > 0 and self.fixed_value is not None:
			t = _GTE+_NoBreakSpace+"{}".format(self.mult * (self.fixed_value))
		elif self.mult > 0:
			t = _GTE+_NoBreakSpace+"{}".format(self.mult * 0)
		else:
			if self.fixed_value is not None:
				t = _LTE+_NoBreakSpace+"{}".format(self.mult * -(self.fixed_value))
			else:
				t = _LTE+_NoBreakSpace+"{}".format(self.mult * 0)
		return t
	def __repr__(self):
		s = "_abandoned...\n"
		s += "     1st_slot:{!s}".format(self.first_slot)
		s += "     1st_mult:{!s}".format(self.first_mult)
		s += "      to_name:{!s}".format(self.to_name)
		s += "      to_slot:{!s}".format(self.to_slot)
		s += "      to_name:{!s}".format(self.to_name)
		s += "         mult:{!s}".format(self.mult)
		s += "  fixed_value:{!s}".format(self.fixed_value)
		return s

def _compute_constrained_d2_loglike_and_bhhh(model, *args, constraints=(), priority_list=None, skip_d2=False):
	# constraints are pre-built by _build_constraints here
	if len(constraints)==0:
		return model.d2_loglike(*args), model.bhhh(*args), model.d_loglike(*args), {}
	try:
		priority_list = model.parameter_priority_list
	except AttributeError:
		priority_list = {}
	if not skip_d2:
		d2 = numpy.asarray(model.d2_loglike(*args))
	d1 = numpy.asarray(model.d_loglike(*args))
	bh = numpy.asarray(model.bhhh(*args))
	abandoned_slots = {}
	for cstrt in constraints:
		is_active = cstrt['is_active'](model.parameter_array)
		if not is_active:
			continue # cstrt not active
		
		if ">=" in cstrt['description']:
			p_hi, p_lo = cstrt['description'].split(">=")
			op = ">="
		elif "<=" in cstrt['description']:
			p_lo, p_hi = cstrt['description'].split("<=")
			op = "<="
		else:
			continue
		jac = numpy.asarray(cstrt['jac'](model.parameter_array)).ravel()
		try:
			hi_slot = numpy.where(numpy.asarray(jac>0).ravel())[0][0]
		except IndexError:
			# ineq constraint is against a fixed number, do not merge just drop
			hi_slot = None
		try:
			lo_slot = numpy.where(numpy.asarray(jac<0).ravel())[0][0]
		except IndexError:
			# ineq constraint is against a fixed number, do not merge just drop
			lo_slot = None

		jac_hi = jac[hi_slot]
		jac_lo = jac[lo_slot]

		mult_1 = -jac_hi/jac_lo
		hi_slot_1 = hi_slot

		mult_2 = -jac_lo/jac_hi
		lo_slot_1 = lo_slot

		loop_detector = set([hi_slot])
		while hi_slot is not None and hi_slot in abandoned_slots:
			jac_hi *= abandoned_slots[hi_slot].mult
			p_hi = abandoned_slots[hi_slot].to_name
			hi_slot = abandoned_slots[hi_slot].to_slot
			if hi_slot in loop_detector:
				raise TypeError('hi_slot loop')
			else:
				loop_detector.add(hi_slot)
		loop_detector = set([lo_slot])
		while lo_slot is not None and lo_slot in abandoned_slots:
			jac_lo *= abandoned_slots[lo_slot].mult
			p_lo = abandoned_slots[lo_slot].to_name
			lo_slot = abandoned_slots[lo_slot].to_slot
			if lo_slot in loop_detector:
				raise TypeError('lo_slot loop')
			else:
				loop_detector.add(lo_slot)

		keep_hi = True
		if p_hi is None:
			p_hi_1 = None
		else:
			p_hi_1 = _NameOrNameTimesNumberOrNumber(p_hi)
		if p_lo is None:
			p_lo_1 = None
		else:
			p_lo_1 = _NameOrNameTimesNumberOrNumber(p_lo)
		if p_lo_1 is not None and p_lo_1.id in priority_list:
			if p_hi_1 is not None and not p_hi_1.id in priority_list:
				keep_hi = False
			else:
				if p_hi_1 is not None and p_lo_1 is not None and priority_list[p_hi_1.id] < priority_list[p_lo_1.id]:
					keep_hi = False


#		if p_hi is not None:
#			p_hi_1 = _NameOrNameTimesNumberOrNumber(p_hi)
#		else:
#			p_hi_1 = None
#		if p_lo is not None:
#			p_lo_1 = _NameOrNameTimesNumberOrNumber(p_lo)
#		else:
#			p_lo_1 = None
#
#		if p_hi_1 is None:
#			keep_hi = False
#		else:
#			if p_lo_1 is not None and p_lo_1.id in priority_list:
#				if p_hi_1 is not None and not p_hi_1.id in priority_list:
#					keep_hi = False
#				else:
#					if p_hi_1 is not None and priority_list[p_hi_1.id] < priority_list[p_lo_1.id]:
#						keep_hi = False

		if hi_slot is not None and lo_slot is not None and keep_hi:
			if not skip_d2:
				d2[hi_slot, :] -= d2[lo_slot, :]*jac_hi/jac_lo
				d2[lo_slot, :] = 0
				d2[:, hi_slot] -= d2[:, lo_slot]*jac_hi/jac_lo
				d2[:, lo_slot] = 0
			bh[hi_slot, :] -= bh[lo_slot, :]*jac_hi/jac_lo
			bh[lo_slot, :] = 0
			bh[:, hi_slot] -= bh[:, lo_slot]*jac_hi/jac_lo
			bh[:, lo_slot] = 0
			d1[hi_slot] -= d1[lo_slot]*jac_hi/jac_lo
			d1[lo_slot] = 0
			abandoned_slots[lo_slot] = _abandoned(to_slot=hi_slot, to_name=p_hi, mult=-jac_hi/jac_lo, this_slot=hi_slot_1, this_mult=mult_1)
		elif hi_slot is not None and lo_slot is not None:
			if not skip_d2:
				d2[lo_slot, :] -= d2[hi_slot, :]*jac_lo/jac_hi
				d2[hi_slot, :] = 0
				d2[:, lo_slot] -= d2[:, hi_slot]*jac_lo/jac_hi
				d2[:, hi_slot] = 0
			bh[lo_slot, :] -= bh[hi_slot, :]*jac_lo/jac_hi
			bh[hi_slot, :] = 0
			bh[:, lo_slot] -= bh[:, hi_slot]*jac_lo/jac_hi
			bh[:, hi_slot] = 0
			d1[lo_slot] -= d1[hi_slot]*jac_lo/jac_hi
			d1[hi_slot] = 0
			abandoned_slots[hi_slot] = _abandoned(to_slot=lo_slot, to_name=p_lo, mult=-jac_lo/jac_hi, this_slot=lo_slot_1, this_mult=mult_2, symbol=_GTE)
		elif hi_slot is None and lo_slot is not None:
			if not skip_d2:
				d2[lo_slot, :] = 0
				d2[:, lo_slot] = 0
			bh[lo_slot, :] = 0
			bh[:, lo_slot] = 0
			d1[lo_slot] = 0
			abandoned_slots[lo_slot] = _abandoned(mult=jac_lo, fixed_value=p_hi)
		elif hi_slot is not None and lo_slot is None:
			if not skip_d2:
				d2[hi_slot, :] = 0
				d2[:, hi_slot] = 0
			bh[hi_slot, :] = 0
			bh[:, hi_slot] = 0
			d1[hi_slot] = 0
			abandoned_slots[hi_slot] = _abandoned(mult=jac_hi, fixed_value=p_lo)
	if model.option.enforce_bounds and model.parameter_bounds() is not None:
		for slot, (min_bound, max_bound) in enumerate(model.parameter_bounds()):
			if min_bound is not None:
				if model.parameter_array[slot] - min_bound < 1e-6:
					if not skip_d2:
						d2[slot, :] = 0
						d2[:, slot] = 0
					bh[slot, :] = 0
					bh[:, slot] = 0
					d1[slot] = 0
					abandoned_slots[slot] = _abandoned(fixed_value=min_bound, mult=1)
			if max_bound is not None:
				if max_bound - model.parameter_array[slot] < 1e-6:
					if not skip_d2:
						d2[slot, :] = 0
						d2[:, slot] = 0
					bh[slot, :] = 0
					bh[:, slot] = 0
					d1[slot] = 0
					abandoned_slots[slot] = _abandoned(fixed_value=max_bound, mult=-1)
	#model.bhhh_constrained = bh
	#model.d_loglike_constrained = d1
	if skip_d2:
		d2 = None
	return d2, bh, d1, abandoned_slots



def _compute_constrained_covariance(model, constraints=()):
	# constraints are pre-built by _build_constraints here
	d2, bh, d1, abandoned_slots = model._compute_constrained_d2_loglike_and_bhhh(constraints=constraints)
	holdfasts = model.parameter_holdfast_array.copy()
	model.t_stat_replacements = [None] * len(model)
	for i in abandoned_slots:
		holdfasts[i] = 1
		if abandoned_slots[i].first_slot is not None:
			t = abandoned_slots[i].symbol+_NoBreakSpace+"{}".format(model[int(abandoned_slots[i].first_slot)].name)
			if abandoned_slots[i].first_mult != 1:
				t += " * {}".format(abandoned_slots[i].first_mult)
		elif abandoned_slots[i].to_slot is None:
			t = abandoned_slots[i].get_fixie()
		else:
			t = abandoned_slots[i].symbol+_NoBreakSpace+"{}".format(model[int(abandoned_slots[i].to_slot)].name)
			if abandoned_slots[i].mult != 1:
				t += " * {}".format(abandoned_slots[i].mult)
		model.t_stat_replacements[i] = t
	hh = holdfasts==0
	from ...linalg import matrix_inverse
	robust_covar = numpy.zeros_like(d2)
	covar = numpy.zeros_like(d2)
	try:
		invhess_squeezed = matrix_inverse(d2[hh,:][:,hh])[:,:]
	except numpy.linalg.linalg.LinAlgError:
		invhess_squeezed = numpy.full_like(d2[hh,:][:,hh], numpy.nan, dtype=numpy.float64)
	bhhh_squeezed = bh[hh,:][:,hh]
	result_squeezed = invhess_squeezed @ bhhh_squeezed @ invhess_squeezed
	robust_covar_i = numpy.zeros([robust_covar.shape[0], invhess_squeezed.shape[1]])
	covar_i = numpy.zeros([covar.shape[0], invhess_squeezed.shape[1]])
	robust_covar_i[hh,:] = result_squeezed[:,:]
	robust_covar[:,hh] = robust_covar_i
	covar_i[hh,:] = -invhess_squeezed[:,:]
	covar[:,hh] = covar_i
	model.covariance_matrix[:,:] = covar
	model.robust_covariance_matrix[:,:] = robust_covar
	model.hessian_matrix[:,:] = d2
	#return covar, robust_covar
	return holdfasts







def parameter_bounds(model):
	bounds = []
	any_finite_bound = False
	for par in model:
		min = par.min_value
		max = par.max_value
		if numpy.isinf(min):
			min = None
		else:
			any_finite_bound = True
		if numpy.isinf(max):
			max = None
		else:
			any_finite_bound = True
		bounds.append( (min, max) )
	if any_finite_bound:
		return bounds

def _bounds_as_constraints(model):
	bounds_c = []
	for par in model:
		min = par.min_value
		max = par.max_value
		if numpy.isinf(min):
			min = None
		else:
			bounds_c.append( "{}>={}".format(par.name,min) )
		if numpy.isinf(max):
			max = None
		else:
			bounds_c.append( "{}<={}".format(par.name,max) )
	return tuple(bounds_c)


def _build_ineq_constraint(gtslot, ltslot, descrip):
	from scipy.sparse import coo_matrix
	fun = lambda x: x[gtslot] - x[ltslot]
	scala = lambda x: (numpy.abs(x[gtslot]) + numpy.abs(x[ltslot]))/2
	constraint = {
		'type':'ineq',
		'fun': fun,
		'jac': lambda x: coo_matrix(([1,-1],([gtslot,ltslot],[0,0])), shape=(len(x),1)).todense().flatten(),
		'description': descrip,
		'is_active': lambda x: fun(x)<1e-4*scala(x) ,
		}
	return constraint

def _build_ineq_constraint_scaled(gtslot, gtmultiple, ltslot, ltmultiple, descrip):
	from scipy.sparse import coo_matrix
	fun = lambda x: (x[gtslot] * gtmultiple) - (x[ltslot] * ltmultiple)
	scala = lambda x: (numpy.abs(x[gtslot] * gtmultiple) + numpy.abs(x[ltslot] * ltmultiple))/2
	constraint = {
		'type':'ineq',
		'fun': fun ,
		'jac': lambda x: coo_matrix(([gtmultiple,-ltmultiple],([gtslot,ltslot],[0,0])), shape=(len(x),1)).todense().flatten(),
		'description': descrip,
		'is_active': lambda x: fun(x)<1e-4*scala(x) ,
		}
	return constraint


def _build_single_gte_constraint(gtslot, pivot, descrip):
	from scipy.sparse import coo_matrix
	fun = lambda x: x[gtslot] - pivot
	scala = lambda x: (numpy.abs(x[gtslot]) + numpy.abs(pivot))/2
	constraint = {
		'type':'ineq',
		'fun': fun ,
		'jac': lambda x: coo_matrix(([1,],([gtslot,],[0,])), shape=(len(x),1)).todense().flatten(),
		'description': descrip,
		'is_active': lambda x: fun(x)<1e-4*scala(x) ,
		}
	return constraint

def _build_single_lte_constraint(ltslot, pivot, descrip):
	from scipy.sparse import coo_matrix
	fun = lambda x: pivot - x[ltslot]
	scala = lambda x: (numpy.abs(x[ltslot]) + numpy.abs(pivot))/2
	constraint = {
		'type':'ineq',
		'fun': fun,
		'jac': lambda x: coo_matrix(([-1,],([ltslot,],[0,])), shape=(len(x),1)).todense().flatten(),
		'description': descrip,
		'is_active': lambda x: fun(x)<1e-4*scala(x) ,
		}
	return constraint



def network_based_constraints(model):
	G = model.networkx_digraph()
	elementals = set(model.alternative_codes())
	constraint_set = {}
	constraints = []
	for n in G.nodes():
		if n not in elementals and n != model.root_id:
			lower = model.parameter_index(model.node[n].param)
			lower_str = 'x[{}]'.format(lower)
			uppers = G.predecessors(n)
			for uppercode in uppers:
				if uppercode != model.root_id:
					upper = model.parameter_index(model.node[uppercode].param)
					upper_str = 'x[{}]'.format(model.parameter_index(model.node[uppercode].param))
					constraint_name = '{} > {}'.format(upper_str,lower_str,)
					if constraint_name not in constraint_set:
						constraint_set[constraint_name] = _build_ineq_constraint(upper,lower,constraint_name )
					#constraints.append(_build_ineq_constraint(upper,lower))
	return [constraint for constraint_name, constraint in constraint_set.items()]

def evaluate_network_based_constraints(model, x=None):
	if x is None:
		x = model.parameter_values()
	return [c['fun'](x) for c in model.network_based_constraints()]

#Constraints definition (only for COBYLA and SLSQP). Each constraint is defined in a dictionary with fields:
#type : str
#Constraint type: ‘eq’ for equality, ‘ineq’ for inequality.
#fun : callable
#The function defining the constraint.
#jac : callable, optional
#The Jacobian of fun (only for SLSQP).
#args : sequence, optional
#Extra arguments to be passed to the function and Jacobian.
#Equality constraint means that the constraint function result is to be zero whereas inequality means that it is to be non-negative. Note that COBYLA only supports inequality constraints.


def _scipy_check_grad(model, x0=None):
	if x0 is None:
		x0 = model.parameter_values()
	from scipy.optimize import check_grad
	return check_grad(model.negative_loglike, model.negative_d_loglike, x0)

