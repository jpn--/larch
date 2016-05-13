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
		ch /= ch_tot[:,numpy.newaxis,:]
		return True
	return False

def maximize_loglike(model, *arg, ctol=1e-6, options={}, metaoptions=None):
	if metaoptions is not None:
		options['options'] = metaoptions
	stat = runstats()
	if not model.Data_UtilityCE_manual.active():
		stat.start_process('setup')
		model.tearDown()
		if not model.is_provisioned() and model._ref_to_db is not None:
			model.provision(idca_avail_ratio_floor = model.option.idca_avail_ratio_floor)
		model.setUp(False)

	from ...metamodel import MetaModel
	if isinstance(model, MetaModel):
		stat.start_process('setup_meta')
		model.setUp()

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
	constraints = ()
	if model.option.enforce_constraints:
		constraints = model.network_based_contraints()
	bounds=None
	if model.option.enforce_bounds:
		bounds=model.parameter_bounds()

	if bounds or constraints:
		ctol = None

	if len(arg):
		ot = model.optimizers(*arg, ctol=ctol)
	else:
		ot = model.optimizers(*_default_optimizers(model), ctol=ctol)

	r = _minimize(lambda z: 0.123999, x0, method=ot, options=options, bounds=bounds, constraints=constraints )
	r.stats.prepend_timing(stat)
	ll = model.loglike()

	if model.option.weight_autorescale and model.get_weight_scale_factor() != 1.0:
		r.stats.start_process("weight unrescale")
		model.restore_scale_weights()
		model.clear_cache()
		ll = model.loglike(cached=False)

	if model.option.calc_std_errors:
		r.stats.start_process("parameter covariance")
		model.calculate_parameter_covariance()
		from ...linalg import possible_overspecification
		overspec = possible_overspecification(model.hessian_matrix, model.parameter_holdfast_array)
		if overspec:
			r.stats.write("WARNING: Model is possibly over-specified (hessian is nearly singular).")
			r.possible_overspecification = []
			for eigval, ox in overspec:
				paramset = list(numpy.asarray(model.parameter_names())[ox])
			r.possible_overspecification.append( (eigval, paramset) )

	r.stats.start_process("cleanup")
	r.stats.number_threads = model.option.threads
	ll = float(ll)
	model._LL_best = ll
	model._LL_current = ll
	r.loglike = ll
	if model.option.calc_null_likelihood:
		r.loglike_null = llnull
	r.stats.end_process()
	model._set_estimation_run_statistics_pickle(r.stats.pickled_dictionary())
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



def _build_ineq_constraint(gtslot, ltslot, descrip):
	from scipy.sparse import coo_matrix
	constraint = {
		'type':'ineq',
		'fun': lambda x: x[gtslot] - x[ltslot] ,
		'jac': lambda x: coo_matrix(([1,-1],([gtslot,ltslot],[0,0])), shape=(len(x),1)).todense().flatten(),
		'description': descrip,
		}
	return constraint


def network_based_contraints(model):
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

def evaluate_network_based_contraints(model, x=None):
	if x is None:
		x = model.parameter_values()
	return [c['fun'](x) for c in model.network_based_contraints()]

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

