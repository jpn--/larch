# cython: language_level=3, embedsignature=True

include "../general_precision.pxi"
from ..general_precision import l4_float_dtype
from ..general_precision cimport l4_float_t

include "fastmath.pxi"
from libc.stdlib cimport malloc, free
from libc.math cimport exp, log

from cython.parallel cimport prange, parallel, threadid

from ..dataframes cimport DataFrames
from .tree_struct cimport TreeStructure
from .controller cimport Model5c
from .mnl cimport _mnl_log_likelihood_from_probability_stride
from .persist_flags cimport *

import numpy
import pandas

import logging
logger = logging.getLogger('L5')

cdef float INFINITY32 = numpy.float('inf')

cimport cython


cdef void _check_for_zero_mu(
		int             n_elemental_alts,
		int             n_nodes,
		l4_float_t[:]   mu,          # input         [n_nodes]  elemental alternatives are ignored
):
	cdef:
		int        parent

	for parent in range(n_elemental_alts, n_nodes):
		if mu[parent] == 0:
			raise ValueError('mu is zero')





@cython.cdivision(True)
@cython.boundscheck(False)
@cython.initializedcheck(False)
cdef void _nl_utility_upstream_v2(
		int             n_elemental_alts,
		int             n_nodes,
		l4_float_t[:]   utility,            # input/output  [n_nodes]
		l4_float_t[:]   mu,                 # input         [n_nodes]  elemental alternatives are ignored
		l4_float_t[:]   alpha,              # input         [n_edges]
		l4_float_t[:]   logalpha,           # input         [n_edges]
		int[:]          first_edge_for_up,  # input         [n_nodes] index of first edge where this node is the up
		int[:]          n_edges_for_up,     # input         [n_nodes] n edge where this node is the up
		int[:]          edge_dn             # input         [n_edges] child on each edge
) nogil:
	cdef:
		int        parent, child, n, n_children_for_parent, edge
		l4_float_t shifter=-1e100
		int        shifter_position=-1
		l4_float_t sum_expU = 0
		l4_float_t x
		l4_float_t z, this_alpha, this_logalpha

	for parent in range(n_elemental_alts, n_nodes):

		utility[parent] = 0
		shifter=-1e100
		shifter_position=-1
		n_children_for_parent = n_edges_for_up[parent]

		# if mu[parent] == 0: # Error Captured earlier

		for n in range(n_children_for_parent):
			edge = first_edge_for_up[parent]+n
			child = edge_dn[edge]
			if utility[child] > -INFINITY32:
				if alpha[edge] >0:
					z = (logalpha[edge] + utility[child]) / mu[parent]
					if z > shifter:
						shifter = z
						shifter_position = child

		#shifter = shifter/mu[parent]

		for n in range(n_children_for_parent):
			edge = first_edge_for_up[parent]+n
			child = edge_dn[edge]
			if utility[child] > -INFINITY32:
				if alpha[edge] >0:
					if shifter_position == child:
						utility[parent] += 1
					else:
						z = ((logalpha[edge] + utility[child]) / mu[parent]) - shifter
						utility[parent] += exp(z)

		utility[parent] = (log(utility[parent]) + shifter) * mu[parent]


@cython.cdivision(True)
cdef void _nl_conditional_logprobability_from_utility(
		int             n_edges,
		l4_float_t*     utility,                    # input  [n_nodes]
		l4_float_t*     mu,                         # input  [n_nodes]
		l4_float_t*     conditional_logprobability, # output [n_edges]
		int*            ups,                        # input  [n_edges]
		int*            dns,                        # input  [n_edges]
		l4_float_t*     alpha_param_values,         # input  [n_edges]
		l4_float_t*     logalpha_param_values,      # input  [n_edges]

) nogil:
	cdef:
		int        parent, child, edge
		l4_float_t sum_expU = 0
		l4_float_t x
		l4_float_t mu_parent

	for edge in range(n_edges):
		parent = ups[edge]
		child  = dns[edge]
		mu_parent = mu[parent]

		if utility[parent] <= -INFINITY32:
			conditional_logprobability[edge] = -INFINITY32
		elif mu_parent != 0:
			conditional_logprobability[edge] = (utility[child] - utility[parent]) / mu_parent
		elif utility[child] == utility[parent]:
			conditional_logprobability[edge] = 0
		else:
			conditional_logprobability[edge] = -INFINITY32

		if alpha_param_values[edge] != 1.0:
			conditional_logprobability[edge] += logalpha_param_values[edge] / mu_parent



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cdef void _nl_d_probability_from_d_utility(
		int             n_edges,
		int             n_params,
		l4_float_t[:]   utility,                    # input  [n_nodes]
		l4_float_t[:,:] d_utility,                  # input  [n_nodes, n_params]
		l4_float_t[:]   mu,                         # input  [n_nodes]
		l4_float_t[:]   scratch,                    # temp   [n_params]
		l4_float_t[:]   log_cond_probability,       # input  [n_edges]
		l4_float_t[:]   probability,                # input  [n_nodes]
		l4_float_t[:,:] d_probability,              # output [n_nodes, n_params]
		int[:]          ups,                        # input  [n_edges]
		int[:]          dns,                        # input  [n_edges]
		int[:]          param_slot_of_mu,           # input  [n_nodes]
		l4_float_t[:]   alpha_param_values,         # input  [n_edges]
		l4_float_t[:]   logalpha_param_values,      # input  [n_edges]
		l4_float_t[:]   array_ch,                   # input/output  [n_nodes]
) nogil:
	cdef:
		int        parent, child, edge, reversi_edge, param
		l4_float_t sum_expU = 0
		l4_float_t x
		l4_float_t multiplier
		l4_float_t mu_parent

	for edge in range(n_edges):
		child  = dns[edge]
		if array_ch[child]:
			parent = ups[edge]
			array_ch[parent] += array_ch[child]

	for param in range(n_params):
		scratch[param] = 0
		for child in range(d_probability.shape[0]):
			d_probability[child,param] = 0

	for reversi_edge in range(n_edges):
		edge = n_edges-reversi_edge-1
		child  = dns[edge]
		if True or array_ch[child]:
			parent = ups[edge]
			for param in range(n_params):
				scratch[param] = (d_utility[child, param] - d_utility[parent, param])

			mu_parent = mu[parent]

			if mu_parent != 0:
				if param_slot_of_mu[parent] >= 0:
					scratch[param_slot_of_mu[parent]] += (utility[parent] - utility[child]) / mu_parent
					if alpha_param_values[edge] != 1.0:
						scratch[param_slot_of_mu[parent]] -= logalpha_param_values[edge] / mu_parent

				multiplier = probability[parent]/mu_parent
			else:
				multiplier = 0

			for param in range(n_params):
				scratch[param] *= multiplier
				scratch[param] += d_probability[parent, param]
				d_probability[child, param] += scratch[param] * exp(log_cond_probability[edge])



@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef int _nl_d_loglike_from_d_probability(
		int             n_params,
		int             n_alts,
		l4_float_t[:]   probability,                # input  [n_nodes]
		l4_float_t[:,:] d_probability,              # input  [n_nodes, n_params]
		l4_float_t[:]   d_LL,                       # output [n_params]
		l4_float_t[:]   array_ch,                   # input/output  [n_nodes]
		l4_float_t      weight,                     # input scalar

		bint            return_bhhh,
		l4_float_t[:]   d_loglike_cum, # input [n_params]
		l4_float_t[:,:] bhhh_cum,      # input [n_alts, n_params]
		l4_float_t*     dLL_temp,      # temp  [n_params]

) nogil:
	cdef int a, i, v, v2, flag=0
	cdef l4_float_t total_probability_a, ch_over_pr, tempvalue, this_ch

	for i in range(n_params):
		d_LL[i] = 0

	for a in range(n_alts):

		this_ch = array_ch[a]

		if this_ch==0:
			continue

		total_probability_a = probability[a]
		if total_probability_a>0:
			ch_over_pr = this_ch/total_probability_a
			for i in range(n_params):
				tempvalue = d_probability[a,i] * ch_over_pr
				dLL_temp[i] = tempvalue / this_ch
				tempvalue *= weight
				d_LL[i] += tempvalue
				d_loglike_cum[i] += tempvalue

			if return_bhhh:
				for v in range(n_params):
					for v2 in range(n_params):
						bhhh_cum[v,v2] += dLL_temp[v] * dLL_temp[v2] * this_ch * weight

		else:
			flag= -1 # ZeroProbWhenChosen

	return flag # 0 on success, -1 on ZeroProbWhenChosen



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.initializedcheck(False)
cdef void _nl_d_utility_upstream_v2(
		int             n_elemental_alts,
		int             n_nodes,
		l4_float_t[:]   utility,                     # input [n_nodes]
		l4_float_t[:]   mu,                          # input [n_nodes]  elemental alternatives are ignored
		int[:]          param_slot_of_mu,            # input [n_nodes]
		l4_float_t[:]   alpha,                       # input [n_edges]
		l4_float_t[:]   logalpha,                       # input [n_edges]
		l4_float_t[:]   conditional_logprobability,  # input [n_edges]
		int             n_edges,
		int             n_params,
		l4_float_t[:,:] dU,           # input/output  [n_nodes, n_params]
		int[:]          ups,                        # input  [n_edges]
		int[:]          dns,                        # input  [n_edges]
) nogil:
	cdef:
		int parent, p, e, child
		l4_float_t  cond_logprob, cond_prob

	for parent in range(n_elemental_alts, n_nodes):
		for p in range(n_params):
			dU[parent, p] = 0

	for e in range(n_edges):
		parent = ups[e]
		child = dns[e]
		cond_logprob = conditional_logprobability[e]
		if cond_logprob > -INFINITY32:
			cond_prob = exp(cond_logprob)

			if child >= n_elemental_alts:
				if param_slot_of_mu[child] >= 0:
					dU[child,param_slot_of_mu[child]] += utility[child]
					dU[child,param_slot_of_mu[child]] /= mu[child]

			if param_slot_of_mu[parent] >= 0:
				if alpha[e] == 1.0:
					dU[parent, param_slot_of_mu[parent]] -= cond_prob * (utility[child])
				else:
					dU[parent, param_slot_of_mu[parent]] -= cond_prob * (utility[child] + mu[child]*logalpha[e])

			for p in range(n_params):
				dU[parent, p] += cond_prob * dU[child, p]


cdef void _nl_total_probability_from_conditional_logprobability(
		int             n_nodes,
		int             n_edges,
		l4_float_t*     total_probability,          # output [n_nodes]
		l4_float_t*     conditional_logprobability, # input  [n_edges]
		int*            ups,                        # input  [n_edges]
		int*            dns,                        # input  [n_edges]
) nogil:

	cdef:
		int e, reverse_edge, parent, child

	total_probability[n_nodes-1] = 1.0
	for e in range(n_nodes-1):
		total_probability[e] = 0.0

	for reverse_edge in range(n_edges):
		e = n_edges-reverse_edge-1
		child = dns[e]
		parent = ups[e]
		if total_probability[parent]:
			total_probability[child] += exp(conditional_logprobability[e]) * total_probability[parent]


# @cython.boundscheck(False)
# @cython.initializedcheck(False)
# cdef void _nl_total_probability_from_conditional_logprobability_safe(
# 		int             n_nodes,
# 		int             n_edges,
# 		l4_float_t[:]   total_probability,          # output [n_nodes]
# 		l4_float_t[:]   conditional_logprobability, # input  [n_edges]
# 		int[:]          ups,                        # input  [n_edges]
# 		int[:]          dns,                        # input  [n_edges]
# ):
#
# 	cdef:
# 		int e, reverse_edge, parent, child
#
# 	total_probability[n_nodes-1] = 1.0
# 	for e in range(n_nodes-1):
# 		total_probability[e] = 0.0
#
# 	for reverse_edge in range(n_edges):
# 		e = n_edges-reverse_edge-1
# 		child = dns[e]
# 		parent = ups[e]
# 		if total_probability[parent]:
# 			total_probability[child] += exp(conditional_logprobability[e]) * total_probability[parent]
#




@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
def nl_d_log_likelihood_from_dataframes_all_rows(
		DataFrames dfs,
		Model5c     model,
		int         num_threads=-1,
		bint        return_dll=True,
		bint        return_bhhh=False,
		int         start_case=0,
		int         stop_case=-1,
		int         step_case=1,
		int         persist=0,
		int         leave_out=-1,
		int         keep_only=-1,
		int         subsample= 1,
		bint        probability_only=False,
):
	cdef:
		int c = 0
		int v, v2
		int n_cases = dfs._n_cases()
		int n_alts  = dfs._n_alts()
		int n_params= dfs._n_model_params
		l4_float_t[:] array_ch
		l4_float_t[:,:] array_ch_wide  # thread-local
		l4_float_t[:]   LL_case
		l4_float_t[:,:] dLL_case
		l4_float_t[:,:] dLL_total # thread-local
		l4_float_t[:,:] dLL_temp
		l4_float_t[:,:] raw_utility
		l4_float_t[:,:] cond_logprobability
		l4_float_t[:,:] total_probability
		l4_float_t[:]   total_probability_case
		l4_float_t[:,:] scratch   # thread-local
		l4_float_t[:,:,:] dU # thread-local
		l4_float_t[:,:,:] dP # thread-local
		l4_float_t[:]   buffer_probability_
		l4_float_t      ll = 0
		l4_float_t      ll_temp
		l4_float_t*     choice
		l4_float_t      weight = 1 # default
		TreeStructure   tree
		l4_float_t[:,:,:] bhhh_total # thread-local
		int             thread_number = 0
		int             storage_size_U
		int             store_number_U
		int             storage_size_CP
		int             store_number_CP
		int             storage_size_P
		int             store_number_P
		int             storage_size_dP
		int             store_number_dP
		int             storage_size_LLc
		int             store_number_LLc
		int             storage_size_dLLc
		int             store_number_dLLc
		int             storage_size_dU
		int             store_number_dU

	if not dfs._is_computational_ready(activate=True):
		raise ValueError('DataFrames is not computational-ready')

	if dfs._data_ch is None and not probability_only:
		raise ValueError('DataFrames does not define data_ch')

	if dfs._data_av is None:
		raise ValueError('DataFrames does not define data_av')

	try:
		if num_threads <= 0:
			num_threads = model._n_threads

		if stop_case<0:
			stop_case = n_cases

		if return_bhhh:
			# must compute dll to get bhhh
			return_dll = True

		storage_size_U    = n_cases if persist & PERSIST_UTILITY            else num_threads
		storage_size_CP   = n_cases if persist & PERSIST_COND_LOG_PROB      else num_threads
		storage_size_P    = n_cases if persist & PERSIST_PROBABILITY        else num_threads
		storage_size_dP   = n_cases if persist & PERSIST_D_PROBABILITY      else num_threads
		storage_size_LLc  = n_cases if persist & PERSIST_LOGLIKE_CASEWISE   else num_threads
		storage_size_dLLc = n_cases if persist & PERSIST_D_LOGLIKE_CASEWISE else num_threads
		storage_size_dU   = n_cases if persist & PERSIST_D_UTILITY          else num_threads

		tree = TreeStructure(model, model._graph)
		_check_for_zero_mu(n_alts, tree.n_nodes, tree.model_mu_param_values)

		scratch             = numpy.zeros([num_threads,n_params], dtype=l4_float_dtype)

		raw_utility         = numpy.zeros([storage_size_U, tree.n_nodes], dtype=l4_float_dtype)
		cond_logprobability = numpy.zeros([storage_size_CP, tree.n_edges], dtype=l4_float_dtype)
		total_probability   = numpy.zeros([storage_size_P, tree.n_nodes], dtype=l4_float_dtype)

		array_ch_wide = numpy.zeros([num_threads, tree.n_nodes], dtype=l4_float_dtype)

		LL_case =  numpy.zeros([storage_size_LLc, ], dtype=l4_float_dtype)

		if return_dll:
			dU = numpy.zeros([storage_size_dU, tree.n_nodes, n_params], dtype=l4_float_dtype)
			dP = numpy.zeros([storage_size_dP, tree.n_nodes, n_params], dtype=l4_float_dtype)
			dLL_case  = numpy.zeros([storage_size_dLLc, n_params], dtype=l4_float_dtype)
			dLL_total = numpy.zeros([num_threads, n_params], dtype=l4_float_dtype)
			dLL_temp  = numpy.zeros([num_threads, n_params], dtype=l4_float_dtype)
		if return_bhhh:
			bhhh_total = numpy.zeros([num_threads,n_params,n_params], dtype=l4_float_dtype)

		with nogil, parallel(num_threads=num_threads):
			thread_number = threadid()

			for c in prange(start_case, stop_case, step_case):

				if leave_out >= 0 and c % subsample == leave_out:
					continue

				if keep_only >= 0 and c % subsample != keep_only:
					continue

				store_number_U    = c if persist & PERSIST_UTILITY            else thread_number
				store_number_CP   = c if persist & PERSIST_COND_LOG_PROB      else thread_number
				store_number_P    = c if persist & PERSIST_PROBABILITY        else thread_number
				store_number_dP   = c if persist & PERSIST_D_PROBABILITY      else thread_number
				store_number_LLc  = c if persist & PERSIST_LOGLIKE_CASEWISE   else thread_number
				store_number_dLLc = c if persist & PERSIST_D_LOGLIKE_CASEWISE else thread_number
				store_number_dU   = c if persist & PERSIST_D_UTILITY          else thread_number

				if return_dll:
					dfs._compute_d_utility_onecase(c,raw_utility[store_number_U,:],dU[store_number_dU],n_alts)
				else:
					dfs._compute_utility_onecase(c,raw_utility[store_number_U,:],n_alts)

				_nl_utility_upstream_v2(
					tree.n_elementals,
					tree.n_nodes,
					raw_utility[store_number_U,:], # in-out [n_nodes]
					tree.model_mu_param_values,  # input  [n_nodes]  elemental alternatives are ignored
					tree.edge_alpha_values,      # input  [n_edges]
					tree.edge_logalpha_values,   # input  [n_edges]
					tree.first_edge_for_up,      # input  [n_nodes] index of first edge where this node is the up
					tree.n_edges_for_up,         # input  [n_nodes] n edge where this node is the up
					tree.edge_dn                 # input  [n_edges] child on each edge
				)

				_nl_conditional_logprobability_from_utility(
						tree.n_edges,
						&raw_utility[store_number_U,0],          # input  [n_nodes]
						&tree.model_mu_param_values[0],        # input  [n_nodes]
						&cond_logprobability[store_number_CP,0],  # output [n_edges]
						&tree.edge_up[0],                      # input  [n_edges]
						&tree.edge_dn[0],                      # input  [n_edges]
						&tree.edge_alpha_values[0],            # input  [n_edges]
						&tree.edge_logalpha_values[0],         # input  [n_edges]
				)

				_nl_total_probability_from_conditional_logprobability(
						tree.n_nodes,
						tree.n_edges,
						&total_probability[store_number_P,0],    # output [n_nodes]
						&cond_logprobability[store_number_CP,0],  # input  [n_edges]
						&tree.edge_up[0],                      # input  [n_edges]
						&tree.edge_dn[0],                      # input  [n_edges]
				)

				if probability_only:
					continue

				if dfs._array_wt is not None:
					weight = dfs._array_wt[c]
				else:
					weight = 1

				ll_temp = _mnl_log_likelihood_from_probability_stride(
					n_alts,
					total_probability[store_number_P,:],        # input [n_alts]
					dfs._array_ch[c,:],                       # input [n_alts]
				) * weight
				LL_case[store_number_LLc] += ll_temp
				ll += ll_temp

				if return_dll:
					_nl_d_utility_upstream_v2(
						tree.n_elementals,
						tree.n_nodes,
						raw_utility[store_number_U,:],          # input [n_nodes]
						tree.model_mu_param_values,           # input [n_nodes]  elemental alternatives are ignored
						tree.model_mu_param_slots,
						tree.edge_alpha_values,               # input [n_nodes,n_nodes]
						tree.edge_logalpha_values,            # input [n_nodes,n_nodes]
						cond_logprobability[store_number_CP,:],  # input [n_edges]
						tree.n_edges,
						n_params,
						dU[store_number_dU],                    # input/output  [n_nodes, n_params]
						tree.edge_up,                         # input  [n_edges]
						tree.edge_dn,                         # input  [n_edges]
					)

					dfs._copy_choice_onecase(c, array_ch_wide[thread_number])

					_nl_d_probability_from_d_utility(
						tree.n_edges,                         # input   int
						n_params,                             # input   int
						raw_utility[store_number_U,:],          # input  [n_nodes]
						dU[store_number_dU],                    # input  [n_nodes, n_params]
						tree.model_mu_param_values,           # input  [n_nodes]
						scratch[thread_number],               # temp   [n_params]
						cond_logprobability[store_number_CP,:],  # input  [n_edges]
						total_probability[store_number_P,:],    # input  [n_nodes]
						dP[store_number_dP],                    # output [n_nodes, n_params]
						tree.edge_up,                         # input  [n_edges]
						tree.edge_dn,                         # input  [n_edges]
						tree.model_mu_param_slots,            # input  [n_nodes]
						tree.edge_alpha_values,               # input  [n_edges]
						tree.edge_logalpha_values,            # input  [n_edges]
						array_ch_wide[thread_number],         # in-out [n_nodes]
					)

					if weight:
						_nl_d_loglike_from_d_probability(
							n_params,                           # input   int
							n_alts,                             # input   int
							total_probability[store_number_P,:],  # input  [n_nodes]
							dP[store_number_dP],                  # input  [n_nodes, n_params]
							dLL_case[store_number_dLLc,:],           # output [n_params]
							dfs._array_ch[c,:],                 # input  [n_nodes]
							weight,

							return_bhhh,
							dLL_total[thread_number],
							bhhh_total[thread_number],
							&dLL_temp[thread_number,0],
						)

		if probability_only:
			ll = numpy.nan

		ll *= dfs._weight_normalization
		if return_dll:
			dll = dLL_total.base.sum(0) * dfs._weight_normalization
		if return_bhhh:
			bhhh = bhhh_total.base.sum(0) * dfs._weight_normalization


		from ..util import Dict
		result = Dict(
			ll=ll,
		)
		if return_dll:
			result.dll = pandas.Series(
				data=dll,
				index=dfs._model_param_names,
			)
		if return_bhhh:
			result.bhhh = bhhh

		if persist & PERSIST_LOGLIKE_CASEWISE:
			result.ll_casewise = LL_case.base
		if persist & PERSIST_UTILITY:
			result.utility = raw_utility.base
		if persist & PERSIST_D_UTILITY:
			if return_dll:
				result.dutility = dU.base
		if persist & PERSIST_COND_LOG_PROB:
			result.conditional_log_prob = cond_logprobability.base
		if persist & PERSIST_PROBABILITY:
			result.probability   = total_probability.base
		if persist & PERSIST_D_PROBABILITY:
			if return_dll:
				result.dprobability = dP.base
		if persist & PERSIST_D_LOGLIKE_CASEWISE:
			if return_dll:
				for v in range(dLL_case.shape[0]):
					for v2 in range(dLL_case.shape[1]):
						dLL_case[v,v2] *= dfs._weight_normalization
				result.dll_casewise=pandas.DataFrame(
					dLL_case.base,
					columns=dfs._model_param_names,
				)

		return result

	except:
		# logger.error(f'c={c}')
		# logger.error(f'n_cases, n_alts={(n_cases, n_alts)}')
		# logger.error(f'array_ch={array_ch}')
		logger.exception('error in nl_log_likelihood_from_dataframes_all_rows')
		raise

