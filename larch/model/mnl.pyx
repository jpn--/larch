# cython: language_level=3, embedsignature=True

include "../general_precision.pxi"
from ..general_precision import l4_float_dtype
from ..general_precision cimport l4_float_t

include "fastmath.pxi"
from libc.stdlib cimport malloc, free
from libc.math cimport exp, log

from cython.parallel cimport prange, parallel, threadid

from ..dataframes cimport DataFrames
from .persist_flags cimport *

import numpy
import pandas

import logging
logger = logging.getLogger('L5')

cdef float INFINITY32 = numpy.float('inf')

cimport cython

ctypedef l4_float_t (*VERT)(l4_float_t x)

cdef float exp_(float x) nogil:
	return exp(x)

cdef float log_(float x) nogil:
	return log(x)





@cython.cdivision(True)
cdef void _mnl_probability_from_utility_approx2(
		int    n_alts,
		l4_float_t* utility,     # input
		l4_float_t* exp_utility, # output
		l4_float_t* probability, # output
):
	cdef:
		int i
		l4_float_t sum_expU = 0
		l4_float_t x

	for i in range(n_alts):
		if utility[i] < -3.402823e38:
			exp_utility[i] = 0
		else:
			x = exp_utility[i] = fasterexp(utility[i])
			sum_expU += x

	if sum_expU == 0:
		for i in range(n_alts):
			probability[i] = 0
	else:
		for i in range(n_alts):
			probability[i] = exp_utility[i] / sum_expU


@cython.cdivision(True)
cdef void _mnl_probability_from_utility_approx(
		int    n_alts,
		l4_float_t* utility,     # input
		l4_float_t* exp_utility, # output
		l4_float_t* probability, # output
):
	cdef:
		int i
		l4_float_t sum_expU = 0
		l4_float_t x

	for i in range(n_alts):
		if utility[i] < -3.402823e38:
			exp_utility[i] = 0
		else:
			x = exp_utility[i] = fastexp(utility[i])
			sum_expU += x

	if sum_expU == 0:
		for i in range(n_alts):
			probability[i] = 0
	else:
		for i in range(n_alts):
			probability[i] = exp_utility[i] / sum_expU


@cython.cdivision(True)
cdef void _mnl_probability_from_utility(
		int    n_alts,
		l4_float_t* utility,     # input
		l4_float_t* exp_utility, # output
		l4_float_t* probability, # output
) nogil:
	cdef:
		int i
		l4_float_t sum_expU = 0
		l4_float_t x

	for i in range(n_alts):
		if utility[i] < -3.402823e38:
			exp_utility[i] = 0
		else:
			x = exp_utility[i] = exp(utility[i])
			sum_expU += x

	if sum_expU == 0:
		for i in range(n_alts):
			probability[i] = 0
	else:
		for i in range(n_alts):
			probability[i] = exp_utility[i] / sum_expU


@cython.cdivision(True)
cdef void _mnl_logsum_from_utility(
		int    n_alts,
		l4_float_t* utility,     # input
		l4_float_t* exp_utility, # output
		l4_float_t* logsum,      # output (scalar)
):
	cdef:
		int i
		l4_float_t sum_expU = 0
		l4_float_t x

	for i in range(n_alts):
		if utility[i] < -3.402823e38:
			exp_utility[i] = 0
		else:
			x = exp_utility[i] = exp(utility[i])
			sum_expU += x

	if sum_expU == 0:
		logsum[0] = -INFINITY32
	else:
		logsum[0] = log(sum_expU)

@cython.cdivision(True)
cdef void _mnl_logsum_from_utility_MU(
		int    n_alts,
		l4_float_t* utility,     # input
		l4_float_t* exp_utility, # output
		l4_float_t* logsum,      # output (scalar)
		l4_float_t  mu           # input
):
	cdef:
		int i
		l4_float_t sum_expU = 0
		l4_float_t x

	if mu==0:
		sum_expU = -INFINITY32
		for i in range(n_alts):
			if utility[i] > sum_expU:
				sum_expU = utility[i]
		logsum[0] = sum_expU
	else:
		for i in range(n_alts):
			if utility[i] < -3.402823e38:
				exp_utility[i] = 0
			else:
				x = exp_utility[i] = exp(utility[i]/mu)
				sum_expU += x

		if sum_expU == 0:
			logsum[0] = -INFINITY32
		else:
			logsum[0] = log(sum_expU)*mu


ctypedef void (*PROB_FROM_UTIL)(
		int         n_alts,
		l4_float_t* utility,     # input
		l4_float_t* exp_utility, # output
		l4_float_t* probability, # output
)




cdef l4_float_t _mnl_log_likelihood_from_probability_approx2(
		int    n_alts,
		l4_float_t* probability, # input [n_alts]
		l4_float_t* choice,      # input [n_alts]
):
	cdef:
		l4_float_t ll = 0
		l4_float_t ch
		int i

	for i in range(n_alts):
		ch = choice[i]
		if ch != 0:
			ll += fasterlog(probability[i]) * ch

	return ll


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cdef l4_float_t _mnl_log_likelihood_from_probability_approx2_stride(
		int    n_alts,
		l4_float_t[:] probability, # input [n_alts]
		l4_float_t[:] choice,      # input [n_alts]
):
	cdef:
		l4_float_t ll = 0
		l4_float_t ch
		int i

	for i in range(n_alts):
		ch = choice[i]
		if ch != 0:
			ll += fasterlog(probability[i]) * ch

	return ll


cdef l4_float_t _mnl_log_likelihood_from_probability_approx(
		int    n_alts,
		l4_float_t* probability, # input [n_alts]
		l4_float_t* choice,      # input [n_alts]
):
	cdef:
		l4_float_t ll = 0
		l4_float_t ch
		int i

	for i in range(n_alts):
		ch = choice[i]
		if ch != 0:
			ll += fastlog(probability[i]) * ch

	return ll

@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cdef l4_float_t _mnl_log_likelihood_from_probability_approx_stride(
		int    n_alts,
		l4_float_t[:] probability, # input [n_alts]
		l4_float_t[:] choice,      # input [n_alts]
):
	cdef:
		l4_float_t ll = 0
		l4_float_t ch
		int i

	for i in range(n_alts):
		ch = choice[i]
		if ch != 0:
			ll += fastlog(probability[i]) * ch

	return ll

cdef l4_float_t _mnl_log_likelihood_from_probability(
		int    n_alts,
		l4_float_t* probability, # input [n_alts]
		l4_float_t* choice,      # input [n_alts]
) nogil:
	cdef:
		l4_float_t ll = 0
		l4_float_t ch
		int i

	for i in range(n_alts):
		ch = choice[i]
		if ch != 0:
			ll += log(probability[i]) * ch

	return ll


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cdef l4_float_t _mnl_log_likelihood_from_probability_stride(
		int    n_alts,
		l4_float_t[:] probability, # input [n_alts]
		l4_float_t[:] choice,      # input [n_alts]
) nogil:
	cdef:
		l4_float_t ll = 0
		l4_float_t ch
		int i

	for i in range(n_alts):
		ch = choice[i]
		if ch != 0:
			ll += log(probability[i]) * ch

	return ll


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cdef l4_float_t _mnl_log_likelihood_from_probability_stride2(
		int    n_alts,
		l4_float_t*   probability, # input [n_alts]
		l4_float_t[:] choice,      # input [n_alts]
):
	cdef:
		l4_float_t ll = 0
		l4_float_t ch
		int i

	for i in range(n_alts):
		ch = choice[i]
		if ch != 0:
			ll += log(probability[i]) * ch

	return ll


ctypedef l4_float_t (*LL_FROM_PROB)(
		int         n_alts,
		l4_float_t[:] probability, # input [n_alts]
		l4_float_t[:] choice,      # input [n_alts]
)









@cython.boundscheck(False)
cdef void _mnl_d_log_likelihood_from_d_utility(
		int             n_alts,
		int             n_params,
		l4_float_t[:]   choice,         # input [n_alts]
		l4_float_t      weight,         # input scalar
		l4_float_t[:,:] dU,             # input [n_alts, n_params]
		l4_float_t*     probability,    # input [n_alts]
		l4_float_t*     d_loglike_case, # output [n_params]
		int             accel,
		bint            return_bhhh,
		l4_float_t[:]   d_loglike_cum,  # input [n_params]
		l4_float_t[:,:] bhhh_cum,       # input [n_alts, n_params]
		l4_float_t*     dLL_temp,       # temp  [n_params]
) nogil:

	cdef:
		int a, i, v
		l4_float_t this_ch, tempvalue, tempvalue2

	# THIS IS WHERE INPUT CAME FROM
	# dfs._compute_d_utility_onecase(
	# 	c,
	# 	U,
	# 	dU,
	# )

	for v in range(n_params):
		d_loglike_case[v] = 0

	if weight == 0:
		return

	for a in range(n_alts):
		this_ch = choice[a] * weight

		if this_ch==0:
			continue

		for v in range(n_params):
			dLL_temp[v] = 0

		for i in range(n_alts):
			tempvalue = ((1 if i==a else 0) - probability[i])
			if tempvalue==0:
				continue
			for v in range(n_params):
				tempvalue2 = dU[i,v] * tempvalue
				dLL_temp[v] += tempvalue2
				tempvalue2 *= this_ch
				d_loglike_case[v] += tempvalue2
				d_loglike_cum[v]  += tempvalue2

		if return_bhhh:
			for v in range(n_params):
				for v2 in range(n_params):
					bhhh_cum[v,v2] += dLL_temp[v] * dLL_temp[v2] * this_ch




@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
def mnl_d_log_likelihood_from_dataframes_all_rows(
		DataFrames  dfs,
		int         num_threads=1,
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
		int v
		int v2
		int n_cases = dfs._n_cases()
		int n_alts  = dfs._n_alts()
		int n_params= dfs._n_model_params
		l4_float_t[:] array_ch
		l4_float_t[:] LL_case
		l4_float_t[:,:] dLL_case, dLL_total, dLL_temp
		l4_float_t[:,:] raw_utility
		l4_float_t[:,:] exp_utility
		l4_float_t[:,:] probability
		l4_float_t[:,:,:] dU
		l4_float_t[:,:,:] bhhh_total
		l4_float_t*     buffer_exp_utility
		l4_float_t*     buffer_probability
		l4_float_t      ll = 0
		l4_float_t      ll_temp
		l4_float_t*     choice
		l4_float_t      weight = 1 # default
		int             thread_number = 0
		int             storage_size_U
		int             store_number_U
		int             storage_size_expU
		int             store_number_expU
		int             storage_size_P
		int             store_number_P
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

		if stop_case<0:
			stop_case = n_cases

		if return_bhhh:
			# must compute dll to get bhhh
			return_dll = True

		storage_size_U    = n_cases if persist & PERSIST_UTILITY            else num_threads
		storage_size_expU = n_cases if persist & PERSIST_EXP_UTILITY        else num_threads
		storage_size_P    = n_cases if persist & PERSIST_PROBABILITY        else num_threads
		storage_size_LLc  = n_cases if persist & PERSIST_LOGLIKE_CASEWISE   else num_threads
		storage_size_dLLc = n_cases if persist & PERSIST_D_LOGLIKE_CASEWISE else num_threads
		storage_size_dU   = n_cases if persist & PERSIST_D_UTILITY          else num_threads

		raw_utility = numpy.zeros([storage_size_U,   n_alts], dtype=l4_float_dtype)
		exp_utility = numpy.zeros([storage_size_expU,n_alts], dtype=l4_float_dtype)
		probability = numpy.zeros([storage_size_P,   n_alts], dtype=l4_float_dtype)

		LL_case  = numpy.zeros([storage_size_LLc,], dtype=l4_float_dtype)

		if return_dll:
			dU = numpy.zeros([storage_size_dU,n_alts, n_params], dtype=l4_float_dtype)
			dLL_case  = numpy.zeros([storage_size_dLLc,n_params], dtype=l4_float_dtype)
			dLL_total = numpy.zeros([num_threads,n_params], dtype=l4_float_dtype)
			dLL_temp  = numpy.zeros([num_threads,n_params], dtype=l4_float_dtype)
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
				store_number_expU = c if persist & PERSIST_EXP_UTILITY        else thread_number
				store_number_P    = c if persist & PERSIST_PROBABILITY        else thread_number
				store_number_LLc  = c if persist & PERSIST_LOGLIKE_CASEWISE   else thread_number
				store_number_dLLc = c if persist & PERSIST_D_LOGLIKE_CASEWISE else thread_number
				store_number_dU   = c if persist & PERSIST_D_UTILITY          else thread_number

				buffer_exp_utility = &exp_utility[store_number_expU,0]
				buffer_probability = &probability[store_number_P,0]

				if dfs._array_wt is not None:
					weight = dfs._array_wt[c]
				else:
					weight = 1

				if return_dll:
					dfs._compute_d_utility_onecase(c,raw_utility[store_number_U],dU[store_number_dU],n_alts)
				else:
					dfs._compute_utility_onecase(c,raw_utility[store_number_U],n_alts)

				_mnl_probability_from_utility(
					n_alts,
					&raw_utility[store_number_U,0],     # input
					buffer_exp_utility, # output
					buffer_probability, # output
				)

				if probability_only:
					continue

				ll_temp = _mnl_log_likelihood_from_probability_stride(
					n_alts,
					probability[store_number_P], # output
					dfs._array_ch[c,:],
				) * weight
				ll += ll_temp
				LL_case[store_number_LLc] += ll_temp

				if return_dll:
					if weight:
						_mnl_d_log_likelihood_from_d_utility(
							n_alts,
							n_params,
							dfs._array_ch[c,:],         # input [n_alts]
							weight,                     # input scalar
							dU[store_number_dU],          # input [n_alts, n_params]
							buffer_probability,         # output [n_alts]
							&dLL_case[store_number_dLLc,0],  # output [n_params]
							0,                          # accelerator
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
		if persist & PERSIST_UTILITY:
			result.utility=raw_utility.base
		if persist & PERSIST_LOGLIKE_CASEWISE:
			result.ll_casewise=LL_case.base
		if persist & PERSIST_EXP_UTILITY:
			result.exp_utility=exp_utility.base
		if persist & PERSIST_PROBABILITY:
			result.probability=probability.base

		if return_dll:
			result.dll = pandas.Series(
				data=dll,
				index=dfs._model_param_names,
			)
			for v in range(dLL_case.shape[0]):
				for v2 in range(dLL_case.shape[1]):
					dLL_case[v,v2] *= dfs._weight_normalization
			if persist & PERSIST_D_LOGLIKE_CASEWISE:
				result.dll_casewise=pandas.DataFrame(
					dLL_case.base,
					columns=dfs._model_param_names,
				)
			if persist & PERSIST_D_UTILITY:
				result.dutility = dU.base
		if return_bhhh:
			result.bhhh = bhhh

		return result

	except:
		logger.error(f'c={c}')
		logger.error(f'n_cases, n_alts={(n_cases, n_alts)}')
		logger.exception('error in mnl_d_log_likelihood_from_dataframes_all_rows')
		raise



@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
def mnl_logsums_from_dataframes_all_rows(
		DataFrames dfs,
		l4_float_t    logsum_parameter,
		l4_float_t[:] logsums, # output
):
	cdef:
		int c = 0
		int v
		int n_cases = dfs._n_cases()
		int n_alts  = dfs._n_alts()
		int n_params= dfs._n_model_params
		l4_float_t[:] array_ch
		l4_float_t[:] dLL_case, dLL_total
		l4_float_t[:] raw_utility
		l4_float_t[:] exp_utility
		l4_float_t[:] probability
		l4_float_t[:,:] dU
		l4_float_t*     buffer_exp_utility
		l4_float_t*     buffer_probability
		l4_float_t      ll = 0
		l4_float_t*     choice
		l4_float_t      weight = 1 # TODO

	if not dfs.is_computational_ready(activate=True):
		raise ValueError('DataFrames is not computational-ready')

	if dfs._data_ch is None:
		raise ValueError('DataFrames does not define data_ch')

	try:
		raw_utility = numpy.zeros([n_alts], dtype=l4_float_dtype)
		exp_utility = numpy.zeros([n_alts], dtype=l4_float_dtype)
		probability = numpy.zeros([n_alts], dtype=l4_float_dtype)

		buffer_exp_utility = &exp_utility[0]
		buffer_probability = &probability[0]

		dU = numpy.zeros([n_alts, n_params], dtype=l4_float_dtype)
		dLL_case  = numpy.zeros([n_params], dtype=l4_float_dtype)
		dLL_total = numpy.zeros([n_params], dtype=l4_float_dtype)

		if logsum_parameter == 1:
			for c in range(n_cases):
				array_ch = dfs._get_choice_onecase(c)
				choice = &array_ch[0]
				dfs._compute_d_utility_onecase(c,raw_utility,dU,dU.shape[0]) #TODO: don't bother with derivative if not needed
				_mnl_logsum_from_utility(
					n_alts,
					&raw_utility[0],     # input
					buffer_exp_utility, # output
					&logsums[c], # output
				)
		else:

			for c in range(n_cases):
				array_ch = dfs._get_choice_onecase(c)
				choice = &array_ch[0]
				dfs._compute_d_utility_onecase(c,raw_utility,dU,dU.shape[0]) #TODO: don't bother with derivative if not needed
				_mnl_logsum_from_utility_MU(
					n_alts,
					&raw_utility[0],     # input
					buffer_exp_utility, # output
					&logsums[c], # output
					logsum_parameter,
				)

	except:
		logger.error(f'c={c}')
		logger.error(f'n_cases, n_alts={(n_cases, n_alts)}')
		logger.error(f'array_ch={array_ch}')
		logger.exception('error in mnl_logsums_from_dataframes_all_rows')
		raise




############

@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
def speed_tester(
		l4_float_t[:] x,
		int accel = 0,
):

	cdef:
		int c = 0
		VERT func_exp
		VERT func_log
		l4_float_t temp

	IF DOUBLE_PRECISION:
		if accel>1:
			func_exp = exp_fast3
			func_log = log
		elif accel>0:
			func_exp = exp_fast2
			func_log = log
		elif accel>-1:
			func_exp = exp
			func_log = log
		else:
			func_exp = exp
			func_log = log
	ELSE:
		if accel>1:
			func_exp = fasterexp
			func_log = fasterlog
		elif accel>0:
			func_exp = fastexp
			func_log = fastlog
		elif accel>-1:
			func_exp = expf
			func_log = logf
		else:
			func_exp = exp_
			func_log = log_


	for c in range(x.shape[0]):
		temp = func_exp(func_log(x[c]))