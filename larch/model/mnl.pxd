# cython: language_level=3

from ..general_precision cimport *

cdef l4_float_t _mnl_log_likelihood_from_probability(
		int    n_alts,
		l4_float_t* probability, # input [n_alts]
		l4_float_t* choice,      # input [n_alts]
) nogil

cdef l4_float_t _mnl_log_likelihood_from_probability_stride(
		int    n_alts,
		l4_float_t[:] probability, # input [n_alts]
		l4_float_t[:] choice,      # input [n_alts]
) nogil
