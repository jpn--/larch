# cython: language_level=3, embedsignature=True

from ..general_precision cimport *
from .parameter_frame cimport ParameterFrame



cdef class AbstractChoiceModel(ParameterFrame):

	cdef:

		object _most_recent_estimation_result
		object _possible_overspecification

		double _cached_loglike_null
		double _cached_loglike_nil
		double _cached_loglike_constants_only
		double _cached_loglike_best

		object _dashboard

