# cython: language_level=3, embedsignature=True

from ..general_precision cimport *

cdef class ParameterRef_C(unicode):
	pass

cdef class DataRef_C(unicode):
	pass

cdef class LinearComponent_C:
	cdef:
		unicode    _param
		unicode    _data
		l4_float_t _scale


cdef class LinearFunction_C:
	cdef:
		object _func
		object _instance
		unicode name


cdef class DictOfLinearFunction_C:
	cdef:
		object _map
		object _instance
		object _alts_validator
		unicode name


cdef class Top:
	cdef public LinearFunction_C _qf
	cdef object arg