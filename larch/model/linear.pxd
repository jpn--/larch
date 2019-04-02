# cython: language_level=3, embedsignature=True

from ..general_precision cimport *

cdef class UnicodeRef_C(unicode):
	pass

cdef class Ref_Gen:
	cdef:
		object _kind

cdef class ParameterRef_C(UnicodeRef_C):
	cdef public unicode _formatting

cdef class DataRef_C(UnicodeRef_C):
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
		unicode private_name


cdef class DictOfLinearFunction_C:
	cdef:
		object _map
		object _instance
		object _alts_validator
		unicode name
		unicode private_name


cdef class GenericContainerCy:
	cdef public:
		LinearFunction_C _lf
		DictOfLinearFunction_C _dlf
		object ident