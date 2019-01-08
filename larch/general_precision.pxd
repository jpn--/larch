# cython: language_level=3


include "general_precision.pxi"

IF DOUBLE_PRECISION:
	ctypedef double l4_float_t
ELSE:
	ctypedef float l4_float_t


