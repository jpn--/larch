# cython: language_level=3, embedsignature=True

import numpy

include "general_precision.pxi"

IF DOUBLE_PRECISION:
	l4_float_dtype = numpy.float64
ELSE:
	l4_float_dtype = numpy.float64

