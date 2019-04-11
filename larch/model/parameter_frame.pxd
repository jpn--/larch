# cython: language_level=3, embedsignature=True

from ..general_precision cimport *
from ..dataframes cimport DataFrames
from .linear cimport LinearFunction_C, DictOfLinearFunction_C

cdef class ParameterFrame:

	cdef:
		unicode _title
		bint    _mangled
		object  _frame                # pandas.DataFrame
		object  _prior_frame_values   # pandas.Series
		object  _display_order
		object  _display_order_tail
		object  _matrixes             # Dict[ndarray]

