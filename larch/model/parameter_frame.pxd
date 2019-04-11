# cython: language_level=3, embedsignature=True

from ..general_precision cimport *

cdef class ParameterFrame:

	cdef public:
		object  _frame                # pandas.DataFrame
		object  _matrixes             # Dict[ndarray]

	cdef:
		unicode _title
		bint    _mangled
		object  _prior_frame_values   # pandas.Series
		object  _display_order
		object  _display_order_tail

