# cython: language_level=3, embedsignature=True

from ..general_precision cimport *
from ..dataframes cimport DataFrames
from .linear cimport LinearFunction_C, DictOfLinearFunction_C
from .abstract_model cimport AbstractChoiceModel


cdef class Model5c(AbstractChoiceModel):

	cdef public:
		DictOfLinearFunction_C _utility_co
		LinearFunction_C _utility_ca
		LinearFunction_C _quantity_ca

	cdef:
		DataFrames _dataframes
		object _dataservice

		object _quantity_scale
		object _logsum_parameter
		object rename_parameters

		str    _choice_ca_var
		object _choice_co_vars
		str    _choice_co_code
		bint   _choice_any
		str    _weight_co_var
		str    _availability_var
		object _availability_co_vars
		bint   _availability_any

		bint _is_clone
		bint _does_not_require_choice

		object _graph

		int _n_threads

