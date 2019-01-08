# cython: language_level=3

from ..general_precision cimport *
from ..dataframes cimport DataFrames

cdef class Model5c:

	cdef:
		DataFrames _dataframes
		object _dataservice

		object _utility_co_functions
		object _utility_ca_function
		object _quantity_ca_function
		object _quantity_scale
		object _logsum_parameter
		object rename_parameters

		str    _choice_ca_var
		object _choice_co_vars
		str    _choice_co_code
		str    _weight_co_var
		str    _availability_var
		object _availability_co_vars

		bint _mangled
		bint _is_clone

		object frame  # pandas.DataFrame

		object _graph

		object _display_order
		object _display_order_tail
		object _possible_overspecification

		object _most_recent_estimation_result
		double _cached_loglike_null
		double _cached_loglike_constants_only
		double _cached_loglike_best

		unicode _title

		object hessian_matrix
		object covariance_matrix
		object robust_covariance_matrix

		int _n_threads

		object dashboard