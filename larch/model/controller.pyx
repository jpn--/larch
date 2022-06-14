# cython: language_level=3, embedsignature=True

include "../general_precision.pxi"
from ..general_precision import l4_float_dtype
from ..general_precision cimport l4_float_t
from .persist_flags cimport *

import sys
import numpy
import pandas
from typing import Union

import logging
from ..log import logger_name
logger = logging.getLogger(logger_name+'.model')


from ..dataframes cimport DataFrames

from ..roles import DictOfStrings

from .linear cimport ParameterRef_C as ParameterRef
from .linear cimport DataRef_C as DataRef

from .abstract_model cimport AbstractChoiceModel

from ..exceptions import MissingDataError, ParameterNotInModelWarning


cdef class Model5c(AbstractChoiceModel):

	def __init__(
			self, *,
			parameters=None,
			alts=None,
			utility_ca=None,
			utility_co=None,
			quantity_ca=None,
			quantity_scale=None,
			graph=None,
			dataservice=None,
			rename_parameters=None,
			frame=None,
			n_threads=-1,
			is_clone=False,
			title=None,
			root_id=0,
	):
		self._dataframes = None
		self._does_not_require_choice = False

		if is_clone:
			self._is_clone = True
			if quantity_scale is None:
				self._quantity_scale = None
			else:
				self._quantity_scale = str(quantity_scale)

		else:
			self._is_clone = False
			if quantity_scale is None:
				self._quantity_scale = None
			else:
				self._quantity_scale = str(quantity_scale)

		self.rename_parameters = DictOfStrings()

		self._cached_loglike_null = 0
		self._cached_loglike_nil = 0
		self._cached_loglike_constants_only = 0
		self._cached_loglike_best = -numpy.inf
		self._most_recent_estimation_result = None

		self.n_threads = n_threads

		self._dataservice = dataservice

		super().__init__(
			parameters=parameters,
			frame=frame,
			title=title,
		)

		if alts is not None and graph is None:
			from .tree import NestingTree
			graph = NestingTree(root_id=root_id)
			if hasattr(alts, 'items'):
				for a, name in alts.items():
					graph.add_node(a, name=name)
			else:
				for a in alts:
					graph.add_node(a)

		self._graph = graph
		if self._dataservice is not None:
			self.graph.suggest_elemental_order(self._dataservice.alternative_codes())

	def __getstate__(self):

		import cloudpickle
		import gzip
		import base64

		mrer = self._most_recent_estimation_result
		if mrer is not None and 'dashboard' in mrer:
			del mrer['dashboard']

		state = dict()
		state["utility_ca                     ".strip()] = (self.utility_ca.copy()              )
		state["utility_co                     ".strip()] = (self.utility_co.copy()              )
		state["quantity_ca                    ".strip()] = (self.quantity_ca.copy()             )
		state["_quantity_scale                ".strip()] = (self._quantity_scale                )
		state["_logsum_parameter              ".strip()] = (self._logsum_parameter              )
		state["rename_parameters              ".strip()] = (self.rename_parameters              )
		state["_choice_ca_var                 ".strip()] = (self._choice_ca_var                 )
		state["_choice_co_vars                ".strip()] = (self._choice_co_vars                )
		state["_choice_co_code                ".strip()] = (self._choice_co_code                )
		state["_choice_any                    ".strip()] = (self._choice_any                    )
		state["_weight_co_var                 ".strip()] = (self._weight_co_var                 )
		state["_availability_var              ".strip()] = (self._availability_var              )
		state["_availability_co_vars          ".strip()] = (self._availability_co_vars          )
		state["_availability_any              ".strip()] = (self._availability_any              )
		state["_frame                         ".strip()] = (self._frame                         )
		state["_graph                         ".strip()] = (self._graph                         )
		state["_display_order                 ".strip()] = (self._display_order                 )
		state["_display_order_tail            ".strip()] = (self._display_order_tail            )
		state["_possible_overspecification    ".strip()] = (self._possible_overspecification    )
		state["_most_recent_estimation_result ".strip()] = (self._most_recent_estimation_result )
		state["_cached_loglike_null           ".strip()] = (self._cached_loglike_null           )
		state["_cached_loglike_nil            ".strip()] = (self._cached_loglike_nil            )
		state["_cached_loglike_constants_only ".strip()] = (self._cached_loglike_constants_only )
		state["_cached_loglike_best           ".strip()] = (self._cached_loglike_best           )
		state["_title                         ".strip()] = (self._title                         )
		state["_matrixes                      ".strip()] = (self._matrixes                      )

		state = cloudpickle.dumps(state)
		state = gzip.compress(state)
		state = base64.b85encode(state)

		return state

	def __setstate__(self, state):

		import cloudpickle
		import gzip
		import base64

		state = base64.b85decode(state)
		state = gzip.decompress(state)
		state = cloudpickle.loads(state)

		(self.utility_ca                     ) = state["utility_ca                     ".strip()]
		(self.utility_co                     ) = state["utility_co                     ".strip()]
		(self.quantity_ca                    ) = state["quantity_ca                    ".strip()]
		(self._quantity_scale                ) = state["_quantity_scale                ".strip()]
		(self._logsum_parameter              ) = state["_logsum_parameter              ".strip()]
		(self.rename_parameters              ) = state["rename_parameters              ".strip()]
		(self._choice_ca_var                 ) = state["_choice_ca_var                 ".strip()]
		(self._choice_co_vars                ) = state["_choice_co_vars                ".strip()]
		(self._choice_co_code                ) = state["_choice_co_code                ".strip()]
		(self._choice_any                    ) = state["_choice_any                    ".strip()]
		(self._weight_co_var                 ) = state["_weight_co_var                 ".strip()]
		(self._availability_var              ) = state["_availability_var              ".strip()]
		(self._availability_co_vars          ) = state["_availability_co_vars          ".strip()]
		(self._availability_any              ) = state["_availability_any              ".strip()]
		(self._frame                         ) = state["_frame                         ".strip()]
		(self._graph                         ) = state["_graph                         ".strip()]
		(self._display_order                 ) = state["_display_order                 ".strip()]
		(self._display_order_tail            ) = state["_display_order_tail            ".strip()]
		(self._possible_overspecification    ) = state["_possible_overspecification    ".strip()]
		(self._most_recent_estimation_result ) = state["_most_recent_estimation_result ".strip()]
		(self._cached_loglike_nil            ) = state["_cached_loglike_nil            ".strip()]
		(self._cached_loglike_null           ) = state["_cached_loglike_null           ".strip()]
		(self._cached_loglike_constants_only ) = state["_cached_loglike_constants_only ".strip()]
		(self._cached_loglike_best           ) = state["_cached_loglike_best           ".strip()]
		(self._title                         ) = state["_title                         ".strip()]
		(self._matrixes                      ) = state["_matrixes                      ".strip()]

		self.unmangle(True)
		self.n_threads = 0
		self._prior_frame_values = None
		# if self._graph is not None:
		# 	self.graph.set_touch_callback(self.mangle)

	def specification_hash(self, digest_size=4):
		"""
		Generate a hex hash on the model specification.

		Parameters
		----------
		digest_size : int, default 4
			Size of hash digest, in bytes.  The resulting
			hex-encoded string is actually double this
			length.

		Returns
		-------
		str
		"""
		from hashlib import blake2b
		h = blake2b(digest_size=digest_size)
		h.update(repr(self.utility_ca).encode())
		h.update(repr(self.utility_co).encode())
		h.update(repr(self.quantity_ca).encode())
		h.update(repr(self.quantity_scale).encode())
		h.update(repr(self.logsum_parameter).encode())
		h.update(repr(self.choice_ca_var).encode())
		h.update(repr(self.choice_co_vars).encode())
		h.update(repr(self.choice_co_code).encode())
		h.update(repr(self.choice_any).encode())
		h.update(repr(self.weight_co_var).encode())
		h.update(repr(self.availability_var).encode())
		h.update(repr(self.availability_co_vars).encode())
		h.update(repr(self.availability_any).encode())
		for i in self.graph.nodes:
			h.update(repr(self.graph.nodes[i]).encode())
		for i in self.graph.edges:
			h.update(repr(self.graph.edges[i]).encode())
		return h.hexdigest()

	def _preload_tree_structure(self):
		"""
		Performance optimization for NL models.

		Warning: do not use this for model parameter estimation!

		This function pre-computes the tree structure arrays.
		It is useful only in high performance applications with
		exceptionally large nesting structures.  It should only
		be called after the nesting tree and all parameter values
		are fully defined and fixed. Editing the nesting tree or
		parameter values after calling this function may cause
		unexpected results including incorrect results, and it
		possibly can result in crashing the entire Python instance.
		"""
		from .tree_struct import TreeStructure
		self.unmangle()
		self._tree_struct = TreeStructure(self, self._graph)

	def _clear_preloaded_tree_structure(self):
		self._tree_struct = None

	@property
	def _model_does_not_require_choice(self):
		return self._does_not_require_choice

	@_model_does_not_require_choice.setter
	def _model_does_not_require_choice(self, value):
		self._does_not_require_choice = bool(value)

	@property
	def n_threads(self):
		return self._n_threads

	@n_threads.setter
	def n_threads(self, value):
		if value <= 0:
			import multiprocessing
			self._n_threads = multiprocessing.cpu_count()
		else:
			self._n_threads = int(value)

	def mangle(self, *args, **kwargs):
		super().mangle(*args, **kwargs)

	def unmangle(self, force=False):
		if force:
			self.mangle()
		if self._mangled or force:
			self._scan_all_ensure_names()
			if self._dataframes is not None:
				self._dataframes._check_data_is_sufficient_for_model(self)
				self._refresh_derived_arrays()
			self._mangled = False

	def _frame_values_have_changed(self):
		# refresh everything # TODO: only refresh what changed
		try:
			if self._dataframes is not None:
				self._dataframes._read_in_model_parameters()
		except AttributeError:
			# the model may not have been ever unmangled yet
			pass


	def _scan_all_ensure_names(self):
		self._scan_utility_ensure_names()
		self._scan_quantity_ensure_names()
		self._scan_logsums_ensure_names()
		super()._scan_all_ensure_names()

	def _scan_utility_ensure_names(self):
		"""
		Scan the utility functions and ensure all named parameters appear in the parameter frame.

		Any named parameters that do not appear in the parameter frame are added.
		"""
		try:
			nameset = set()
			u_co_dataset = set()

			_utility_co_postprocessed = self._utility_co_postprocess
			for altcode, linear_function in self._utility_co.items():
				for component in linear_function:
					nameset.add(self.__p_rename(component.param))
					try:
						u_co_dataset.add(str(component.data))
					except:
						import warnings
						warnings.warn(f'bad data in altcode {altcode}')
						raise
			#self._u_co_varindex = pandas.Index( u_co_dataset )

			linear_function_ca = self._utility_ca_postprocess
			#self._u_ca_varindex = pandas.Index( str(component.data) for component in linear_function_ca)
			for component in linear_function_ca:
				nameset.add(self.__p_rename(component.param))

			self._ensure_names(nameset)
		except:
			logger.exception('error in Model5c._scan_utility_ensure_names')
			raise

	def _scan_quantity_ensure_names(self):
		if self._quantity_ca is not None:
			nameset = set()

			for component in self._quantity_ca:
				nameset.add(self.__p_rename(component.param))

			self._ensure_names(nameset)

	def _scan_logsums_ensure_names(self):
		nameset = set()

		if self._graph is not None:
			for nodecode in self._graph.topological_sorted_no_elementals:
				if nodecode != self._graph._root_id:
					param_name = str(self._graph.nodes[nodecode]['parameter'])
					# if param_name in self._snapped_parameters:
					# 	snapped = self._snapped_parameters[param_name]
					# 	param_name = str(snapped._is_scaled_parameter()[0])
					nameset.add(self.__p_rename(param_name))

		if self._quantity_ca is not None and len(self._quantity_ca)>0:
			# for nodecode in self._graph.elementals:
			# 	try:
			# 		param_name = str(self._graph.nodes[nodecode]['parameter'])
			# 	except KeyError:
			# 		pass
			# 	else:
			# 		nameset.add(self.__p_rename(param_name))

			if self.quantity_scale is not None:
				nameset.add(self.__p_rename(self.quantity_scale))

		if self.logsum_parameter is not None:
			nameset.add(self.__p_rename(self.logsum_parameter))

		self._ensure_names(nameset, nullvalue=1, initvalue=1, min=0.001, max=1)



	@property
	def quantity_scale(self):
		return self._quantity_scale

	@quantity_scale.setter
	def quantity_scale(self, value):
		if self._is_clone:
			raise PermissionError('cannot edit quantity_scale when is_clone is True')
		if value is None:
			self._quantity_scale = None
		else:
			self._quantity_scale = str(value)
		self.mangle()

	@quantity_scale.deleter
	def quantity_scale(self):
		if self._is_clone:
			raise PermissionError('cannot edit quantity_scale when is_clone is True')
		self._quantity_scale = None
		self.mangle()





	@property
	def logsum_parameter(self):
		return self._logsum_parameter

	@logsum_parameter.setter
	def logsum_parameter(self, value):
		self._logsum_parameter = str(value)
		self.mangle()

	@logsum_parameter.deleter
	def logsum_parameter(self):
		self._logsum_parameter = None
		self.mangle()




	def __p_rename(self,x):
		x = str(x)
		if x in self.rename_parameters:
			return str(self.rename_parameters[x])
		return str(x)

	@property
	def _utility_co_postprocess(self):
		return self._utility_co
		# u = DictOfLinearFunction_C()
		# warning_queue = {}
		# if self._utility_co is not None:
		# 	keys = list(self._utility_co.keys())
		# 	u_found = {k:set() for k in keys}
		# 	for altkey in keys:
		# 		u[altkey] = LinearFunction_C()
		# 	for altkey in keys:
		# 		for n, i in enumerate(self._utility_co[altkey]):
		# 			try:
		# 				i_d = i.data
		# 			except:
		# 				logger.error(f"n={n}, altkey={altkey}, len(self.utility_co[altkey])={len(self._utility_co[altkey])}")
		# 				raise
		# 			if i_d in u_found[altkey]:
		# 				if i_d not in warning_queue:
		# 					warning_queue[i_d] = set()
		# 				warning_queue[i_d].add(altkey)
		# 			u_found[altkey].add(i_d)
		# 			# if i.param in self._snapped_parameters:
		# 			# 	u[altkey] += self._snapped_parameters[i.param] * i_d * i.scale
		# 			# else:
		# 			# 	u[altkey] += i.param * i_d * i.scale
		# 			u[altkey].append(i.param * i_d * i.scale)
		# if warning_queue:
		# 	import warnings
		# 	for i_d, altkeys in warning_queue.items():
		# 		if len(altkeys) < 6:
		# 			k = ','.join(str(j) for j in altkeys)
		# 		else:
		# 			k = (','.join(str(j) for j in sorted(altkeys)[:3])
		# 				+ f", & {len(altkeys)-3} more")
		# 		warnings.warn(f"found duplicate X({i_d}) in utility_co[{k}]")
		# return u

	@property
	def _utility_ca_postprocess(self):
		# if len(self._snapped_parameters):
		# 	u = LinearFunction()
		# 	for n, i in enumerate(self.utility_ca):
		# 		if i.param in self._snapped_parameters:
		# 			u += self._snapped_parameters[i.param] * i.data * i.scale
		# 		else:
		# 			u += i.param * i.data * i.scale
		# 	return u
		# else:
		# 	return self.utility_ca
		return self._utility_ca

	@property
	def _quantity_ca_postprocess(self):
		# if len(self._snapped_parameters):
		# 	u = LinearFunction()
		# 	for n, i in enumerate(self.quantity_ca):
		# 		if i.param in self._snapped_parameters:
		# 			u += self._snapped_parameters[i.param] * i.data * i.scale
		# 		else:
		# 			u += i.param * i.data * i.scale
		# 	return u
		# else:
		# 	return self.quantity_ca
		return self._quantity_ca

	def _refresh_derived_arrays(self):
		self._dataframes._link_to_model_structure(self)

	@property
	def dataservice(self):
		return self._dataservice

	@dataservice.setter
	def dataservice(self, x):
		try:
			x.validate_dataservice(self.required_data())
		except AttributeError:
			raise TypeError("proposed dataservice does not implement 'validate_dataservice'")
		self._dataservice = x

	@property
	def dataframes(self):
		return self._dataframes

	def set_dataframes(
			self,
			DataFrames x,
			bint check_sufficiency=True,
			*,
			bint raw=False,
	):
		"""

		Parameters
		----------
		x : larch.DataFrames
		check_sufficiency : bool, default True
			Run a check

		Returns
		-------

		"""
		if raw:
			self._dataframes = x
			return

		x.computational = True
		self.clear_best_loglike()

		#self.unmangle() # don't do a full unmangle here, it will fail if the old data is incomplete
		if self._mangled:
			self._scan_all_ensure_names()
		if check_sufficiency:
			x._check_data_is_sufficient_for_model(self)
		self._dataframes = x
		self._refresh_derived_arrays()
		self._dataframes._read_in_model_parameters()

	@dataframes.setter
	def dataframes(self, x):
		if isinstance(x, pandas.DataFrame):
			x = DataFrames(x)
		self.set_dataframes(x)

	@property
	def n_cases(self):
		"""int : The number of cases in the attached dataframes."""
		if self._dataframes is None:
			raise MissingDataError("no dataframes are set")
		return self._dataframes.n_cases

	def total_weight(self):
		"""
		The total weight of cases in the attached dataframes.

		Returns
		-------
		float
		"""
		if self._dataframes is None:
			raise MissingDataError("no dataframes are set")
		return self._dataframes.total_weight()

	def load_data(self, dataservice=None, autoscale_weights=True, log_warnings=True):
		"""Load dataframes as required from the dataservice.

		This method prepares the data for estimation. It is used to
		pre-process the data, extracting the required values, pre-computing
		the values of fixed expressions, and assembling the results into
		contiguous arrays suitable for computing the log likelihood values
		efficiently.

		Parameters
		----------
		dataservice : DataService, optional
			A dataservice from which to load data.  If a dataservice
			has not been previously defined for this model, this
			argument is not optional.
		autoscale_weights : bool, default True
			If True and data_wt is not None, the loaded dataframes will
			have the weights automatically scaled such that the average
			value for data_wt is 1.0.  See `autoscale_weights` for more
			information.
		log_warnings : bool, default True
			Emit warnings in the logger if choice, avail, or weight is
			not included in `req_data` but is set in the dataservice, and
			thus returned by default even though it was not requested.

		Raises
		------
		ValueError
			If no dataservice is given nor pre-defined.
		"""
		if dataservice is not None:
			self._dataservice = dataservice
		if self._dataservice is not None:
			self.dataframes = self._dataservice.make_dataframes(
				self.required_data(),
				log_warnings=log_warnings,
			)
			if autoscale_weights and self.dataframes.data_wt is not None:
				self.dataframes.autoscale_weights()
		else:
			raise ValueError('dataservice is not defined')

	def dataframes_from_idce(self, ce, choice, autoscale_weights=True):
		"""
		Create DataFrames from a single `idce` format DataFrame.

		Parameters
		----------
		ce : pandas.DataFrame
			The data
		choice : str
			The name of the choice column.
		autoscale_weights : bool, default True
			Also call autoscale_weights on the result after loading.

		"""
		self.dataframes = DataFrames(
			ce = ce[self.required_data().ca],
			ch = ce[choice].unstack().fillna(0),
		)
		if autoscale_weights and self.dataframes.data_wt is not None:
			self.dataframes.autoscale_weights()

	def __d_log_likelihood_from_dataframes_all_rows(
			self,
			bint        return_dll=True,
			bint        return_bhhh=False,
			int         start_case=0,
			int         stop_case=-1,
			int         step_case=1,
			int         persist=0,
			int         leave_out=-1,
			int         keep_only=-1,
			int         subsample= 1,
			bint        probability_only=False,
	):
		if self.is_mnl() and not (persist & PERSIST_D_PROBABILITY):
			from .mnl import mnl_d_log_likelihood_from_dataframes_all_rows
			y = mnl_d_log_likelihood_from_dataframes_all_rows(
				self._dataframes,
				num_threads=self.n_threads,
				return_dll=return_dll,
				return_bhhh=return_bhhh,
				start_case=start_case,
				stop_case=stop_case,
				step_case=step_case,
				persist=persist,
				leave_out=leave_out,
				keep_only=keep_only,
				subsample=subsample,
				probability_only=probability_only,
			)
		else:
			if self.graph is None:
				if persist & PERSIST_D_PROBABILITY:
					raise ValueError('must have a graph to use PERSIST_D_PROBABILITY')
				else:
					raise ValueError('no graph defined for NL computations')
			from .nl import nl_d_log_likelihood_from_dataframes_all_rows
			y = nl_d_log_likelihood_from_dataframes_all_rows(
				self._dataframes,
				self,
				return_dll=return_dll,
				return_bhhh=return_bhhh,
				start_case=start_case,
				stop_case=stop_case,
				step_case=step_case,
				persist=persist,
				leave_out=leave_out,
				keep_only=keep_only,
				subsample=subsample,
				probability_only=probability_only,
			)
		return y

	def loglike2(
			self,
			x=None,
			*,
			start_case=0,
			stop_case=-1,
			step_case=1,
			persist=0,
			leave_out=-1,
			keep_only=-1,
			subsample=-1,
			return_series=True,
			probability_only=False,
	):
		"""
		Compute a log likelihood value and it first derivative.

		Parameters
		----------
		x : {'null', 'init', 'best', array-like, dict, scalar}, optional
			Values for the parameters.  See :ref:`set_values` for details.
		start_case : int, default 0
			The first case to include in the log likelihood computation.  To include all
			cases, start from 0 (the default).
		stop_case : int, default -1
			One past the last case to include in the log likelihood computation.  This is processed as usual for
			Python slicing and iterating, and negative values count backward from the end.  To include all cases,
			end at -1 (the default).
		step_case : int, default 1
			The step size of the case iterator to use in likelihood calculation.  This is processed as usual for
			Python slicing and iterating.  To include all cases, step by 1 (the default).
		persist : int, default 0
			Whether to return a variety of internal and intermediate arrays in the result dictionary.
			If set to 0, only the final `ll` value is included.
		leave_out, keep_only, subsample : int, optional
			Settings for cross validation calculations.
			If `leave_out` and `subsample` are set, then case rows where rownumber % subsample == leave_out are dropped.
			If `keep_only` and `subsample` are set, then only case rows where rownumber % subsample == keep_only are used.
		return_series : bool
			Deprecated, no effect.  Derivatives are always returned as a Series.

		Returns
		-------
		dictx
			The log likelihood is given by key 'll' and the first derivative by key 'dll'.
			Other arrays are also included if `persist` is set to True.

		"""
		self.__prepare_for_compute(x)
		y = self.__d_log_likelihood_from_dataframes_all_rows(
			return_dll=True,
			return_bhhh=False,
			start_case=start_case,
			stop_case=stop_case,
			step_case=step_case,
			persist=persist,
			leave_out=leave_out,
			keep_only=keep_only,
			subsample=subsample,
			probability_only=probability_only,
		)
		if start_case==0 and stop_case==-1 and step_case==1:
			self._check_if_best(y.ll)
		# if return_series and 'dll' in y and not isinstance(y['dll'], (pandas.DataFrame, pandas.Series)):
		# 	y['dll'] = pandas.Series(y['dll'], index=self.frame.index, )
		return y


	def loglike2_bhhh(
			self,
			x=None,
			*,
			return_series=False,
			start_case=0, stop_case=-1, step_case=1,
			persist=0,
			leave_out=-1, keep_only=-1, subsample=-1,
	):
		"""
		Compute a log likelihood value, it first derivative, and the BHHH approximation of the Hessian.

		The `BHHH algorithm <https://en.wikipedia.org/wiki/Berndt–Hall–Hall–Hausman_algorithm>`
		employs a matrix computated as the sum of the casewise outer product of the gradient, to
		approximate the hessian matrix.

		Parameters
		----------
		x : {'null', 'init', 'best', array-like, dict, scalar}, optional
			Values for the parameters.  See :ref:`set_values` for details.
		start_case : int, default 0
			The first case to include in the log likelihood computation.  To include all
			cases, start from 0 (the default).
		stop_case : int, default -1
			One past the last case to include in the log likelihood computation.  This is processed as usual for
			Python slicing and iterating, and negative values count backward from the end.  To include all cases,
			end at -1 (the default).
		step_case : int, default 1
			The step size of the case iterator to use in likelihood calculation.  This is processed as usual for
			Python slicing and iterating.  To include all cases, step by 1 (the default).
		persist : int, default False
			Whether to return a variety of internal and intermediate arrays in the result dictionary.
			If set to 0, only the final `ll` value is included.
		leave_out, keep_only, subsample : int, optional
			Settings for cross validation calculations.
			If `leave_out` and `subsample` are set, then case rows where rownumber % subsample == leave_out are dropped.
			If `keep_only` and `subsample` are set, then only case rows where rownumber % subsample == keep_only are used.
		return_series : bool
			Deprecated, no effect.  Derivatives are always returned as a Series.

		Returns
		-------
		dictx
			The log likelihood is given by key 'll', the first derivative by key 'dll', and the BHHH matrix by 'bhhh'.
			Other arrays are also included if `persist` is set to True.

		"""
		self.__prepare_for_compute(x)
		y = self.__d_log_likelihood_from_dataframes_all_rows(
			return_dll=True,
			return_bhhh=True,
			start_case=start_case,
			stop_case=stop_case,
			step_case=step_case,
			persist=persist,
			leave_out=leave_out,
			keep_only=keep_only,
			subsample=subsample,
		)
		if start_case==0 and stop_case==-1 and step_case==1:
			self._check_if_best(y.ll)
		if return_series and 'dll' in y and not isinstance(y['dll'], (pandas.DataFrame, pandas.Series)):
			y['dll'] = pandas.Series(y['dll'], index=self._frame.index, )
		if return_series and 'bhhh' in y and not isinstance(y['bhhh'], pandas.DataFrame):
			y['bhhh'] = pandas.DataFrame(y['bhhh'], index=self._frame.index, columns=self._frame.index)
		return y

	def d_probability(
			self,
			x=None,
			start_case=0, stop_case=-1, step_case=1,
			leave_out=-1, keep_only=-1, subsample=-1,
	):
		"""
		Compute the partial derivative of probability w.r.t. the parameters.

		Parameters
		----------
		x
		start_case
		stop_case
		step_case
		leave_out
		keep_only
		subsample

		Returns
		-------
		ndarray
		"""
		return self.loglike2(
			x=x,
			start_case=start_case, stop_case=stop_case, step_case=step_case,
			leave_out=leave_out, keep_only=keep_only, subsample=subsample,
			persist=PERSIST_D_PROBABILITY,
		).dprobability


	def __prepare_for_compute(self, x=None, allow_missing_ch=False, allow_missing_av=False):
		missing_ch, missing_av = False, False
		if self._dataframes is None:
			raise MissingDataError('dataframes is not set, maybe you need to call `load_data` first?')
		if not self._dataframes.is_computational_ready(activate=True):
			raise ValueError('DataFrames is not computational-ready')
		if x is not None:
			self.set_values(x)
		self.unmangle()
		self._dataframes._read_in_model_parameters()
		if self._dataframes._data_ch is None:
			if allow_missing_ch or self._does_not_require_choice:
				missing_ch = True
			else:
				raise MissingDataError('model.dataframes does not define data_ch')
		if self._dataframes._data_av is None:
			if allow_missing_av:
				missing_av = True
			else:
				raise MissingDataError('model.dataframes does not define data_av')
		return missing_ch, missing_av


	def loglike(
			self,
			x=None,
			*,
			start_case=0, stop_case=-1, step_case=1,
			persist=0,
			leave_out=-1, keep_only=-1, subsample=-1,
			probability_only=False,
	):
		"""
		Compute a log likelihood value.

		Parameters
		----------
		x : {'null', 'init', 'best', array-like, dict, scalar}, optional
			Values for the parameters.  See :ref:`set_values` for details.
		start_case : int, default 0
			The first case to include in the log likelihood computation.  To include all
			cases, start from 0 (the default).
		stop_case : int, default -1
			One past the last case to include in the log likelihood computation.  This is processed as usual for
			Python slicing and iterating, and negative values count backward from the end.  To include all cases,
			end at -1 (the default).
		step_case : int, default 1
			The step size of the case iterator to use in likelihood calculation.  This is processed as usual for
			Python slicing and iterating.  To include all cases, step by 1 (the default).
		persist : int, default 0
			Whether to return a variety of internal and intermediate arrays in the result dictionary.
			If set to 0, only the final `ll` value is included.
		leave_out, keep_only, subsample : int, optional
			Settings for cross validation calculations.
			If `leave_out` and `subsample` are set, then case rows where rownumber % subsample == leave_out are dropped.
			If `keep_only` and `subsample` are set, then only case rows where rownumber % subsample == keep_only are used.
		probability_only : bool, default False
			Compute only the probability and ignore the likelihood.  If this setting is active, the
			dataframes need not include the "actual" choice data.

		Returns
		-------
		float or array or dictx
			The log likelihood as a float, when `probability_only` is False and `persist` is 0.
			The probability as an array, when `probability_only` is True and `persist` is 0.
			A dictx is returned if `persist` is non-zero.

		"""
		self.__prepare_for_compute(x, allow_missing_ch=probability_only)
		y = self.__d_log_likelihood_from_dataframes_all_rows(
			return_dll=False,
			return_bhhh=False,
			start_case=start_case,
			stop_case=stop_case,
			step_case=step_case,
			persist=persist,
			leave_out=leave_out,
			keep_only=keep_only,
			subsample=subsample,
			probability_only=probability_only,
		)
		if start_case==0 and stop_case==-1 and step_case==1:
			self._check_if_best(y.ll)
		if probability_only:
			return y.probability
		if persist:
			return y
		return y.ll


	def logsums(self, x=None, arr=None):
		"""
		Returns the model logsums.

		Parameters
		----------
		x : {'null', 'init', 'best', array-like, dict, scalar}, optional
			Values for the parameters.  See :ref:`set_values` for details.
		arr : ndarray, optional
			Output array

		Returns
		-------
		arr
		"""
		self.__prepare_for_compute(x, allow_missing_ch=True)
		from .mnl import mnl_logsums_from_dataframes_all_rows
		if arr is None:
			arr = numpy.zeros([self._dataframes._n_cases()], dtype=l4_float_dtype)
		cdef l4_float_t logsum_parameter = 1
		if self._logsum_parameter is not None:
			logsum_parameter = self.get_value(self._logsum_parameter)
		if self.is_mnl():
			mnl_logsums_from_dataframes_all_rows(self._dataframes, logsum_parameter, arr)
		else:
			arr[:] = self.loglike(persist=PERSIST_UTILITY).utility[:,-1]
		return arr

	def exputility(self, x=None, return_dataframe=None):
		try:
			arr = self.loglike(persist=PERSIST_EXP_UTILITY).exp_utility
		except KeyError:
			raise NotImplementedError("exputility is not available for this model structure")
		if return_dataframe == 'names':
			return pandas.DataFrame(
				data=arr,
				columns=self._dataframes._alternative_names,
				index=self._dataframes._data_co.index if self._dataframes._data_co is not None else None,
			)
		elif return_dataframe:
			return pandas.DataFrame(
				data=arr,
				columns=self._dataframes._alternative_codes,
				index=self._dataframes._data_co.index if self._dataframes._data_co is not None else None,
			)
		else:
			return arr

	def utility(self, x=None, return_dataframe=None):
		arr = self.loglike(persist=PERSIST_UTILITY).utility
		if return_dataframe == 'names':
			return pandas.DataFrame(
				data=arr,
				columns=self._dataframes._alternative_names if self.is_mnl() else self.graph.standard_sort_names,
				index=self._dataframes._data_co.index if self._dataframes._data_co is not None else None,
			)
		elif return_dataframe:
			return pandas.DataFrame(
				data=arr,
				columns=self._dataframes._alternative_codes if self.is_mnl() else self.graph.standard_sort,
				index=self._dataframes._data_co.index if self._dataframes._data_co is not None else None,
			)
		else:
			return arr

	def probability(
			self,
			x=None,
			start_case=0,
			stop_case=-1,
			step_case=1,
			return_dataframe=False,
			include_nests=False,
	):
		"""
		Compute probability values.

		Parameters
		----------
		x : {'null', 'init', 'best', array-like, dict, scalar}, optional
			Values for the parameters.  See :ref:`set_values` for details.
		start_case : int, default 0
			The first case to include in the log likelihood computation.  To include all
			cases, start from 0 (the default).
		stop_case : int, default -1
			One past the last case to include in the log likelihood computation.  This is processed as usual for
			Python slicing and iterating, and negative values count backward from the end.  To include all cases,
			end at -1 (the default).
		step_case : int, default 1
			The step size of the case iterator to use in likelihood calculation.  This is processed as usual for
			Python slicing and iterating.  To include all cases, step by 1 (the default).
		return_dataframe : {'names', True, False, 'idco', 'idca', 'idce'}, default False
			Format for the results.  If True or 'idco', a pandas.DataFrame is returned, with case indexes and alternative
			code columns.  If 'names', the alternative names are used for the columns.
			If set to False, the results are returned as a numpy array.
			If 'idca', the resulting dataframe is stacked, such that a single column is included and
			there is a two-level MultiIndex with caseids and alternative codes, respectively.
			If 'idce', the resulting dataframe is stacked and unavailable alternatives are removed.
		include_nests : bool, default False
			Whether to include the nests section in a nested model.  This argument is ignored for MNL models
			as the probability array is naturally limited to only the elemental alternatives.

		Returns
		-------
		array or DataFrame

		"""
		try:
			# if include_nests and return_dataframe is not in (False, 'names'):
			# 	raise ValueError('cannot use both `include_nests` and `return_dataframe`')
			arr = self.loglike(x=x, persist=PERSIST_PROBABILITY, start_case=start_case, stop_case=stop_case, step_case=step_case, probability_only=True)
			if not include_nests:
				arr = arr[:,:self._dataframes._n_alts()]
			if return_dataframe:
				idx = self._dataframes._data_co.index if self._dataframes._data_co is not None else None
				if idx is not None:
					if stop_case == -1:
						stop_case = len(idx)
					idx = idx[start_case:stop_case:step_case]

			if return_dataframe == 'names':
				return pandas.DataFrame(
					data=arr,
					columns=self._dataframes._alternative_names if not include_nests else self.graph.standard_sort_names,
					index=idx,
				)
			elif return_dataframe:
				result = pandas.DataFrame(
					data=arr,
					columns=self._dataframes._alternative_codes if not include_nests else self.graph.standard_sort,
					index=idx,
				)
				if return_dataframe == 'idce':
					return result.stack()[self._dataframes._data_av.stack().astype(bool).values]
				elif return_dataframe == 'idca':
					return result.stack()
				else:
					return result
			else:
				return arr
		except:
			logger.exception('error in probability')
			raise




	@property
	def choice_ca_var(self):
		"""str : An |idca| variable giving the choices as indicator values."""
		return self._choice_ca_var

	@choice_ca_var.setter
	def choice_ca_var(self, x):
		if x is not None:
			x = str(x)
		if self._choice_ca_var != x:
			self.mangle()
		self._choice_ca_var = x
		if x is not None:
			self._choice_co_vars = None
			self._choice_co_code = None
			self._choice_any = False

	@property
	def choice_co_vars(self):
		"""Dict[int,str] : A mapping giving |idco| expressions that evaluate to indicator values.

		Each key represents an alternative code number, and the associated expression
		gives the name of an |idco| variable or some function of |idco| variables that
		indicates whether that alternative was chosen.
		"""
		if self._choice_co_vars:
			return self._choice_co_vars
		else:
			return None

	@choice_co_vars.setter
	def choice_co_vars(self, x):
		if isinstance(x, dict):
			if self._choice_co_vars != x:
				self.mangle()
			self._choice_co_vars = x
			self._choice_ca_var = None
			self._choice_co_code = None
			self._choice_any = False
		elif x is None:
			if self._choice_co_vars != x:
				self.mangle()
			self._choice_co_vars = x
		else:
			raise TypeError('choice_co_vars must be a dictionary')

	@choice_co_vars.deleter
	def choice_co_vars(self):
		self._choice_co_vars = None

	@property
	def choice_co_code(self):
		"""str : An |idco| variable giving the choices as alternative id's."""
		if self._choice_co_code:
			return self._choice_co_code
		else:
			return None

	@choice_co_code.setter
	def choice_co_code(self, x):
		if isinstance(x, str):
			if self._choice_co_code != x:
				self.mangle()
			self._choice_co_code = x
			self._choice_co_vars = None
			self._choice_ca_var = None
			self._choice_any = False
		elif x is None:
			if self._choice_co_code != x:
				self.mangle()
			self._choice_co_code = x
		else:
			raise TypeError('choice_co_vars must be a str')

	@choice_co_code.deleter
	def choice_co_code(self):
		if self._choice_co_code is not None:
			self.mangle()
		self._choice_co_code = None

	@property
	def choice_any(self):
		if self._choice_any:
			return True
		else:
			return False

	@choice_any.setter
	def choice_any(self, x):
		if x:
			self._choice_any = True
			self._choice_co_code = None
			self._choice_co_vars = None
			self._choice_ca_var = None
		else:
			self._choice_any = False

	@choice_any.deleter
	def choice_any(self):
		self._choice_any = False


	@property
	def weight_co_var(self):
		return self._weight_co_var

	@weight_co_var.setter
	def weight_co_var(self, x):
		self._weight_co_var = x

	@property
	def availability_ca_var(self):
		"""str : An |idca| variable or expression indicating if alternatives are available."""
		return self._availability_var

	@availability_ca_var.setter
	def availability_ca_var(self, x):
		if x is not None:
			x = str(x)
		if self._availability_var != x:
			self.mangle()
		self._availability_var = x
		self._availability_co_vars = None
		self._availability_any = False

	@property
	def availability_var(self):
		"""str : An |idca| variable or expression indicating if alternatives are available.

		Deprecated, prefer `availability_ca_var` for clarity.
		"""
		return self._availability_var

	@availability_var.setter
	def availability_var(self, x):
		if x is not None:
			x = str(x)
		if self._availability_var != x:
			self.mangle()
		self._availability_var = x
		self._availability_co_vars = None
		self._availability_any = False

	@property
	def availability_co_vars(self):
		"""Dict[int,str] : A mapping giving |idco| expressions that evaluate to availability indicators.

		Each key represents an alternative code number, and the associated expression
		gives the name of an |idco| variable or some function of |idco| variables that
		indicates whether that alternative is available.
		"""
		x = self._availability_co_vars
		if x == 1 or str(x) == '1' or x is True:
			try:
				alts = self.dataframes.alternative_codes()
			except:
				try:
					alts = self.dataservice.alternative_codes()
				except:
					raise ValueError('must define alternatives to have them be all available from co data')
			x = {i:1 for i in alts}
		return x

	@availability_co_vars.setter
	def availability_co_vars(self, x):
		from typing import Mapping
		if not isinstance(x, Mapping):
			raise TypeError(f'availability_co_vars must be dict not {type(x)}')
		if self._availability_co_vars != x:
			self.mangle()
		self._availability_co_vars = x
		self._availability_var = None
		self._availability_any = False

	@property
	def availability_any(self):
		"""bool : A flag indicating whether availability should be inferred from the data.

		This only applies to DataFrames-based models, as the Dataset interface does
		not include a mechanism for the data to self-describe an availability feature.
		"""
		return self._availability_any

	@availability_any.setter
	def availability_any(self, x):
		self._availability_any = True
		self._availability_co_vars = None
		self._availability_var = None


	@property
	def graph(self):
		if self._graph is None and not self._is_clone:
			try:
				self.initialize_graph()
			except ValueError:
				import warnings
				warnings.warn('cannot initialize graph, must define dataframes, dataservice, or give alternative_codes explicitly')
		return self._graph

	@graph.setter
	def graph(self, x):
		self._graph = x

	def initialize_graph(self, dataframes=None, alternative_codes=None, alternative_names=None, root_id=0):
		"""
		Write a nesting tree graph for a MNL model.

		Parameters
		----------
		dataframes : DataFrames, optional
			Use this to determine the included alternatives.
		alternative_codes : array-like, optional
			Explicitly give alternative codes. Ignored if `dataframes` is given
			or if the model has dataframes or a dataservice already set.
		alternative_names : array-like, optional
			Explicitly give alternative names. Ignored if `dataframes` is given
			or if the model has dataframes or a dataservice already set.
		root_id : int, default 0
			The id code of the root node.

		Raises
		------
		ValueError
			The model is unable to infer the alternative codes to use.  This can
			be avoided by giving alternative codes explicitly or having previously
			set dataframes or a dataservice that will give the alternative codes.
		"""

		if dataframes is not None:
			self.dataframes = dataframes

		if self._dataframes is None and alternative_codes is None and self._dataservice is None:
			raise ValueError('must define dataframes, dataservice, or give alternative_codes explicitly')

		if self._dataframes is not None and alternative_codes is not None:
			import warnings
			warnings.warn('alternative_codes are ignored when dataframes are set')
			alternative_codes = self._dataframes._alternative_codes

		if alternative_codes is None:
			if self._dataframes is None and self._dataservice is not None:
				alternative_codes = self._dataservice.alternative_codes()
				alternative_names = self._dataservice.alternative_names()
			else:
				alternative_codes = self._dataframes._alternative_codes
				alternative_names = self._dataframes._alternative_names

		from .tree import NestingTree
		g = NestingTree(root_id=root_id)
		# print(root_id)
		if alternative_names is None:
			for a in alternative_codes:
				# print(a)
				g.add_node(a)
		else:
			for a, name in zip(alternative_codes, alternative_names):
				g.add_node(a, name=name)
		self.graph = g

	def is_mnl(self):
		"""
		Check if this model is a MNL model

		Returns
		-------
		bool
		"""
		if self._graph is None:
			return True
		if len(self._graph) - len(self._graph.elementals) == 1:
			return True
		return False

	def graph_descrip(self, which='nodes'):
		"""
		Generate DataFrames that describe this graph.

		Parameters
		----------
		which : {'nodes','edges'}
			Which type of description to return.

		Returns
		-------
		DataFrame
			Describing nodes or edges
		"""
		if which in ('edge','edges','link','links'):
			ef = pandas.DataFrame.from_dict(self.graph.edges, orient='index')
			ef.index.names = ['up', 'down']
			return ef.fillna("")

		nf = pandas.DataFrame.from_dict(self.graph.nodes, orient='index')
		from itertools import islice
		max_c = 8
		max_p = 8
		for n in nf.index:
			children = [e[1] for e in islice(self.graph.out_edges(n), max_c)]
			if len(children) > max_c-1:
				children[-1] = '...'
			if children:
				nf.loc[n,'children'] = ", ".join(str(c) for c in children)
		for n in nf.index:
			parents = [e[0] for e in islice(self.graph.in_edges(n), max_p)]
			if len(parents) > max_p-1:
				parents[-1] = '...'
			if parents:
				nf.loc[n,'parents'] = ", ".join(str(c) for c in parents)
		nf.drop(columns='root', errors='ignore', inplace=True)
		return nf.fillna("")

	###########
	# Scoring
	#
	# These scoring mechanisms are popular in machine learning applications.

	def top_k_accuracy(self, k=1, x=None):
		"""
		Compute top-K accuracy for the model.

		Parameters
		----------
		k : int or iterable of ints
			The ranking to compute.
		x : {'null', 'init', 'best', array-like, dict, scalar}, optional
			Values for the parameters.  See :ref:`set_values` for details.

		Returns
		-------
		float
			Fraction of cases where the chosen alternative is ranked at k or better.

		Notes
		-----
		Top
		"""
		from ..scoring import top_k_accuracy, probability_to_rank
		if self._dataframes._array_wt is None:
			return top_k_accuracy(
				probability_to_rank(self.probability(x=x)),
				self._dataframes._data_ch,
				k=k,
			)
		else:
			return top_k_accuracy(
				probability_to_rank(self.probability(x=x)),
				self.dataframes.array_ch() * self.dataframes.array_wt(),
				k=k,
			)


	def logloss(self, x=None):
		"""
		Logloss is the average per-case (per unit weight for weighted data) negative log likelihood.
		"""
		return -self.loglike(x=x) / self._dataframes.total_weight()
