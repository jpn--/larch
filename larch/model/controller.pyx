# cython: language_level=3, embedsignature=True

include "../general_precision.pxi"
from ..general_precision import l4_float_dtype
from ..general_precision cimport l4_float_t

import numpy
import pandas
from typing import Union

import logging
logger = logging.getLogger('L5.model')


from ..dataframes cimport DataFrames

from ..roles import LinearFunction, LinearComponent, ParameterRef, DictOfLinearFunction, DictOfStrings, DataRef

def _empty_parameter_frame(names, nullvalue=0, initvalue=0, max=None, min=None):

	cdef int len_names = 0 if names is None else len(names)

	min_ = min if min is not None else -numpy.inf
	max_ = max if max is not None else numpy.inf
	return pandas.DataFrame(index=names, data=dict(
			value = numpy.full(len_names, fill_value=initvalue, dtype=l4_float_dtype),
			minimum = numpy.full(len_names, fill_value=min_, dtype=l4_float_dtype),
			maximum = numpy.full(len_names, fill_value=max_, dtype=l4_float_dtype),
			nullvalue = numpy.full(len_names, fill_value=nullvalue, dtype=l4_float_dtype),
			initvalue = numpy.full(len_names, fill_value=initvalue, dtype=l4_float_dtype),
			holdfast = numpy.zeros(len_names, dtype=numpy.int8),
			note = numpy.zeros(len_names, dtype='<U128')
		), columns=['value', 'initvalue', 'nullvalue', 'minimum', 'maximum', 'holdfast', 'note'])


class MissingDataError(ValueError):
	pass


cdef class Model5c:

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
	):

		self._dataframes = None

		if is_clone:
			self._is_clone = True
			self._utility_co_functions = utility_co
			self._utility_ca_function = utility_ca
			self._quantity_ca_function = quantity_ca
			if quantity_scale is None:
				self._quantity_scale = None
			else:
				self._quantity_scale = str(quantity_scale)

		else:
			self._is_clone = False
			self._utility_co_functions = DictOfLinearFunction(utility_co, touch_callback=self.mangle)
			self._utility_ca_function  = LinearFunction(utility_ca, touch_callback=self.mangle)
			self._quantity_ca_function  = LinearFunction(quantity_ca, touch_callback=self.mangle)
			if quantity_scale is None:
				self._quantity_scale = None
			else:
				self._quantity_scale = str(quantity_scale)

		self.rename_parameters = DictOfStrings(touch_callback=self.mangle)

		self._cached_loglike_null = 0
		self._cached_loglike_constants_only = 0
		self._cached_loglike_best = -numpy.inf
		self._most_recent_estimation_result = None

		self.n_threads = n_threads

		self._dataservice = dataservice

		if frame is not None:
			self.frame = frame
		else:
			self.frame = _empty_parameter_frame(parameters)

		self._graph = graph

		self._scan_all_ensure_names()
		self.mangle()

	@property
	def title(self):
		if self._title is None:
			return "Untitled"
		else:
			return self._title

	@title.setter
	def title(self, value):
		self._title = str(value)

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
		self.clear_best_loglike()
		self._mangled = True
		#self._clear_cached_values()

	def unmangle(self, force=False):
		if self._mangled or force:
			self._scan_all_ensure_names()
			if self._dataframes is not None:
				self._dataframes._check_data_is_sufficient_for_model(self)
				self._refresh_derived_arrays()
			self._mangled = False



	def _scan_all_ensure_names(self):
		self._scan_utility_ensure_names()
		self._scan_quantity_ensure_names()
		self._scan_logsums_ensure_names()
		self.frame.sort_index(inplace=True)

	def _scan_utility_ensure_names(self):
		"""
		Scan the utility functions and ensure all named parameters appear in the parameter frame.

		Any named parameters that do not appear in the parameter frame are added.
		"""
		try:
			nameset = set()
			u_co_dataset = set()

			_utility_co_postprocessed = self._utility_co_postprocess
			for altcode, linear_function in _utility_co_postprocessed.items():
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
		nameset = set()

		#self._q_ca_varindex = pandas.Index( str(component.data) for component in self._quantity_ca_postprocess)
		for component in self._quantity_ca_function:
			nameset.add(self.__p_rename(component.param))

		self._ensure_names(nameset)

	def _scan_logsums_ensure_names(self):
		nameset = set()

		if self._graph is not None:
			for nodecode in self._graph.topological_sorted_no_elementals:
				if nodecode != self._graph._root_id:
					param_name = str(self._graph.node[nodecode]['parameter'])
					# if param_name in self._snapped_parameters:
					# 	snapped = self._snapped_parameters[param_name]
					# 	param_name = str(snapped._is_scaled_parameter()[0])
					nameset.add(self.__p_rename(param_name))

		if len(self._quantity_ca_function)>0:
			# for nodecode in self._graph.elementals:
			# 	try:
			# 		param_name = str(self._graph.node[nodecode]['parameter'])
			# 	except KeyError:
			# 		pass
			# 	else:
			# 		nameset.add(self.__p_rename(param_name))

			if self.quantity_scale is not None:
				nameset.add(self.__p_rename(self.quantity_scale))

		if self.logsum_parameter is not None:
			nameset.add(self.__p_rename(self.logsum_parameter))

		self._ensure_names(nameset, nullvalue=1, initvalue=1, min=0.001, max=1)

	def _ensure_names(self, names, **kwargs):
		existing_names = set(self.frame.index)
		nameset = set(names)
		missing_names = nameset - existing_names
		if missing_names:
			self.frame = self.frame.append(_empty_parameter_frame([n for n in names if (n in missing_names)], **kwargs), verify_integrity=True)

	def __contains__(self, item):
		return (item in self.pf.index) or (item in self.rename_parameters)

	@property
	def pf(self):
		self.unmangle()
		return self.frame

	def pf_sort(self):
		self.unmangle()
		self.frame.sort_index(inplace=True)
		self.mangle()
		self.unmangle()
		return self.frame

	@property
	def pvals(self):
		self.unmangle()
		return self.frame['value'].values.copy()

	@property
	def pnames(self):
		self.unmangle()
		return self.frame.index.values.copy()

	def pvalue(self, parameter_name, apply_formatting=False, default_value=None):
		"""
		Get the value of a parameter or a parameter expression.

		Parameters
		----------
		parameter_name : str or ParameterRef
		apply_formatting : bool

		Returns
		-------

		"""
		from .roles import _param_math_binaryop
		from numbers import Number
		try:
			if isinstance(parameter_name, _param_math_binaryop):
				if apply_formatting:
					return parameter_name.strf(self)
				else:
					return parameter_name.value(self)
			elif isinstance(parameter_name, dict):
				result = type(parameter_name)()
				for k,v in parameter_name.items():
					result[k] = self.pvalue(v, apply_formatting=apply_formatting)
				return result
			elif isinstance(parameter_name, (set, tuple, list)):
				result = dict()
				for k in parameter_name:
					result[k] = self.pvalue(k) #? apply_formatting=apply_formatting ?
				return result
			elif isinstance(parameter_name, Number):
				return parameter_name
			else:
				return self.pf.loc[self.__p_rename(parameter_name),'value']
		except KeyError:
			if default_value is not None:
				return default_value
			raise

	def set_value(self, name, value=None, **kwargs):
		if isinstance(name, ParameterRef):
			name = str(name)
		if name not in self.frame.index:
			self.unmangle()
		if value is not None:
			# value = numpy.float64(value)
			# self.frame.loc[name,'value'] = value
			kwargs['value'] = value
		for k,v in kwargs.items():
			if k in self.frame.columns:
				if k == 'holdfast':
					v = numpy.int8(v)
				else:
					v = l4_float_dtype(v)
				self.frame.loc[name, k] = v

			# update init values when they are implied
			if k=='value':
				if 'initvalue' not in kwargs and pandas.isnull(self.frame.loc[name, 'initvalue']):
					self.frame.loc[name, 'initvalue'] = l4_float_dtype(v)

		# update null values when they are implied
		if 'nullvalue' not in kwargs and pandas.isnull(self.frame.loc[name, 'nullvalue']):
			self.frame.loc[name, 'nullvalue'] = 0

		# refresh everything # TODO: only refresh what changed
		try:
			if self._dataframes is not None:
				self._dataframes._read_in_model_parameters()
		except AttributeError:
			# the model may not have been ever unmangled yet
			pass

	def lock_value(self, name, value, note=None):
		if isinstance(name, ParameterRef):
			name = str(name)
		if value is 'null':
			value = self.pf.loc[name, 'nullvalue']
		self.set_value(name, value, holdfast=1, initvalue=value, nullvalue=value, minimum=value, maximum=value)
		if note is not None:
			self.frame.loc[name, 'note'] = note

	def lock_values(self, *names, note=None, **kwargs):
		"""Set a fixed value for one or more model parameters.

		Positional arguments give the names of parameters to fix at the current value.
		Keyword arguments name the parameter and give a value to set as fixed.

		Other Parameters
		----------------
		note : str, optional
			Add a note to all parameters fixed with this comment.  The same note is added to
			all the parameters.
		"""
		for name in names:
			value = self.get_value(name)
			self.lock_value(name, value, note=note)
		for name, value in kwargs.items():
			self.lock_value(name, value, note=note)

	def get_value(self, name):
		if isinstance(name, ParameterRef):
			return name.value(self)
		return self.frame.loc[name,'value']

	def get_value_x(self, name, default):
		if name is None:
			return default
		try:
			return self.frame.loc[name,'value']
		except KeyError:
			return default

	def get_holdfast_x(self, name, default):
		if name is None:
			return default
		try:
			return self.frame.loc[name,'holdfast']
		except KeyError:
			return default


	def get_slot_x(self, name, holdfast_invalidates=False):
		"""Get the position of a named parameter within the parameters index.

		Parameters
		----------
		name : str
			Name of parameter to find.
		holdfast_invalidates : bool, default False
			If the parameter is holdfast, return -1 (as if there is no parameter).

		Returns
		-------
		int
			Position of parameter, or -1 if it is not found.
		"""
		if name is None:
			return -1
		try:
			result = self.frame.index.get_loc(name)
		except KeyError:
			return -1
		if holdfast_invalidates:
			if self.frame.loc[name,'holdfast']:
				return -1
		return result

	def __getitem__(self, name):
		try:
			return self.frame.loc[name,:]
		except:
			logger.exception("error in Model5c.__getitem__")
			raise

	def set_values(self, values=None, **kwargs):
		"""
		Set the parameter values.

		Parameters
		----------
		values : {'null', 'init', 'best', array-like, dict, scalar}, optional
			New values to set for the parameters.
			If 'null' or 'init', the current values are set equal to the null or initial values given in
			the 'nullvalue' or 'initvalue' column of the parameter frame, respectively.
			If 'best', the current values are set equal to the values given in the 'best' column of the
			parameter frame, if that columns exists, otherwise a ValueError exception is raised.
			If given as array-like, the array must be a vector with length equal to the length of the
			parameter frame, and the given vector will replace the current values.  If given as a dictionary,
			the dictionary is used to update `kwargs` before they are processed.
		kwargs : dict
			Any keyword arguments (or if `values` is a dictionary) are used to update the included named parameters
			only.  A warning will be given if any key of the dictionary is not found among the existing named
			parameters in the parameter frame, and the value associated with that key is ignored.  Any parameters
			not named by key in this dictionary are not changed.

		Notes
		-----
		Setting parameters both in the `values` argument and through keyword assignment is not explicitly disallowed,
		although it is not recommended and may be disallowed in the future.

		Raises
		------
		ValueError
			If setting to 'best' but there is no 'best' column in the `pf` parameters DataFrame.
		"""

		if isinstance(values, str):
			if values.lower() == 'null':
				values = self.frame.loc[:,'nullvalue'].values
			elif values.lower() == 'init':
				values = self.frame.loc[:,'initvalue'].values
			elif values.lower() == 'initial':
				values = self.frame.loc[:, 'initvalue'].values
			elif values.lower() == 'best':
				if 'best' not in self.frame.columns:
					raise ValueError('best is not stored')
				else:
					values = self.frame.loc[:, 'best'].values
		if isinstance(values, dict):
			kwargs.update(values)
			values = None
		if values is not None:
			if not isinstance(values, (int,float)):
				if len(values) != len(self.frame):
					raise ValueError(f'gave {len(values)} values, needs to be exactly {len(self.frame)} values')
			# only change parameters that are not holdfast
			free_parameters= self.frame['holdfast'].values == 0
			if isinstance(values, (int,float)):
				self.frame.loc[free_parameters, 'value'] = l4_float_dtype(values)
			else:
				self.frame.loc[free_parameters, 'value'] = numpy.asanyarray(values)[free_parameters]
		if len(kwargs):
			if self._mangled:
				self._scan_all_ensure_names()
			for k,v in kwargs.items():
				if k in self.frame.index:
					if self.frame.loc[k,'holdfast'] == 0:
						self.frame.loc[k, 'value'] = v
				else:
					import warnings
					warnings.warn(f'{k} not in model')
		# refresh everything # TODO: only refresh what changed
		try:
			if self._dataframes is not None:
				self._dataframes._read_in_model_parameters()
			#_refresh_derived_arrays()
		except AttributeError:
			# the model may not have been ever unmangled yet
			pass

	def get_values(self):
		return self.pf.loc[:,'value'].copy()

	def shift_values(self, shifts):
		self.set_values(self.pvals + shifts)
		return self

	def update_values(self, values):
		self.set_values(values)
		return self

	@property
	def utility_ca(self):
		if self._is_clone:
			return self._utility_ca_function
		return LinearFunction(touch_callback=self.mangle) + self._utility_ca_function

	@utility_ca.setter
	def utility_ca(self, value):
		if self._is_clone:
			raise PermissionError('cannot edit utility_ca when is_clone is True')
		if isinstance(value, (LinearComponent, ParameterRef)):
			value = LinearFunction() + value
		elif value is None or value == 0:
			value = LinearFunction()
		if not isinstance(value, LinearFunction):
			raise TypeError('needs LinearFunction')
		self._utility_ca_function = value
		self._utility_ca_function.set_touch_callback(self.mangle)
		self.mangle()

	@utility_ca.deleter
	def utility_ca(self):
		if self._is_clone:
			raise PermissionError('cannot edit utility_ca when is_clone is True')
		self._utility_ca_function = LinearFunction()
		self._utility_ca_function.set_touch_callback(self.mangle)
		self.mangle()

	@property
	def utility_co(self):
		return self._utility_co_functions

	@utility_co.setter
	def utility_co(self, value):
		if self._is_clone:
			raise PermissionError('cannot edit utility_co when is_clone is True')
		if not isinstance(value, (dict, DictOfLinearFunction)):
			raise TypeError('needs [dict] of {key:LinearFunction}')
		value = value.copy()
		for k in value.keys():
			if isinstance(value[k], (LinearComponent, ParameterRef)):
				value[k] = LinearFunction() + value[k]
			if not isinstance(value[k], LinearFunction):
				raise TypeError('needs dict of {key:[LinearFunction]}')
		self._utility_co_functions = DictOfLinearFunction(value, touch_callback=self.mangle)


	@property
	def quantity_ca(self):
		if self._is_clone:
			return self._quantity_ca_function
		return LinearFunction(touch_callback=self.mangle) + self._quantity_ca_function

	@quantity_ca.setter
	def quantity_ca(self, value):
		if self._is_clone:
			raise PermissionError('cannot edit quantity_ca when is_clone is True')
		if isinstance(value, (LinearComponent, ParameterRef)):
			value = LinearFunction() + value
		if not isinstance(value, LinearFunction):
			raise TypeError('needs LinearFunction')
		self._quantity_ca_function = value
		self._quantity_ca_function.set_touch_callback(self.mangle)
		self.mangle()

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


	def utility_functions(self, subset=None, resolve_parameters=True):
		from xmle import Elem
		x = Elem('div')
		t = x.elem('table', style="margin-top:1px;", attrib={'class':'floatinghead'})
		if len(self.utility_co):
			# t.elem('caption', text=f"Utility Functions",
			# 	   style="caption-side:top;text-align:left;font-family:Roboto;font-weight:700;"
			# 			 "font-style:normal;font-size:100%;padding:0px;color:black;")
			t_head = t.elem('thead')
			tr = t_head.elem('tr')
			tr.elem('th', text="alt")
			tr.elem('th', text='formula')
			t_body = t.elem('tbody')
			for j in self.utility_co.keys():
				if subset is None or j in subset:
					tr = t_body.elem('tr')
					tr.elem('td', text=str(j))
					utilitycell = tr.elem('td')
					utilitycell.elem('div')
					anything = False
					if len(self.utility_ca):
						utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + "
						utilitycell << list(self.utility_ca.__xml__(linebreaks=True, resolve_parameters=self if resolve_parameters else None))
						anything = True
					if j in self.utility_co:
						if anything:
							utilitycell << Elem('br')
						v = self.utility_co[j]
						utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + "
						utilitycell << list(v.__xml__(linebreaks=True, resolve_parameters=self if resolve_parameters else None))
						anything = True
					if len(self.quantity_ca):
						if anything:
							utilitycell << Elem('br')
						if self.quantity_scale:
							utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + "
							from .roles import ParameterRef
							utilitycell << list(ParameterRef(self.quantity_scale).__xml__(resolve_parameters=self if resolve_parameters else None))
							utilitycell[-1].tail = (utilitycell[-1].tail or "") + " * log("
						else:
							utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + log("
						content = self.quantity_ca.__xml__(linebreaks=True, lineprefix="  ",
														   exponentiate_parameters=True, resolve_parameters=self if resolve_parameters else None)
						utilitycell << list(content)
						utilitycell.elem('br', tail=")")
		else:
			# there is no differentiation by alternatives, just give one formula
			# t.elem('caption', text=f"Utility Function",
			# 	   style="caption-side:top;text-align:left;font-family:Roboto;font-weight:700;"
			# 			 "font-style:normal;font-size:100%;padding:0px;color:black;")
			tr = t.elem('tr')
			utilitycell = tr.elem('td')
			utilitycell.elem('div')
			anything = False
			if len(self.utility_ca):
				utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + "
				utilitycell << list(self.utility_ca.__xml__(linebreaks=True, resolve_parameters=self if resolve_parameters else None))
				anything = True
			if len(self.quantity_ca):
				if anything:
					utilitycell << Elem('br')
				if self.quantity_scale:
					utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + "
					from ..roles import ParameterRef
					utilitycell << list(ParameterRef(self.quantity_scale).__xml__(resolve_parameters=self if resolve_parameters else None))
					utilitycell[-1].tail = (utilitycell[-1].tail or "") + " * log("
				else:
					utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + log("
				content = self.quantity_ca.__xml__(linebreaks=True, lineprefix="  ", exponentiate_parameters=True, resolve_parameters=self if resolve_parameters else None)
				utilitycell << list(content)
				utilitycell.elem('br', tail=")")
		return x



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
		u = DictOfLinearFunction()
		keys = list(self.utility_co.keys())
		u_found = {k:set() for k in keys}
		for altkey in keys:
			u[altkey] = LinearFunction()
		for altkey in keys:
			for n, i in enumerate(self.utility_co[altkey]):
				try:
					i_d = i.data
				except:
					logger.error(f"n={n}, altkey={altkey}, len(self.utility_co[altkey])={len(self.utility_co[altkey])}")
					raise
				while i_d in u_found[altkey]:
					i_d = i_d + DataRef('0')
				u_found[altkey].add(i_d)
				# if i.param in self._snapped_parameters:
				# 	u[altkey] += self._snapped_parameters[i.param] * i_d * i.scale
				# else:
				# 	u[altkey] += i.param * i_d * i.scale
				u[altkey] += i.param * i_d * i.scale
		return u

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
		return self.utility_ca

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
		return self.quantity_ca

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

	def _set_dataframes(self, DataFrames x):
		x.computational = True
		self.clear_best_loglike()
		self.unmangle()
		x._check_data_is_sufficient_for_model(self)
		self._dataframes = x
		self._refresh_derived_arrays()
		self._dataframes._read_in_model_parameters()

	@dataframes.setter
	def dataframes(self, x):
		if isinstance(x, pandas.DataFrame):
			x = DataFrames(x)
		self._set_dataframes(x)

	def load_data(self, dataservice=None, autoscale_weights=True):
		if dataservice is not None:
			self._dataservice = dataservice
		if self._dataservice is not None:
			self.dataframes = self._dataservice.make_dataframes(self.required_data())
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

	def loglike2(self, x=None, *, start_case=0, stop_case=-1, step_case=1, persist=0, leave_out=-1, keep_only=-1, subsample=-1, return_series=True):
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
		Dict
			The log likelihood is given by key 'll' and the first derivative by key 'dll'.
			Other arrays are also included if `persist` is set to True.

		"""
		self.__prepare_for_compute(x)
		if self.is_mnl():
			from .mnl import mnl_d_log_likelihood_from_dataframes_all_rows
			y = mnl_d_log_likelihood_from_dataframes_all_rows(
				self._dataframes,
				num_threads=self.n_threads,
				return_dll=True,
				return_bhhh=False,
				start_case=start_case,
				stop_case=stop_case,
				step_case=step_case,
				persist=persist,
				leave_out=leave_out,
				keep_only=keep_only,
				subsample=subsample,
			)
		else:
			from .nl import nl_d_log_likelihood_from_dataframes_all_rows
			y = nl_d_log_likelihood_from_dataframes_all_rows(
				self._dataframes,
				self,
				return_dll=True,
				return_bhhh=False,
				start_case=start_case,
				stop_case=stop_case,
				step_case=step_case,
				persist=persist,
				leave_out=leave_out,
				keep_only=keep_only,
				subsample=subsample,
			)
		if start_case==0 and stop_case==-1:
			self.__check_if_best(y.ll)
		# if return_series and 'dll' in y and not isinstance(y['dll'], (pandas.DataFrame, pandas.Series)):
		# 	y['dll'] = pandas.Series(y['dll'], index=self.frame.index, )
		return y

	def _loglike2_tuple(self, *args, **kwargs):
		"""
		Compute a log likelihood value and it first derivative.

		This is a convenience function that returns these values in a 2-tuple instead of a Dict,
		for compatibility with scipy.optimize.  It accepts all the same input arguments as :ref:`loglike2`.

		"""
		result = self.loglike2(*args, **kwargs)
		return result.ll, result.dll

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

		Returns
		-------
		Dict
			The log likelihood is given by key 'll', the first derivative by key 'dll', and the BHHH matrix by 'bhhh'.
			Other arrays are also included if `persist` is set to True.

		"""
		self.__prepare_for_compute(x)
		if self.is_mnl():
			from .mnl import mnl_d_log_likelihood_from_dataframes_all_rows
			y = mnl_d_log_likelihood_from_dataframes_all_rows(
				self._dataframes,
				num_threads=self.n_threads,
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
		else:
			from .nl import nl_d_log_likelihood_from_dataframes_all_rows
			y = nl_d_log_likelihood_from_dataframes_all_rows(
				self._dataframes,
				self,
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
		if start_case==0 and stop_case==-1:
			self.__check_if_best(y.ll)
		if return_series and 'dll' in y and not isinstance(y['dll'], (pandas.DataFrame, pandas.Series)):
			y['dll'] = pandas.Series(y['dll'], index=self.frame.index, )
		if return_series and 'bhhh' in y and not isinstance(y['bhhh'], pandas.DataFrame):
			y['bhhh'] = pandas.DataFrame(y['bhhh'], index=self.frame.index, columns=self.frame.index)
		return y

	def _loglike2_bhhh_tuple(self, *args, **kwargs):
		"""
		Compute a log likelihood value, its first derivative, and the BHHH approximation of the Hessian.

		This is a convenience function that returns these values in a 3-tuple instead of a Dict,
		for compatibility with scipy.optimize.  It accepts all the same input arguments as :ref:`loglike2_bhhh`.

		"""
		result = self.loglike2_bhhh(*args, **kwargs)
		return result.ll, result.dll, result.bhhh

	def bhhh(self, x=None, *, return_series=False):
		return self.loglike2_bhhh(x=x,return_series=return_series).bhhh

	def _bhhh_simple_direction(self, *args, **kwargs):
		bhhh_inv = self._free_slots_inverse_matrix(self.bhhh(*args))
		return numpy.dot(self.d_loglike(*args), bhhh_inv)

	def simple_step_bhhh(self, steplen=1.0, printer=None, leave_out=-1, keep_only=-1, subsample=-1):
		"""
		Makes one step using the BHHH algorithm.

		Parameters
		----------
		steplen: float
		printer: callable

		Returns
		-------
		loglike, convergence_tolerance
		"""
		current = self.pvals.copy()
		current_ll, current_dll, current_bhhh = self._loglike2_bhhh_tuple(leave_out=leave_out,
																		  keep_only=keep_only,
																		  subsample=subsample,
																		  )
		bhhh_inv = self._free_slots_inverse_matrix(current_bhhh)
		direction = numpy.dot(current_dll, bhhh_inv)
		tolerance = numpy.dot(direction, current_dll)
		while True:
			self.set_values(current + direction * steplen)
			val = self.loglike()
			if val > current_ll: break
			steplen *= 0.5
			if steplen < 0.01: break
		if val <= current_ll:
			self.set_values(current)
			raise RuntimeError("simple step bhhh failed")
		if printer is not None:
			printer("simple step bhhh {} to gain {}".format(steplen, val - current_ll))
		return val, tolerance

	def jumpstart_bhhh(self, steplen=0.5, jumpstart=0, jumpstart_split=5, leave_out=-1, keep_only=-1, subsample=-1):
		"""
		Jump start optimization

		Parameters
		----------
		steplen
		jumpstart
		jumpstart_split

		"""
		for jump in range(jumpstart):
			j_pvals = self.pvals.copy()
			n_cases = self._dataframes._n_cases()
			jump_breaks = list(range(0,n_cases, n_cases//jumpstart_split+(1 if n_cases%jumpstart_split else 0))) + [-1]
			for j0,j1 in zip(jump_breaks[:-1],jump_breaks[1:]):
				j_ll, current_dll, current_bhhh = self._loglike2_bhhh_tuple(start_case=j0, stop_case=j1,
																			leave_out=leave_out, keep_only=keep_only,
																			subsample=subsample)
				bhhh_inv = self._free_slots_inverse_matrix(current_bhhh)
				direction = numpy.dot(current_dll, bhhh_inv)
				self.set_values(j_pvals + direction * steplen)



	def simple_fit_bhhh(
			self,
			steplen=1.0,
			printer=None,
			ctol=1e-4,
			maxiter=100,
			callback=None,
			jumpstart=0,
			jumpstart_split=5,
			minimum_steplen=0.0001,
			leave_out=-1,
			keep_only=-1,
			subsample=-1,
	):
		"""
		Makes a series of steps using the BHHH algorithm.

		Parameters
		----------
		steplen: float
		printer: callable

		Returns
		-------
		loglike, convergence_tolerance, n_iters, steps
		"""
		current_pvals = self.pvals.copy()
		iter = 0
		steps = []

		if jumpstart:
			self.jumpstart_bhhh(jumpstart=jumpstart, jumpstart_split=jumpstart_split)
			iter += jumpstart

		current_ll, current_dll, current_bhhh = self._loglike2_bhhh_tuple(leave_out=leave_out,
																		  keep_only=keep_only,
																		  subsample=subsample,
																		  )
		bhhh_inv = self._free_slots_inverse_matrix(current_bhhh)
		direction = numpy.dot(current_dll, bhhh_inv)
		tolerance = numpy.dot(direction, current_dll)

		while abs(tolerance) > ctol and iter < maxiter:
			iter += 1
			while True:
				self.set_values(current_pvals + direction * steplen)
				proposed_ll, proposed_dll, proposed_bhhh = self._loglike2_bhhh_tuple(leave_out=leave_out,
																					 keep_only=keep_only,
																					 subsample=subsample,
																					 )
				if proposed_ll > current_ll: break
				steplen *= 0.5
				if steplen < minimum_steplen: break
			if proposed_ll <= current_ll:
				self.set_values(current_pvals)
				raise RuntimeError(f"simple step bhhh failed\ndirection = {str(direction)}")
			if printer is not None:
				printer("simple step bhhh {} to gain {}".format(steplen, proposed_ll - current_ll))
			steps.append(steplen)

			current_ll, current_dll, current_bhhh = proposed_ll, proposed_dll, proposed_bhhh
			current_pvals = self.pvals.copy()
			if callback is not None:
				callback(current_pvals)
			bhhh_inv = self._free_slots_inverse_matrix(current_bhhh)
			direction = numpy.dot(current_dll, bhhh_inv)
			tolerance = numpy.dot(direction, current_dll)

		if abs(tolerance) <= ctol:
			message = "Optimization terminated successfully."
		else:
			message = f"Optimization terminated after {iter} iterations."

		return current_ll, tolerance, iter, numpy.asarray(steps), message

	def _bhhh_direction_and_convergence_tolerance(self, *args):
		bhhh_inv = self._free_slots_inverse_matrix(self.bhhh(*args))
		_1 = self.d_loglike(*args)
		direction = numpy.dot(_1, bhhh_inv)
		return direction, numpy.dot(direction, _1)

	def loglike3(self, x=None, **kwargs):
		"""
		Compute a log likelihood value, it first derivative, and the finite-difference approximation of the Hessian.

		See :ref:`loglike2` for a description of arguments.

		Returns
		-------
		Dict
			The log likelihood is given by key 'll', the first derivative by key 'dll', and the second derivative by 'd2ll'.
			Other arrays are also included if `persist` is set to True.

		"""
		part = self.loglike2(x=x, **kwargs)
		from ..math.optimize import approx_fprime
		part['d2ll'] = approx_fprime(self.pvals, lambda y: self.d_loglike(y, **kwargs))
		return part

	def neg_loglike2(self, x=None, start_case=0, stop_case=-1, step_case=1, leave_out=-1, keep_only=-1, subsample=-1):
		result = self.loglike2(
			x=x,
			start_case=start_case, stop_case=stop_case, step_case=step_case,
			leave_out=leave_out, keep_only=keep_only, subsample=subsample
		)
		return (-result.ll, -result.dll)

	def neg_loglike3(self, *args, **kwargs):
		from ..util import Dict
		result = Dict()
		part = self.loglike3(*args, **kwargs)
		return (-result.ll, -result.dll, -result.d2ll)

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
			if allow_missing_ch:
				missing_ch = True
			else:
				raise MissingDataError('model.dataframes does not define data_ch')
		if self._dataframes._data_av is None:
			if allow_missing_av:
				missing_av = True
			else:
				raise MissingDataError('model.dataframes does not define data_av')
		return missing_ch, missing_av

	def __check_if_best(self, computed_ll):
		if computed_ll > self._cached_loglike_best:
			self._cached_loglike_best = computed_ll
			self.frame['best'] = self.frame['value']

	def clear_best_loglike(self):
		if 'best' in self.frame.columns:
			del self.frame['best']
		self._cached_loglike_best = -numpy.inf

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
		Dict or float or array
			The log likelihood or the probability.  Other arrays are also included if `persist` is set to True.

		"""
		self.__prepare_for_compute(x, allow_missing_ch=probability_only)
		if self.is_mnl():
			from .mnl import mnl_d_log_likelihood_from_dataframes_all_rows
			y = mnl_d_log_likelihood_from_dataframes_all_rows(
				self._dataframes,
				num_threads=self.n_threads,
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
		else:
			from .nl import nl_d_log_likelihood_from_dataframes_all_rows
			y = nl_d_log_likelihood_from_dataframes_all_rows(
				self._dataframes,
				self,
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
			self.__check_if_best(y.ll)
		if probability_only:
			return y.probability
		if persist:
			return y
		return y.ll

	def d_loglike(self, x=None, *, start_case=0, stop_case=-1, step_case=1, leave_out=-1, keep_only=-1, subsample=-1,):
		"""
		Compute the first derivative of log likelihood with respect to the parameters.

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
		leave_out, keep_only, subsample : int, optional
			Settings for cross validation calculations.
			If `leave_out` and `subsample` are set, then case rows where rownumber % subsample == leave_out are dropped.
			If `keep_only` and `subsample` are set, then only case rows where rownumber % subsample == keep_only are used.

		Returns
		-------
		pandas.Series
			First derivatives of log likelihood with respect to the parameters (given as the index for the Series).

		"""
		return self.loglike2(x,start_case=start_case,stop_case=stop_case,step_case=step_case,
							 leave_out=leave_out, keep_only=keep_only, subsample=subsample,).dll

	def d2_loglike(self, x=None, *, start_case=0, stop_case=-1, step_case=1, leave_out=-1, keep_only=-1, subsample=-1,):
		"""
		Compute the (approximate) second derivative of log likelihood with respect to the parameters.

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
		leave_out, keep_only, subsample : int, optional
			Settings for cross validation calculations.
			If `leave_out` and `subsample` are set, then case rows where rownumber % subsample == leave_out are dropped.
			If `keep_only` and `subsample` are set, then only case rows where rownumber % subsample == keep_only are used.

		Returns
		-------
		pandas.DataFrame
			First derivatives of log likelihood with respect to the parameters (given as both the columns and
			index for the DataFrame).

		"""
		return self.loglike3(x,start_case=start_case,stop_case=stop_case,step_case=step_case,
							 leave_out=leave_out, keep_only=keep_only, subsample=subsample,).d2ll

	def check_d_loglike(self, stylize=True, skip_zeros=False):
		"""
		Check that the analytic and finite-difference gradients are approximately equal.

		Primarily used for debugging.

		Parameters
		----------
		stylize : bool, default True
			See :ref:`check_gradient` for details.
		skip_zeros : bool, default False
			See :ref:`check_gradient` for details.

		Returns
		-------
		pandas.DataFrame or Stylized DataFrame
		"""
		from ..math.optimize import check_gradient
		IF DOUBLE_PRECISION:
			epsilon=numpy.sqrt(numpy.finfo(float).eps)
		ELSE:
			epsilon=0.0001
		return check_gradient(self.loglike, self.d_loglike, self.pvals.copy(), names=self.pnames, stylize=stylize, skip_zeros=skip_zeros, epsilon=epsilon)

	def logsums(self, x=None, arr=None):
		self.unmangle()
		if x is not None:
			self.set_values(x)
		from .mnl import mnl_logsums_from_dataframes_all_rows
		if arr is None:
			arr = numpy.zeros([self._dataframes._n_cases()], dtype=l4_float_dtype)
		cdef l4_float_t logsum_parameter = 1
		if self._logsum_parameter is not None:
			logsum_parameter = self.get_value(self._logsum_parameter)
		mnl_logsums_from_dataframes_all_rows(self._dataframes, logsum_parameter, arr)
		return arr

	def exputility(self, x=None, return_dataframe=None):
		arr = self.loglike(persist=True).exp_utility
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
		return numpy.log(self.exputility(x=x, return_dataframe=return_dataframe))

	def probability(self, x=None, start_case=0, stop_case=-1, step_case=1, return_dataframe=False,):
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
		return_dataframe : {'names', True, False, 'idce'}, default False
			Format for the results.  If True, a pandas.DataFrame is returned, with case indexes and alternative
			code columns.  If 'names', the alternative names are used for the columns.
			If set to False, the results are returned as a numpy array.
			If 'idce', the resulting dataframe is unstacked and unavailable alternatives are removed.

		Returns
		-------
		array or DataFrame

		"""
		try:
			arr = self.loglike(x=x, persist=True, start_case=start_case, stop_case=stop_case, step_case=step_case, probability_only=True)
			if return_dataframe == 'names':
				return pandas.DataFrame(
					data=arr,
					columns=self._dataframes._alternative_names,
					index=self._dataframes._data_co.index if self._dataframes._data_co is not None else None,
				)
			elif return_dataframe:
				result = pandas.DataFrame(
					data=arr,
					columns=self._dataframes._alternative_codes,
					index=self._dataframes._data_co.index if self._dataframes._data_co is not None else None,
				)
				if return_dataframe == 'idce':
					return result.stack()[self._dataframes._data_av.stack().astype(bool).values]
				else:
					return result
			else:
				return arr
		except:
			logger.exception('error in probability')
			raise


	def required_data(self):
		"""
		What data is required in DataFrames for this model to be used.

		Returns
		-------
		Dict
		"""
		try:
			from ..util import Dict
			req_data = Dict()

			if self._utility_ca_function is not None and len(self._utility_ca_function):
				if 'ca' not in req_data:
					req_data.ca = set()
				for i in self._utility_ca_function:
					req_data.ca.add(str(i.data))

			if self._quantity_ca_function is not None and len(self._quantity_ca_function):
				if 'ca' not in req_data:
					req_data.ca = set()
				for i in self._quantity_ca_function:
					req_data.ca.add(str(i.data))

			if self._utility_co_functions is not None and len(self._utility_co_functions):
				if 'co' not in req_data:
					req_data.co = set()
				for alt, func in self._utility_co_functions.items():
					for i in func:
						if str(i.data)!= '1':
							req_data.co.add(str(i.data))

			if 'ca' in req_data:
				req_data.ca = list(sorted(req_data.ca))
			if 'co' in req_data:
				req_data.co = list(sorted(req_data.co))

			if self.choice_ca_var:
				req_data.choice_ca = self.choice_ca_var
			elif self.choice_co_vars:
				req_data.choice_co = self.choice_co_vars
			elif self.choice_co_code:
				req_data.choice_co_code = self.choice_co_code

			if self.weight_co_var:
				req_data.weight_co = self.weight_co_var

			if self.availability_var:
				req_data.avail_ca = self.availability_var
			elif self.availability_co_vars:
				req_data.avail_co = self.availability_co_vars

			return req_data
		except:
			logger.exception("error in required_data")


	@property
	def choice_ca_var(self):
		#self.unmangle()
		return self._choice_ca_var

	@choice_ca_var.setter
	def choice_ca_var(self, x):
		#self.mangle()
		self._choice_ca_var = x

	@property
	def choice_co_vars(self):
		#self.unmangle()
		if self._choice_co_vars:
			return self._choice_co_vars
		else:
			return None

	@choice_co_vars.setter
	def choice_co_vars(self, x):
		#self.mangle()
		if isinstance(x, dict):
			self._choice_co_vars = x
		elif x is None:
			self._choice_co_vars = x
		else:
			raise TypeError('choice_co_vars must be a dictionary')

	@choice_co_vars.deleter
	def choice_co_vars(self):
		self._choice_co_vars = None

	@property
	def choice_co_code(self):
		#self.unmangle()
		if self._choice_co_code:
			return self._choice_co_code
		else:
			return None

	@choice_co_code.setter
	def choice_co_code(self, x):
		#self.mangle()
		if isinstance(x, str):
			self._choice_co_code = x
		elif x is None:
			self._choice_co_code = x
		else:
			raise TypeError('choice_co_vars must be a str')

	@choice_co_code.deleter
	def choice_co_code(self):
		self._choice_co_code = None


	@property
	def weight_co_var(self):
		#self.unmangle()
		return self._weight_co_var

	@weight_co_var.setter
	def weight_co_var(self, x):
		#self.mangle()
		self._weight_co_var = x

	@property
	def availability_var(self):
		#self.unmangle()
		return self._availability_var

	@availability_var.setter
	def availability_var(self, x):
		#self.mangle()
		self._availability_var = x

	@property
	def availability_co_vars(self):
		#self.unmangle()
		return self._availability_co_vars

	@availability_co_vars.setter
	def availability_co_vars(self, x):
		#self.mangle()
		self._availability_co_vars = x


	def maximize_loglike(
			self,
			method='bhhh',
			quiet=False,
			screen_update_throttle=2,
			final_screen_update=True,
			check_for_overspecification=True,
			return_tags=False,
			reuse_tags=None,
			iteration_number=0,
			iteration_number_tail="",
			options=None,
			maxiter=None,
			jumpstart=0,
			jumpstart_split=5,
			leave_out=-1,
			keep_only=-1,
			subsample=-1,
			**kwargs,
	):
		from ..util.timesize import Timer
		from scipy.optimize import minimize
		from .. import _doctest_mode_
		from ..util.rate_limiter import NonBlockingRateLimiter
		from ..util.display import display_head, display_p

		if self.dataframes is None:
			raise ValueError("you must load data first -- try Model.load_data()")

		if _doctest_mode_:
			from ..model import Model
			if type(self) == Model:
				self.unmangle()
				self.frame.sort_index(inplace=True)
				self.unmangle(True)

		if options is None:
			options = {}
		if maxiter is not None:
			options['maxiter'] = maxiter

		timer = Timer()
		if isinstance(screen_update_throttle, NonBlockingRateLimiter):
			throttle_gate = screen_update_throttle
		else:
			throttle_gate = NonBlockingRateLimiter(screen_update_throttle)

		if throttle_gate and not quiet and not _doctest_mode_:
			if reuse_tags is None:
				tag1 = display_head(f'Iteration 000 {iteration_number_tail}', level=3)
				tag2 = display_p(f'LL = {self.loglike()}')
				tag3 = display_p('...')
			else:
				tag1, tag2, tag3 = reuse_tags
		else:
			tag1 = lambda *ax, **kx: None
			tag2 = lambda *ax, **kx: None
			tag3 = lambda *ax, **kx: None

		def callback(x):
			nonlocal iteration_number, throttle_gate
			iteration_number += 1
			if throttle_gate:
				#clear_output(wait=True)
				tag1.update(f'Iteration {iteration_number:03} {iteration_number_tail}')
				tag2.update(f'LL = {self.loglike()}')
				tag3.update(self.pf)

		if quiet or _doctest_mode_:
			callback = None

		try:
			if method=='bhhh':
				max_iter = options.get('maxiter',100)
				stopping_tol = options.get('ctol',1e-5)

				current_ll, tolerance, iter_bhhh, steps_bhhh, message = self.simple_fit_bhhh(
					ctol=stopping_tol,
					maxiter=max_iter,
					callback=callback,
					jumpstart=jumpstart,
					jumpstart_split=jumpstart_split,
					leave_out=leave_out,
					keep_only=keep_only,
					subsample=subsample,
				)
				raw_result = {
					'loglike':current_ll,
					'x': self.pvals,
					'tolerance':tolerance,
					'steps':steps_bhhh,
					'message':message,
				}
			else:
				raw_result = minimize(
					self.neg_loglike2,
					self.pvals,
					args=(0, -1, 1, leave_out, keep_only, subsample), # start_case, stop_case, step_case, leave_out, keep_only, subsample
					method=method,
					jac=True,
					#bounds=self.pbounds(),
					#hess=self.bhhh,
					callback=callback,
					options=options,
					**kwargs
				)
		except:
			tag1.update(f'Iteration {iteration_number:03} [Exception] {iteration_number_tail}', force=True)
			tag3.update(self.pf, force=True)
			raise

		timer.stop()

		if final_screen_update and not quiet and not _doctest_mode_:
			tag1.update(f'Iteration {iteration_number:03} [Converged] {iteration_number_tail}', force=True)
			tag2.update(f'LL = {self.loglike()}', force=True)
			tag3.update(self.pf, force=True)

		#if check_for_overspecification:
		#	self.check_for_possible_overspecification()

		from ..util import Dict
		result = Dict()
		for k,v in raw_result.items():
			if k == 'fun':
				result['loglike'] = -v
			elif k == 'jac':
				result['d_loglike'] = pandas.Series(-v, index=self.pnames)
			elif k == 'x':
				result['x'] = pandas.Series(v, index=self.pnames)
			else:
				result[k] = v
		result['elapsed_time'] = timer.elapsed()
		result['method'] = method
		result['n_cases'] = self._dataframes._n_cases()
		result['iteration_number'] = iteration_number

		if 'loglike' in result:
			result['logloss'] = -result['loglike'] / self._dataframes._n_cases()

		if _doctest_mode_:
			result['__verbose_repr__'] = True
		if return_tags:
			return result, tag1, tag2, tag3

		# try:
		# 	parent_model = self._original_parallel_model
		# except AttributeError:
		# 	pass
		# else:
		# 	parent_model._parallel_model_results[self.title] = result['loglike']

		self._most_recent_estimation_result = result

		return result

	def cross_validate(self, cv=5, *args, **kwargs):
		"""
		A simple but well optimized cross-validated log likelihood.

		This method assumes that cases are already ordered randomly.
		It is optimized to avoid shuffling or copying the source data.  As such, it is not
		directly compatible with the cross-validation tools in scikit-learn.  If the larch
		discrete choice model is stacked or ensembled with other machine learning tools,
		the cross-validation tools in scikit-learn are preferred, even though they are potentially not
		as memory efficient.

		Parameters
		----------
		cv : int
			The number of folds in k-fold cross-validation.

		Returns
		-------
		float
			The log likelihood as computed from the holdout folds.
		"""
		ll_cv = 0
		for fold in range(cv):
			i = self.maximize_loglike(leave_out=fold, subsample=cv, quiet=True, *args, **kwargs)
			ll_cv += self.loglike(keep_only=fold, subsample=cv)
			self.frame[f'cv_{fold:03d}'] = self.frame['value']
		return ll_cv

	def noop(self):
		print("No op!")

	def loglike_null(self):
		if self._cached_loglike_null != 0:
			return self._cached_loglike_null
		else:
			current_parameters = self.get_values()
			self.set_values('null')
			self._cached_loglike_null = self.loglike()
			self.set_values(current_parameters)
			return self._cached_loglike_null

	def loglike_constants_only(self):
		raise NotImplementedError
		# try:
		# 	return self._cached_loglike_constants_only
		# except AttributeError:
		# 	mm = self.constants_only_model()
		# 	from ..warning import ignore_warnings
		# 	with ignore_warnings():
		# 		result = mm.maximize_loglike(quiet=True, final_screen_update=False, check_for_overspecification=False)
		# 	self._cached_loglike_constants_only = result.loglike
		# 	return self._cached_loglike_constants_only

	def estimation_statistics(self, recompute_loglike_null=True):

		from xmle import Elem
		div = Elem('div')
		table = div.put('table')

		thead = table.put('thead')
		tr = thead.put('tr')
		tr.put('th', text='Statistic')
		tr.put('th', text='Aggregate')
		tr.put('th', text='Per Case')


		tbody = table.put('tbody')

		tr = thead.put('tr')
		tr.put('td', text='Number of Cases')
		tr.put('td', text=str(self._dataframes._n_cases()), colspan='2')

		mostrecent = self._most_recent_estimation_result
		if mostrecent is not None:
			tr = thead.put('tr')
			tr.put('td', text='Log Likelihood at Convergence')
			tr.put('td', text="{:.2f}".format(mostrecent.loglike))
			tr.put('td', text="{:.2f}".format(mostrecent.loglike / self._dataframes._n_cases()))

		ll_z = self._cached_loglike_null
		if ll_z == 0:
			if recompute_loglike_null:
				ll_z = self.loglike_null()
				self.loglike()
			else:
				ll_z = 0
		if ll_z != 0:
			tr = thead.put('tr')
			tr.put('td', text='Log Likelihood at Null Parameters')
			tr.put('td', text="{:.2f}".format(ll_z))
			tr.put('td', text="{:.2f}".format(ll_z / self._dataframes._n_cases()))
			if mostrecent is not None:
				tr = thead.put('tr')
				tr.put('td', text='Rho Squared w.r.t. Null Parameters')
				rsz = 1.0 - (mostrecent.loglike / ll_z)
				tr.put('td', text="{:.3f}".format(rsz), colspan='2')

		ll_c = self._cached_loglike_constants_only
		if ll_c != 0:
			tr = thead.put('tr')
			tr.put('td', text='Log Likelihood at Constants Only')
			tr.put('td', text="{:.2f}".format(ll_c))
			tr.put('td', text="{:.2f}".format(ll_c / self._dataframes._n_cases()))
			if mostrecent is not None:
				tr = thead.put('tr')
				tr.put('td', text='Rho Squared w.r.t. Constants Only')
				rsc = 1.0 - (mostrecent.loglike / ll_c)
				tr.put('td', text="{:.3f}".format(rsc), colspan='2')

		# for parallel_title, parallel_loglike in self._parallel_model_results.items():
		# 	tr = thead.put('tr')
		# 	tr.put('td', text=f'Log Likelihood {parallel_title}')
		# 	tr.put('td', text="{:.2f}".format(parallel_loglike))
		# 	tr.put('td', text="{:.2f}".format(parallel_loglike / self._dataframes._n_cases()))
		# 	if mostrecent is not None:
		# 		tr = thead.put('tr')
		# 		tr.put('td', text=f'Rho Squared w.r.t. {parallel_title}')
		# 		rsc = 1.0 - (mostrecent.loglike / parallel_loglike)
		# 		tr.put('td', text="{:.3f}".format(rsc), colspan='2')

		return div

	@property
	def most_recent_estimation_result(self):
		return self._most_recent_estimation_result


	def _free_slots_inverse_matrix(self, matrix):
		try:
			take = numpy.full_like(matrix, True, dtype=bool)
			dense_s = len(self.pvals)
			for i in range(dense_s):
				ii = self.pnames[i]
				#if self.pf.loc[ii, 'holdfast'] or (self.pf.loc[ii, 'value'] >= self.pf.loc[ii, 'maximum']) or (self.pf.loc[ii, 'value'] <= self.pf.loc[ii, 'minimum']):
				if self.pf.loc[ii, 'holdfast']:
					take[i, :] = False
					take[:, i] = False
					dense_s -= 1
			hess_taken = matrix[take].reshape(dense_s, dense_s)
			from ..linalg import general_inverse
			try:
				invhess = general_inverse(hess_taken)
			except numpy.linalg.linalg.LinAlgError:
				invhess = numpy.full_like(hess_taken, numpy.nan, dtype=numpy.float64)
			result = numpy.full_like(matrix, 0, dtype=numpy.float64)
			result[take] = invhess.reshape(-1)
			return result
		except:
			logger.exception("error in Model5c._free_slots_inverse_matrix")
			raise

	def calculate_parameter_covariance(self, status_widget=None, preserve_hessian=False):
		hess = -self.d2_loglike()

		from ..model.possible_overspec import compute_possible_overspecification, PossibleOverspecification
		overspec = compute_possible_overspecification(hess, self.pf.loc[:,'holdfast'])
		if overspec:
			import warnings
			warnings.warn("WARNING: Model is possibly over-specified (hessian is nearly singular).", category=PossibleOverspecification)
			possible_overspecification = []
			for eigval, ox, eigenvec in overspec:
				if eigval == 'LinAlgError':
					possible_overspecification.append((eigval, [ox, ], ["", ]))
				else:
					paramset = list(numpy.asarray(self.pf.index)[ox])
					possible_overspecification.append((eigval, paramset, eigenvec[ox]))
			self._possible_overspecification = possible_overspecification

		take = numpy.full_like(hess, True, dtype=bool)
		dense_s = len(self.pvals)
		for i in range(dense_s):
			ii = self.pnames[i]
			if self.pf.loc[ii, 'holdfast'] or (self.pf.loc[ii, 'value'] >= self.pf.loc[ii, 'maximum']) or (
				self.pf.loc[ii, 'value'] <= self.pf.loc[ii, 'minimum']):
				take[i, :] = False
				take[:, i] = False
				dense_s -= 1
		if dense_s > 0:
			hess_taken = hess[take].reshape(dense_s, dense_s)
			from ..linalg import general_inverse
			try:
				invhess = general_inverse(hess_taken)
			except numpy.linalg.linalg.LinAlgError:
				invhess = numpy.full_like(hess_taken, numpy.nan, dtype=numpy.float64)
			covariance_matrix = numpy.full_like(hess, 0, dtype=numpy.float64)
			covariance_matrix[take] = invhess.reshape(-1)
			# robust...
			try:
				bhhh_taken = self.bhhh()[take].reshape(dense_s, dense_s)
			except NotImplementedError:
				pass
			else:
				# import scipy.linalg.blas
				# temp_b_times_h = scipy.linalg.blas.dsymm(float(1), invhess, bhhh_taken)
				# robusto = scipy.linalg.blas.dsymm(float(1), invhess, temp_b_times_h, side=1)
				robusto = numpy.dot(numpy.dot(invhess, bhhh_taken), invhess)
				robust_covariance_matrix = numpy.full_like(hess, 0, dtype=numpy.float64)
				robust_covariance_matrix[take] = robusto.reshape(-1)
		else:
			covariance_matrix = numpy.full_like(hess, 0, dtype=numpy.float64)
			robust_covariance_matrix = numpy.full_like(hess, 0, dtype=numpy.float64)
		self.pf['std err'] = numpy.sqrt(covariance_matrix.diagonal())
		self.pf['t stat'] = (self.pf['value'] - self.pf['nullvalue']) / self.pf['std err']
		self.pf['robust std err'] = numpy.sqrt(robust_covariance_matrix.diagonal())
		self.pf['robust t stat'] = (self.pf['value'] - self.pf['nullvalue']) / self.pf['robust std err']

		if preserve_hessian:
			self.hessian_matrix = hess

		self.covariance_matrix = covariance_matrix
		self.robust_covariance_matrix = robust_covariance_matrix

	@property
	def possible_overspecification(self):
		return self._possible_overspecification

	@property
	def ordering(self):
		disp_order = self._display_order if self._display_order is not None else ()
		return disp_order + self.ordering_tail

	@ordering.setter
	def ordering(self, x):
		tail = set(self.ordering_tail)
		self._display_order = tuple(y for y in x if y not in tail)

	@property
	def ordering_tail(self):
		return self._display_order_tail if self._display_order_tail is not None else ()

	@ordering_tail.setter
	def ordering_tail(self, x):
		self._display_order_tail = tuple(x)

	def pfo(self):
		if self.ordering is None:
			return self.pf
		paramset = set(self.pf.index)
		out = []
		import re
		if self.ordering:
			for category in self.ordering:
				category_name = category[0]
				category_params = []
				for category_pattern in category[1:]:
					category_params.extend(sorted(i for i in paramset if re.match(category_pattern, i) is not None))
					paramset -= set(category_params)
				out.append( [category_name, category_params] )
		if len(paramset):
			out.append( ['Other', sorted(paramset)] )

		tuples = []
		for c,pp in out:
			for p in pp:
				tuples.append((c,p))

		ix = pandas.MultiIndex.from_tuples(tuples, names=['Category','Parameter'])

		f = self.pf
		f = f.reindex(ix.get_level_values(1))
		f.index = ix
		return f




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

	def initialize_graph(self, dataframes=None, alternative_codes=None):
		"""
		Write a nesting tree graph for a MNL model.

		Parameters
		----------
		dataframes : DataFrames, optional
			Use this to determine the included alternatives.
		alternative_codes : array-like, optional
			Explicitly give alternative codes. Ignored if `dataframes` is given
			or if the model has dataframes or a dataservice already set.

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

		if alternative_codes is None:
			if self._dataframes is None and self._dataservice is not None:
				alternative_codes = self._dataservice.alternative_codes()
			else:
				alternative_codes = self._dataframes._alternative_codes

		from .tree import NestingTree
		g = NestingTree()
		for a in alternative_codes:
			g.add_node(a)

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



	###########
	# Scoring
	#
	# These scoring mechanisms are popular in machine learning applications.

	def top_k_accuracy(self, x=None, *, k=1):
		"""
		Compute top-K accuracy for the model.

		Parameters
		----------
		k : int or iterable of ints
			The ranking to compute.

		Returns
		-------
		float
			Fraction of cases where the chosen alternative is ranked at k or better.

		Notes
		-----
		Top
		"""
		from .scoring import top_k_accuracy, probability_to_rank
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
		return -self.loglike(x=x) / self._dataframes._n_cases() / self._dataframes._weight_normalization

