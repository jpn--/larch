# cython: language_level=3, embedsignature=True

include "../general_precision.pxi"
from ..general_precision import l4_float_dtype

import sys
import numpy
import pandas

from ..roles import DictOfStrings
from .linear import ParameterRef_C, LinearComponent_C, LinearFunction_C
from .linear_math import ParameterOp

import logging
from ..log import logger_name
logger = logging.getLogger(logger_name+'.model')

NBSP = " " # non=breaking space

def _empty_parameter_frame(names, nullvalue=0, initvalue=0, max=None, min=None):

	cdef int len_names = 0 if names is None else len(names)

	min_ = min if min is not None else -numpy.inf
	max_ = max if max is not None else numpy.inf

	data=dict(
			value = numpy.full(len_names, fill_value=initvalue, dtype=l4_float_dtype),
			minimum = numpy.full(len_names, fill_value=min_, dtype=l4_float_dtype),
			maximum = numpy.full(len_names, fill_value=max_, dtype=l4_float_dtype),
			nullvalue = numpy.full(len_names, fill_value=nullvalue, dtype=l4_float_dtype),
			initvalue = numpy.full(len_names, fill_value=initvalue, dtype=l4_float_dtype),
			holdfast = numpy.zeros(len_names, dtype=numpy.int8),
			note = numpy.zeros(len_names, dtype='<U128')
		)
	ix = names if names is not None else []
	columns=['value', 'initvalue', 'nullvalue', 'minimum', 'maximum', 'holdfast', 'note']
	try:
		r = pandas.DataFrame(
			index=ix,
			data=data,
			columns=columns
		)
	except Exception as err:
		print(err, file=sys.stderr)
		raise

	return r.copy()


from ..exceptions import ParameterNotInModelWarning


cdef class ParameterFrame:

	def __init__(
			self, *,
			parameters=None,
			frame=None,
			title=None,
	):
		self._title = title

		if frame is not None:
			self._frame = frame
		else:
			fr = _empty_parameter_frame(parameters or [])
			self._frame = fr

		self._prior_frame_values = self._frame['value'].copy()
		self._mangled = True
		self._display_order = None
		self._display_order_tail = None
		self._matrixes = dict()

	def _check_if_frame_values_changed(self):
		"""
		Check if frame values have changed since the last time this was called.

		Triggers a call to `frame_values_have_changed` if there is a change to the
		frame values.
		"""
		try:
			if self._prior_frame_values is None or (
				not (
						numpy.array_equal(self._prior_frame_values.values, self._frame['value'].values)
					and numpy.array_equal(self._prior_frame_values.index, self._frame['value'].index)
				)
			):
				self._frame_values_have_changed()
				self._prior_frame_values = self._frame['value'].copy()
		except:
			logger.exception('error in _check_if_frame_values_changed')
			raise

	def _frame_values_have_changed(self):
		raise NotImplementedError("abstract base class, use a derived class instead")

	def _ensure_names(self, names, **kwargs):
		existing_names = set(self._frame.index)
		nameset = set(names)
		missing_names = nameset - existing_names
		if missing_names:
			self._frame = self._frame.append(
				_empty_parameter_frame([n for n in names if (n in missing_names)], **kwargs),
				verify_integrity=True,
				sort=False
			)

	def _scan_all_ensure_names(self):
		self._frame.sort_index(inplace=True)

	@property
	def title(self):
		if self._title is None:
			return "Untitled"
		else:
			return self._title

	@title.setter
	def title(self, value):
		self._title = str(value)

	def __getstate__(self):
		import cloudpickle
		import gzip
		import base64
		state = dict()
		state["_frame                         ".strip()] = (self._frame                         )
		state["_display_order                 ".strip()] = (self._display_order                 )
		state["_display_order_tail            ".strip()] = (self._display_order_tail            )
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
		(self._frame                         ) = state["_frame                         ".strip()]
		(self._display_order                 ) = state["_display_order                 ".strip()]
		(self._display_order_tail            ) = state["_display_order_tail            ".strip()]
		(self._title                         ) = state["_title                         ".strip()]
		(self._matrixes                      ) = state["_matrixes                      ".strip()]
		self._prior_frame_values = None
		self.unmangle(True)

	def mangle(self, *args, **kwargs):
		self._mangled = True

	def _is_mangled(self):
		return self._mangled

	def unmangle(self, force=False):
		if self._mangled or force:
			self._scan_all_ensure_names()
			self._mangled = False

	def set_frame(self, frame):
		"""
		Assign a new parameter frame.

		If the frame to be set evaluates as equal to the existing
		frame, this method will not change the existing frame and will not
		trigger a `mangle`.

		Parameters
		----------
		frame : pandas.DataFrame
		"""
		if not isinstance(frame, pandas.DataFrame):
			raise ValueError(f'frame must be pandas.DataFrame, not {type(frame)}')
		if not self._frame.equals(frame):
			self._frame = frame
			self.mangle()

	@property
	def pf(self):
		"""
		pandas.DataFrame : The parameter frame, unmangling on access.
		"""
		self.unmangle()
		return self._frame

	def pf_sort(self):
		"""
		Sort (on index, i.e. parameter name) and return the parameter frame.

		Returns
		-------
		pandas.DataFrame
		"""
		self.unmangle()
		self._frame.sort_index(inplace=True)
		self.unmangle(True)
		return self._frame

	@property
	def pvals(self):
		"""ndarray : A copy of the current value of the parameters."""
		self.unmangle()
		return self._frame['value'].values.copy()

	@property
	def pbounds(self):
		"""scipy.optimize.Bounds : A copy of the current min-max bounds of the parameters."""
		self.unmangle()
		from scipy.optimize import Bounds
		return Bounds(
			self._frame['minimum'].values.copy(),
			self._frame['maximum'].values.copy(),
		)

	@property
	def pnames(self):
		"""ndarray : A copy of the current names of the parameters."""
		self.unmangle()
		return self._frame.index.values.copy()

	def pvalue(self, parameter_name, apply_formatting=False, default_value=None, log_errors=True):
		"""
		Get the value of a parameter or a parameter expression.

		Parameters
		----------
		parameter_name : str or ParameterOp
			The named parameter, or parameter operation, to evaluate.
		apply_formatting : bool, default False
			Whether to string-format the result.
		default_value : numeric, optional
			A default value to return if the evaluation fails.

		Returns
		-------
		numeric or str
			Returns a numeric value (when `apply_formatting` is False) or
			a formatted string.

		Raises
		------
		KeyError
			When the named parameter, or some named part of a ParameterOp,
			is not found in the parameter frame.
		"""
		from ..roles import _param_math_binaryop
		from .linear_math import _ParameterOp
		from numbers import Number
		try:
			if isinstance(parameter_name, (_ParameterOp, ParameterRef_C)):
				if apply_formatting:
					return parameter_name.string(self)
				else:
					return parameter_name.value(self)
			elif hasattr(parameter_name, 'as_pmath'):
				if apply_formatting:
					return parameter_name.as_pmath().string(self)
				else:
					return parameter_name.as_pmath().value(self)
			elif isinstance(parameter_name, _param_math_binaryop):
				if apply_formatting:
					return parameter_name.strf(self)
				else:
					return parameter_name.value(self)
			elif isinstance(parameter_name, dict):
				result = type(parameter_name)()
				for k,v in parameter_name.items():
					result[k] = self.pvalue(v, apply_formatting=apply_formatting, default_value=default_value)
				return result
			elif isinstance(parameter_name, (set, tuple, list)):
				result = dict()
				for k in parameter_name:
					result[k] = self.pvalue(k, apply_formatting=apply_formatting, default_value=default_value)
				return result
			elif isinstance(parameter_name, Number):
				if apply_formatting:
					return str(parameter_name)
				else:
					return parameter_name
			else:
				if apply_formatting:
					return str(self.pf.loc[parameter_name,'value'])
				else:
					return self.pf.loc[parameter_name,'value']
		except KeyError:
			if default_value is not None:
				return default_value
			if log_errors:
				logger.exception("error in ParameterFrame.pvalue")
			raise

	def pformat(self, parameter_name, apply_formatting=True, default_value='NA'):
		"""
		Get the value of a parameter or a parameter expression.

		Parameters
		----------
		parameter_name : str or ParameterOp
			The named parameter, or parameter operation, to evaluate.
		apply_formatting : bool, default True
			Whether to string-format the result.
		default_value : numeric or str, default 'NA'
			A default value to return if the evaluation fails.

		Returns
		-------
		numeric or str
			Returns a numeric value (when `apply_formatting` is False) or
			a formatted string.

		Raises
		------
		KeyError
			When the named parameter, or some named part of a ParameterOp,
			is not found in the parameter frame.
		"""
		return self.pvalue(
			parameter_name,
			apply_formatting=apply_formatting,
			default_value=default_value,
		)

	def set_value(self, name, value=None, **kwargs):
		"""
		Set the value for a single model parameter.

		This function will set the current value of a parameter.
		Unless explicitly instructed with an alternate value,
		the new value will also be saved as the "initial" value
		of the parameter.

		Parameters
		----------
		name : str
			The name of the parameter to set to a fixed value.
		value : float
			The numerical value to set for the parameter.
		initvalue : float, optional
			If given, this value is used to indicate the initial value
			for the parameter, which may be different from the
			current value.
		nullvalue : float, optional
			If given, this will overwrite any existing null value for
			the parameter.  If not given, the null value for the
			parameter is not changed.

		"""
		if isinstance(name, ParameterRef_C):
			name = str(name)
		if name not in self._frame.index:
			self.unmangle()
		if value is not None:
			# value = numpy.float64(value)
			# self._frame.loc[name,'value'] = value
			kwargs['value'] = value
		for k,v in kwargs.items():
			if k in self._frame.columns:
				if self._frame.dtypes[k] == 'float64':
					v = numpy.float64(v)
				elif self._frame.dtypes[k] == 'float32':
					v = numpy.float32(v)
				elif self._frame.dtypes[k] == 'int8':
					v = numpy.int8(v)
				self._frame.loc[name, k] = v

			# update init values when they are implied
			if k=='value':
				if 'initvalue' not in kwargs and pandas.isnull(self._frame.loc[name, 'initvalue']):
					self._frame.loc[name, 'initvalue'] = l4_float_dtype(v)

		# update null values when they are implied
		if 'nullvalue' not in kwargs and pandas.isnull(self._frame.loc[name, 'nullvalue']):
			self._frame.loc[name, 'nullvalue'] = 0

		self._check_if_frame_values_changed()

	def lock_value(self, name, value, note=None, change_check=True):
		"""
		Set a fixed value for a model parameter.

		Parameters with a fixed value (i.e., with "holdfast" set to 1)
		will not be changed during estimation by the likelihood
		maximization algorithm.

		Parameters
		----------
		name : str
			The name of the parameter to set to a fixed value.
		value : float
			The numerical value to set for the parameter.
		note : str, optional
			A note as to why this parameter is set to a fixed value.
			This will not affect the mathematical treatment of the
			parameter in any way, but may be useful for reporting.
		change_check : bool, default True
			Whether to trigger a check to see if any parameter frame
			values have changed.  Can be set to false to skip this
			check if you know that the values have not changed or want
			to delay this check for later, but this may result in
			problems if the check is needed but not triggered before
			certain other modeling tasks are performed.

		"""
		if isinstance(name, ParameterRef_C):
			name = str(name)
		if value is 'null':
			value = self.pf.loc[name, 'nullvalue']
		self.set_value(name, value, holdfast=1, initvalue=value, nullvalue=value, minimum=value, maximum=value)
		if note is not None:
			self._frame.loc[name, 'note'] = note
		if change_check:
			self._check_if_frame_values_changed()

	def lock_values(self, *names, note=None, **kwargs):
		"""
		Set a fixed value for one or more model parameters.

		Positional arguments give the names of parameters to fix at the current value.
		Keyword arguments name the parameter and give a value to set as fixed.

		This is a convenience method to efficiently lock
		multiple parameters simultaneously.

		Other Parameters
		----------------
		note : str, optional
			Add a note to all parameters fixed with this comment.  The same note is added to
			all the parameters.
		"""
		for name in names:
			value = self.get_value(name)
			self.lock_value(name, value, note=note, change_check=False)
		for name, value in kwargs.items():
			self.lock_value(name, value, note=note, change_check=False)
		self._check_if_frame_values_changed()

	def get_value(self, name, *, default=None):
		if name is None and default is not None:
			return default
		if isinstance(name, dict):
			return {k:self.get_value(v) for k,v in name.items()}
		try:
			if isinstance(name, (ParameterRef_C, ParameterOp)):
				return name.value(self)
			if isinstance(name, (LinearComponent_C,LinearFunction_C)):
				return name.as_pmath().value(self)
			return self._frame.loc[name,'value']
		except KeyError:
			if default is not None:
				return default
			else:
				raise

	def get_holdfast(self, name, *, default=None):
		if name is None and default is not None:
			return default
		try:
			return self._frame.loc[name,'holdfast']
		except KeyError:
			if default is not None:
				return default
			else:
				raise

	def get_slot_x(self, name, holdfast_invalidates=False):
		"""
		Get the position of a named parameter within the parameters index.

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
			result = self._frame.index.get_loc(name)
		except KeyError:
			return -1
		if holdfast_invalidates:
			if self._frame.loc[name,'holdfast']:
				return -1
		return result

	def __getitem__(self, name):
		try:
			return self._frame.loc[name,:]
		except:
			logger.exception("error in ParameterFrame.__getitem__")
			raise

	def set_values(self, values=None, **kwargs):
		"""
		Set the parameter values for one or more parameters.

		Parameters
		----------
		values : {'null', 'init', 'best', array-like, dict, scalar}, optional
			New values to set for the parameters.
			If 'null' or 'init', the current values are set
			equal to the null or initial values given in
			the 'nullvalue' or 'initvalue' column of the
			parameter frame, respectively.
			If 'best', the current values are set equal to
			the values given in the 'best' column of the
			parameter frame, if that columns exists,
			otherwise a ValueError exception is raised.
			If given as array-like, the array must be a
			vector with length equal to the length of the
			parameter frame, and the given vector will replace
			the current values.  If given as a dictionary,
			the dictionary is used to update `kwargs` before
			they are processed.
		kwargs : dict
			Any keyword arguments (or if `values` is a
			dictionary) are used to update the included named
			parameters only.  A warning will be given if any key of
			the dictionary is not found among the existing named
			parameters in the parameter frame, and the value
			associated with that key is ignored.  Any parameters
			not named by key in this dictionary are not changed.

		Notes
		-----
		Setting parameters both in the `values` argument and
		through keyword assignment is not explicitly disallowed,
		although it is not recommended and may be disallowed in the future.

		Raises
		------
		ValueError
			If setting to 'best' but there is no 'best' column in
			the `pf` parameters DataFrame.
		"""

		if isinstance(values, str):
			if values.lower() == 'null':
				values = self._frame.loc[:,'nullvalue'].values
			elif values.lower() == 'init':
				values = self._frame.loc[:,'initvalue'].values
			elif values.lower() == 'initial':
				values = self._frame.loc[:, 'initvalue'].values
			elif values.lower() == 'best':
				if 'best' not in self._frame.columns:
					raise ValueError('best is not stored')
				else:
					values = self._frame.loc[:, 'best'].values
		if isinstance(values, dict):
			kwargs.update(values)
			values = None
		if values is not None:
			if not isinstance(values, (int,float)):
				if len(values) != len(self._frame):
					raise ValueError(f'gave {len(values)} values, needs to be exactly {len(self._frame)} values')
			# only change parameters that are not holdfast
			free_parameters= self._frame['holdfast'].values == 0
			if isinstance(values, (int,float)):
				self._frame.loc[free_parameters, 'value'] = l4_float_dtype(values)
			else:
				self._frame.loc[free_parameters, 'value'] = numpy.asanyarray(values)[free_parameters]
		if len(kwargs):
			if self._mangled:
				self._scan_all_ensure_names()
			for k,v in kwargs.items():
				if k in self._frame.index:
					if self._frame.loc[k,'holdfast'] == 0:
						self._frame.loc[k, 'value'] = v
				else:
					import warnings
					warnings.warn(f'{k} not in model', category=ParameterNotInModelWarning)
		self._check_if_frame_values_changed()

	def get_values(self):
		return self.pf.loc[:,'value'].copy()

	def shift_values(self, shifts):
		self.set_values(self.pvals + shifts)
		return self

	def update_values(self, values):
		self.set_values(values)
		return self

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
		if self.ordering is None or self.ordering == ():
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
	def hessian_matrix(self):
		mat = self._matrixes.get('hessian_matrix', None)
		if mat is None:
			return None
		return pandas.DataFrame(
			mat, columns=self._frame.index, index=self._frame.index,
		)

	@property
	def covariance_matrix(self):
		mat = self._matrixes.get('covariance_matrix', None)
		if mat is None:
			return None
		return pandas.DataFrame(
			mat, columns=self._frame.index, index=self._frame.index,
		)

	@property
	def robust_covariance_matrix(self):
		mat = self._matrixes.get('robust_covariance_matrix', None)
		if mat is None:
			return None
		return pandas.DataFrame(
			mat, columns=self._frame.index, index=self._frame.index,
		)

	@property
	def constrained_covariance_matrix(self):
		mat = self._matrixes.get('constrained_covariance_matrix', None)
		if mat is None:
			return None
		return pandas.DataFrame(
			mat, columns=self._frame.index, index=self._frame.index,
		)

	def __parameter_table_section(self, pname):

		from xmle import Elem

		pname_str = str(pname)
		pf = self.pf
		# if pname in self.rename_parameters:
		# 	colspan = 0
		# 	if 'std_err' in pf.columns:
		# 		colspan += 1
		# 	if 't_stat' in pf.columns:
		# 		colspan += 1
		# 	if 'nullvalue' in pf.columns:
		# 		colspan += 1
		# 	return [
		# 		Elem('td', text="{:.4g}".format(pf.loc[self.rename_parameters[pname],'value'])),
		# 		Elem('td', text="= "+self.rename_parameters[pname], colspan=str(colspan)),
		# 	]
		if pf.loc[pname_str,'holdfast']:
			colspan = 0
			if 'std_err' in pf.columns:
				colspan += 1
			if 't_stat' in pf.columns:
				colspan += 1
			if 'nullvalue' in pf.columns:
				colspan += 1
			# if pf.loc[pname_str,'holdfast'] == holdfast_reasons.asc_but_never_chosen:
			# 	return [
			# 		Elem('td', text="{:.4g}".format(pf.loc[pname_str,'value'])),
			# 		Elem('td', text="fixed value, never chosen", colspan=str(colspan)),
			# 	]
			# elif pf.loc[pname_str, 'holdfast'] == holdfast_reasons.snap_to_constant_eq:
			# 	return [
			# 		Elem('td', text="{:.4g}".format(pf.loc[pname_str, 'value'])),
			# 		Elem('td', text="fixed value", colspan=str(colspan)),
			# 	]
			# elif pf.loc[pname_str, 'holdfast'] == holdfast_reasons.snap_to_constant_le:
			# 	return [
			# 		Elem('td', text="{:.4g}".format(pf.loc[pname_str, 'value'])),
			# 		Elem('td', text="≤ {:.4g}".format(pf.loc[pname_str, 'value']), colspan=str(colspan)),
			# 	]
			# elif pf.loc[pname_str, 'holdfast'] == holdfast_reasons.snap_to_constant_ge:
			# 	return [
			# 		Elem('td', text="{:.4g}".format(pf.loc[pname_str, 'value'])),
			# 		Elem('td', text="≥ {:.4g}".format(pf.loc[pname_str, 'value']), colspan=str(colspan)),
			# 	]
			# elif pf.loc[pname_str, 'note'] != "" and not pandas.isnull(pf.loc[pname_str, 'note']):
			# 	return [
			# 		Elem('td', text="{:.4g}".format(pf.loc[pname_str,'value'])),
			# 		Elem('td', text=pf.loc[pname_str, 'note'], colspan=str(colspan)),
			# 	]
			# else:
			return [
				Elem('td', text="{:.4g}".format(pf.loc[pname_str,'value'])),
				Elem('td', text="fixed value", colspan=str(colspan), style="text-align: left;"),
			]
		else:
			result = [ Elem('td', text="{:.4g}".format(pf.loc[pname_str,'value'])) ]
			if 'std_err' in pf.columns:
				result += [ Elem('td', text="{:#.3g}".format(pf.loc[pname_str, 'std_err'])), ]
			if 't_stat' in pf.columns:
				result += [ Elem('td', text="{:#.2f}".format(pf.loc[pname_str, 't_stat'])), ]
			if 'nullvalue' in pf.columns:
				result += [ Elem('td', text="{:#.2g}".format(pf.loc[pname_str, 'nullvalue'])), ]
			return result


	# def pfo(self):
	# 	if self.ordering is None:
	# 		return self.pf
	# 	paramset = set(self.pf.index)
	# 	out = []
	# 	import re
	# 	if self.ordering:
	# 		for category in self.ordering:
	# 			category_name = category[0]
	# 			category_params = []
	# 			for category_pattern in category[1:]:
	# 				category_params.extend(sorted(i for i in paramset if re.match(category_pattern, i) is not None))
	# 				paramset -= set(category_params)
	# 			out.append( [category_name, category_params] )
	# 	if len(paramset):
	# 		out.append( ['Other', sorted(paramset)] )
	#
	# 	tuples = []
	# 	for c,pp in out:
	# 		for p in pp:
	# 			tuples.append((c,p))
	#
	# 	ix = pandas.MultiIndex.from_tuples(tuples, names=['Category','Parameter'])
	#
	# 	f = self.pf
	# 	f = f.reindex(ix.get_level_values(1))
	# 	f.index = ix
	# 	return f

	def parameter_summary(self, output='df'):
		"""
		Create a tabular summary of parameter values.

		This will generate a small table of parameters statistics,
		containing:

		*	Parameter Name (and Category, if applicable)
		*	Estimated Value
		*	Standard Error of the Estimate (if known)
		*	t Statistic (if known)
		*	Null Value
		*	Binding Constraints (if applicable)

		Parameters
		----------
		output : {'df','xml'}
			The format of the output.  The default, 'df', creates a
			pandas DataFrame.  Alternatively, use 'xml' to create
			a table as a xmle.Elem (this format is no longer preferred).

		Returns
		-------
		pandas.DataFrame or xmle.Elem

		"""
		try:

			pfo = self.pfo()
			if output == 'xml':

				any_categories = not (self.ordering is None or self.ordering == ())
				if not any_categories:
					ordered_p = [("",i) for i in pfo.index]
				else:
					ordered_p = list(pfo.index)

				any_colons = False
				for rownum in range(len(ordered_p)):
					if ":" in ordered_p[rownum][1]:
						any_colons = True
						break

				from xmle import ElemTable

				div = ElemTable('div')
				table = div.put('table')

				thead = table.put('thead')
				tr = thead.put('tr')
				if any_categories:
					tr.put('th', text='Category', style="text-align: left;")
				if any_colons:
					tr.put('th', text='Parameter', colspan='2', style="text-align: left;")
				else:
					tr.put('th', text='Parameter', style="text-align: left;")

				tr.put('th', text='Value')
				if 'std_err' in pfo.columns:
					tr.put('th', text='Std Err')
				if 't_stat' in pfo.columns:
					tr.put('th', text='t Stat')
				if 'nullvalue' in pfo.columns:
					tr.put('th', text='Null Value')

				tbody = table.put('tbody')

				swallow_categories = 0
				swallow_subcategories = 0

				for rownum in range(len(ordered_p)):
					tr = tbody.put('tr')
					if not any_categories:
						pass
					elif swallow_categories > 0:
						swallow_categories -= 1
					else:
						nextrow = rownum + 1
						while nextrow<len(ordered_p) and ordered_p[nextrow][0] == ordered_p[rownum][0]:
							nextrow += 1
						swallow_categories = nextrow - rownum - 1
						if swallow_categories:
							tr.put('th', text=ordered_p[rownum][0], rowspan=str(swallow_categories+1), style="vertical-align: top; text-align: left;")
						else:
							tr.put('th', text=ordered_p[rownum][0], style="vertical-align: top; text-align: left;")
					parameter_name = ordered_p[rownum][1]
					if ":" in parameter_name:
						parameter_name = parameter_name.split(":",1)
						if swallow_subcategories > 0:
							swallow_subcategories -= 1
						else:
							nextrow = rownum + 1
							while nextrow < len(ordered_p) and ":" in ordered_p[nextrow][1] and ordered_p[nextrow][1].split(":",1)[0] == parameter_name[0]:
								nextrow += 1
							swallow_subcategories = nextrow - rownum - 1
							if swallow_subcategories:
								tr.put('th', text=parameter_name[0], rowspan=str(swallow_subcategories+1), style="vertical-align: top; text-align: left;")
							else:
								tr.put('th', text=parameter_name[0], style="vertical-align: top; text-align: left;")
						tr.put('th', text=parameter_name[1], style="vertical-align: top; text-align: left;")
					else:
						if any_colons:
							tr.put('th', text=parameter_name, colspan="2", style="vertical-align: top; text-align: left;")
						else:
							tr.put('th', text=parameter_name, style="vertical-align: top; text-align: left;")
					tr << self.__parameter_table_section(ordered_p[rownum][1])

				return div

			else:
				columns = [i for i in ['value','std_err','t_stat','nullvalue', 'constrained'] if i in pfo.columns]
				result = pfo[columns].rename(
					columns={
						'value':'Value',
						'std_err':'Std Err',
						't_stat':'t Stat',
						'nullvalue':'Null Value',
						'constrained': 'Constrained'
					}
				)
				monospace_cols = []
				if 't Stat' in result.columns:
					result.insert(result.columns.get_loc('t Stat')+1, 'Signif', "")
					result.loc[numpy.absolute(result['t Stat']) > 1.9600, 'Signif'] = "*"
					result.loc[numpy.absolute(result['t Stat']) > 2.5758, 'Signif'] = "**"
					result.loc[numpy.absolute(result['t Stat']) > 3.2905, 'Signif'] = "***"
					_fmt_t = lambda x: f"{x:0< 4.2f}".replace(" ",NBSP) if numpy.isfinite(x) else NBSP+"NA"
					result['t Stat'] = result['t Stat'].apply(_fmt_t)
					result.loc[result['t Stat'] == NBSP+"NA", 'Signif'] = ""
					monospace_cols.append('t Stat')
					monospace_cols.append('Signif')
				if 'likelihood ratio' in pfo:
					non_finite_t = ~numpy.isfinite(pfo['t_stat'])
					result.loc[numpy.absolute((numpy.isfinite(pfo['likelihood ratio']))&non_finite_t), 'Signif'] = "[]"
					result.loc[numpy.absolute(((pfo['likelihood ratio']) > 1.9207)&non_finite_t), 'Signif'] = "[*]"
					result.loc[numpy.absolute(((pfo['likelihood ratio']) > 3.3174)&non_finite_t), 'Signif'] = "[**]"
					result.loc[numpy.absolute(((pfo['likelihood ratio']) > 5.4138)&non_finite_t), 'Signif'] = "[***]"
				if 'Std Err' in result.columns:
					_fmt_s = lambda x: f"{x: #.3g}".replace(" ",NBSP) if numpy.isfinite(x) else NBSP+"NA"
					result['Std Err'] = result['Std Err'].apply(_fmt_s)
					monospace_cols.append('Std Err')
				if 'Value' in result.columns:
					result['Value'] = result['Value'].apply(lambda x: f"{x: #.3g}".replace(" ",NBSP))
					monospace_cols.append('Value')
				if 'Constrained' in result.columns:
					result['Constrained'] = result['Constrained'].str.replace("\n","<br>")
				if 'Null Value' in result.columns:
					monospace_cols.append('Null Value')
				if result.index.nlevels > 1:
					pnames = result.index.get_level_values(-1)
				else:
					pnames = result.index
				styles = [
					dict(selector="th", props=[
						("vertical-align", "top"),
						("text-align", "left"),
					]),
					dict(selector="td", props=[
						("vertical-align", "top"),
						("text-align", "left"),
					]),

				]
				return result.style.set_table_styles(styles).format({'Null Value':"{: .2f}"}).applymap(lambda x:"font-family:monospace", subset=monospace_cols)

		except Exception as err:
			logger.exception("error in parameter_summary")
			raise