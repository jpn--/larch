# cython: language_level=3

from __future__ import print_function

include "general_precision.pxi"
from .general_precision import l4_float_dtype
from .general_precision cimport l4_float_t

from libc.string cimport memset
cimport cython

import pandas
import numpy
import inspect

import logging
logger = logging.getLogger('L5')

from .model.controller cimport Model5c
from numpy.math cimport expf, logf
from libc.math cimport exp, log

cdef float INFINITY32 = numpy.float('inf')

class MissingDataError(Exception):
	pass

class DuplicateColumnNames(ValueError):
	pass

def _initialize_or_validate_shape(arr, shape, dtype):
	if arr is None:
		return numpy.zeros(shape, dtype=dtype)
	else:
		if tuple(arr.shape) != tuple(shape):
			raise ValueError(f"{arr.shape} != {shape}")
		return arr

def _check_dataframe_of_dtype(df, dtype):
	if df is None:
		return False
	if not isinstance(df, pandas.DataFrame):
		return False
	if not all(df.dtypes == dtype):
		return False
	return True


def _ensure_dataframe_of_dtype(df, dtype, label, warn_on_convert=True):
	if df is None:
		return df
	if not isinstance(df, pandas.DataFrame):
		raise TypeError(f'{label} must be a DataFrame')
	if not all(df.dtypes == dtype):
		if warn_on_convert:
			logger.warning(f'converting {label} to {dtype}')
		df = pandas.DataFrame(
			data=df.values.astype(dtype),
			columns=df.columns,
			index=df.index,
		)
	return df

def _ensure_dataframe_of_dtype_c_contiguous(df, dtype, label):
	if df is None:
		return df
	if not isinstance(df, pandas.DataFrame):
		raise TypeError(f'{label} must be a DataFrame')
	if not all(df.dtypes == dtype):
		logger.warning(f'converting {label} to {dtype}')
		df = pandas.DataFrame(
			data=numpy.ascontiguousarray(df.values.astype(dtype)),
			columns=df.columns,
			index=df.index,
		)
	if not df.values.flags['C_CONTIGUOUS']:
		logger.warning(f'converting {label} to c-contiguous')
		df = pandas.DataFrame(
			data=numpy.ascontiguousarray(df.values.astype(dtype)),
			columns=df.columns,
			index=df.index,
		)
	return df

def _ensure_no_duplicate_column_names(df):
	cols, counts = numpy.unique(df.columns, return_counts=True)
	if counts.max()>1:
		dupes = str(cols[numpy.where(counts>1)])
		raise DuplicateColumnNames(dupes)

def _df_values(df, re_shape=None, dtype=None):
	if df is None:
		return None
	elif re_shape is not None:
		if dtype is None:
			return df.values.reshape(*re_shape)
		else:
			return df.values.reshape(*re_shape).astype(dtype)
	else:
		if dtype is None:
			return df.values
		else:
			return df.values.astype(dtype)

def _df_values_c_contiguous(df, re_shape=None, dtype=None):
	if df is None:
		return None
	elif re_shape is not None:
		if dtype is None:
			return numpy.ascontiguousarray(df.values.reshape(*re_shape))
		else:
			return numpy.ascontiguousarray(df.values.reshape(*re_shape).astype(dtype))
	else:
		if dtype is None:
			return numpy.ascontiguousarray(df.values)
		else:
			return numpy.ascontiguousarray(df.values.astype(dtype))


def _to_categorical(y, num_classes=None, dtype='float32'):
	"""Converts a class vector (integers) to binary class matrix.

	E.g. for use with categorical_crossentropy.

	Parameters
	----------
	y: class vector to be converted into a matrix
		(integers from 0 to num_classes).
	num_classes: total number of classes.
	dtype: The data type expected by the input, as a string
		(`float32`, `float64`, `int32`...)

	Returns
	-------
	A binary matrix representation of the input. The classes axis
		is placed last.

	Notes
	-----
	This function is from keras.utils
	"""
	y = numpy.array(y, dtype='int')
	input_shape = y.shape
	if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
		input_shape = tuple(input_shape[:-1])
	y = y.ravel()
	if not num_classes:
		num_classes = numpy.max(y) + 1
	n = y.shape[0]
	categorical = numpy.zeros((n, num_classes), dtype=dtype)
	categorical[numpy.arange(n), y] = 1
	output_shape = input_shape + (num_classes,)
	categorical = numpy.reshape(categorical, output_shape)
	return categorical

def to_categorical(arr, index=None, dtype='float32'):

	if isinstance(arr, (pandas.DataFrame, pandas.Series)) and index is None:
		index = arr.index

	codes, positions = numpy.unique(arr, return_inverse=True)
	return pandas.DataFrame(
		data=_to_categorical(positions, dtype=dtype),
		columns=codes,
		index=index,
	)

def categorical_expansion(s, column=None, inplace=False, drop=False):
	"""
	Expand a pandas Series into a DataFrame containing a categorical dummy variables.

	This is sometimes called "one-hot" encoding.

	Parameters
	----------
	s : pandas.Series or pandas.DataFrame
		The input data
	column : str, optional
		If `s` is given as a DataFrame, expand this column
	inplace : bool, default False
		If true, add the result directly as new columns in `s`.
	drop : bool, default False
		If true, drop the existing column from `s`. Has no effect if
		`inplace` is not true.

	Returns
	-------
	pandas.DataFrame
		Only if inplace is false.
	"""
	if isinstance(s, pandas.DataFrame):
		input = s
		if column is not None and column not in s.columns:
			raise KeyError(f'key not found "{column}"')
		if len(s.columns) == 1:
			s = s.iloc[:, 0]
			column = s.columns[0]
		elif column is not None:
			s = s.loc[:, column]
	else:
		input = None

	onehot = to_categorical(s)
	onehot.columns = [f'{s.name}=={_}' for _ in onehot.columns]
	if inplace and input is not None:
		input[onehot.columns] = onehot
		if drop:
			input.drop([column], axis=1)
	else:
		return onehot

def crack_idca(df:pandas.DataFrame, caseid_col=True):
	"""
	Split an :ref:`idca` DataFrame into :ref:`idca` and :ref:`idco` parts.

	Parameters
	----------
	df: pandas.DataFrame
		DataFrame to split
	caseid_col: str, optional
		Name of the case indentifier column. If omitted, the first level of the MultiIndex is used.

	Returns
	-------
	idca: DataFrame
	idco: DataFrame
	"""
	if caseid_col is None or caseid_col is True:
		try:
			caseid_col = df.index.names[0]
		except:
			raise ValueError('cannot infer caseid_col')
	g = df.groupby([caseid_col])
	g_std = g.std()
	idca_columns = g_std.any()
	idco_columns = ~idca_columns
	return df[idca_columns[idca_columns].index], g[idco_columns[idco_columns].index].first()


cdef class DataFrames:

	def __init__(
			self,
			*,
			co=None,
			ca=None,
			ce=None,
			av=None,
			ch=None,
			wt=None,

			# Alternative keyword names
			data_co=None,
			data_ca=None,
			data_ce=None,
			data_av=None,
			data_ch=None,
			data_wt=None,

			alt_names = None,
			alt_codes = None,

			crack = False,

			av_name = None,
			ch_name = None,
			wt_name = None,
	):

		try:
			co = co if co is not None else data_co
			ca = ca if ca is not None else data_ca
			ce = ce if ce is not None else data_ce
			av = av if av is not None else data_av
			ch = ch if ch is not None else data_ch
			wt = wt if wt is not None else data_wt

			if crack and co is None:
				if ca is not None:
					ca, co = crack_idca(ca, crack)
				elif ce is not None:
					ce, co = crack_idca(ce, crack)

			if co is not None:
				_n_cases= len(co)
			elif ca is not None:
				_n_cases= len(ca) / len(ca.index.levels[1])
			else:
				_n_cases= 0

			if ch is not None and len(ch.shape) == 1 and ch.shape[0] == _n_cases:
				logger.warning('one-hot encoding choice array')
				ch = to_categorical(ch)

			if alt_codes is None:
				if ca is not None:
					self._alternative_codes = ca.index.levels[1]
				elif ce is not None:
					self._alternative_codes = ce.index.levels[1]
				elif ch is not None:
					self._alternative_codes = ch.columns
				else:
					self._alternative_codes = pandas.Index([])
			else:
				self._alternative_codes = pandas.Index(alt_codes)

			self._alternative_names = alt_names

			self.data_co = co
			self.data_ca = ca
			self.data_ce = ce
			self.data_ch = ch
			self.data_wt = wt
			if av is None and self.data_ce is not None:
				av = pandas.DataFrame(data=(self._array_ce_reversemap.base>=0), columns=self.alternative_codes(), index=self.caseindex)
			if av is True or (isinstance(av, (int, float)) and av==1):
				self.data_av = pandas.DataFrame(data=1, columns=self.alternative_codes(), index=self.caseindex)
			else:
				if isinstance(av, pandas.DataFrame) and isinstance(av.index, pandas.MultiIndex) and len(av.index.levels)==2 and av.shape[1]==1:
					av = av.iloc[:,0]
				if isinstance(av, pandas.Series) and isinstance(av.index, pandas.MultiIndex) and len(av.index.levels)==2:
					av = av.unstack()
				self.data_av = av

			self._weight_normalization = 1

			self._data_av_name = av_name
			self._data_ch_name = ch_name
			self._data_wt_name = wt_name
		except:
			logger.exception('error in constructing DataFrames')
			raise

	@classmethod
	def from_idce(cls, ce, choice=None, columns=None, autoscale_weights=True, crack=False):
		"""
		Create DataFrames from a single `idce` format DataFrame.

		Parameters
		----------
		ce : pandas.DataFrame
			The data
		choice : str, optional
			The name of the choice column. If not given, data_ch will not be populated.
		columns : list, optional
			Import only these columns.  If not given, will import into data_ce all columns except the choice,
			which is imported separately.
		autoscale_weights : bool, default True
			Also call autoscale_weights on the result after loading.
		crack : bool, default False
			Split `ce` into :ref:`idca` and :ref:`idco` parts.
		"""
		co = None
		if crack:
			ce, co = crack_idca(ce, crack)

		if columns is None:
			columns = list(ce.columns)
			if choice is not None and choice in ce.columns:
				columns.remove(choice)
		result = cls(
			ce = ce[columns],
			ch = ce[choice].unstack().fillna(0) if choice is not None else None,
			co = co,
			ch_name = choice
		)
		if autoscale_weights:
			result.autoscale_weights()
		return result

	def __repr__(self):
		return f"<larch.DataFrames ({self.n_cases} cases, {self.n_alts} alts)>"

	def info(self, out=None):
		print(f"larch.DataFrames:", file=out)
		print(f"  n_cases: {self.n_cases}", file=out)
		print(f"  n_alts: {self.n_alts}", file=out)
		if self.data_ca is not None:
			print(f"  data_ca:", file=out)
			for col in self.data_ca.columns:
				print(f"    - {col}", file=out)
		elif self.data_ce is not None:
			print(f"  data_ce:", file=out)
			for col in self.data_ce.columns:
				print(f"    - {col}", file=out)
		else:
			print(f"  data_ca: <not populated>", file=out)
		if self.data_co is not None:
			print(f"  data_co:", file=out)
			for col in self.data_co.columns:
				print(f"    - {col}", file=out)
		else:
			print(f"  data_co: <not populated>", file=out)
		if self.data_av is not None:
			print(f"  data_av: {self._data_av_name or '<populated>'}", file=out)
		if self.data_ch is not None:
			print(f"  data_ch: {self._data_ch_name or '<populated>'}", file=out)
		if self.data_wt is not None:
			print(f"  data_wt: {self._data_wt_name or '<populated>'}", file=out)

	def statistics(self, title="Data Statistics", header_level=2, graph=None):
		from ..util.addict import adict_report
		from ..util.statistics import statistics_for_dataframe
		result = adict_report(__title=title, __depth=header_level)
		if self.data_co is not None:
			result.data_co = statistics_for_dataframe(self.data_co)
		if self.data_ca is not None:
			result.data_ca = statistics_for_dataframe(self.data_ca)
		if self.data_ce is not None:
			result.data_ce = statistics_for_dataframe(self.data_ce)
		if self.data_ch is not None and self.data_av is not None:
			result.choices = self.choice_avail_summary(graph=graph)
		return result

	def choice_avail_summary(self, graph=None):

		if graph is None:
			ch_ = self.data_ch.unstack()
			av_ = self.data_av
		else:
			ch_ = self.data_ch_cascade(graph)
			av_ = self.data_av_cascade(graph)

		ch_.columns = ch_.columns.droplevel(0)

		ch = ch_.sum()
		av = av_.sum()

		if self.data_wt is not None:
			ch_w = pandas.Series((ch_.values * self.data_wt.values).sum(0), index=ch_.columns)
			av_w = pandas.Series((av_.values * self.data_wt.values).sum(0), index=av_.columns)
			show_wt = numpy.any(ch != ch_w)
		else:
			ch_w = ch
			av_w = av
			show_wt = False

		ch_.values[av_.values > 0] = 0
		if ch_.values.sum() > 0:
			ch_but_not_av = ch_.sum()
			if self.data_wt is not None:
				ch_but_not_av_w = pandas.Series((ch_.values * self.data_wt.values).sum(0), index=ch_.columns)
			else:
				ch_but_not_av_w = ch_but_not_av
		else:
			ch_but_not_av = None
			ch_but_not_av_w = None

		from collections import OrderedDict
		od = OrderedDict()

		if self.alternative_names() is not None:
			od['name'] = pandas.Series(self.alternative_names(), index=self.alternative_codes())

		if show_wt:
			od['chosen weighted'] = ch_w
			od['chosen unweighted'] = ch
			od['available weighted'] = av_w
			od['available unweighted'] = av
		else:
			od['chosen'] = ch
			od['available'] = av
		if ch_but_not_av is not None:
			if show_wt:
				od['chosen but not available weighted'] = ch_but_not_av_w
				od['chosen but not available unweighted'] = ch_but_not_av
			else:
				od['chosen but not available'] = ch_but_not_av

		result = pandas.DataFrame.from_dict(od)

		totals = result.sum()

		for tot in (
				'chosen',
				'chosen weighted',
				'chosen unweighted',
				'chosen but not available',
				'chosen but not available weighted',
				'chosen but not available unweighted',
				'chosen thus available',
				'not available so not chosen'
		):
			if tot in totals:
				result.loc['< Total All Alternatives >', tot] = totals[tot]

		result.loc['< Total All Alternatives >', pandas.isnull(result.loc['< Total All Alternatives >', :])] = ""
		result.drop('_root_', errors='ignore', inplace=True)
		return result

	def alternative_names(self):
		return self._alternative_names

	def alternative_codes(self):
		return self._alternative_codes

	@property
	def n_alts(self):
		return self._n_alts()

	@property
	def n_cases(self):
		return self._n_cases()

	@property
	def n_params(self):
		if self._n_model_params is None:
			return 0
		return self._n_model_params

	@property
	def n_vars_ca(self):
		return 0 if self.data_ca is None else self.data_ca.shape[1]

	@property
	def n_vars_co(self):
		return 0 if self.data_co is None else self.data_co.shape[1]

	@property
	def param_names(self):
		return self._model_param_names

	cdef int _n_alts(self):
		if self._alternative_codes is None:
			return 0
		return len(self._alternative_codes)

	cdef int _n_cases(self):
		if self._data_co is not None:
			return len(self.data_co)
		elif self._data_ca is not None:
			return len(self.data_ca) / self.n_alts
		elif self._data_ce is not None:
			return self._array_ce_reversemap.shape[0]
		else:
			return 0

	@property
	def caseindex(self):
		if self._data_co is not None:
			return self.data_co.index
		elif self._data_ch is not None:
			return self.data_ch.index
		elif self._data_wt is not None:
			return self.data_wt.index
		elif self._data_av is not None:
			return self.data_av.index
		elif self._data_ca is not None:
			return self._data_ca.index.levels[0][numpy.unique(self._data_ca.index.labels[0])]
		elif self._data_ce is not None:
			return self._data_ce.index.levels[0][numpy.unique(self._data_ce.index.labels[0])]
		else:
			return 0


	@property
	def data_ca(self):
		return self._data_ca

	@data_ca.setter
	def data_ca(self, df:pandas.DataFrame):
		if df is not None:
			_ensure_no_duplicate_column_names(df)
			self._data_ca = _ensure_dataframe_of_dtype(df, l4_float_dtype, 'data_ca')
			if self._alternative_codes is None and self._data_ca is not None:
				self._alternative_codes = self._data_ca.index.levels[1]
			self._array_ca = _df_values(self.data_ca, (self.n_cases, self.n_alts, -1))
		else:
			self._data_ca = None
			self._array_ca = None

	def data_ca_as_ce(self):
		"""
		Reformat any idca data into idce format.

		This function condenses the idca data into an idce format DataFrame by dropping unavailable
		alternatives.
		It is usually not needed to estimate or apply a discrete choice model using built-in Larch
		functions, but this may be convenient in interfacing with external tools (e.g. pre-learning
		with xgboost for a stacked model) or to improve memory efficiency.

		Returns
		-------
		pandas.DataFrame
		"""
		if self._data_ca is None:
			return None
		if self._data_av is None:
			raise ValueError('data_av is not defined')
		return self._data_ca[self._data_av.stack().astype(bool).values]


	@property
	def data_co(self):
		return self._data_co

	@data_co.setter
	def data_co(self, df:pandas.DataFrame):
		if df is not None:
			_ensure_no_duplicate_column_names(df)
			self._data_co = _ensure_dataframe_of_dtype(df, l4_float_dtype, 'data_co')
			self._array_co = _df_values(self.data_co)
		else:
			self._data_co = None
			self._array_co = None

	def data_co_as_ce(self):
		"""
		Reformat any idco data into idce format.

		This function replicates the idco data for each case onto every row of an idce format DataFrame.
		It is generally not needed to estimate or apply a discrete choice model using built-in Larch
		functions, but this may be convenient in interfacing with external tools (e.g. pre-learning
		with xgboost for a stacked model).

		Returns
		-------
		pandas.DataFrame
		"""
		if self.data_co is None:
			return None
		cdef int c,a,v,row,n_vars
		n_vars = self._array_co.shape[1]
		arr = numpy.zeros( [len(self.data_ce), n_vars], dtype=l4_float_dtype )
		for c in range(self._array_ce_reversemap.shape[0]):
			for a in range(self._array_ce_reversemap.shape[1]):
				row = self._array_ce_reversemap[c,a]
				if row >= 0:
					for v in range(n_vars):
						arr[row,v] = self._array_co[c,v]
		return pandas.DataFrame(arr, index=self.data_ce.index, columns=['weight'])

	@property
	def data_ce(self):
		return self._data_ce

	@data_ce.setter
	def data_ce(self, df:pandas.DataFrame):
		cdef int64_t i, c, a, min_case_x
		if df is None:
			self._data_ce = None
			self._array_ce = None
			self._array_ce_caseindexes = None
			self._array_ce_altindexes = None
			self._array_ce_reversemap = None
		else:
			_ensure_no_duplicate_column_names(df)
			if not df.index.is_monotonic_increasing:
				df = df.sort_index()
			self._data_ce = _ensure_dataframe_of_dtype(df, l4_float_dtype, 'data_ce')
			self._array_ce = _df_values(self.data_ce)

			unique_labels, new_labels =numpy.unique(self.data_ce.index.labels[0], return_inverse=True)
			self._array_ce_caseindexes = new_labels

			# min_case_x = self.data_ce.index.labels[0].min()
			# if min_case_x == 0:
			# 	self._array_ce_caseindexes = self.data_ce.index.labels[0]
			# else:
			# 	self._array_ce_caseindexes = self.data_ce.index.labels[0] - min_case_x
			self._array_ce_altindexes  = self.data_ce.index.labels[1]
			self._array_ce_reversemap = numpy.full([self._array_ce_caseindexes.max()+1, self._array_ce_altindexes.max()+1], -1, dtype=numpy.int64)
			for i in range(len(self._array_ce_caseindexes)):
				c = self._array_ce_caseindexes[i]
				a = self._array_ce_altindexes[i]
				self._array_ce_reversemap[c, a] = i
			#self._array_ce_reversemap[self._array_ce_caseindexes, self._array_ce_altindexes] = numpy.arange(len(self._array_ce_caseindexes), dtype=numpy.int64)

	@property
	def data_av(self):
		return self._data_av

	@data_av.setter
	def data_av(self, df:pandas.DataFrame):
		self._data_av = _ensure_dataframe_of_dtype(df, numpy.int8, 'data_av', warn_on_convert=False)
		self._array_av = _df_values(self.data_av, (self.n_cases, self.n_alts))

	def data_av_cascade(self, graph):
		result = pandas.DataFrame(
			data=numpy.int8(0),
			columns=graph.standard_sort,
			index=self.data_av.index,
		)
		result.iloc[:,:graph.n_elementals()] = self.data_av
		for dn in graph.standard_sort:
			for up in graph.predecessors(dn):
				result[up] |= result[dn]
		return result

	@property
	def data_ch(self):
		return self._data_ch

	@data_ch.setter
	def data_ch(self, df:pandas.DataFrame):
		self._data_ch = _ensure_dataframe_of_dtype(df, l4_float_dtype, 'data_ch')
		self._array_ch = _df_values(self.data_ch, (self.n_cases, self.n_alts))

	def data_ch_cascade(self, graph):
		result = pandas.DataFrame(
			data=l4_float_dtype(0),
			columns=graph.standard_sort,
			index=self.data_ch.index,
		)
		result.iloc[:,:graph.n_elementals()] = self.data_ch
		for dn in graph.standard_sort:
			for up in graph.predecessors(dn):
				result[up] += result[dn]
		return result

	def data_ch_as_ce(self):
		"""
		Reformat choice data into idce format.

		This function condenses the choice data, normally stored in idca format, into an idce format
		DataFrame by dropping unavailable alternatives.
		It is usually not needed to estimate or apply a discrete choice model using built-in Larch
		functions, but this may be convenient in interfacing with external tools (e.g. pre-learning
		with xgboost for a stacked model) or to improve memory efficiency.

		Returns
		-------
		pandas.DataFrame
		"""
		if self._data_ch is None:
			return None
		if self.data_ce is None:
			if self._data_av is None:
				raise NotImplementedError('not implemented when data_ce and data_av are None')
			return self._data_ch[self._data_av.stack().astype(bool).values]

		arr = numpy.zeros( [len(self.data_ce)], dtype=l4_float_dtype )
		cdef int c,a,row
		for c in range(self._array_ce_reversemap.shape[0]):
			for a in range(self._array_ce_reversemap.shape[1]):
				row = self._array_ce_reversemap[c,a]
				if row >= 0:
					arr[row] = self._array_ch[c,a]
		return pandas.DataFrame(arr, index=self.data_ce.index, columns=['choice'])

	@property
	def data_wt(self):
		return self._data_wt

	@data_wt.setter
	def data_wt(self, df:pandas.DataFrame):
		self._data_wt = _ensure_dataframe_of_dtype(df, l4_float_dtype, 'data_wt')
		self._array_wt = _df_values(self.data_wt, (self.n_cases, ))

	def data_wt_as_ce(self):
		if self.data_wt is None:
			return None
		arr = numpy.zeros( [len(self.data_ce)], dtype=l4_float_dtype )
		cdef int c,a,row
		for c in range(self._array_ce_reversemap.shape[0]):
			for a in range(self._array_ce_reversemap.shape[1]):
				row = self._array_ce_reversemap[c,a]
				if row >= 0:
					arr[row] = self._array_wt[c]
		return pandas.DataFrame(arr, index=self.data_ce.index, columns=['weight'])

	@property
	def _data_ca_or_ce(self):
		if self._data_ca is not None:
			return self._data_ca
		elif self._data_ce is not None:
			return self._data_ce
		else:
			raise MissingDataError('neither ca nor ce is defined')

	def array_ca(self, dtype=None, force=False):
		if force and self.data_ca is None:
			dtype = dtype if dtype is not None else numpy.float32
			return numpy.empty( (self.n_cases, self.n_alts, 0), dtype=dtype)
		return _df_values(self.data_ca, (self.n_cases, self.n_alts, -1), dtype=dtype)

	def array_co(self, dtype=None, force=False):
		if force and self.data_co is None:
			dtype = dtype if dtype is not None else numpy.float32
			return numpy.empty( (self.n_cases, 0), dtype=dtype)
		return _df_values(self.data_co, dtype=dtype)

	def array_ce(self, dtype=None, force=False):
		if force and self.data_ce is None:
			dtype = dtype if dtype is not None else numpy.float32
			return numpy.empty( (self.n_cases, 0), dtype=dtype)
		return _df_values(self.data_ce, dtype=dtype)

	@property
	def array_ce_caseindexes(self):
		return self._array_ce_caseindexes

	@property
	def array_ce_altindexes(self):
		return self._array_ce_altindexes

	@property
	def array_ce_reversemap(self):
		return self._array_ce_reversemap

	def array_av(self, dtype=None):
		return _df_values(self.data_av, (self.n_cases, self.n_alts, ), dtype=dtype)

	def array_ch(self, dtype=None):
		return _df_values(self.data_ch, (self.n_cases, self.n_alts, ), dtype=dtype)

	def array_ch_as_ce(self, dtype=None):
		return _df_values(self.data_ch_as_ce(), None, dtype=dtype)

	def array_wt(self, dtype=None):
		return _df_values(self.data_wt, dtype=dtype)

	def array_wt_as_ce(self, dtype=None):
		return _df_values(self.data_wt_as_ce(), None, dtype=dtype)

	def array_not_av(self, dtype='float32'):
		arr = self.array_av(dtype=dtype)-1
		arr[arr!=0] = -2e22 # sufficiently negative to zero out any utility
		return arr


	cdef int _check_data_is_sufficient_for_model(
			self,
			Model5c model,
	) except -1:
		"""

		Parameters
		----------
		model : Model

		Returns
		-------

		"""
		missing_data = set()

		import logging
		logger = logging.getLogger("L4").error

		def missing(y):
			if y not in missing_data:
				logger(y)
				missing_data.add(y)

		if model._utility_ca_function is not None and len(model._utility_ca_function):
			if self._data_ca_or_ce is None:
				missing(f'idca data missing for utility')
			else:
				for i in model._utility_ca_function:
					if str(i.data) not in self._data_ca_or_ce:
						missing(f'idca utility variable missing: {i.data}')

		if model._quantity_ca_function is not None and len(model._quantity_ca_function):
			if self._data_ca_or_ce is None:
				missing(f'idca data missing for quantity')
			else:
				for i in model._quantity_ca_function:
					if str(i.data) not in self._data_ca_or_ce:
						missing(f'idca quantity variable missing: {i.data}')

		if model._utility_co_functions is not None and len(model._utility_co_functions):
			if self._data_co is None:
				missing(f'idco data missing for utility')
			else:
				for alt, func in model._utility_co_functions.items():
					for i in func:
						if str(i.data) not in self._data_co and str(i.data)!= '1':
							missing(f'idco utility variable missing: {i.data}')

		if missing_data:
			if len(missing_data) < 5:
				raise MissingDataError(f'{len(missing_data)} things missing:\n  '+'\n  '.join(str(_) for _ in missing_data))
			else:
				missing_examples = [missing_data.pop() for _ in range(5)]
				raise MissingDataError(f'{len(missing_data)+5} things missing, for example:\n  '+'\n  '.join(str(_) for _ in missing_examples))

		return 0

	def check_data_is_sufficient_for_model( self, Model5c model, ):
		self._check_data_is_sufficient_for_model(model)

	cdef void _link_to_model_structure(
			self,
			Model5c model,
	):
		cdef:
			int j,n
			int len_model_utility_ca

		try:
			self._model = model
			self._n_model_params = len(model.frame)
			self._model_param_names = model.frame.index

			#self.du = _initialize_or_validate_shape(du, [self_.n_alts, len(model.pf)], dtype=l4_float_dtype)
			self.check_data_is_sufficient_for_model(model)

			if model._quantity_ca_function is not None:
				len_model_utility_ca = len(model._quantity_ca_function)
				self.model_quantity_ca_param_value = numpy.zeros([len_model_utility_ca], dtype=l4_float_dtype)
				self.model_quantity_ca_param_holdfast = numpy.zeros([len_model_utility_ca], dtype=numpy.int8)
				self.model_quantity_ca_param       = numpy.zeros([len_model_utility_ca], dtype=numpy.int32)
				self.model_quantity_ca_data        = numpy.zeros([len_model_utility_ca], dtype=numpy.int32)
				for n,i in enumerate(model._quantity_ca_function):
					self.model_quantity_ca_param[n] = model.frame.index.get_loc(str(i.param))
					self.model_quantity_ca_data [n] = self.data_ca.columns.get_loc(str(i.data))
				if model._quantity_scale is not None:
					self.model_quantity_scale_param = model.frame.index.get_loc(str(model._quantity_scale))
				else:
					self.model_quantity_scale_param = -1
			else:
				len_model_utility_ca = 0
				self.model_quantity_ca_param_value = numpy.zeros([len_model_utility_ca], dtype=l4_float_dtype)
				self.model_quantity_ca_param_holdfast = numpy.zeros([len_model_utility_ca], dtype=numpy.int8)
				self.model_quantity_ca_param       = numpy.zeros([len_model_utility_ca], dtype=numpy.int32)
				self.model_quantity_ca_data        = numpy.zeros([len_model_utility_ca], dtype=numpy.int32)
				self.model_quantity_scale_param = -1

			if model._utility_ca_function is not None:
				len_model_utility_ca = len(model._utility_ca_function)
				self.model_utility_ca_param_value = numpy.zeros([len_model_utility_ca], dtype=l4_float_dtype)
				self.model_utility_ca_param_holdfast=numpy.zeros([len_model_utility_ca], dtype=numpy.int8)
				self.model_utility_ca_param       = numpy.zeros([len_model_utility_ca], dtype=numpy.int32)
				self.model_utility_ca_data        = numpy.zeros([len_model_utility_ca], dtype=numpy.int32)
				for n,i in enumerate(model._utility_ca_function):
					self.model_utility_ca_param[n] = model.frame.index.get_loc(str(i.param))
					self.model_utility_ca_data [n] = self._data_ca_or_ce.columns.get_loc(str(i.data))
			else:
				len_model_utility_ca = 0
				self.model_utility_ca_param_value = numpy.zeros([len_model_utility_ca], dtype=l4_float_dtype)
				self.model_utility_ca_param_holdfast=numpy.zeros([len_model_utility_ca], dtype=numpy.int8)
				self.model_utility_ca_param       = numpy.zeros([len_model_utility_ca], dtype=numpy.int32)
				self.model_utility_ca_data        = numpy.zeros([len_model_utility_ca], dtype=numpy.int32)

			if model._utility_co_functions is not None:
				len_co = sum(len(_) for _ in model._utility_co_functions.values())
				self.model_utility_co_alt         = numpy.zeros([len_co], dtype=numpy.int32)
				self.model_utility_co_param_value = numpy.zeros([len_co], dtype=l4_float_dtype)
				self.model_utility_co_param_holdfast = numpy.zeros([len_co], dtype=numpy.int8)
				self.model_utility_co_param       = numpy.zeros([len_co], dtype=numpy.int32)
				self.model_utility_co_data        = numpy.zeros([len_co], dtype=numpy.int32)

				j = 0
				for alt, func in model._utility_co_functions.items():
					altindex = self._alternative_codes.get_loc(alt)
					for i in func:
						self.model_utility_co_alt  [j] = altindex
						self.model_utility_co_param[j] = model.frame.index.get_loc(str(i.param))
						if i.data == '1':
							self.model_utility_co_data [j] = -1
						else:
							self.model_utility_co_data [j] = self._data_co.columns.get_loc(str(i.data))
						j += 1
			else:
				len_co = 0
				self.model_utility_co_alt         = numpy.zeros([len_co], dtype=numpy.int32)
				self.model_utility_co_param_value = numpy.zeros([len_co], dtype=l4_float_dtype)
				self.model_utility_co_param_holdfast = numpy.zeros([len_co], dtype=numpy.int8)
				self.model_utility_co_param       = numpy.zeros([len_co], dtype=numpy.int32)
				self.model_utility_co_data        = numpy.zeros([len_co], dtype=numpy.int32)

		except:
			import logging
			logger = logging.getLogger('L4')
			logger.exception('error in DataFrames._link_to_model_structure')
			raise

	def link_to_model_parameters(
			self,
			model,
			logger=None,
	):
		cdef:
			int j,n
			int len_model_utility_ca
			l4_float_t[:] pvalues
			int8_t[:] hvalues

		try:
			pvalues = model.pf['value'].values.astype(l4_float_dtype)
			hvalues = model.pf['holdfast'].values.astype(numpy.int8)

			for n in range(self.model_quantity_ca_param_value.shape[0]):
				IF DOUBLE_PRECISION:
					self.model_quantity_ca_param_value[n] = exp(pvalues[self.model_quantity_ca_param[n]])
				ELSE:
					self.model_quantity_ca_param_value[n] = expf(pvalues[self.model_quantity_ca_param[n]])
				self.model_quantity_ca_param_holdfast[n] = hvalues[self.model_quantity_ca_param[n]]

			for n in range(self.model_utility_ca_param_value.shape[0]):
				self.model_utility_ca_param_value[n]    = pvalues[self.model_utility_ca_param[n]]
				self.model_utility_ca_param_holdfast[n] = hvalues[self.model_utility_ca_param[n]]

			for n in range(self.model_utility_co_param_value.shape[0]):
				self.model_utility_co_param_value[n]    = pvalues[self.model_utility_co_param[n]]
				self.model_utility_co_param_holdfast[n] = hvalues[self.model_utility_co_param[n]]

		except:
			if logger is None:
				import logging
				logger = logging.getLogger('L4')
			logger.exception('error in DataFrames.link_to_model_parameters')
			raise

	cdef void _read_in_model_parameters(
			self,
	):
		cdef:
			int j,n
			int len_model_utility_ca
			l4_float_t[:] pvalues

		try:
			pvalues = self._model.frame['value'].values.astype(l4_float_dtype)
			hvalues = self._model.frame['holdfast'].values.astype(numpy.int8)

			for n in range(self.model_quantity_ca_param_value.shape[0]):
				IF DOUBLE_PRECISION:
					self.model_quantity_ca_param_value[n] = exp(pvalues[self.model_quantity_ca_param[n]])
				ELSE:
					self.model_quantity_ca_param_value[n] = expf(pvalues[self.model_quantity_ca_param[n]])
				self.model_quantity_ca_param_holdfast[n] = hvalues[self.model_quantity_ca_param[n]]

			if self.model_quantity_scale_param >= 0:
				self.model_quantity_scale_param_value    = pvalues[self.model_quantity_scale_param]
				self.model_quantity_scale_param_holdfast = hvalues[self.model_quantity_scale_param]
			else:
				self.model_quantity_scale_param_value = 1
				self.model_quantity_scale_param_holdfast = 1

			for n in range(self.model_utility_ca_param_value.shape[0]):
				self.model_utility_ca_param_value[n]    = pvalues[self.model_utility_ca_param[n]]
				self.model_utility_ca_param_holdfast[n] = hvalues[self.model_utility_ca_param[n]]

			for n in range(self.model_utility_co_param_value.shape[0]):
				self.model_utility_co_param_value[n]    = pvalues[self.model_utility_co_param[n]]
				self.model_utility_co_param_holdfast[n] = hvalues[self.model_utility_co_param[n]]

		except:
			import logging
			logger = logging.getLogger('L4')
			logger.exception('error in DataFrames._read_in_model_parameters')
			raise

	def read_in_model_parameters(self):
		self._read_in_model_parameters()

	def _debug_access(self):
		from .util import Dict
		return Dict(
			model_utility_ca_param_value    = self.model_utility_ca_param_value.base,
			model_utility_ca_param_holdfast = self.model_utility_ca_param_holdfast.base,
			model_utility_co_param_value    = self.model_utility_co_param_value.base,
			model_utility_co_param_holdfast = self.model_utility_co_param_holdfast.base,
			model_quantity_scale_param_value= self.model_quantity_scale_param_value,
			model_quantity_scale_param_holdfast = self.model_quantity_scale_param_holdfast,
			model_quantity_ca_param_value   = self.model_quantity_ca_param_value.base,
			model_quantity_ca_param_holdfast= self.model_quantity_ca_param_holdfast.base,

			model_utility_ca_param   = self.model_utility_ca_param ,
			model_utility_ca_data    = self.model_utility_ca_data  ,
			model_utility_co_alt     = self.model_utility_co_alt   ,
			model_utility_co_param   = self.model_utility_co_param ,
			model_utility_co_data    = self.model_utility_co_data  ,
			model_quantity_ca_param  = self.model_quantity_ca_param,
			model_quantity_ca_data   = self.model_quantity_ca_data ,
		)


	def array_to_ce(self, arr):
		"""
		Convert a single case-alt array to an :ref:`idce` Series.

		Parameters
		----------
		arr: array-like

		Returns
		-------
		pandas.Series
		"""
		if self._data_ce is None:
			raise ValueError('no data_ce set')
		cdef int c,a,row
		result = pandas.Series(
			index = self.data_ce.index,
			data = 0,
			dtype = arr.dtype,
		)
		for c in range(self._n_cases()):
			for a in range(self._n_alts()):
				row = self._array_ce_reversemap[c,a]
				if row >= 0:
					result.values[row] = arr[c,a]
		return result


	def compute_utility_onecase(
			self,
			int c,
			l4_float_t[:] U,
			int n_alts=-1,
	):
		if n_alts==-1:
			n_alts = U.shape[1]
		self._compute_utility_onecase(c, U, n_alts)


	@cython.boundscheck(False)
	@cython.initializedcheck(False)
	@cython.cdivision(True)
	cdef void _compute_d_utility_onecase(
			self,
			int c,
			l4_float_t[:]   U,
			l4_float_t[:,:] dU,
			int n_alts,
	) nogil:
		"""
		Compute utility and d_utility w.r.t. parameters, writing to externally defined `U` and `dU` array.
		
		Parameters
		----------
		c : int
			The case index to compute.
		U : l4_float_t[n_nodes]
			Output array
		dU : l4_float_t[n_nodes,n_params]
			Output array
		n_alts : int
			Number of elemental alternatives. Must be equal to or less than `n_nodes` dimensions of 
			output arrays.
		"""

		cdef:
			int i,j,k, altindex
			int64_t row = -2
			l4_float_t  _temp, _temp_data, _max_U=0

		#memset(&dU[0,0], 0, sizeof(l4_float_t) * dU.size)
		U[:] = 0
		dU[:,:] = 0

		for j in range(n_alts):

			if self._array_ce_reversemap is not None:
				if c >= self._array_ce_reversemap.shape[0] or j >= self._array_ce_reversemap.shape[1]:
					row = -1
				else:
					row = self._array_ce_reversemap[c,j]

			if self._array_av[c,j] and row!=-1:

				if self.model_quantity_ca_param.shape[0]:
					for i in range(self.model_quantity_ca_param.shape[0]):
						if row >= 0:
							_temp = self._array_ce[row, self.model_quantity_ca_data[i]]
						else:
							_temp = self._array_ca[c, j, self.model_quantity_ca_data[i]]
						_temp *= self.model_quantity_ca_param_value[i]
						U[j] += _temp
						if not self.model_quantity_ca_param_holdfast[i]:
							dU[j,self.model_quantity_ca_param[i]] += _temp * self.model_quantity_scale_param_value

					for i in range(self.model_quantity_ca_param.shape[0]):
						if not self.model_quantity_ca_param_holdfast[i]:
							dU[j,self.model_quantity_ca_param[i]] /= U[j]

					IF DOUBLE_PRECISION:
						_temp = log(U[j])
					ELSE:
						_temp = logf(U[j])
					U[j] = _temp * self.model_quantity_scale_param_value
					if (self.model_quantity_scale_param >= 0) and not self.model_quantity_scale_param_holdfast:
						dU[j,self.model_quantity_scale_param] += _temp

				for i in range(self.model_utility_ca_param.shape[0]):
					if row >= 0:
						_temp = self._array_ce[row, self.model_utility_ca_data[i]]
					else:
						_temp = self._array_ca[c, j, self.model_utility_ca_data[i]]
					U[j] += _temp * self.model_utility_ca_param_value[i]
					if not self.model_utility_ca_param_holdfast[i]:
						dU[j,self.model_utility_ca_param[i]] += _temp
			else:
				U[j] = -INFINITY32

		for i in range(self.model_utility_co_alt.shape[0]):
			altindex = self.model_utility_co_alt[i]
			if self._array_av[c,altindex]:
				if self.model_utility_co_data[i] == -1:
					U[altindex] += self.model_utility_co_param_value[i]
					if not self.model_utility_co_param_holdfast[i]:
						dU[altindex,self.model_utility_co_param[i]] += 1
				else:
					_temp = self._array_co[c, self.model_utility_co_data[i]]
					U[altindex] += _temp * self.model_utility_co_param_value[i]
					if not self.model_utility_co_param_holdfast[i]:
						dU[altindex,self.model_utility_co_param[i]] += _temp

		# Keep exp(U) from generating overflow
		for j in range(n_alts):
			if U[j] > _max_U:
				_max_U = U[j]
		if _max_U > 500:
			for j in range(n_alts):
				U[j] -= _max_U


	@cython.boundscheck(False)
	@cython.initializedcheck(False)
	@cython.cdivision(True)
	cdef void _compute_utility_onecase(
			self,
			int c,
			l4_float_t[:]   U,
			int n_alts,
	) nogil:
		"""
		Compute d_utility w.r.t. parameters, writing to externally defined `du` array.
		
		Parameters
		----------
		c : int
			The case index to compute.
		U : l4_float_t[n_alts]
			output array
		"""

		cdef:
			int i,j,k, altindex
			int64_t row = -2
			l4_float_t  _temp, _temp_data, _max_U

		U[:] = 0
		_max_U = 0

		for j in range(n_alts):

			if self._array_ce_reversemap is not None:
				if c >= self._array_ce_reversemap.shape[0] or j >= self._array_ce_reversemap.shape[1]:
					row = -1
				else:
					row = self._array_ce_reversemap[c,j]

			if self._array_av[c,j] and row!=-1:

				if self.model_quantity_ca_param.shape[0]:
					for i in range(self.model_quantity_ca_param.shape[0]):
						if row >= 0:
							_temp = self._array_ce[row, self.model_quantity_ca_data[i]]
						else:
							_temp = self._array_ca[c, j, self.model_quantity_ca_data[i]]
						_temp *= self.model_quantity_ca_param_value[i]
						U[j] += _temp

					IF DOUBLE_PRECISION:
						_temp = log(U[j])
					ELSE:
						_temp = logf(U[j])
					U[j] = _temp * self.model_quantity_scale_param_value

				for i in range(self.model_utility_ca_param.shape[0]):
					if row >= 0:
						_temp = self._array_ce[row, self.model_utility_ca_data[i]]
					else:
						_temp = self._array_ca[c, j, self.model_utility_ca_data[i]]
					U[j] += _temp * self.model_utility_ca_param_value[i]
			else:
				U[j] = -INFINITY32

		for i in range(self.model_utility_co_alt.shape[0]):
			altindex = self.model_utility_co_alt[i]
			if self._array_av[c,altindex]:
				if self.model_utility_co_data[i] == -1:
					U[altindex] += self.model_utility_co_param_value[i]
				else:
					_temp = self._array_co[c, self.model_utility_co_data[i]]
					U[altindex] += _temp * self.model_utility_co_param_value[i]

		# Keep exp(U) from generating overflow
		for j in range(n_alts):
			if U[j] > _max_U:
				_max_U = U[j]
		if _max_U > 500:
			for j in range(n_alts):
				U[j] -= _max_U


	def compute_d_utility_onecase(
			self,
			int c,
			l4_float_t[:] U,
			l4_float_t[:,:] dU,
	):
		self._compute_d_utility_onecase(c, U, dU, dU.shape[0])


	@cython.boundscheck(False)
	@cython.initializedcheck(False)
	@cython.cdivision(True)
	@cython.wraparound(False)
	cdef l4_float_t[:] _get_choice_onecase(
			self,
			int c,
	) nogil:
		return self._array_ch[c,:]

	@cython.boundscheck(False)
	@cython.initializedcheck(False)
	@cython.cdivision(True)
	@cython.wraparound(False)
	cdef void _copy_choice_onecase(
			self,
			int c,
			l4_float_t[:] into_array,
	) nogil:
		cdef:
			int i
			int readcap = self._array_ch.shape[1]
			int writecap = into_array.shape[0]
		if readcap < writecap:
			for i in range(readcap):
				into_array[i] = self._array_ch[c,i]
			for i in range(readcap, writecap):
				into_array[i] = 0
		else:
			for i in range(writecap):
				into_array[i] = self._array_ch[c,i]

	def what_is_up(self):
		return (
			self.model_quantity_scale_param_value,
			self.model_quantity_ca_param_value.base,

		)

	def dump(self, filename, **kwargs):
		"""Persist this DataFrames object into one file.

		Parameters
		-----------
		filename: str, pathlib.Path, or file object.
			The file object or path of the file in which it is to be stored.
			The compression method corresponding to one of the supported filename
			extensions ('.z', '.gz', '.bz2', '.xz' or '.lzma') will be used
			automatically.
		compress: int from 0 to 9 or bool or 2-tuple, optional
			Optional compression level for the data. 0 or False is no compression.
			Higher value means more compression, but also slower read and
			write times. Using a value of 3 is often a good compromise.
			See the notes for more details.
			If compress is True, the compression level used is 3.
			If compress is a 2-tuple, the first element must correspond to a string
			between supported compressors (e.g 'zlib', 'gzip', 'bz2', 'lzma'
			'xz'), the second element must be an integer from 0 to 9, corresponding
			to the compression level.

		Returns
		-------
		filenames: list of strings
			The list of file names in which the data is stored. If
			compress is false, each array is stored in a different file.

		See Also
		--------
		DataFrames.load : corresponding loader

		Notes
		-----
		Only the dataframes are persisted to disk. Features of the model, if one is linked,
		are not saved with this function and should be saved independently.

		"""
		storage_dict = {}
		if self.data_av is not None:
			storage_dict['av'] = self.data_av
		if self.data_ca is not None:
			storage_dict['ca'] = self.data_ca
		if self.data_ce is not None:
			storage_dict['ce'] = self.data_ce
		if self.data_ch is not None:
			storage_dict['ch'] = self.data_ch
		if self.data_co is not None:
			storage_dict['co'] = self.data_co
		if self.data_wt is not None:
			storage_dict['wt'] = self.data_wt
		from sklearn.externals import joblib
		return joblib.dump(storage_dict, filename, **kwargs)

	@classmethod
	def load(cls, filename):
		"""Reconstruct a DataFrames object from a file persisted with DataFrames.dump.

		Parameters
		-----------
		filename: str, pathlib.Path, or file object.
			The file object or path of the file from which to load the object

		Returns
		-------
		result: DataFrames object
			The object stored in the file.

		See Also
		--------
		DataFrames.dump : function to save a DataFrames

		"""

		from sklearn.externals import joblib
		storage_dict = joblib.load(filename)
		return cls(**storage_dict)


	def standardize(self, with_mean=True, with_std=True, DataFrames same_as=None):
		"""
		Standardize the data in idco, idca, and idce arrays.

		This will adjust the data values such that the mean is zero and the stdev is 1.
		For idca data, only values for available alternatives are considered to compute
		the mean and stdev.

		Parameters
		----------
		with_mean : bool
		with_std : bool
		same_as : DataFrames, optional
			An existing DataFrames object that has already been standardized.  The same
			adjustment factors will be used, instead of fitting new ones.  This may result
			in the mean and/or stdev not coming out at the target values, but can be useful
			to standardize test or validation data the same as training data.

		Returns
		-------

		"""
		if same_as is not None:
			self._std_scaler_co = same_as._std_scaler_co
			self._std_scaler_ca = same_as._std_scaler_ca
			self._std_scaler_ce = same_as._std_scaler_ce

			if self.data_ca is not None:
				self.data_ca.values[:] = self._std_scaler_ca.transform(self.data_ca.values)

			if self.data_co is not None:
				self.data_co.values[:] = self._std_scaler_co.transform(self.data_co.values)

			if self.data_ce is not None:
				self.data_ce.values[:] = self._std_scaler_ce.transform(self.data_ce.values)

		else:
			self._std_scaler_co = StandardScalerExcludeUnitRange(with_mean=with_mean, with_std=with_std)
			self._std_scaler_ca = StandardScalerExcludeUnitRange(with_mean=with_mean, with_std=with_std)
			self._std_scaler_ce = StandardScalerExcludeUnitRange(with_mean=with_mean, with_std=with_std)

			mask = (self.data_av.values==0)

			if self.data_ca is not None:
				self._std_scaler_ca.fit(self.data_ca.values, Xmask=mask)
				self.data_ca.values[:] = self._std_scaler_ca.transform(self.data_ca.values)

			if self.data_co is not None:
				self._std_scaler_co.fit(self.data_co.values, Xmask=mask)
				self.data_co.values[:] = self._std_scaler_co.transform(self.data_co.values)

			if self.data_ce is not None:
				self._std_scaler_ce.fit(self.data_ce.values, Xmask=mask)
				self.data_ce.values[:] = self._std_scaler_ce.transform(self.data_ce.values)

	def autoscale_weights(self):

		cdef int i,j
		cdef bint need_to_extract_wgt_from_ch = False
		cdef l4_float_t scale_level, temp

		for i in range(self._array_ch.shape[0]):
			temp = 0
			for j in range(self._array_ch.shape[1]):
				temp += self._array_ch[i,j]
			if temp < 0.99999999 or temp > 1.00000001:
				need_to_extract_wgt_from_ch = True
				break

		if need_to_extract_wgt_from_ch and self._data_wt is None:
			self.data_wt = pandas.DataFrame(
				data=1.0,
				index=self.caseindex,
				columns=['computed_weight'],
			)

		if need_to_extract_wgt_from_ch:
			while i < self._array_ch.shape[0]:
				temp = 0
				for j in range(self._array_ch.shape[1]):
					temp += self._array_ch[i,j]
				if temp != 0:
					for j in range(self._array_ch.shape[1]):
						self._array_ch[i,j] /= temp
				self._array_wt[i] *= temp
				i += 1

		if self._data_wt is None:
			return 1.0

		total_weight = self._array_wt.base.sum()
		scale_level = total_weight / self._n_cases()

		for i in range(self._array_wt.shape[0]):
			self._array_wt[i] /= scale_level

		self._weight_normalization *= scale_level

		if scale_level < 0.99999 or scale_level > 1.00001:
			logger.warning(f'rescaled array of weights by a factor of {scale_level}')

		return scale_level

	def unscale_weights(self):

		cdef int i
		cdef l4_float_t scale_level

		scale_level = self._weight_normalization

		for i in range(self._array_wt.shape[0]):
			self._array_wt[i] *= scale_level

		self._weight_normalization /= scale_level

		return scale_level

	@property
	def weight_normalization(self):
		return self._weight_normalization

	@weight_normalization.deleter
	def weight_normalization(self):
		if self._weight_normalization != 1.0:
			logger.warning(f'dropping weight normalization factor of {self._weight_normalization}')
		self._weight_normalization = 1.0

	@property
	def std_scaler_co(self):
		return self._std_scaler_co

	@property
	def std_scaler_ca(self):
		return self._std_scaler_ca

	@property
	def std_scaler_ce(self):
		return self._std_scaler_ce

	def make_mnl(self):
		"""
		Generate a simple MNL model that uses the entire data_ca and data_co dataframes.

		Returns
		-------
		Model
		"""
		from . import Model
		m5 = Model()
		from larch.roles import P, X, PX
		if self.data_co is not None:
			for a in self.alternative_codes()[1:]:
				m5.utility_co[a] = sum(PX(j) for j in self.data_co.columns)
		if self.data_ca is not None:
			m5.utility_ca = sum(PX(j) for j in self.data_ca.columns)
		m5.dataframes = self
		return m5

	def split(self, splits, method='simple'):
		"""
		Generate a train/test or similar multi-part split of the data.

		Parameters
		----------
		splits : int or list
			If an int, gives the number of evenly sized splits to create. If a list, gives
			the relative size of the splits.
		method : {'simple', 'shuffle'}
			If simple, the data is assumed to be adequately shuffled already and splits are
			made of contiguous sections of data.  This is memory efficient.  Choose 'shuffle'
			to randomly shuffle the cases while splitting; data will be copied to new memory.

		Returns
		-------
		list of DataFrames
		"""
		try:
			if isinstance(splits, int):
				n_splits = splits
				splits = [1.0/splits] * splits
			else:
				n_splits = len(splits)

			if n_splits <= 1:
				raise ValueError('no splitting required')

			splits = numpy.asarray(splits)
			splits = splits/splits.sum()

			logger.debug(f'splitting dataframe {splits}')

			cum_splits = splits.cumsum()
			uniform_seq = numpy.arange(self.n_cases) / self.n_cases

			membership = (uniform_seq.reshape(-1,1) <= cum_splits.reshape(1,-1))
			membership[:,1:] ^= membership[:,:-1]

			logger.debug(f'   membership.shape {membership.shape}')
			logger.debug(f'   membership0 {membership.sum(0)}')
			membership_sum1 = membership.sum(1)
			logger.debug(f'   membership1 {membership_sum1.min()} to {membership_sum1.max()}')

			result = []
			for s in range(n_splits):
				logger.debug(f'  split {s} data prep')
				these_positions = membership[:,s].reshape(-1)
				these_caseids   = self.caseindex[these_positions]
				data_co=None if self.data_co is None else self.data_co.iloc[these_positions,:]
				data_ca=None if self.data_ca is None else self.data_ca.loc[these_caseids,:]
				data_ce=None if self.data_ce is None else self.data_ce.loc[these_caseids,:]
				data_av=None if self.data_av is None else self.data_av.iloc[these_positions,:]
				data_ch=None if self.data_ch is None else self.data_ch.iloc[these_positions,:]
				data_wt=None if self.data_wt is None else self.data_wt.iloc[these_positions,:]

				logger.debug(f'  split {s} factory')
				result.append(self.__class__(
					data_co=data_co,
					data_ca=data_ca,
					data_ce=data_ce,
					data_av=data_av,
					data_ch=data_ch,
					data_wt=data_wt,
					alt_names = self.alternative_names(),
					alt_codes = self.alternative_codes(),
				))
			logger.debug(f'done splitting dataframe {splits}')
			return result
		except:
			logger.exception('error in DataFrames.split')
			raise


	def make_dataframes(self, req_data, *, selector=None, float_dtype=numpy.float64):
		"""Create a DataFrames object that will satisfy a data request.

		Parameters
		----------
		req_data : Dict or str
			The requested data. The keys for this dictionary may include {'ca', 'co',
			'choice_ca', 'choice_co', 'weight_co', 'avail_ca', 'standardize'}.
			Currently, the keys {'choice_co_code', 'avail_co'} are not implemented and
			will raise an error.
			Other keys are silently ignored.
		selector : array-like[bool] or slice, optional
			If given, the selector filters the cases. This argument can only be given
			as a keyword argument.
		float_dtype : dtype, default float64
			The dtype to use for all float-type arrays.  Note that the availability
			arrays are always returned as int8 regardless of the float type.
			This argument can only be given
			as a keyword argument.

		Returns
		-------
		DataFrames
			This object should satisfy the request.
		"""

		if selector is not None:
			raise NotImplementedError

		if isinstance(req_data, str):
			from .util import Dict
			import textwrap
			req_data = Dict.load(textwrap.dedent(req_data))

		if 'ca' in req_data:
			from .util.dataframe import columnize
			df_ca = columnize(self._data_ca_or_ce, list(req_data['ca']), inplace=False, dtype=float_dtype)
		else:
			df_ca = None

		if 'co' in req_data:
			df_co = columnize(self._data_co, list(req_data['co']), inplace=False, dtype=float_dtype)
		else:
			df_co = None

		if 'choice_ca' in req_data:
			name_ch = req_data['choice_ca']
			if name_ch == self._data_ch_name:
				df_ch = self._data_ch
			else:
				try:
					df_ch = columnize(self._data_ca_or_ce, [name_ch], inplace=False, dtype=float_dtype)
				except NameError:
					df_ch = self._data_ch
		elif 'choice_co' in req_data:
			alts = self.alternative_codes()
			cols = [req_data['choice_co'].get(a, '0') for a in alts]
			try:
				df_ch = columnize(self._data_co, cols, inplace=False, dtype=float_dtype)
			except NameError:
				df_ch = self._data_ch
			else:
				df_ch.columns = alts
		elif 'choice_co_code' in req_data:
			raise NotImplementedError('choice_co_code')
		else:
			df_ch = None

		weight_normalization = 1

		if 'weight_co' in req_data:
			try:
				df_wt = columnize(self._data_co, [req_data['weight_co']], inplace=False, dtype=float_dtype)
			except NameError:
				df_wt = self._data_wt
				weight_normalization = self._weight_normalization
		else:
			df_wt = None

		if df_wt is None and self._data_wt is not None:
			logger.warning('req_data does not request weight_co but it is set and being provided')
			df_wt = self._data_wt
			weight_normalization = self._weight_normalization

		if 'avail_ca' in req_data:
			try:
				df_av = columnize(self._data_ca_or_ce, [req_data['avail_ca']], inplace=False, dtype=numpy.int8)
			except NameError:
				df_av = self._data_av
		elif 'avail_co' in req_data:
			raise NotImplementedError('avail_co')
		else:
			df_av = None

		result = DataFrames(
			co=df_co,
			ce=df_ca,
			av=df_av,
			ch=df_ch,
			wt=df_wt,
			alt_codes=self.alternative_codes(),
			alt_names=self.alternative_names(),
		)

		result._weight_normalization = weight_normalization

		if 'standardize' in req_data and req_data.standardize:
			result.standardize()

		return result

	def validate_dataservice(self, req_data):
		pass

from sklearn.preprocessing import StandardScaler

class StandardScalerExcludeUnitRange(StandardScaler):

	def fit(self, X, y=None, Xmask=None):
		if Xmask is not None:
			Xmask_ = Xmask.reshape(-1)
			try:
				X = numpy.ma.masked_array(X, Xmask_)
			except numpy.ma.MaskError:
				_, Xmask_ = numpy.broadcast_arrays(X, Xmask_[..., None])
				X = numpy.ma.masked_array(X, Xmask_)
		Xq = X.reshape(-1, X.shape[-1])

		# result = super().fit(Xq, y)
		self.mean_ = numpy.nanmean(Xq, axis=0)
		self.scale_ = numpy.nanstd(Xq, axis=0)
		self.scale_[self.scale_==0] = 1.0

		# Do not scale when the min and max for a variable are exactly 0 and/or 1
		nonscale = ((Xq.max(0) == 0)|(Xq.max(0) == 1)) & ((Xq.min(0) == 0)|(Xq.min(0) == 1))
		if self.with_mean:
			self.mean_[nonscale] = 0
		if self.with_std:
			self.scale_[nonscale] = 1
		return self
