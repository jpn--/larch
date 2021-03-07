# cython: language_level=3, embedsignature=True

from __future__ import print_function

include "general_precision.pxi"
from .general_precision import l4_float_dtype
from .general_precision cimport l4_float_t

from libc.string cimport memset
cimport cython

import pandas
import numpy
import inspect
from typing import Mapping, Sequence, Union

import logging
from .log import logger_name
logger = logging.getLogger(logger_name)

from .model.controller cimport Model5c
from numpy.math cimport expf, logf
from libc.math cimport exp, log

from .util.dataframe import columnize
from .util.multiindex import remove_unused_level

cdef float INFINITY32 = numpy.float64('inf')

from .exceptions import MissingDataError, DuplicateColumnNames


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
		raise TypeError(f'{label} must be a DataFrame, not a {type(df)}')
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
	if len(cols)>0 and counts.max()>1:
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

def _fast_check_multiindex_equality(i, j):
	if not isinstance(i, pandas.MultiIndex):
		return False
	if not isinstance(j, pandas.MultiIndex):
		return False
	if len(i.codes) != len(j.codes):
		return False
	for k in range(len(i.codes)):
		if not numpy.array_equal(i.codes[k], j.codes[k]):
			return False
	if len(i.levels) != len(j.levels):
		return False
	for k in range(len(i.levels)):
		if not numpy.array_equal(i.levels[k], j.levels[k]):
			return False
	return True

def _data_check(x, df, valid_shorts=()):
	if str(x) in df:
		return True
	if str(x) in valid_shorts:
		return True
	return False


def _to_categorical(y, num_classes=None, dtype='float32'):
	"""
	Converts a class vector (integers) to binary class matrix.

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
	_to_co = idco_columns[idco_columns].index
	_keep_ca = [c for c in df.columns if c not in _to_co]
	return df[_keep_ca], g[_to_co].first()

def force_crack_idca(df:pandas.DataFrame, caseid_col=True):
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
	idco: DataFrame
	"""
	if caseid_col is None or caseid_col is True:
		try:
			caseid_col = df.index.names[0]
		except:
			raise ValueError('cannot infer caseid_col')
	g = df.groupby([caseid_col])
	return g.first()


def _infer_name(thing):
	if thing is None:
		return None
	if isinstance(thing, str):
		return thing
	if hasattr(thing, 'name'):
		name = thing.name
		if isinstance(name, str):
			return name
		if callable(name):
			return _infer_name(name())
		return str(name)
	return None


def get_dataframe_format(df):
	'''Check the format of a dataframe.

	This function assumes the input dataframe is in |idco|, |idca|, or
	|idce| formats, and returns the format found.

	Parameters
	----------
	df : pandas.DataFrame
		The dataframe to inspect

	Returns
	-------
	str
		one of {'idco', 'idca', 'idce'}
	'''
	if isinstance(df.index, pandas.MultiIndex) and df.index.nlevels==2:
		# The df is idca or idce format
		if len(df) < len(df.index.levels[0]) * len(df.index.levels[1]):
			return 'idce'
		if not df.index.is_monotonic_increasing:
			return 'idce'
		return 'idca'
	elif isinstance(df.index, pandas.Index) and not isinstance(df.index, pandas.MultiIndex):
		return 'idco'


cdef class DataFrames:
	"""A structured class to hold multi-format discrete choice data.

	Parameters
	----------
	co : pandas.DataFrame
		A dataframe containing |idco| format data, with one row per case.
		The index contains the caseid's.
	ca : pandas.DataFrame
		A dataframe containing |idca| format data, with one row per alternative.
		The index should be a two-level multi-index, with the first level
		containing the caseid's and the second level containing the altid's.
	av : pandas.DataFrame or pandas.Series or True, optional
		Alternative availability data.  This can be given as a pandas.DataFrame
		in |idco| format, with one row per case and one column per alternative,
		where the index contains the caseid's, and the columns contain the altid's.
		Or, it can be given as a pandas.Series in |idca| format, with one row
		per alternative, and an index that is a two-level multi-index, with the
		first level containing the caseid's and the second level containing the
		altid's. Or, set to `True` to make all alternatives available for all
		cases.  If not given, then `data_av` will not be defined unless it can
		be inferred from missing rows in `ca`.
	ch : pandas.DataFrame or pandas.Series or str, optional
		Choice data. This can be given as a pandas.DataFrame
		in |idco| format, with one row per case and one column per alternative,
		where the index contains the caseid's, and the columns contain the altid's.
		Or, it can be given as a pandas.Series in |idca| format, with one row
		per alternative, and an index that is a two-level multi-index, with the
		first level containing the caseid's and the second level containing the
		altid's. Or, if given as a str, then that named column is found in the `ca`
		dataframe if it appears there and used as the choice.  Otherwise, if
		the named column is found in the `co` dataframe, then the codes in that column
		are used to identify the choices.  If not given, `data_ch` is not set.
	wt : pandas.DataFrame or pandas.Series or str, optional
		Case weights. This can only be given in |idco| format, either as a
		pandas.DataFrame with a single column, or as a pandas.Series. Or, if given
		as a str, then that named column is found in the `co` or `ca`
		dataframe if it appears there and used as the weight. If not given,
		`data_wt` is not set.
	alt_names : Sequence[str]
		A sequence of alternative names as str.
	alt_codes : Sequence[int]
		A sequence of alternative codes.
	crack : bool, default False
		Whether to pre-process `ca` data to identify variables that do
		not vary within cases, and move them to a new `co` dataframe.  This can
		result in more computationally efficient model estimation, but the cracking
		process can be slow for large data sets.
	av_name : str, optional
		A name to use for the availability variable.  If not given, it is inferred
		from the `av` argument if possible.
	ch_name : str, optional
		A name to use for the choice variable.  If not given, it is inferred from
		the `ch` argument if possible.
	wt_name : str, optional
		A name to use for the weight variable.  If not given, it is inferred from
		the `wt` argument if possible.
	autoscale_weights : bool, default False
		Call `autoscale_weights` on the DataFrames after initialization. Note that
		this will not only scale an explicitly given `wt`, but it will also
		extract implied weights from the `ch` as well.
	"""

	def __init__(
			self,
			*args,
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

			av_as_ce = None,
			ch_as_ce = None,

			sys_alts = None,
			computational = False,

			caseindex_name = '_caseid_',
			altindex_name = '_altid_',

			autoscale_weights=False,
	):

		try:
			if len(args) > 1:
				raise ValueError('DataFrames accepts at most one positional argument')
			elif len(args) == 1:
				fmt = get_dataframe_format(args[0])
				if fmt == 'idca':
					if ca is not None:
						raise ValueError('cannot give both positional and keyword data_ca')
					ca = args[0]
				elif fmt == 'idce':
					if ce is not None:
						raise ValueError('cannot give both positional and keyword data_ce')
					ce = args[0]
				elif fmt == 'idco':
					if co is not None:
						raise ValueError('cannot give both positional and keyword data_co')
					co = args[0]

			self._data_co = None
			self._data_ca = None
			self._data_ce = None
			self._data_ch = None
			self._data_wt = None
			self._data_av = None

			co = co if co is not None else data_co
			ca = ca if ca is not None else data_ca
			ce = ce if ce is not None else data_ce
			av = av if av is not None else data_av
			ch = ch if ch is not None else data_ch
			wt = wt if wt is not None else data_wt

			self._caseindex_name = caseindex_name
			self._altindex_name = altindex_name

			if co is not None:
				# If co is given, ensure that it has a one level (single)index,
				# and that index is converted to integer data type.
				if co.index.nlevels!=1:
					raise ValueError('co must have one level index')
				try:
					co.index = co.index.astype(int)
				except Exception as err:
					raise ValueError('co index must have integer values') from err

			if ce is not None:
				# If ce is given, ensure that it has a two level multi-index,
				# and both levels are converted to integer data types.
				if not isinstance(ce.index, pandas.MultiIndex) or ce.index.nlevels!=2:
					raise ValueError('ce must have two level multi-index')
				try:
					ce = ce.set_index(ce.index.set_levels(ce.index.levels[0].astype(int), 0))
				except Exception as err:
					raise ValueError('ce multi-index must have integer case values') from err
				try:
					ce = ce.set_index(ce.index.set_levels(ce.index.levels[1].astype(int), 1))
				except Exception as err:
					if alt_codes is None and alt_names is None:
						alt_names = list(ce.index.levels[1])
						alt_codes = numpy.arange(len(alt_names))+1
						ce = ce.set_index(ce.index.set_levels(alt_codes, 1))
					else:
						raise ValueError('ce multi-index must have integer values') from err

			if ca is not None:
				# If ca is given, ensure that it has a two level multi-index,
				# and both levels are converted to integer data types.
				if not isinstance(ca.index, pandas.MultiIndex) or ca.index.nlevels!=2:
					raise ValueError('ca must have two level multi-index')
				try:
					ca = ca.set_index(ca.index.set_levels(ca.index.levels[0].astype(int), 0))
				except Exception as err:
					raise ValueError('ca multi-index must have integer values') from err
				try:
					ca = ca.set_index(ca.index.set_levels(ca.index.levels[1].astype(int), 1))
				except Exception as err:
					if alt_codes is None and alt_names is None:
						alt_names = list(ca.index.levels[1])
						alt_codes = numpy.arange(len(alt_names))+1
						ca = ca.set_index(ca.index.set_levels(alt_codes, 1))
					else:
						raise ValueError('ca multi-index must have integer values') from err

			# reassign ca and ce as needed
			if ca is None and ce is not None:
				if get_dataframe_format(ce) == 'idca':
					ca, ce = ce, None
			elif ca is not None and ce is None:
				if get_dataframe_format(ca) == 'idce':
					ca, ce = None, ca


			self._computational = computational

			if isinstance(ch, str) and ch_name is None:
				ch, ch_name = None, ch
			if isinstance(wt, str) and wt_name is None:
				wt, wt_name = None, wt

			# Infer names from input data is possible
			if av_name is None:
				av_name = _infer_name(av)
			if ch_name is None:
				ch_name = _infer_name(ch)
			if wt_name is None:
				wt_name = _infer_name(wt)

			if crack and co is None:
				if ca is not None:
					logger.debug(" DataFrames ~ cracking idca")
					ca, co = crack_idca(ca, crack)
				elif ce is not None:
					logger.debug(" DataFrames ~ cracking idce")
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

			if ch is None and ch_as_ce is not None and ce is not None:
				logger.debug(" DataFrames ~ building ch_as_ce")
				_ch_temp = pandas.DataFrame(numpy.asarray(ch_as_ce), index=ce.index, columns=['choice'])
				ch = _ch_temp.unstack().fillna(0)
				ch.columns = ch.columns.droplevel(0)

			if ch is None and ch_name is not None and ce is not None and ch_name in ce.columns:
				logger.debug(" DataFrames ~ pulling ch from ce")
				_ch_temp = ce[ch_name]
				ch = _ch_temp.unstack().fillna(0)

			if ch is None and ch_name is not None and ca is not None and ch_name in ca.columns:
				logger.debug(" DataFrames ~ pulling ch from ca")
				_ch_temp = ca[ch_name]
				ch = _ch_temp.unstack().fillna(0)

			if av is None and av_as_ce is not None and ce is not None:
				logger.debug(" DataFrames ~ building av_as_ce")
				_av_temp = pandas.DataFrame(numpy.asarray(av_as_ce), index=ce.index, columns=['avail'])
				av = _av_temp.unstack().fillna(0)
				av.columns = av.columns.droplevel(0)

			if wt_name is not None and wt is None:
				if co is not None and wt_name in co.columns:
					logger.debug(" DataFrames ~ pulling wt from co")
					wt = co[wt_name]
				elif ca is not None and wt_name in ca.columns:
					logger.debug(" DataFrames ~ build wt from ca")
					_, wt = force_crack_idca(ca[wt_name], crack if ((crack is not None) and (crack is not False)) else True)
				elif ce is not None and wt_name in ce.columns:
					logger.debug(" DataFrames ~ build wt from ce")
					_, wt = force_crack_idca(ce[wt_name], crack if ((crack is not None) and (crack is not False)) else True)

			logger.debug(" DataFrames ~ set alternative_codes")
			self._ensure_consistent_alternative_codes(alt_codes)
			if ca is not None:
				self._ensure_consistent_alternative_codes(ca.index.levels[1])
			if ce is not None:
				self._ensure_consistent_alternative_codes(ce.index.levels[1])

			if ch is None and ch_name is not None and co is not None and ch_name in co.columns:
				choicecodes = columnize(co, [ch_name], inplace=False, dtype=int)
				if self.alternative_codes() is None:
					self._ensure_consistent_alternative_codes(numpy.unique(choicecodes))
				_ch_temp = pandas.DataFrame(
					0,
					columns=self.alternative_codes(),
					index=co.index,
					dtype=numpy.float64,
				)
				for c in _ch_temp.columns:
					_ch_temp.loc[:,c] = (choicecodes==c).astype(numpy.float64)
				ch = _ch_temp

			logger.debug(" DataFrames ~ set alternative_names")
			self._alternative_names = alt_names

			logger.debug(" DataFrames ~ assign core data")
			self.data_co = co
			self.data_ca = ca
			self.data_ce = ce
			if isinstance(ch, pandas.DataFrame) and isinstance(ch.index, pandas.MultiIndex) and len(ch.index.levels)==2 and ch.shape[1]==1:
				logger.debug(" DataFrames ~ change ch to Series")
				ch = ch.iloc[:,0]
			if isinstance(ch, pandas.Series) and isinstance(ch.index, pandas.MultiIndex) and len(ch.index.levels)==2:
				if self.data_ca is not None and _fast_check_multiindex_equality(self.data_ca.index, ch.index):
					logger.debug(" DataFrames ~ unstack ch (fast)")
					ch = pandas.DataFrame(
						data=ch.values.reshape(-1, len(self.alternative_codes())),
						index=self.caseindex,
						columns=self.alternative_codes(),
					)
				else:
					logger.debug(" DataFrames ~ unstack ch (slow)")
					ch = ch.unstack().fillna(0)

			if ch is not None:
				# check if there are missing columns for known alt codes
				if self._alternative_codes is not None:
					if len(self._alternative_codes) > len(ch.columns):
						if set(self._alternative_codes).issuperset(ch.columns):
							ch = ch.reindex(columns=self._alternative_codes).fillna(0)
				self._ensure_consistent_alternative_codes(ch.columns)
			if self._alternative_codes is None:
				self._alternative_codes = pandas.Index([])

			logger.debug(" DataFrames ~ assign aux data")
			self.data_ch = ch
			try:
				self.data_wt = wt
			except ValueError:
				self.data_wt = force_crack_idca(wt)
			if av is None and self.data_ce is not None:
				logger.debug(" DataFrames ~ build av from ce")
				av = pandas.DataFrame(data=(self._array_ce_reversemap.base>=0), columns=self.alternative_codes(), index=self.caseindex)
			if isinstance(av, dict) and self.data_co is not None:
				_av_temp = pandas.DataFrame(
					0,
					columns=self.alternative_codes(),
					index=self.caseindex,
					dtype=numpy.int8,
				)
				for _altcode, _av_def in av.items():
					_av_temp.loc[:,_altcode] = columnize(self.data_co, [_av_def], inplace=False, dtype=numpy.int8)
				av = _av_temp
			if isinstance(av, str) and self.data_ca is not None:
				av = pandas.DataFrame(
					data=columnize(self.data_ca, [av], inplace=False, dtype=numpy.int8).values.reshape(-1, len(self.alternative_codes())),
					index=self.caseindex,
					columns=self.alternative_codes(),
				)
			if av is True or (isinstance(av, (int, float)) and av==1):
				if self.n_alts == 0:
					raise ValueError('cannot declare all alternatives are available without defining alternative codes')
				logger.debug(" DataFrames ~ initialize av as 1")
				self.data_av = pandas.DataFrame(data=1, columns=self.alternative_codes(), index=self.caseindex, dtype=numpy.int8)
			else:
				if isinstance(av, pandas.DataFrame) and isinstance(av.index, pandas.MultiIndex) and len(av.index.levels)==2 and av.shape[1]==1:
					logger.debug(" DataFrames ~ change av to Series")
					av = av.iloc[:,0]
				if isinstance(av, pandas.Series) and isinstance(av.index, pandas.MultiIndex) and len(av.index.levels)==2:
					if self.data_ca is not None and _fast_check_multiindex_equality(self.data_ca.index, av.index):
						logger.debug(" DataFrames ~ unstack av (fast)")
						av = pandas.DataFrame(
							data=av.values.reshape(-1, len(self.alternative_codes())),
							index=self.caseindex,
							columns=self.alternative_codes(),
						)
					else:
						logger.debug(" DataFrames ~ unstack av (slow)")
						av = av.unstack()
				self.data_av = av

			self._weight_normalization = 1.0

			logger.debug(" DataFrames ~ assign names")
			self._data_av_name = av_name
			self._data_ch_name = ch_name
			self._data_wt_name = wt_name

			self._systematic_alternatives = sys_alts

			if autoscale_weights:
				self.autoscale_weights()

		except:
			logger.exception('error in constructing DataFrames')
			raise

	@classmethod
	def from_idce(cls, ce, choice=None, columns=None, autoscale_weights=True, crack=False):
		"""
		Create DataFrames from a single `idce` format DataFrame.

		Note: This constructor is deprecated in favor of the plain
		__init__ constructor for DataFrames, which can now automatically
		recognize the difference between `idca` and `idce` data.

		Parameters
		----------
		ce : pandas.DataFrame
			The data to use, in `idce` format, which contains one row
			for each *available* alternative for each case (also known
			as "tall" or "idcase-idalt" format data).  This DataFrame
			must have a two-level MultiIndex, where the first level
			gives the caseid's and the second level gives the altid's.
		choice : str, optional
			The name of the choice column. If not given, data_ch will
			not be populated.
		columns : list, optional
			Import only these columns.  If not given, will import into
			data_ce all columns except the choice, which is imported
			separately.
		autoscale_weights : bool, default True
			Also call autoscale_weights on the result after loading.
		crack : bool, default False
			Split `ce` into :ref:`idca` and :ref:`idco` parts, by
			identifying variables that do not vary within cases, and
			moving them to a `co` dataframe.  This can result in more
			computationally efficient model estimation, but the cracking
			process can be slow for large data sets.

		Raises
		------
		TypeError
			When `ce` does not have a two-level MultiIndex.
		"""
		if not isinstance(ce.index, pandas.MultiIndex):
			raise TypeError(f'input data must have case-alt set as a MultiIndex, not {type(ce.index)}')
		if not len(ce.index.levels) == 2:
			raise TypeError(f'input data must have 2 level MultiIndex (case,alt), not {len(ce.index.levels)} levels')

		co = None
		if crack:
			ce, co = crack_idca(ce, crack)

		caseindex_name, altindex_name = ce.index.names
		if caseindex_name is None:
			caseindex_name = '_caseid_'
		if altindex_name is None:
			altindex_name = '_altid_'

		if columns is None:
			columns = list(ce.columns)
			if choice is not None and choice in ce.columns:
				columns.remove(choice)
		result = cls(
			ce = ce[columns],
			ch = ce[choice].unstack().fillna(0) if choice is not None else None,
			co = co,
			ch_name = choice,
			caseindex_name=caseindex_name,
			altindex_name=altindex_name,
		)
		if autoscale_weights:
			result.autoscale_weights()
		return result

	def __repr__(self):
		return f"<larch.DataFrames ({self.n_cases} cases, {self.n_alts} alts)>"

	def info(self, verbose=False, out=None):
		"""Print info about this DataFrames.

		Parameters
		----------
		verbose : bool, default False
			Print a more verbose report
		out : file-like, optional
			A file-like object to which the report will be written. If not given,
			this method will print to stdout.
		"""
		if not self.is_computational_ready():
			print(f"larch.DataFrames:  (not computation-ready)", file=out)
		else:
			print(f"larch.DataFrames:", file=out)
		print(f"  n_cases: {self.n_cases}", file=out)
		print(f"  n_alts: {self.n_alts}", file=out)
		if self.data_ca is not None:
			if verbose:
				max_col = max(len(str(k)) for k in self.data_ca.columns)
				count = self.data_ca.count()
				dtype = self.data_ca.dtypes
				print(f"  data_ca:", file=out)
				for c,col in enumerate(self.data_ca.columns):
					print(f"    - {col:{max_col}s} ({count.iloc[c]} non-null {dtype.iloc[c]})", file=out)
			else:
				print(f"  data_ca: {len(self.data_ca.columns)} variables", file=out)
		elif self.data_ce is not None:
			if verbose:
				max_col = max(len(str(k)) for k in self.data_ce.columns)
				count = self.data_ce.count()
				dtype = self.data_ce.dtypes
				print(f"  data_ce: {len(self.data_ce)} rows", file=out)
				for c,col in enumerate(self.data_ce.columns):
					print(f"    - {col:{max_col}s} ({count.iloc[c]} non-null {dtype.iloc[c]})", file=out)
			else:
				print(f"  data_ce: {len(self.data_ce.columns)} variables, {len(self.data_ce)} rows", file=out)
		else:
			print(f"  data_ca: <not populated>", file=out)
		if self.data_co is not None:
			if verbose:
				max_col = max(len(str(k)) for k in self.data_co.columns)
				count = self.data_co.count()
				dtype = self.data_co.dtypes
				print(f"  data_co:", file=out)
				for c,col in enumerate(self.data_co.columns):
					print(f"    - {col:{max_col}s} ({count.iloc[c]} non-null {dtype.iloc[c]})", file=out)
			else:
				print(f"  data_co: {len(self.data_co.columns)} variables", file=out)
		else:
			print(f"  data_co: <not populated>", file=out)
		if self.data_av is not None:
			print(f"  data_av: {self._data_av_name or '<populated>'}", file=out)
		if self.data_ch is not None:
			print(f"  data_ch: {self._data_ch_name or '<populated>'}", file=out)
		if self.data_wt is not None:
			if self.weight_normalization < 0.99999999 or self.weight_normalization > 1.00000001:
				print(f"  data_wt: {self._data_wt_name or '<populated>'} (/ {self.weight_normalization})", file=out)
			else:
				print(f"  data_wt: {self._data_wt_name or '<populated>'}", file=out)

	@property
	def computational(self):
		return self._computational

	@computational.setter
	def computational(self, value):
		value = bool(value)
		original = self._computational
		try:
			if not self._computational and value:
				self._computational = value
				self.data_ca = self._data_ca
				self.data_ce = self._data_ce
				self.data_co = self._data_co
		except:
			self._computational = original

	cdef bint _is_computational_ready(self, bint activate) nogil:
		if self._computational:
			return True
		with gil:
			if self._data_ca is not None:
				if not _check_dataframe_of_dtype(self._data_ca, l4_float_dtype):
					return False
			if self._data_ce is not None:
				if not _check_dataframe_of_dtype(self._data_ce, l4_float_dtype):
					return False
			if self._data_co is not None:
				if not _check_dataframe_of_dtype(self._data_co, l4_float_dtype):
					return False
		if activate:
			self._computational = True
		return True

	def is_computational_ready(self, bint activate=False):
		"""Check if this DataFrames is or can be computational with no data conversion.

		Parameters
		----------
		activate : bool, default False
			If this DataFrames is computational-ready, setting this parameter as True
			will make this DataFrames computational.

		Returns
		-------
		bool
		"""
		return self._is_computational_ready(activate)

	def to_feathers(self, filename, components=None):
		"""
		Output data to a collection of Feather files.

		Parameters
		----------
		filename : path-like
			The base filename for the output files.  A collection
			of like-named files differing by only file extension
			will be created.
		components : subset of {'co','ca','ce','wt','av','ch','meta'}
			Only these data components will be exported.

		"""
		import pyarrow as pa
		import pyarrow.feather as pf
		import pickle
		try:

			if components is None:
				components = {'co','ca','ce','wt','av','ch','meta'}
			else:
				components = set(components)

			if not self.is_computational_ready(True):
				self.computational = True

			if 'meta' in components:
				alt_data = {}
				altcodes = self.alternative_codes()
				if altcodes is not None:
					alt_data['altcodes'] = altcodes
				altnames = self.alternative_names()
				if altnames is not None:
					alt_data['altnames'] = altnames
				tb = pa.table(alt_data)
				metadata = tb.schema.metadata or {}
				caseindex = self.caseindex
				if caseindex is not None:
					ci = pickle.dumps(caseindex)
					metadata[b'CASEINDEX'] = pa.compress(ci, asbytes=True)
					metadata[b'CASEINDEXBYTES'] = str(len(ci))
				if self.weight_normalization != 1.0:
					metadata[b'WGT_NORM'] = self.weight_normalization.hex()
				new_schema = tb.schema.with_metadata(metadata)
				tb = tb.cast(new_schema)
				pf.write_feather(tb, str(filename)+".metadata")

			# core data
			def segment_out(seg):
				array_ = getattr(self, f'array_{seg}')()
				if array_ is not None:
					if seg == 'ce':
						pf.write_feather(self.data_ce.reset_index(), str(filename)+f".data_ce")
					else:
						if array_.flags['F_CONTIGUOUS']:
							tb = pa.table([array_.T.reshape(-1)], [f'data_{seg}'])
							transpose = b'Y'
						else:
							tb = pa.table([array_.reshape(-1)], [f'data_{seg}'])
							transpose = b'N'
						metadata = tb.schema.metadata or {}
						metadata[b'T'] = transpose
						if 'meta' in components:
							data_ = getattr(self, f'data_{seg}')
							columns = pickle.dumps(data_.columns)
							metadata[b'COLUMNS'] = pa.compress(columns, asbytes=True)
							metadata[b'COLUMNSBYTES'] = str(len(columns))
						new_schema = tb.schema.with_metadata(metadata)
						tb = tb.cast(new_schema)
						pf.write_feather(tb, str(filename)+f".data_{seg}")
			for seg in components:
				if seg == 'meta': continue
				segment_out(seg)
			# if self.array_co() is not None and 'co' in components:
			# 	tb = pa.table([self.array_co().T.reshape(-1)], ['data_co'])
			# 	if 'meta' in components:
			# 		metadata = tb.schema.metadata or {}
			# 		columns = pickle.dumps(self.data_co.columns)
			# 		metadata[b'COLUMNS'] = pa.compress(columns, asbytes=True)
			# 		metadata[b'COLUMNSBYTES'] = str(len(columns))
			# 		new_schema = tb.schema.with_metadata(metadata)
			# 		tb = tb.cast(new_schema)
			# 	pf.write_feather(tb, str(filename)+".data_co")
			# if self.array_ca() is not None and 'ca' in components:
			# 	tb = pa.table([self.array_ca().T.reshape(-1)], ['data_ca'])
			# 	if 'meta' in components:
			# 		metadata = tb.schema.metadata or {}
			# 		columns = pickle.dumps(self.data_ca.columns)
			# 		metadata[b'COLUMNS'] = pa.compress(columns, asbytes=True)
			# 		metadata[b'COLUMNSBYTES'] = str(len(columns))
			# 		new_schema = tb.schema.with_metadata(metadata)
			# 		tb = tb.cast(new_schema)
			# 	pf.write_feather(tb, str(filename)+".data_ca")
			# if self.array_ce() is not None and 'ce' in components:
			# 	tb = pa.table([self.array_ce().T.reshape(-1)], ['data_ce'])
			# 	if 'meta' in components:
			# 		metadata = tb.schema.metadata or {}
			# 		columns = pickle.dumps(self.data_ce.columns)
			# 		metadata[b'COLUMNS'] = pa.compress(columns, asbytes=True)
			# 		metadata[b'COLUMNSBYTES'] = str(len(columns))
			# 		new_schema = tb.schema.with_metadata(metadata)
			# 		tb = tb.cast(new_schema)
			# 	pf.write_feather(tb, str(filename)+".data_ce")
			# if self.array_av() is not None and 'av' in components:
			# 	tb = pa.table([self.array_av().T.reshape(-1)], ['data_av'])
			# 	pf.write_feather(tb, str(filename)+".data_av")
			# if self.array_ch() is not None and 'ch' in components:
			# 	tb = pa.table([self.array_ch().T.reshape(-1)], ['data_ch'])
			# 	pf.write_feather(tb, str(filename)+".data_ch")
			# if self.array_wt() is not None and 'wt' in components:
			# 	tb = pa.table([self.array_wt().reshape(-1)], ['data_wt'])
			# 	pf.write_feather(tb, str(filename)+".data_wt")
		except:
			logger.exception("error in to_feathers")
			raise

	@classmethod
	def from_feathers(cls, filename, components=None):
		import pyarrow as pa
		import pyarrow.feather as pf
		import pickle
		import os

		if components is None:
			components = {'co','ca','ce','wt','av','ch', 'meta'}
		else:
			components = set(components)

		def from_metadata(tb, tag):
			raw_ = tb.schema.metadata.get(tag, None)
			len_ = int(tb.schema.metadata.get(tag+b'BYTES', 0))
			if raw_:
				return pickle.loads(pa.decompress(raw_, len_))
			else:
				return None

		filename_meta = str(filename)+".metadata"
		if not os.path.exists(filename_meta):
			raise FileNotFoundError(filename_meta)
		tb = pf.read_table(filename_meta)
		if 'altcodes' in tb.column_names:
			altcodes = tb['altcodes'].to_numpy()
		else:
			altcodes = None
		if 'altnames' in tb.column_names:
			altnames = tb['altnames'].to_numpy()
		else:
			altnames = None

		caseindex = from_metadata(tb, b'CASEINDEX')
		raw_wgtnorm = tb.schema.metadata.get(b'WGT_NORM', None)
		if raw_wgtnorm:
			wgtnorm = float.fromhex(raw_wgtnorm)
		else:
			wgtnorm = 1.0

		kwargs = {}
		def segment_in(seg):
			try:
				filename_seg = str(filename)+f".data_{seg}"
				if os.path.exists(filename_seg):
					if seg == 'ce':
						df = pf.read_feather(filename_seg)
						df = df.set_index(list(df.columns[:2]))
					else:
						tb = pf.read_table(filename_seg)
						columns = from_metadata(tb, b'COLUMNS')
						transpose = tb.schema.metadata.get(b'T', b'N')
						if transpose == b'Y':
							arr = tb[f'data_{seg}'].to_numpy().reshape(len(columns), -1).T
						else:
							arr = tb[f'data_{seg}'].to_numpy().reshape(-1, len(columns))
						if seg == 'ca':
							idx = pandas.MultiIndex.from_product([
							    caseindex,
							    altcodes,
							], names=['_caseid_', '_altid_'])
						else:
							idx = caseindex
						df = pandas.DataFrame(arr, columns=columns, index=idx)
						if seg in ('ch','wt'):
							df = df.copy() # these two arrays must be writeable
					kwargs[seg] = df
			except Exception as err:
				raise ValueError(f"error in reading '{seg}'") from err
		for seg in components:
			if seg == 'meta': continue
			segment_in(seg)

		result = cls(
			alt_codes=altcodes,
			alt_names=altnames,
			**kwargs,
		)
		result.weight_normalization = wgtnorm
		return result

	def inject_feathers(self, filename, components=None):
		"""
		Read data from a collection of Feather files.

		This method overwrites the existing arrays of data
		for the given components.  The shape and structure
		of the imported data must exactly match the existing
		data, including the number of cases.  This requirement
		is intended to allow significant speed optimizations.
		If you need to change any dimension of the data, instead
		load a new DataFrames object.

		Parameters
		----------
		filename : path-like
			The base filename for the output files.  A collection
			of like-named files differing by only file extension
			will be created.
		components : subset of {'co','ca','ce','wt','av','ch','meta'}
			Only these data components will be imported.

		"""
		import pyarrow.feather as pf
		import os

		if components is None:
			components = {'co','ca','ce','wt','av','ch'}
		else:
			components = set(components)

		# core data
		array_co = self.array_co()
		if array_co is not None and 'co' in components:
			filename_co = str(filename)+".data_co"
			tb = pf.read_table(filename_co)
			transpose = tb.schema.metadata.get(b'T', b'N')
			if transpose == b'Y':
				arr = tb['data_co'].to_numpy().reshape(array_co.shape[1], array_co.shape[0]).T
			else:
				arr = tb['data_co'].to_numpy().reshape(array_co.shape[0], array_co.shape[1])
			array_co[:] = arr[:]

		array_ca = self.array_ca()
		if array_ca is not None and 'ca' in components:
			filename_ca = str(filename)+".data_ca"
			tb = pf.read_table(filename_ca)
			transpose = tb.schema.metadata.get(b'T', b'N')
			if transpose == b'Y':
				arr = tb['data_ca'].to_numpy().reshape(array_ca.shape[2], array_ca.shape[1], array_ca.shape[0]).T
			else:
				arr = tb['data_ca'].to_numpy().reshape(array_ca.shape[0], array_ca.shape[1], array_ca.shape[2])
			array_ca[:] = arr[:]

		array_ce = self.array_ce()
		if array_ce is not None and 'ce' in components:
			filename_ce = str(filename)+".data_ce"
			df = pf.read_feather(filename_ce)
			df = df.set_index(list(df.columns[:2]))
			arr = df.to_numpy().reshape(array_ce.shape[0], array_ce.shape[1])
			array_ce[:] = arr[:]

		array_av = self.array_av()
		if array_av is not None and 'av' in components:
			filename_av = str(filename)+".data_av"
			tb = pf.read_table(filename_av)
			transpose = tb.schema.metadata.get(b'T', b'N')
			if transpose == b'Y':
				arr = tb['data_av'].to_numpy().reshape(array_av.shape[1], array_av.shape[0]).T
			else:
				arr = tb['data_av'].to_numpy().reshape(array_av.shape[0], array_av.shape[1])
			array_av[:] = arr[:]

		array_ch = self.array_ch()
		if array_ch is not None and 'ch' in components:
			filename_ch = str(filename)+".data_ch"
			tb = pf.read_table(filename_ch)
			transpose = tb.schema.metadata.get(b'T', b'N')
			if transpose == b'Y':
				arr = tb['data_ch'].to_numpy().reshape(array_ch.shape[1], array_ch.shape[0]).T
			else:
				arr = tb['data_ch'].to_numpy().reshape(array_ch.shape[0], array_ch.shape[1])
			array_ch[:] = arr[:]

		array_wt = self.array_wt()
		if array_wt is not None and 'wt' in components:
			filename_wt = str(filename)+".data_wt"
			tb = pf.read_table(filename_wt)
			arr = tb['data_wt'].to_numpy().reshape(array_wt.shape[0])
			array_wt[:] = arr[:]


	def statistics(self, title="Data Statistics", header_level=2, graph=None):
		from xmle import Reporter, NumberedCaption
		from .util.statistics import statistics_for_dataframe
		result = Reporter()
		result.hn_(header_level, title)
		if self.data_co is not None:
			result.hn_(header_level+1, 'CO Data')
			result << statistics_for_dataframe(self.data_co)
		if self.data_ca is not None:
			result.hn_(header_level+1, 'CA Data')
			result << statistics_for_dataframe(self.data_ca)
		if self.data_ce is not None:
			result.hn_(header_level+1, 'CE Data')
			result << statistics_for_dataframe(self.data_ce)
		if self.data_ch is not None and self.data_av is not None:
			result.hn_(header_level+1, 'Choices')
			result << self.choice_avail_summary(graph=graph)
		return result

	def choice_avail_summary(self, graph=None, availability_co_vars=None):
		"""
		Generate a summary of choice and availability statistics.

		Parameters
		----------
		graph : networkx.DiGraph, optional
			The nesting graph.
		availability_co_vars : dict, optional
			Also attach the definition of the availability conditions.

		Returns
		-------
		pandas.DataFrame
		"""
		try:
			if graph is None:
				if self.data_ch is not None:
					ch_ = self.data_ch.copy()
				else:
					ch_ = None
				av_ = self.data_av
			else:
				ch_ = self.data_ch_cascade(graph)
				av_ = self.data_av_cascade(graph)

			# ch_.columns = ch_.columns.droplevel(0)
			if ch_ is not None:
				ch = ch_.sum()
			else:
				ch = None

			if av_ is not None:
				av = av_.sum()
			else:
				av = None

			if self.data_wt is not None:
				if ch_ is not None:
					ch_w = pandas.Series((ch_.values * self.data_wt.values).sum(0), index=ch_.columns)
				else:
					ch_w = None
				if av_ is not None:
					av_w = pandas.Series((av_.values * self.data_wt.values).sum(0), index=av_.columns)
				else:
					av_w = None
				show_wt = numpy.any(ch != ch_w)
			else:
				ch_w = ch
				av_w = av
				show_wt = False

			if av_ is not None:
				ch_.values[av_.values > 0] = 0
			if ch_ is not None and ch_.values.sum() > 0:
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

			if graph is not None:
				od['name'] = pandas.Series(
					graph.standard_sort_names,
					index=graph.standard_sort,
				)

			if availability_co_vars is not None:
				od['availability condition'] = availability_co_vars

			result = pandas.DataFrame.from_dict(od)
			if graph is not None:
				totals = result.loc[graph.root_id, :]
				result.drop(index=graph.root_id, inplace=True)
			else:
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
		except:
			logger.exception('error in choice_avail_summary')
			raise

	def alternative_names(self):
		"""The alternative names."""
		return self._alternative_names

	def set_alternative_names(self, names:Union[Mapping,Sequence]):
		"""
		Set the alternative names.

		Parameters
		----------
		names : Mapping or Sequence
			If a mapping, with keys as the codes
			that appear in `alternative_codes`, and values that are
			the names, these will be used.  Any missing codes will be labeled with the string
			representation of the code.
			If given as a sequence, the names must be in the same order as the codes
			that appear in `alternative_codes`.
		"""
		if isinstance(names, Mapping):
			self._alternative_names = [names.get(i, str(i)) for i in self.alternative_codes()]
		elif isinstance(names, Sequence):
			self._alternative_names = names
		else:
			raise ValueError('must give a sequence or mapping')

	def alternative_codes(self):
		"""The alternative codes."""
		return self._alternative_codes

	def _ensure_consistent_alternative_codes(self, alt_codes, result='raise'):
		try:
			if alt_codes is None:
				return True
			alt_codes = pandas.Index(alt_codes)
			if self._alternative_codes is None:
				self._alternative_codes = alt_codes
				return True
			if alt_codes.shape != self._alternative_codes.shape:
				if result == 'raise':
					raise ValueError(f'shape of alt_codes ({alt_codes.shape}) does not match existing ({self._alternative_codes.shape})')
				return False
			if numpy.any(alt_codes != self._alternative_codes):
				if result == 'raise':
					raise ValueError(f'values of alt_codes does not match existing\nnew:{alt_codes}\nold:{self._alternative_codes}')
				return False
			return True
		except:
			import logging
			from .log import logger_name
			logger = logging.getLogger(logger_name)
			logger.exception('error in DataFrames._ensure_consistent_alternative_codes')
			logger.error(f"alt_codes = {alt_codes}")
			logger.error(f"self._alternative_codes = {self._alternative_codes}")
			raise

	@property
	def n_alts(self):
		"""The number of alternatives."""
		return self._n_alts()

	@property
	def n_cases(self):
		"""The number of cases."""
		return self._n_cases()

	def total_weight(self):
		"""The total weight of cases."""
		if self._data_wt is None:
			return float(self.n_cases)
		return float(self._data_wt.sum() * self._weight_normalization)

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
		elif self._data_ch is not None:
			return len(self.data_ch)
		elif self._data_av is not None:
			return len(self.data_av)
		elif self._data_wt is not None:
			return len(self.data_wt)
		else:
			return 0

	@property
	def caseindex(self):
		"""The indexes of the cases.
		"""
		if self._data_co is not None:
			return self.data_co.index
		elif self._data_ch is not None:
			return self.data_ch.index
		elif self._data_wt is not None:
			return self.data_wt.index
		elif self._data_av is not None:
			return self.data_av.index
		elif self._data_ca is not None:
			return self._data_ca.index.levels[0][numpy.unique(self._data_ca.index.codes[0])]
		elif self._data_ce is not None:
			return self._data_ce.index.levels[0][numpy.unique(self._data_ce.index.codes[0])]
		else:
			return None


	@property
	def data_ca(self):
		"""A pandas.DataFrame in idca format.

		This DataFrame should have a two-level MultiIndex as the index, where
		the first level is the caseids and the second level is the
		alternative codes.
		"""
		return self._data_ca

	@data_ca.setter
	def data_ca(self, df:pandas.DataFrame):
		if df is None:
			self._data_ca = None
			self._array_ca = None
		else:
			if isinstance(df, pandas.Series):
				df = pandas.DataFrame(df)
			if not isinstance(df, pandas.DataFrame):
				raise TypeError('data_ca must be a pandas.DataFrame or pandas.Series')
			_ensure_no_duplicate_column_names(df)

			# Check for 2 level multiindex
			if not isinstance(df.index, pandas.MultiIndex):
				raise ValueError('data_ca.index must be pandas.MultiIndex')
			if df.index.nlevels != 2:
				raise ValueError(f'data_ca.index must be a 2 level pandas.MultiIndex, not {df.index.nlevels} levels')

			# Change level names if requested
			caseindex_name = self._caseindex_name or df.index.names[0]
			altindex_name = self._altindex_name or df.index.names[1]
			df = df.set_index(df.index.set_names([caseindex_name, altindex_name]))

			if self._computational:
				self._data_ca = _ensure_dataframe_of_dtype(df, l4_float_dtype, 'data_ca')
				self._array_ca = _df_values(self.data_ca, (self.n_cases, self.n_alts, -1))
			else:
				self._data_ca = df
				self._array_ca = None
			if self._alternative_codes is None and self._data_ca is not None:
				self._alternative_codes = self._data_ca.index.levels[1]

			if self._data_ca is not None and self._data_ce is not None:
				self.data_ce = None

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
		if self._data_ce is not None:
			return self._data_ce
		if self._data_ca is None:
			return None
		if self._data_av is None:
			raise ValueError('data_av is not defined')
		return self._data_ca[self._data_av.stack().astype(bool).values]


	@property
	def data_co(self):
		"""A pandas.DataFrame in |idco| format.

		This DataFrame should have a simple :class:`pandas.Index` as the index, where
		the index values are is the caseids.
		"""
		return self._data_co

	@data_co.setter
	def data_co(self, df:pandas.DataFrame):
		if df is None:
			self._data_co = None
			self._array_co = None
		else:
			if isinstance(df, pandas.Series):
				df = pandas.DataFrame(df)
			if not isinstance(df, pandas.DataFrame):
				raise TypeError('data_co must be a pandas.DataFrame or pandas.Series')
			_ensure_no_duplicate_column_names(df)

			# Change index name if requested
			caseindex_name = self._caseindex_name or df.index.names[0]
			df = df.set_index(df.index.set_names(caseindex_name))

			if self._computational:
				self._data_co = _ensure_dataframe_of_dtype(df, l4_float_dtype, 'data_co')
				self._array_co = _df_values(self.data_co)
			else:
				self._data_co = df
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
			if isinstance(df, pandas.Series):
				df = pandas.DataFrame(df)
			if not isinstance(df, pandas.DataFrame):
				raise TypeError('data_ce must be a pandas.DataFrame or pandas.Series')
			_ensure_no_duplicate_column_names(df)

			# Check for 2 level multiindex
			if not isinstance(df.index, pandas.MultiIndex):
				raise ValueError('data_ce.index must be pandas.MultiIndex')
			if df.index.nlevels != 2:
				raise ValueError('data_ce.index must be a 2 level pandas.MultiIndex')

			# Change level names if requested
			caseindex_name = self._caseindex_name or df.index.names[0]
			altindex_name = self._altindex_name or df.index.names[1]
			df = df.set_index(df.index.set_names([caseindex_name, altindex_name]))

			if not df.index.is_monotonic_increasing:
				df = df.sort_index()
			if self._computational:
				self._data_ce = _ensure_dataframe_of_dtype(df, l4_float_dtype, 'data_ce')
				self._array_ce = _df_values(self.data_ce)
			else:
				self._data_ce = df
				self._array_ce = None

			unique_labels, new_labels =numpy.unique(self.data_ce.index.codes[0], return_inverse=True)
			self._array_ce_caseindexes = new_labels

			# min_case_x = self.data_ce.index.labels[0].min()
			# if min_case_x == 0:
			# 	self._array_ce_caseindexes = self.data_ce.index.labels[0]
			# else:
			# 	self._array_ce_caseindexes = self.data_ce.index.labels[0] - min_case_x
			self._array_ce_altindexes  = self.data_ce.index.codes[1]
			self._array_ce_reversemap = numpy.full([self._array_ce_caseindexes.max()+1, self._array_ce_altindexes.max()+1], -1, dtype=numpy.int64)
			for i in range(len(self._array_ce_caseindexes)):
				c = self._array_ce_caseindexes[i]
				a = self._array_ce_altindexes[i]
				self._array_ce_reversemap[c, a] = i
			#self._array_ce_reversemap[self._array_ce_caseindexes, self._array_ce_altindexes] = numpy.arange(len(self._array_ce_caseindexes), dtype=numpy.int64)

			if self._data_ca is not None and self._data_ce is not None:
				self.data_ca = None

	def data_ce_as_ca(self, promote=False):
		"""
		Reformat any idce data into idca format.

		This function expands the idce data into an idca format DataFrame by adding all-zero value
		rows for all unavailable alternatives.

		Parameters
		----------
		promote : str, optional
			If given, permanently change this data from idce to idca in this DataFrames,
			and add a column with this name that indicates the availability of alternatives.

		Returns
		-------
		pandas.DataFrame
		"""
		if promote:
			self.data_ce[promote] = numpy.int8(1)
		result = self.data_ce.unstack().fillna(0).stack()
		try:
			result = result.astype(self.data_ce.dtypes)
		except ValueError:
			pass
		if promote:
			self.data_ca = result
		return result

	@property
	def data_av(self):
		return self._data_av

	@data_av.setter
	def data_av(self, df:pandas.DataFrame):
		self._data_av = _ensure_dataframe_of_dtype(df, numpy.int8, 'data_av', warn_on_convert=False)
		self._array_av = _df_values(self.data_av, (self.n_cases, self.n_alts))

	def data_av_as_ce(self):
		"""
		Reformat avail data into idce format.

		Returns
		-------
		pandas.DataFrame
		"""
		if self._data_av is None:
			return None
		if self.data_ce is None:
			if self._data_av is None:
				raise NotImplementedError('not implemented when data_ce and data_av are None')
			return self._data_av[self._data_av.stack().astype(bool).values]

		arr = numpy.zeros( [len(self.data_ce)], dtype=numpy.int8 )
		cdef int c,a,row
		for c in range(self._array_ce_reversemap.shape[0]):
			for a in range(self._array_ce_reversemap.shape[1]):
				row = self._array_ce_reversemap[c,a]
				if row >= 0:
					arr[row] = self._array_av[c,a]
		return pandas.DataFrame(arr, index=self.data_ce.index, columns=['avail'])


	def data_av_cascade(self, graph):
		"""
		Create an extra wide dataframe with availability rolled up to nests.

		This has all the same elemental alternatives as the original
		`data_av` dataframe, plus columns for all nests.  For each
		nest, if any component is available, then the nest is also
		indicated as available.

		Parameters
		----------
		graph : NestingTree

		Returns
		-------
		pandas.DataFrame
		"""
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

	def set_data_ch_wide(self, df, graph):
		"""
		Write an extra wide choice dataframe with choices also on nests.

		Use this function to overwrite the existing loaded choices
		with an alternative wider set of choices.  This can be used
		to manually over-ride choices, and to make choices explicitly
		at the nest level instead of at the elemental alternative
		level.  Use with care, as some data quality checks are skipped
		when the choice array is wide.

		Parameters
		----------
		df : pandas.DataFrame
		graph : NestingTree

		"""
		self._data_ch = _ensure_dataframe_of_dtype(df, l4_float_dtype, 'data_ch')
		self._array_ch = _df_values(self.data_ch, (self.n_cases, len(graph)))

	def data_ch_cascade(self, graph):
		"""
		Create an extra wide dataframe with choices rolled up to nests.

		This has all the same elemental alternatives as the original
		`data_ch` dataframe, plus columns for all nests. For each
		nest, the chosen-ness is given as the sum of the chosen-ness
		of all nest components.

		Parameters
		----------
		graph : NestingTree

		Returns
		-------
		pandas.DataFrame
		"""

		result = pandas.DataFrame(
			data=l4_float_dtype(0),
			columns=graph.standard_sort,
			index=self.data_ch.index,
		)
		result.iloc[:,:self.data_ch.shape[1]] = self.data_ch
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
		if isinstance(df, pandas.Series):
			df=pandas.DataFrame(df)
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

	@property
	def _data_ca_or_ce_type(self):
		if self._data_ce is not None:
			return 'ce'
		else:
			return 'ca'

	@property
	def data_ca_or_ce(self):
		if self._data_ca is not None:
			return self._data_ca
		elif self._data_ce is not None:
			return self._data_ce
		return None

	def data_co_combined(self):
		"""Return a combined DataFrame in idco format that includes idco and idca data.

		Returns
		-------
		pandas.DataFrame

		"""
		ca = self.data_ca_or_ce

		if ca is not None and self._data_co is not None:
			result = pandas.DataFrame(
				self.data_co.values,
				index=self.data_co.index,
				columns=pandas.MultiIndex.from_product([self.data_co.columns, ['*']])
			).merge(
				ca.unstack(),
				left_index=True, right_index=True,
			)
		elif ca is not None:
			result = ca.unstack()
		elif self._data_co is not None:
			result = self.data_co
		return result

	def data_ca_combined(self):
		"""Return a combined DataFrame in idca format that includes idco and idca data.

		Returns
		-------
		pandas.DataFrame

		Raises
		------
		NotImplementedError
			If neither data_ca nor data_ce is defined; in this case there is nothing to
			combine and it is generally more efficient to just use data_co anyhow.
		"""
		ca = self.data_ca_or_ce
		if ca is not None and self.data_co is not None:
			idx_names = ca.index.names
			result = ca.merge(
				self.data_co,
				left_on=idx_names[0],
				right_index=True,
			)
			result.index.names = idx_names
		elif ca is not None:
			result = ca
		else:
			raise NotImplementedError("data_ca is not defined, just use data_co")
		return result

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
		return _df_values(self.data_ch, (self.n_cases, -1, ), dtype=dtype)

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
		Check that model probabilities can be found with the attached data.

		Parameters
		----------
		model : Model

		Returns
		-------
		int
			If all checks pass, zero is returned.
			
		Raises
		------
		MissingDataError
			When some critical data is missing.  This includes
			not defining `ca` or `co` data when it is needed,
			or missing particular required variables from these
			dataframes.  
		ValueError
			When the columns of `data_ch` are not aligned with
			the alternatives given in `model.graph`.
		"""
		missing_data = set()

		import logging
		from .log import logger_name
		logger = logging.getLogger(logger_name).error

		def missing(y):
			if y not in missing_data:
				logger(y)
				missing_data.add(y)

		if model._utility_ca is not None and len(model._utility_ca):
			if self.data_ca_or_ce is None:
				missing(f'idca data missing for utility')
			else:
				for i in model._utility_ca:
					if not _data_check(i.data, self._data_ca_or_ce):
						missing(f'idca utility variable missing: {i.data}')

		if model._quantity_ca is not None and len(model._quantity_ca):
			if self.data_ca_or_ce is None:
				missing(f'idca data missing for quantity')
			else:
				for i in model._quantity_ca:
					if not _data_check(i.data, self._data_ca_or_ce):
						missing(f'idca quantity variable missing: {i.data}')

		if model._utility_co is not None and len(model._utility_co):
			if self._data_co is None:
				missing(f'idco data missing for utility')
			else:
				for alt, func in model._utility_co.items():
					for i in func:
						if not _data_check(i.data, self._data_co, {'1'}):
							missing(f'idco utility variable missing: {i.data}')

		if missing_data:
			if len(missing_data) < 5:
				raise MissingDataError(f'{len(missing_data)} things missing:\n  '+'\n  '.join(str(_) for _ in missing_data))
			else:
				missing_examples = [missing_data.pop() for _ in range(5)]
				raise MissingDataError(f'{len(missing_data)+5} things missing, for example:\n  '+'\n  '.join(str(_) for _ in missing_examples))

		# check data is well aligned
		if self._data_ch is not None and model._graph is not None:
			try:
				data_eq_elementals = numpy.all(self._data_ch.columns == model._graph.elementals)
			except ValueError:
				data_eq_elementals = False
			if not data_eq_elementals:
				try:
					data_eq_fullgraph = numpy.all(self._data_ch.columns == model._graph.standard_sort)
				except ValueError:
					data_eq_fullgraph = False
				if not data_eq_fullgraph:
					raise ValueError("data_ch columns not aligned with graph.elementals or graph.standard_sort")

		return 0

	def check_data_is_sufficient_for_model( self, Model5c model, ):
		"""
		Check that probabilities can be found from the attached data.

		Parameters
		----------
		model : Model

		Raises
		------
		MissingDataError
			When some critical data is missing.  This includes
			not defining `ca` or `co` data when it is needed,
			or missing particular required variables from these
			dataframes.
		ValueError
			When the columns of `data_ch` are not aligned with
			the alternatives given in `model.graph`.
		"""
		self._check_data_is_sufficient_for_model(model)

	cdef void _link_to_model_structure(
			self,
			Model5c model,
	) except *:
		cdef:
			int j,n
			int len_model_utility_ca

		try:
			self._model = model
			self._n_model_params = len(model._frame)
			self._model_param_names = model._frame.index

			#self.du = _initialize_or_validate_shape(du, [self_.n_alts, len(model.pf)], dtype=l4_float_dtype)
			self.check_data_is_sufficient_for_model(model)

			if model._quantity_ca is not None:
				len_model_utility_ca = len(model._quantity_ca)
				self.model_quantity_ca_param_value = numpy.zeros([len_model_utility_ca], dtype=l4_float_dtype)
				self.model_quantity_ca_param_scale = numpy.ones([len_model_utility_ca], dtype=l4_float_dtype)
				self.model_quantity_ca_param_holdfast = numpy.zeros([len_model_utility_ca], dtype=numpy.int8)
				self.model_quantity_ca_param       = numpy.zeros([len_model_utility_ca], dtype=numpy.int32)
				self.model_quantity_ca_data        = numpy.zeros([len_model_utility_ca], dtype=numpy.int32)
				for n,i in enumerate(model._quantity_ca):
					self.model_quantity_ca_param[n] = model._frame.index.get_loc(str(i.param))
					self.model_quantity_ca_data [n] = self._data_ca_or_ce.columns.get_loc(str(i.data))
					self.model_quantity_ca_param_scale[n] = i.scale
				if model._quantity_scale is not None:
					self.model_quantity_scale_param = model._frame.index.get_loc(str(model._quantity_scale))
				else:
					self.model_quantity_scale_param = -1
			else:
				len_model_utility_ca = 0
				self.model_quantity_ca_param_value = numpy.zeros([len_model_utility_ca], dtype=l4_float_dtype)
				self.model_quantity_ca_param_scale = numpy.ones([len_model_utility_ca], dtype=l4_float_dtype)
				self.model_quantity_ca_param_holdfast = numpy.zeros([len_model_utility_ca], dtype=numpy.int8)
				self.model_quantity_ca_param       = numpy.zeros([len_model_utility_ca], dtype=numpy.int32)
				self.model_quantity_ca_data        = numpy.zeros([len_model_utility_ca], dtype=numpy.int32)
				self.model_quantity_scale_param = -1

			if model._utility_ca is not None:
				len_model_utility_ca = len(model._utility_ca)
				self.model_utility_ca_param_value = numpy.zeros([len_model_utility_ca], dtype=l4_float_dtype)
				self.model_utility_ca_param_scale = numpy.ones([len_model_utility_ca], dtype=l4_float_dtype)
				self.model_utility_ca_param_holdfast=numpy.zeros([len_model_utility_ca], dtype=numpy.int8)
				self.model_utility_ca_param       = numpy.zeros([len_model_utility_ca], dtype=numpy.int32)
				self.model_utility_ca_data        = numpy.zeros([len_model_utility_ca], dtype=numpy.int32)
				for n,i in enumerate(model._utility_ca):
					self.model_utility_ca_param[n] = model._frame.index.get_loc(str(i.param))
					self.model_utility_ca_data [n] = self._data_ca_or_ce.columns.get_loc(str(i.data))
					self.model_utility_ca_param_scale[n] = i.scale
			else:
				len_model_utility_ca = 0
				self.model_utility_ca_param_value = numpy.zeros([len_model_utility_ca], dtype=l4_float_dtype)
				self.model_utility_ca_param_scale = numpy.ones([len_model_utility_ca], dtype=l4_float_dtype)
				self.model_utility_ca_param_holdfast=numpy.zeros([len_model_utility_ca], dtype=numpy.int8)
				self.model_utility_ca_param       = numpy.zeros([len_model_utility_ca], dtype=numpy.int32)
				self.model_utility_ca_data        = numpy.zeros([len_model_utility_ca], dtype=numpy.int32)

			if model._utility_co is not None and len(model._utility_co):
				len_co = sum(len(_) for _ in model._utility_co.values())
				self.model_utility_co_alt         = numpy.zeros([len_co], dtype=numpy.int32)
				self.model_utility_co_param_value = numpy.zeros([len_co], dtype=l4_float_dtype)
				self.model_utility_co_param_scale = numpy.ones([len_co], dtype=l4_float_dtype)
				self.model_utility_co_param_holdfast = numpy.zeros([len_co], dtype=numpy.int8)
				self.model_utility_co_param       = numpy.zeros([len_co], dtype=numpy.int32)
				self.model_utility_co_data        = numpy.zeros([len_co], dtype=numpy.int32)

				j = 0

				param_loc = {}
				for _n,_pname in enumerate(model._frame.index):
					param_loc[_pname] = _n
				data_loc = {}
				for _n,_dname in enumerate(self._data_co.columns):
					data_loc[_dname] = _n

				for alt, func in model._utility_co.items():
					altindex = self._alternative_codes.get_loc(alt)
					for i in func:
						self.model_utility_co_alt  [j] = altindex
						self.model_utility_co_param[j] = param_loc[str(i.param)] # model._frame.index.get_loc(str(i.param))
						self.model_utility_co_param_scale[j] = i.scale
						if i.data == '1':
							self.model_utility_co_data [j] = -1
						else:
							self.model_utility_co_data [j] = data_loc[str(i.data)] # self._data_co.columns.get_loc(str(i.data))
						j += 1
			else:
				len_co = 0
				self.model_utility_co_alt         = numpy.zeros([len_co], dtype=numpy.int32)
				self.model_utility_co_param_value = numpy.zeros([len_co], dtype=l4_float_dtype)
				self.model_utility_co_param_scale = numpy.ones([len_co], dtype=l4_float_dtype)
				self.model_utility_co_param_holdfast = numpy.zeros([len_co], dtype=numpy.int8)
				self.model_utility_co_param       = numpy.zeros([len_co], dtype=numpy.int32)
				self.model_utility_co_data        = numpy.zeros([len_co], dtype=numpy.int32)

		except:
			import logging
			from .log import logger_name
			logger = logging.getLogger(logger_name)
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
				from .log import logger_name
				logger = logging.getLogger(logger_name)
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
			pvalues = self._model._frame['value'].values.astype(l4_float_dtype)
			hvalues = self._model._frame['holdfast'].values.astype(numpy.int8)

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
			from .log import logger_name
			logger = logging.getLogger(logger_name)
			logger.exception('error in DataFrames._read_in_model_parameters')
			raise

	def read_in_model_parameters(self):
		self._read_in_model_parameters()

	def _debug_arrays(self):
		from .util import dictx
		return dictx(
			model_utility_ca_param_value    = self.model_utility_ca_param_value.base,
			model_utility_ca_param_holdfast = self.model_utility_ca_param_holdfast.base,
			model_utility_co_param_value    = self.model_utility_co_param_value.base,
			model_utility_co_param_holdfast = self.model_utility_co_param_holdfast.base,
			model_quantity_scale_param_value= self.model_quantity_scale_param_value,
			model_quantity_scale_param_holdfast = self.model_quantity_scale_param_holdfast,
			model_quantity_ca_param_value   = self.model_quantity_ca_param_value.base,
			model_quantity_ca_param_holdfast= self.model_quantity_ca_param_holdfast.base,

			model_utility_ca_param   = self.model_utility_ca_param.base ,
			model_utility_ca_data    = self.model_utility_ca_data.base  ,
			model_utility_co_alt     = self.model_utility_co_alt.base   ,
			model_utility_co_param   = self.model_utility_co_param.base ,
			model_utility_co_data    = self.model_utility_co_data.base  ,
			model_quantity_ca_param  = self.model_quantity_ca_param.base,
			model_quantity_ca_data   = self.model_quantity_ca_data.base ,
			model_quantity_scale_param = self.model_quantity_scale_param,
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
			l4_float_t[:]   Q=None,
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

		if not self._is_computational_ready(activate=True):
			return

		#memset(&dU[0,0], 0, sizeof(l4_float_t) * dU.size)
		U[:] = 0
		dU[:,:] = 0
		if Q is not None:
			Q[:] = 0

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
						_temp *= self.model_quantity_ca_param_value[i] * self.model_quantity_ca_param_scale[i]
						U[j] += _temp
						if not self.model_quantity_ca_param_holdfast[i]:
							dU[j,self.model_quantity_ca_param[i]] += _temp * self.model_quantity_scale_param_value

					for i in range(self.model_quantity_ca_param.shape[0]):
						if not self.model_quantity_ca_param_holdfast[i]:
							dU[j,self.model_quantity_ca_param[i]] /= U[j]

					if Q is not None:
						Q[j] = U[j]
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
					_temp *= self.model_utility_ca_param_scale[i]
					U[j] += _temp * self.model_utility_ca_param_value[i]
					if not self.model_utility_ca_param_holdfast[i]:
						dU[j,self.model_utility_ca_param[i]] += _temp
			else:
				U[j] = -INFINITY32

		for i in range(self.model_utility_co_alt.shape[0]):
			altindex = self.model_utility_co_alt[i]
			if self._array_av[c,altindex]:
				if self.model_utility_co_data[i] == -1:
					U[altindex] += self.model_utility_co_param_value[i] * self.model_utility_co_param_scale[i]
					if not self.model_utility_co_param_holdfast[i]:
						dU[altindex,self.model_utility_co_param[i]] += self.model_utility_co_param_scale[i]
				else:
					_temp = self._array_co[c, self.model_utility_co_data[i]] * self.model_utility_co_param_scale[i]
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
			l4_float_t[:]   Q=None,
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

		if not self._is_computational_ready(activate=True):
			return

		U[:] = 0
		_max_U = 0
		if Q is not None:
			Q[:]=0

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
						_temp *= self.model_quantity_ca_param_value[i] * self.model_quantity_ca_param_scale[i]
						U[j] += _temp

					if Q is not None:
						Q[j] = U[j]
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
					_temp *= self.model_utility_ca_param_scale[i]
					U[j] += _temp * self.model_utility_ca_param_value[i]
			else:
				U[j] = -INFINITY32

		for i in range(self.model_utility_co_alt.shape[0]):
			altindex = self.model_utility_co_alt[i]
			if self._array_av[c,altindex]:
				if self.model_utility_co_data[i] == -1:
					U[altindex] += self.model_utility_co_param_value[i] * self.model_utility_co_param_scale[i]
				else:
					_temp = self._array_co[c, self.model_utility_co_data[i]] * self.model_utility_co_param_scale[i]
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



	@cython.boundscheck(False)
	@cython.initializedcheck(False)
	@cython.cdivision(True)
	cdef void _is_zero_quantity_onecase(
			self,
			int c,
			int8_t[:] Q,
			int n_alts,
	) nogil:
		"""
		Check whether each alt in a specific case has zero quantity.
		
		Parameters
		----------
		c : int
			The case index to check.
		Q : int[n_alts]
			input/output array.  Only non-zero input values are checked
		n_alts : int
			The number of alternatives to check
		"""

		cdef:
			int i,j
			int64_t row = -2
			l4_float_t  _temp

		if not self._is_computational_ready(activate=True):
			return

		for j in range(n_alts):

			if Q[j]:

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
							if _temp:
								Q[j] = 0
								break # stop searching for non-zero quant vars in this alt

				else:
					Q[j] = 0 # don't flag as non-zero if not available

	@cython.boundscheck(False)
	@cython.initializedcheck(False)
	@cython.cdivision(True)
	cdef bint _is_zero_quantity_onecase_onealt(
			self,
			int c,
			int j,
	) nogil:
		"""
		Check whether a specific case-alt has zero quantity.
		
		Parameters
		----------
		c : int
			The case index to check.
		j : int
			The alt index to check.
		"""

		cdef:
			int i
			int64_t row = -2
			l4_float_t  _temp
			bint result = True

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
					if _temp:
						result = False
						break # stop searching for non-zero quant vars in this alt

		else:
			result = False # don't flag as non-zero if not available

		return result


	def get_zero_quantity_ca(self):
		"""
		Find all alternatives with zeros across all quantity values.

		Returns
		-------
		pandas.DataFrame
		"""

		z = self.data_av.copy()

		cdef int8_t[:,:] _z_check = z.values
		cdef int ncases = self.n_cases
		cdef int n_alts = _z_check.shape[1]
		cdef int c
		cdef int q_shape

		try:
			q_shape = self.model_quantity_ca_param.shape[0]
		except AttributeError:
			raise ValueError("no quantity defined, cannot get zeros on empty set")

		if q_shape == 0:
			raise ValueError("no quantity defined, cannot get zeros on empty set")

		for c in range(ncases):
			self._is_zero_quantity_onecase( c, _z_check[c], n_alts, )

		return z


	def dump(self, filename, **kwargs):
		"""
		Persist this DataFrames object into one file.

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
		import joblib
		return joblib.dump(storage_dict, filename, **kwargs)

	@classmethod
	def load(cls, filename):
		"""
		Reconstruct a DataFrames object from a file persisted with DataFrames.dump.

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

		import joblib
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

	def scale_weights(self, scale):
		"""
		Scale the weights by a fixed exogenous value.

		If weights are embedded in the choice variable,
		they are extracted (so the total choice in each case
		is 0.0 or 1.0, and the weight is isolated in the
		data_wt terms) before any scaling is applied.

		Parameters
		----------
		scale : float
			The fixed exogenous scale to apply to the weights.

		Returns
		-------
		scale
		"""

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
			self._data_wt_name = 'computed_weight'

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

		if self._data_wt is None and scale == 1.0:
			return 1.0

		scale_level = scale

		for i in range(self._array_wt.shape[0]):
			self._array_wt[i] /= scale_level

		self._weight_normalization *= scale_level

		if scale_level < 0.99999 or scale_level > 1.00001:
			logger.warning(f'rescaled array of weights by a factor of {scale_level}')

		return scale_level

	def autoscale_weights(self):
		"""
		Scale the weights so the average weight is 1.

		If weights are embedded in the choice variable,
		they are extracted (so the total choice in each case
		is 0.0 or 1.0, and the weight is isolated in the
		data_wt terms) before any scaling is applied.

		Returns
		-------
		scale
		"""

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
			self._data_wt_name = 'computed_weight'

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
		if numpy.isnan(total_weight):
			total_weight = self.data_wt.sum()
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

	@weight_normalization.setter
	def weight_normalization(self, float value):
		if value <= 0:
			raise ValueError('weight_normalization must be strictly positive')
		self._weight_normalization = value

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
		method : {'simple', 'shuffle', 'copy'}
			If simple, the data is assumed to be adequately shuffled already and splits are
			made of contiguous sections of data.  This may be more memory efficient.  Choose 'shuffle'
			to randomly shuffle the cases while splitting; data will be copied to new memory.
			Choose 'copy' to adopt the simple approach but still copy the underlying data,
			which may help minimize "SettingWithCopy" warnings from pandas.

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

			if method == 'shuffle':
				numpy.random.shuffle(uniform_seq)

			membership = (uniform_seq.reshape(-1,1) < cum_splits.reshape(1,-1))
			membership[:,1:] ^= membership[:,:-1]

			logger.debug(f'   membership.shape {membership.shape}')
			logger.debug(f'   membership0 {membership.sum(0)}')
			membership_sum1 = membership.sum(1)
			logger.debug(f'   membership1 {membership_sum1.min()} to {membership_sum1.max()}')

			result = []
			for s in range(n_splits):
				logger.debug(f'  split {s} data prep')
				these_positions = membership[:,s].reshape(-1)
				data_co=None if self.data_co is None else self.data_co.iloc[these_positions,:]

				if self.data_ca is None:
					data_ca = None
				else:
					these_positions_2 = numpy.in1d(self.data_ca.index.codes[0], numpy.where(these_positions))
					data_ca=self.data_ca.iloc[these_positions_2,:]
				if self.data_ce is None:
					data_ce = None
				else:
					these_positions_2 = numpy.in1d(self.data_ce.index.codes[0], numpy.where(these_positions))
					data_ce=self.data_ce.iloc[these_positions_2,:]
					data_ce.index = remove_unused_level(data_ce.index, 0)

				data_av=None if self.data_av is None else self.data_av.iloc[these_positions,:]
				data_ch=None if self.data_ch is None else self.data_ch.iloc[these_positions,:]
				data_wt=None if self.data_wt is None else self.data_wt.iloc[these_positions,:]

				if method == 'copy':
					data_co=None if data_co is None else data_co.copy(deep=False)
					data_ca=None if data_ca is None else data_ca.copy(deep=False)
					data_ce=None if data_ce is None else data_ce.copy(deep=False)
					data_av=None if data_av is None else data_av.copy(deep=False)
					data_ch=None if data_ch is None else data_ch.copy(deep=False)
					data_wt=None if data_wt is None else data_wt.copy(deep=False)

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
					sys_alts=self.sys_alts,
					ch_name=self._data_ch_name,
					wt_name=self._data_wt_name,
					av_name=self._data_av_name,
				))
			logger.debug(f'done splitting dataframe {splits}')
			return result
		except:
			logger.exception('error in DataFrames.split')
			raise

	def make_idca(self, *columns, selector=None, float_dtype=numpy.float64):
		"""
		Extract a set of idca values into a new dataframe.

		Parameters
		----------
		columns : tuple of str
			A tuple (or other iterable) giving the expressions to extract
			as :ref:`idco` format variables.
		selector : None or slice
			If given, use this to slice the caseids used to build
			the array.
		float_dtype : dtype, default float64
			The dtype to use for all float-type arrays.
			This argument can only be given
			as a keyword argument.

		Other Parameters
		----------------
		dtype : str or dtype
			Describe the data type you would like the output array to adopt, probably
			numpy.int64, numpy.float64, or numpy.bool.

		Returns
		-------
		data : pandas.DataFrame
			A DataFrame with data of specified dtype
		"""
		if selector is not None:
			raise NotImplementedError
		return columnize(self._data_ca_or_ce, list(columns), inplace=False, dtype=float_dtype)

	def make_idco(self, *columns, selector=None, float_dtype=numpy.float64):
		"""
		Extract a set of idco values into a new dataframe.

		Parameters
		----------
		columns : tuple of str
			A tuple (or other iterable) giving the expressions to extract
			as :ref:`idco` format variables.
		selector : None or slice
			If given, use this to slice the caseids used to build
			the array.
		float_dtype : dtype, default float64
			The dtype to use for all float-type arrays.
			This argument can only be given
			as a keyword argument.

		Other Parameters
		----------------
		dtype : str or dtype
			Describe the data type you would like the output array to adopt, probably
			numpy.int64, numpy.float64, or numpy.bool.

		Returns
		-------
		data : pandas.DataFrame
			A DataFrame with data of specified dtype
		"""
		if selector is not None:
			raise NotImplementedError
		return columnize(self._data_co, list(columns), inplace=False, dtype=float_dtype)

	def selector_co(self, co_expr):
		"""
		Filter a DataFrames object based on an idco selector expression.

		Parameters
		----------
		co_expr : str
			An epxression to evaluate on the idco data that results in a boolean
			selection filter.

		Returns
		-------
		DataFrames
			Containing only those cases that meet the selector expression.
		"""
		selector = columnize(
			self.data_co,
			[co_expr],
			inplace=False,
			dtype=bool,
		).iloc[:,0]

		df_co = None
		df_ca = None
		df_av = None
		df_ch = None
		df_wt = None

		if self.data_co is not None:
			df_co = self.data_co[selector]

		if self.data_wt is not None:
			df_wt = self.data_wt[selector]

		if self.data_ca is not None:
			df_ca = self.data_ca.unstack()[selector].stack()
		elif self.data_ce is not None:
			df_ca = self.data_ce.unstack()[selector].stack()

		if self.data_av is not None:
			df_av = self.data_av[selector]

		if self.data_ch is not None:
			df_ch = self.data_ch[selector]

		result = DataFrames(
			co=df_co,
			# ce=df_ca, # dynamically set below
			av=df_av,
			ch=df_ch,
			wt=df_wt,
			alt_codes=self.alternative_codes(),
			alt_names=self.alternative_names(),
			sys_alts=self._systematic_alternatives,
			**{self._data_ca_or_ce_type: df_ca},
		)

		result._weight_normalization = self._weight_normalization
		return result

	def make_dataframes(
			self,
			req_data,
			*,
			selector=None,
			float_dtype=numpy.float64,
			log_warnings=True,
			explicit=False,
	):
		"""
		Create a DataFrames object that will satisfy a data request.

		Parameters
		----------
		req_data : Dict or str
			The requested data. The keys for this dictionary may include {'ca', 'co',
			'choice_ca', 'choice_co', 'choice_co_code', 'choice_any', 'weight_co', 'avail_ca',
			'avail_co', 'standardize'}. Other keys are silently ignored.
		selector : array-like[bool] or slice, optional
			If given, the selector filters the cases. This argument can only be given
			as a keyword argument.
		float_dtype : dtype, default float64
			The dtype to use for all float-type arrays.  Note that the availability
			arrays are always returned as int8 regardless of the float type.
			This argument can only be given
			as a keyword argument.
		log_warnings : bool, default True
			Emit warnings in the logger if choice, avail, or weight is not included in
			`req_data` but is set in this DataFrames and thus returned by default even
			though it was not requested.
		explicit : bool, default False
			Only include data that is explicitly requested.  If set to True, the
			choice, avail, or weight will not be included, even if set in this
			DataFrames, unless explicitly included in the request.

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
			df_ca = columnize(
				self._data_ca_or_ce,
				list(req_data['ca']),
				inplace=False,
				dtype=float_dtype,
				backing=self._data_co,
			)
		else:
			df_ca = None

		if 'co' in req_data:
			df_co = columnize(
				self._data_co,
				list(req_data['co']),
				inplace=False,
				dtype=float_dtype,
			)
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
			choicecodes = columnize(self._data_co, [req_data['choice_co_code']], inplace=False, dtype=int)
			df_ch = pandas.DataFrame(
				0,
				columns=self.alternative_codes(),
				index=self._data_co.index,
				dtype=float_dtype,
			)
			for c in df_ch.columns:
				df_ch.loc[:,c] = (choicecodes==c).astype(float_dtype)
		elif 'choice_any' in req_data:
			if self._data_ch is None:
				raise MissingDataError('req_data includes "choice_any" but no choice data is set')
			df_ch = self._data_ch
		elif self._data_ch is not None and not explicit:
			if log_warnings:
				logger.warning('req_data does not request {choice_ca,choice_co,choice_co_code} but '
							   'choice is set and being provided')
			df_ch = self._data_ch
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

		if df_wt is None and self._data_wt is not None and not explicit:
			if log_warnings:
				logger.warning(
					'req_data does not request weight_co '
					'but it is set and being provided'
				)
			df_wt = self._data_wt
			weight_normalization = self._weight_normalization

		if 'avail_ca' in req_data:
			try:
				df_av = columnize(self._data_ca_or_ce, [req_data['avail_ca']], inplace=False, dtype=numpy.int8)
			except NameError:
				df_av = self._data_av
			except MissingDataError:
				df_av = self._data_av
		elif 'avail_co' in req_data:
			alts = self.alternative_codes()
			if len(alts) == 0:
				raise ValueError("must define alternative_codes to use avail_co")
			cols = {a:req_data['avail_co'].get(a, '0') for a in alts}
			try:
				df_av = columnize(self._data_co, cols, inplace=False, dtype=numpy.int8)
			except NameError:
				logger.exception('NameError in avail_co')
				raise
				#df_av = self._data_av
			# else:
			# 	df_av.columns = alts
		elif 'avail_any' in req_data:
			if self._data_av is None:
				raise MissingDataError(
					'req_data includes "avail_any" but no availability data is set'
				)
			df_av = self._data_av
		elif self._data_av is not None and not explicit:
			if log_warnings:
				logger.warning(
					'req_data does not request avail_ca or avail_co '
					'but it is set and being provided'
				)
			df_av = self._data_av
		else:
			df_av = None

		result = DataFrames(
			co=df_co,
			# ce=df_ca, # dynamically set below
			av=df_av,
			ch=df_ch,
			wt=df_wt,
			alt_codes=self.alternative_codes(),
			alt_names=self.alternative_names(),
			sys_alts=self._systematic_alternatives,
			**{self._data_ca_or_ce_type: df_ca},
			caseindex_name=self._caseindex_name,
			altindex_name=self._altindex_name,
		)



		result._weight_normalization = weight_normalization

		if 'standardize' in req_data and req_data.standardize:
			result.standardize()

		return result

	def validate_dataservice(self, req_data):
		pass

	def new_systematic_alternatives(
			self,
			groupby,
			name='alternative_code',
			padding_levels=4,
			groupby_prefixes=None,
			overwrite=False,
			complete_features_list=None,
	):
		"""
		Create new systematic alternatives.

		Parameters
		----------
		groupby : str or list
			The column or columns that will define the new alternative codes. Every unique combination
			of values in these columns will be grouped together in such a way as to allow a nested logit
			tree to be build with the unique values defining the nest memberships.  As such, the order
			of columns given in this list is meaningful.
		name : str
			Name of the new alternative codes column.
		padding_levels : int, default 4
			Space for this number of "extra" levels is reserved in each mask, beyond the number of
			detected categories.  This is critical if these alternative codes will be used for OGEV models
			that require extra nodes at levels that are cross-nested.
		overwrite : bool, default False
			Should existing variable with `name` be overwritten. If False and it already exists, a
			'_v#' suffix is added.
		complete_features_list : dict, optional
			Give a complete, ordered enumeration of all possible feature values for each `groupby` level.
			If any level is not specifically identified, the unique values observed in this dataset are used.
			This argument can be important for OGEV models (e.g. when grouping by time slot but some time slots
			have no alternatives present) or when train and test data may not include a completely overlapping
			set of feature values.

		Returns
		-------
		pandas.DataFrames

		"""
		from .model.systematic_alternatives import new_alternative_codes

		df = self._data_ca_or_ce

		caseindex = self.data_ce.index.names[0]
		if caseindex is None:
			caseindex = "_caseid_"

		df1, sa1 = new_alternative_codes(
			self.data_ce,
			groupby=groupby,
			caseindex=caseindex,
			name=name,
			padding_levels=padding_levels,
			groupby_prefixes=groupby_prefixes,
			overwrite=overwrite,
			complete_features_list=complete_features_list,
		)

		cdef DataFrames dfs
		dfs = type(self)(
			ce = df1,
			alt_names = sa1.altnames,
			alt_codes = sa1.altcodes,
			co = self.data_co,
			av_as_ce = self.data_av_as_ce(),
			wt = self.data_wt,
			ch_as_ce = self.data_ch_as_ce(),
			sys_alts = sa1,
			ch_name = self._data_ch_name,
			wt_name = self._data_wt_name,
			av_name = self._data_av_name,
			caseindex_name=self._caseindex_name,
			altindex_name=self._altindex_name,
		)
		dfs.weight_normalization = self.weight_normalization

		return dfs

	@property
	def sys_alts(self):
		"""The SystematicAlternatives instance used to create this DataFrames."""
		return self._systematic_alternatives

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
