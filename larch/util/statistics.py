import numpy
import pandas
import enum
from .. import warning
from .histograms import sizable_histogram_figure, seems_like_discrete_data
from . import dictx
from .arraytools import scalarize
import numpy.ma as ma
from .common_functions import parse_piece


def statistics_for_array(
		arr,
		histogram=True,
		varname=None,
		ch_weights=None,
		avail=None,
		dictionary=None,
		flatten=False,
		**kwargs,
):
	"""
	Generate a statistical analysis of an array.

	Parameters
	----------
	arr : ndarray
		The array to analyze.
	histogram : bool or numeric
		If this evaluates as True, a histogram for the data array
		will be generated. If given as a number, this indicates the
		size and level of detail in the histogram.
	varname : str, optional
		A label for the x-axis of the histogram.  Only appears if
		the size is 3 or greater.  The varname is also used to
		detect piecewise linear segments, when given as
		"piece(label, lowerbound, upperbound)". Note this function
		does not *process* the array to extract a linear segment,
		this just flags the histogram generator to know that this
		is a linear segment.
	ch_weights : ndarray, optional
		If given, this indicates the choice-weights of the observations.
		Must be the same shape as `arr`. Useful to analyze whether the
		distribution of the data appears different for chosen
		alternatives than for all alternatives.
	avail : ndarray, optional
		If given, this indicates the availability of the observations.
		Must be the same shape as `arr`. If missing, all finite values
		in `arr` are considered available.
	dictionary : dict, optional
		A dictionary of {value: str} pairs, which give labels to use on
		the histogram instead of the code values stored in `arr`.
	flatten : bool, default False
		Indicates if `arr` (and `ch_weights` if given) should be
		flattened before processing.

	Other keyword parameters are passed through to
	:meth:`sizable_histogram_figure`.

	Returns
	-------
	dictx

	"""
	a = arr

	if flatten:
		a = a.reshape(-1)
		if ch_weights is not None:
			ch_weights = ch_weights.reshape(-1)

	stats = dictx()

	if varname is not None:
		stats.description = varname

	base_def, lowbound, highbound, piece_exactly = parse_piece(varname)

	try:
		nans = numpy.isnan(a)
	except TypeError:
		can_nan = False
	else:
		can_nan = True

	stats.n = scalarize(a.shape[0])
	if can_nan:
		stats.minimum = scalarize(numpy.nanmin(a, axis=0))
		stats.maximum = scalarize(numpy.nanmax(a, axis=0))
		with warning.ignore_warnings():
			stats.median = scalarize(numpy.nanmedian(a, axis=0))

	if ch_weights is not None:
		a_ch_weighted = ma.masked_array(a, mask=(ch_weights>0))
		stats.n_chosen = scalarize(a_ch_weighted.count(axis=0))
		if can_nan:
			stats.minimum_chosen = scalarize(numpy.nanmin(a_ch_weighted, axis=0))
			stats.maximum_chosen = scalarize(numpy.nanmax(a_ch_weighted, axis=0))
			with warning.ignore_warnings():
				stats.median_chosen = scalarize(numpy.nanmedian(a_ch_weighted, axis=0))

	a_masked_is_category = None

	if avail is None and can_nan:
		a_masked = ma.masked_array(a, mask=~numpy.isfinite(a))
	elif avail is not None:
		a_masked = ma.masked_array(a, mask=~avail)
	elif isinstance(a, pandas.Series) and isinstance(a.dtype, pandas.CategoricalDtype):
		a_masked = ma.masked_array(a, mask=pandas.isnull(a))
		a_masked_is_category = True
	else:
		a_masked = a

	if ch_weights is not None:
		if avail is None and can_nan:
			ch_weightsx = ma.masked_array(ch_weights, mask=~numpy.isfinite(a))
		elif avail is not None:
			ch_weightsx = ma.masked_array(ch_weights, mask=~avail)
		elif isinstance(a, pandas.Series) and isinstance(a.dtype, pandas.CategoricalDtype):
			ch_weightsx = ma.masked_array(ch_weights, mask=pandas.isnull(a))
		else:
			ch_weightsx = ch_weights
	else:
		ch_weightsx = None

	if a_masked_is_category is None:
		try:
			a_masked_is_category = isinstance(a_masked.dtype, pandas.CategoricalDtype)
		except:
			a_masked_is_category = False

	if (can_nan and histogram) or a_masked_is_category:
		with warning.ignore_warnings():

			discrete_ = kwargs.pop('discrete', None)
			try:
				if a_masked_is_category:
					discrete_ = True
			except TypeError:
				pass
			if discrete_ is None:
				discrete_ = seems_like_discrete_data(a_masked, dictionary)

			if lowbound is not None or highbound is not None:
				stats.histogram = sizable_histogram_figure(
					a_masked, sizer=histogram,
					title=None, xlabel=base_def, ylabel='Frequency',
					ch_weights=ch_weightsx,
					piecerange=(lowbound, highbound),
					**kwargs
				)
			else:
				stats.histogram = sizable_histogram_figure(
					a_masked, sizer=histogram,
					title=None, xlabel=varname, ylabel='Frequency',
					ch_weights=ch_weightsx,
					discrete=discrete_,
					**kwargs
				)
	if can_nan:
		stats.mean = scalarize(numpy.mean(a_masked, axis=0))
		stats.stdev = scalarize(numpy.std(a_masked, axis=0))
		stats.zeros = scalarize(numpy.sum(numpy.logical_not(a_masked), axis=0))
		with warning.ignore_warnings():
			stats.positives = scalarize(numpy.sum(a_masked > 0, axis=0))
			stats.negatives = scalarize(numpy.sum(a_masked < 0, axis=0))
		n_nans = scalarize(numpy.sum(numpy.isnan(a), axis=0))
		if numpy.any(n_nans):
			stats.nans = n_nans
		n_infs = scalarize(numpy.sum(numpy.isinf(a), axis=0))
		if numpy.any(n_infs):
			stats.infs = n_infs
		a_masked.mask |= ( a_masked ==0)
		stats.nonzero_minimum = scalarize(numpy.min(a_masked, axis=0))
		stats.nonzero_maximum = scalarize(numpy.max(a_masked, axis=0))
		stats.nonzero_mean = scalarize(numpy.mean(a_masked, axis=0))
		stats.nonzero_stdev = scalarize(numpy.std(a_masked, axis=0))
	elif a_masked_is_category:
		stats.nulls = pandas.isnull(a).sum()
		the_mode = tuple(a.mode().values)
		if len(the_mode) == 1:
			the_mode = the_mode[0]
		stats.mode = the_mode

	if dictionary is not None:
		stats.dictionary = dictionary

	return stats

def statistics_for_array5(arr, histogram=5, *args, **kwargs):
	"""
	Generate a statistical analysis of an array.

	Parameters
	----------
	arr : ndarray
		The array to analyze.
	histogram : bool or numeric, default 5
		If this evaluates as True, a histogram for the data array
		will be generated. If given as a number, this indicates
		the size and level of detail in the histogram.
	varname : str, optional
		A label for the x-axis of the histogram.  Only appears if
		the size is 3 or greater.  The varname is also used to
		detect piecewise linear segments, when given as
		"piece(label, lowerbound, upperbound)". Note this function
		does not *process* the array to extract a linear segment,
		this just flags the histogram generator to know that this
		is a linear segment.
	ch_weights : ndarray, optional
		If given, this indicates the choice-weights of the
		observations.  Must be the same shape as `arr`. Useful to
		analyze whether the distribution of the data appears
		different for chosen alternatives than for all alternatives.
	avail : ndarray, optional
		If given, this indicates the availability of the observations.
		Must be the same shape as `arr`. If missing, all finite values
		in `arr` are considered available.
	dictionary : dict, optional
		A dictionary of {value: str} pairs, which give labels to use
		on the histogram instead of the code values stored in `arr`.
	flatten : bool, default False
		Indicates if `arr` (and `ch_weights` if given) should be
		flattened before processing.

	Other keyword parameters are passed through to
	:meth:`sizable_histogram_figure`.

	Returns
	-------
	dictx

	"""
	return statistics_for_array(arr, histogram=histogram, *args, **kwargs)

def statistics_for_dataframe(df, histogram=True, ch_weights=None, avail=None, **kwargs):
	s = {}
	for col in df.columns:
		if isinstance(df.dtypes[col], pandas.CategoricalDtype):
			s[col] = statistics_for_array(
				df[col],
				varname=col,
				histogram=histogram,
				ch_weights=ch_weights,
				avail=avail,
				**kwargs,
			)
		elif numpy.issubdtype(df.dtypes[col], numpy.floating):
			s[col] = statistics_for_array(
				df[col],
				varname=col,
				histogram=histogram,
				ch_weights=ch_weights,
				avail=avail,
				**kwargs,
			)
		elif numpy.issubdtype(df.dtypes[col], numpy.integer):
			s[col] = statistics_for_array(
				df[col],
				varname=col,
				histogram=histogram,
				ch_weights=ch_weights,
				avail=avail,
				**kwargs,
			)
		else:
			s[col] = statistics_for_array(
				df[col],
				varname=col,
				histogram=histogram,
				ch_weights=ch_weights,
				avail=avail,
				**kwargs,
			)

	from .dataframe import DataFrameViewer
	result = DataFrameViewer.from_dict(s, orient='index')
	try:
		result = result.drop('description', axis=1)
	except:
		pass
	return result


def uniques(s, dictionary=None):
	"""
	Count the frequency for each unique value in a pandas.Series.

	Parameters
	----------
	s : pandas.Series
		The values to count.
	dictionary : Mapping or Enum, optional
		A mapping converting the actual values in the series
		to a more meaningful value (e.g. changes code numbers
		to names).  If an Enum class is passed, the values
		are treated as keys, and the names as values.

	Returns
	-------
	pandas.Series
	"""
	action = s
	len_action = len(action)
	try:
		action = action[~numpy.isnan(action)]
	except TypeError:
		num_nan = 0
	else:
		num_nan = len_action - len(action)
	x = numpy.unique(action, return_counts=True)
	if dictionary is None:
		y = pandas.Series(x[1], x[0])
	else:
		if isinstance(dictionary, enum.EnumMeta):
			dictionary = {v: k for k, v in dictionary.__members__.items()}
		y = pandas.Series(x[1], [dictionary.get(j, j) for j in x[0]])
	if num_nan:
		y[numpy.nan] = num_nan
	return y


def invmap(s, mapping):
	"""
	Apply an inverse map.

	When the values in `s` are the values of a Mapping,
	but you want them to be the keys instead.  Note
	that values need not be unique, so an arbitrary
	key with the correct value will result for non-unique
	inverse mappings.

	Parameters
	----------
	s : array-like
	mapping : Mapping or Enum, optional
		A mapping. If an Enum class is passed, the
		__members__ is used.

	Returns
	-------

	"""
	if isinstance(mapping, enum.EnumMeta):
		mapping = mapping.__members__
	inv_map = {v: k for k, v in mapping.items()}
	return s.map(inv_map)