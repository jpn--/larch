import numpy
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

	if avail is None and can_nan:
		ax = ma.masked_array(a, mask=~numpy.isfinite(a))
	elif avail is not None:
		ax = ma.masked_array(a, mask=~avail)
	else:
		ax = a

	if ch_weights is not None:
		if avail is None and can_nan:
			ch_weightsx = ma.masked_array(ch_weights, mask=~numpy.isfinite(a))
		elif avail is not None:
			ch_weightsx = ma.masked_array(ch_weights, mask=~avail)
		else:
			ch_weightsx = ch_weights
	else:
		ch_weightsx = None

	if can_nan and histogram:
		with warning.ignore_warnings():

			discrete_ = kwargs.pop('discrete', None)
			if discrete_ is None:
				discrete_ = seems_like_discrete_data(ax, dictionary)

			if lowbound is not None or highbound is not None:
				stats.histogram = sizable_histogram_figure(
					ax, sizer=histogram,
					title=None, xlabel=base_def, ylabel='Frequency',
					ch_weights=ch_weightsx,
					piecerange=(lowbound, highbound),
					**kwargs
				)
			else:
				stats.histogram = sizable_histogram_figure(
					ax, sizer=histogram,
					title=None, xlabel=varname, ylabel='Frequency',
					ch_weights=ch_weightsx,
					discrete=discrete_,
					**kwargs
				)
	if can_nan:
		stats.mean = scalarize(numpy.mean(ax, axis=0))
		stats.stdev = scalarize(numpy.std(ax, axis=0))
		stats.zeros = scalarize(numpy.sum(numpy.logical_not(ax), axis=0))
		with warning.ignore_warnings():
			stats.positives = scalarize(numpy.sum(ax > 0, axis=0))
			stats.negatives = scalarize(numpy.sum(ax < 0, axis=0))
		n_nans = scalarize(numpy.sum(numpy.isnan(a), axis=0))
		if numpy.any(n_nans):
			stats.nans = n_nans
		n_infs = scalarize(numpy.sum(numpy.isinf(a), axis=0))
		if numpy.any(n_infs):
			stats.infs = n_infs
		ax.mask |= ( ax ==0)
		stats.nonzero_minimum = scalarize(numpy.min(ax, axis=0))
		stats.nonzero_maximum = scalarize(numpy.max(ax, axis=0))
		stats.nonzero_mean = scalarize(numpy.mean(ax, axis=0))
		stats.nonzero_stdev = scalarize(numpy.std(ax, axis=0))

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
		if numpy.issubdtype(df.dtypes[col], numpy.floating):
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