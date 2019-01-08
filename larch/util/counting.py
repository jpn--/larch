
import pandas, numpy
from ..data_services.h5.h5pod.generic import CArray

def tally(arr):

	if isinstance(arr, CArray):
		arr_ = arr
		arr = arr[:]
		values, freqs = numpy.unique(arr[~pandas.isnull(arr)], return_counts=True)
		try:
			d = arr_.DICTIONARY
		except:
			d = {}
		s = pandas.Series(freqs, index=[d.get(i,i) for i in values]).sort_values(ascending=False)
		nan_count = pandas.isnull(arr).sum()
		if nan_count:
			s[numpy.NaN] = nan_count
		return s
	else:
		values, freqs = numpy.unique(  arr[~pandas.isnull(arr)], return_counts=True)
		s = pandas.Series(freqs, index=values).sort_values(ascending=False)
		nan_count = pandas.isnull(arr).sum()
		if nan_count:
			s[numpy.NaN] = nan_count
		return s

def tally_weighted(arr, wgt):

	if isinstance(arr, CArray):
		raise NotImplementedError()
	else:
		arr = arr.reshape(-1)
		wgt = wgt.reshape(-1)

		values, inv = numpy.unique(arr[~pandas.isnull(arr)], return_inverse=True)
		s = pandas.Series(numpy.bincount(inv, wgt), index=values).sort_values(ascending=False)
		nan_count = wgt[pandas.isnull(arr)].sum()
		if nan_count:
			s[numpy.NaN] = nan_count
		return s