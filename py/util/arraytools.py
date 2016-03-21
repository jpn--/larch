

import numpy
from ..core import LarchError


class NonUniqueLookup(LarchError):
	pass


def is_sorted_and_unique(arr):
	arr = arr.reshape(-1)
	return numpy.all(arr[:-1]<arr[1:])

def is_sorted(arr):
	arr = arr.reshape(-1)
	return numpy.all(arr[:-1]<=arr[1:])

def label_to_index(labels, arr):
	"""Convert an array of lookup-able values into indexes.
	
	If you have an array of lookup-able values (e.g., TAZ identifiers) and you
	want to convert them to 0-based indexes for use in accessing matrix data,
	this is the function for you.
	
	Parameters
	----------
	labels : 1d array-like
		An array of labels.
	arr : array-like
		An array of values that appear in the label array. This method uses
		numpy.digitize to process values, so any target value that appears in `arr` but does
		not appear in the labels will be assigned the index of the smallest label
		value that is greater than the target, or the maximum label value if no label value
		is greater than the target.
		
	Returns
	-------
	array
		An array of index (int) values, with the same shape as `arr`.
		
	Raises
	------
	OMXNonUniqueLookup
		When the lookup does not contain a set of unique values, this tool is not appropriate.
	
	"""
	try:
		if len(labels)==0:
			return labels
	except TypeError:
		return labels
	labels = numpy.asarray(labels)
	if is_sorted_and_unique(labels):
		return numpy.digitize(arr, labels, right=True)
	uniq_labels, uniq_indexes = numpy.unique(labels, return_inverse=True)
	if len(uniq_labels) != len(labels):
		raise NonUniqueLookup("lookup '{}' does not have unique labels for each item".format(lookupname))
	index_malordered = numpy.digitize(arr, uniq_labels, right=True)
	return uniq_indexes[index_malordered]
