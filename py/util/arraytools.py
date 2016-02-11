

import numpy




def is_sorted_and_unique(arr):
	arr = arr.reshape(-1)
	return numpy.all(arr[:-1]<arr[1:])

def is_sorted(arr):
	arr = arr.reshape(-1)
	return numpy.all(arr[:-1]<=arr[1:])

