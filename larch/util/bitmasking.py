import numpy, pandas


def define_masks(bitmask_sizes):
	"""

	Parameters
	----------
	bitmask_sizes : iterable

	Returns
	-------
	list
	"""

	shifts = ((numpy.cumsum(bitmask_sizes[::-1]))[::-1]).astype(int)
	shifts[:-1] = shifts[1:]
	shifts[-1] = 0

	masks = [
		((2**bitmask_sizes[b])-1)
		<<shifts[b]
		for b in range(len(bitmask_sizes))
	]

	return masks


def define_min_masks_from_sets( sets ):
	"""

	Parameters
	----------
	sets : iterable of iterable

	Returns
	-------
	list
	"""
	bitmask_sizes = [ int(numpy.ceil(numpy.log2(len(numpy.unique(s))))) for s in sets ]
	return define_masks(bitmask_sizes)


def define_min_masks_from_df( df ):
	"""

	Parameters
	----------
	df : pandas.DataFrame

	Returns
	-------
	list
	"""
	bitmask_sizes = [ int(numpy.ceil(numpy.log2(len(numpy.unique(df[s]))))) for s in df.columns ]
	return define_masks(bitmask_sizes)
