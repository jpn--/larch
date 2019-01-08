
from .generic import *

class H5Pod0A(H5Pod):
	"""
	A HDF5 group node containing :ref:`id0a` format data.

	The 0A pod is conceptually similar to the CA pod, although cases share a single
	row of data instead of varying individually.  Each array of data is stored as an array, where
	each array has one row of values attributable to all observations,
	and columns attributable to individual choice alternatives.  The common use case is an OMX
	matrix file containing lookups, where the observations all share a single lookup.

	"""

	def __init__(self, filename, n_cases=-1, n_alts=-1, *args, **kwargs):
		super().__init__(*args, filename=filename, **kwargs)
		self._n_cases = n_cases
		self._n_alts = n_alts

	@property
	def podtype(self):
		return 'id0a'


	def __getitem__(self, item):

		if isinstance(item, tuple) and len(item)>=2 and isinstance(item[-1], slice):
			names, slice_ = item[:-1], item[-1]
		else:
			names = item
			slice_ = None

		# convert a single name string to a one item list
		if isinstance(names, str):
			names = [names, ]

		dtype = numpy.float64

		result = numpy.zeros( [selector_len_for(slice_, self.shape[0]), *self.shape[1:], len(names)], dtype=dtype)

		for i, cmd in enumerate(names):
			temp = self._evaluate_single_item(cmd, None).squeeze()
			result[...,i] = temp[None,:]
		return result

	@property
	def shape(self):
		"""The shape of the pod.

		"""
		return (self._n_cases,) + self.metashape[1:]

	@property
	def metashape(self):
		"""The shape of the underlying skims.

		"""
		return (1, self._n_alts)

	def load_data_item(self, name, result, selector=None):
		"""Load a slice of the pod arrays into an array in memory"""
		# convert a single name string to a one item list

		from ...general import _sqz_same
		_sqz_same(result.shape, [selector_len_for(selector, self.shape[0]), *self.shape[1:]])

		temp = self._evaluate_single_item(name, None).squeeze()
		if selector is not None:
			result[:] = temp[None, :]
		else:
			result[:] = temp[None, :]
		return result

