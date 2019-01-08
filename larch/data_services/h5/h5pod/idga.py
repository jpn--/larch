
from .generic import *
from tables import NoSuchNodeError

class H5PodGA(H5Pod):
	"""
	A HDF5 group node containing :ref:`idga` format data.

	The GA pod is conceptually similar to the CA pod, although cases are included
	by a lookup key instead of individually.  Each array of data is stored as an array, where
	each array has rows of values attributable to arbitrary groups of casewise observations,
	and columns attributable to individual choice alternatives.  The common use case is an OMX
	matrix file containing skims, where the case-wise observations each pull a single row
	from the skims.

	The H5PodGA has one special attribute , naming the row-index to use for each case.  This
	name must exist in :ref:`idco` data.

	"""

	def __init__(self, filename, rowindexes, *args, **kwargs):
		super().__init__(*args, filename=filename, **kwargs)
		if isinstance(rowindexes, str):
			self.rowindexes = self._groupnode._v_children[rowindexes][:]
		elif isinstance(rowindexes, numpy.ndarray):
			self.rowindexes = rowindexes.squeeze()
		else:
			raise TypeError('rowindexes must be str or ndarray')

	@property
	def podtype(self):
		return 'idga'


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
			temp = self._evaluate_single_item(cmd, None)
			if slice_ is not None:
				result[...,i] = temp[self.rowindexes[slice_],:]
			else:
				result[...,i] = temp[self.rowindexes,:]
		return result

	@property
	def shape(self):
		"""The shape of the pod.

		"""
		return tuple(self.rowindexes.shape) + self.metashape[1:]

	@property
	def metashape(self):
		"""The shape of the underlying skims.

		"""
		try:
			return tuple(self._groupnode._v_attrs.SHAPE)
		except (AttributeError, NoSuchNodeError):
			return tuple(self._groupnode._v_parent._v_attrs.SHAPE)

	def load_data_item(self, name, result, selector=None):
		"""Load a slice of the pod arrays into an array in memory"""
		# convert a single name string to a one item list

		from ...general import _sqz_same
		_sqz_same(result.shape, [selector_len_for(selector, self.shape[0]), *self.shape[1:]])

		temp = self._evaluate_single_item(name, None)
		if selector is not None:
			result[:] = temp[self.rowindexes[selector], :]
		else:
			result[:] = temp[self.rowindexes, :]
		return result

