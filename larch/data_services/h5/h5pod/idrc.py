import numpy
from .idco import H5PodCO
from ...general import selector_len_for, _sqz_same

class H5PodRC(H5PodCO):

	def __init__(self, rowindexes, colindexes, *args, **kwargs):
		super().__init__(*args, **kwargs)
		if isinstance(rowindexes, str):
			self.rowindexes = self._groupnode._v_children[rowindexes][:]
		elif isinstance(rowindexes, numpy.ndarray):
			self.rowindexes = rowindexes
		elif isinstance(rowindexes, int) and isinstance(colindexes, (str, numpy.ndarray)):
			self.rowindexes = None  # fix in a moment
		else:
			raise TypeError('rowindexes must be str, int, or ndarray')

		if isinstance(colindexes, str):
			self.colindexes = self._groupnode._v_children[colindexes][:]
		elif isinstance(colindexes, numpy.ndarray):
			self.colindexes = colindexes
		elif isinstance(colindexes, int) and self.rowindexes is not None:
			self.colindexes = numpy.full_like(self.rowindexes, colindexes, dtype=int)
		else:
			raise TypeError('colindexes must be str, int, or ndarray')

		if self.rowindexes is None:
			self.rowindexes = numpy.full_like(self.colindexes, rowindexes, dtype=int)

	@property
	def podtype(self):
		return 'idrc'


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
				result[...,i] = temp[self.rowindexes[slice_],self.colindexes[slice_]]
			else:
				result[...,i] = temp[self.rowindexes,self.colindexes]
		return result

	def _load_into(self, names, slc, result):
		if len(self._altcodes) == 0:
			raise ValueError("alternatives vector not set")
		# convert a single name string to a one item list
		if isinstance(names, str):
			names = [names, ]
		names = [self._get_list_from_bunchname(i) for i in names]

		# check that all names are actually lists or tuples of the same length
		first_len = len(names[0])
		for cmd in names[1:]:
			assert( len(cmd) == first_len )

		assert (tuple(result.shape) == tuple([selector_len_for(slc, self.shape[0]), *self.shape[1:], len(names)]))
		for i, cmd in enumerate(names):
			temp = self._evaluate_single_item(cmd, None)
			if slc is not None:
				result[...,i] = temp[self.rowindexes[slc],self.colindexes[slc]]
			else:
				result[...,i] = temp[self.rowindexes,self.colindexes]
		return result

	@property
	def shape(self):
		"""The shape of the pod.

		"""
		return tuple(self.rowindexes.shape)

	@property
	def metashape(self):
		"""The shape of the underlying skims.

		"""
		return tuple(self._groupnode._v_attrs.SHAPE)

	def load_data_item(self, name, result, selector=None):
		"""Load a slice of the pod arrays into an array in memory"""
		# convert a single name string to a one item list

		from ...general import _sqz_same
		_sqz_same(result.shape, [selector_len_for(selector, self.shape[0]), *self.shape[1:]])

		temp = self._evaluate_single_item(name, None)
		if selector is not None:
			result[:] = temp[self.rowindexes[selector], self.colindexes[selector]]
		else:
			result[:] = temp[self.rowindexes, self.colindexes]
		return result

	def as_idca(self):
		ret = H5PodRCasCA(filename=self, rowindexes=self.rowindexes, colindexes=self.colindexes, ident=self.ident+"_as_idca")
		return ret

class H5PodRCasCA(H5PodRC):

	@property
	def podtype(self):
		return 'idca'

	def load_data_item(self, name, result, selector=None):
		"""Load a slice of the pod arrays into an array in memory"""
		# convert a single name string to a one item list

		from ...general import _sqz_same
		_sqz_same(result.shape, [selector_len_for(selector, self.shape[0]), *self.shape[1:]])

		temp = self._evaluate_single_item(name, None)
		if selector is not None:
			result[:,:] = temp[self.rowindexes[selector], self.colindexes[selector]][:,None]
		else:
			result[:,:] = temp[self.rowindexes, self.colindexes][:,None]
		return result

	@property
	def shape(self):
		"""The shape of the pod.

		"""
		return super().shape + (self.trailing_dim,)

	@property
	def trailing_dim(self):
		try:
			return self._trailing_dim
		except AttributeError:
			return -1

	@trailing_dim.setter
	def trailing_dim(self, value):
		self._trailing_dim = int(value)
