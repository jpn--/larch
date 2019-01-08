import numpy
from .exceptions import NoKnownShape, NoKnownType
from ..util.text_manip import truncate_path_for_display
from .general import selector_len_for
import uuid

class Pod():
	"""
	This is the abstract base class for Pods.
	"""

	def __init__(self, ident=None):
		if ident is None:
			self.ident = uuid.uuid1()
		else:
			self.ident = ident

	def load_data_item(self, name, result, selector=None):
		"""
		Load data into an existing array.

		The existing array must be the correct shape to receive the data. Errors can
		occur if the existing array is the incorrect shape.

		Parameters
		----------
		name : str
			The identifier for the data that will be loaded. This can be
			the natural name of a data item in this pod, or some expression
			that can be evaluated. The exact form of allowable expressions is
			dependent on the particular Pod implementation.
		result : ndarray
			This is the array (typically a slice of an array)
			into which the data will be loaded. It must be given and it
			must be the correct shape.
		selector : slice or array-like, optional
			This will slice the first dimension (cases) of the result.
			There may be a performace hit for using slices.

		Raises
		------
		NameError
			This pod cannot independently resolve the given name.
		"""
		raise NotImplementedError("overload this function")

	def _shape_for_get_data_item(self, selector=None):
		return [selector_len_for(selector, self.shape[0]), *self.shape[1:]]

	def get_data_item(self, name, selector=None, dtype=None):
		"""
		Get data in a new array.

		Parameters
		----------
		name : str
			The identifier for the data that will be loaded. This can be
			the natural name of a data item in this pod, or some expression
			that can be evaluated. The exact form of allowable expressions is
			dependent on the particular Pod implementation.
		selector : slice or array-like, optional
			This will slice the first dimension (cases) of the result.
			There may be a performace hit for using slices.
		dtype : dtype, optional
			The dtype for the array to return. If the dtype is not given and not
			inferable from the data itself, float64 will be used.

		Returns
		-------
		result : ndarray
			This is the array (typically a slice of an array)
			into which the data has beem loaded.

		Raises
		------
		NameError
			This pod cannot independently resolve the given name.
		"""
		try:
			natural_dtype = self.dtype_of(name)
		except:
			natural_dtype = numpy.float64
		result = numpy.zeros(self._shape_for_get_data_item(selector), dtype=dtype or natural_dtype)
		self.load_data_item(name, result, selector=selector)
		return result

	def get_data_dictionary(self, name):
		raise NotImplementedError("overload this function")

	@property
	def podtype(self):
		raise NotImplementedError("overload this function")

	@property
	def dims(self):
		"""The number of dimensions in a full data item"""
		raise NotImplementedError("overload this function")

	@property
	def shape(self):
		"""The shape of a full data item available in this pod"""
		raise NotImplementedError("overload this function")

	@shape.setter
	def shape(self, x):
		raise NotImplementedError("overload this function")

	@property
	def n_cases(self):
		"""The size of the first dimension in a full data item"""
		return self.shape[0]

	def names(self):
		"""Natural names of data items available in this pod"""
		raise NotImplementedError("overload this function")

	def nameset(self):
		"""Natural names of data items available in this pod, as a set"""
		n = self.names()
		if isinstance(n, set):
			return n
		return set(n)

	def __contains__(self, item):
		return (item in self.nameset())

	def dtype_of(self, name):
		"""dtype of raw data for a particular named data item"""
		raise NotImplementedError("overload this function")

	@property
	def filename(self):
		"""The filename of the underlying file (read-only)"""
		raise NotImplementedError("overload this function")

	@property
	def internalname(self):
		raise NotImplementedError("overload this function")


	def __repr__(self):
		from ..util.text_manip import max_len
		s = f"<larch.data_services.{self.__class__.__name__}>"
		try:
			filename = self.filename
		except NotImplementedError:
			pass
		else:
			s += f"\n |  file: {truncate_path_for_display(filename)}"
		try:
			internalnametype, internalnamevalue = self.internalname
		except NotImplementedError:
			pass
		else:
			s += f"\n |  {internalnametype}: {internalnamevalue}"
		try:
			shape = self.shape
		except (AttributeError, NoKnownShape, NotImplementedError):
			pass
		else:
			s += f"\n |  shape: {shape}"
		names = self.names()
		if len(names):
			s += "\n |  data:"
			just = max_len(names)
			for i in names:
				try:
					node_dtype = self.dtype_of(i)
				except NoKnownType:
					node_dtype = "<no dtype>"
				s += "\n |    {0:{2}s} ({1})".format(i, node_dtype, just)
		else:
			s += "\n |  data: <empty>"
		return s

	@property
	def durable_mask(self):
		"""A mask identifier for this pod, used to optimize reloading data"""
		try:
			return self._loading_mask
		except AttributeError:
			return numpy.uint32(0)

	@durable_mask.setter
	def durable_mask(self, value):
		self._loading_mask = numpy.uint32(value)

	@property
	def ident(self):
		"""An identifier for this pod, typically a `str`"""
		return self._ident

	@ident.setter
	def ident(self, x):
		self._ident = str(x)

