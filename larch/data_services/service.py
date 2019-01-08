import numpy
import pandas
from .pod import Pod
from .podlist import Pods, PodsCA, EmptyPodsError
from .general import _sqz_same, _sqz, selector_len_for



class DataService():
	"""A DataService is a collection of data Pod that provides data for Larch models.

	Parameters
	----------
	altids : sequence, optional
		A sequence of alternative identifiers (typically integers).
	altnames : sequence, optional
		A sequence of alternative names (typically strings). If given it must
		have the same length as the altids.
	pods : iterable of larch.Pod, optional
		Include these :class:`Pod` in the DataService. The pods can also be given as
		positional arguments to the constructor.
	"""


	def __init__(self, *args, altids=None, altnames=None, pods=None, broadcastable=True, selector=None):

		self._master_n_cases = None
		self._master_altids = [] if altids is None else altids
		self._master_altnames = [] if altnames is None else altnames

		self._pods_idco = Pods()
		self._pods_idca = PodsCA(n_alts= -1 if altids is None else len(altids) )
		self._pods_idce = Pods()

		self._pod_library = {}

		for pod in args:
			if pod is not None:
				self.add_pod(pod, broadcastable=broadcastable)

		if pods is not None:
			for pod in pods:
				self.add_pod(pod, broadcastable=broadcastable)

		self._default_selector = selector

	@property
	def selector(self):
		return self._default_selector

	@selector.setter
	def selector(self, value):
		if value is None:
			self._default_selector = None
		else:
			if not isinstance(value, (str, slice, numpy.ndarray, list, tuple)):
				raise TypeError
			if isinstance(value, str):
				self._default_selector = None
				value = self.array_idco(value, dtype=bool).squeeze()
				self._default_selector = value
			elif isinstance(value, (tuple, list)):
				self._default_selector = None
				value = self.array_idco(*value, dtype=bool).all(axis=1)
				self._default_selector = value
			else:
				if len(value)==0:
					self._default_selector = None
				else:
					self._default_selector = value

	def __getattr__(self, item):
		if item in self._pod_library:
			return self._pod_library[item]
		raise AttributeError( f"DataService object has no attribute '{item}'")

	def library_keys(self):
		return self._pod_library.keys()

	@property
	def n_cases(self):
		"""The number of cases represented by this DataService."""
		return self._master_n_cases

	@property
	def n_alts(self):
		"""The number of alternatives represented by this DataService."""
		return len(self._master_altids)

	def add_pod(self, pod:Pod, *, broadcastable=True):
		if pod.podtype in ('idco', 'idrc'):
			if self._master_n_cases is None:
				self._master_n_cases = pod.n_cases
			elif self._master_n_cases != pod.n_cases:
				raise ValueError(
					f"incompatible n_cases, have {self._master_n_cases} adding {pod.n_cases} in pod {pod!r}")
			self._pods_idco.append(pod)
			if broadcastable:
				try:
					pod_ca = pod.as_idca( )
				except AttributeError:
					pass
				else:
					self._pods_idca.append(pod_ca)

		elif pod.podtype in ('idca', 'idcs', 'idga', 'id0a'):
			if self._master_n_cases is None:
				self._master_n_cases = pod.n_cases
			elif self._master_n_cases != pod.n_cases:
				raise ValueError(
					f"incompatible n_cases, have {self._master_n_cases} adding {pod.n_cases} in pod {pod!r}")
			self._pods_idca.append(pod)
		elif pod.podtype=='idce':
			if self._master_n_cases is None:
				self._master_n_cases = pod.n_cases
			elif self._master_n_cases != pod.n_cases:
				raise ValueError(
					f"incompatible n_cases, have {self._master_n_cases} adding {pod.n_cases} in pod {pod!r}")
			self._pods_idce.append(pod)
		else:
			raise TypeError(f'podtype {pod.podtype} unknown')

		self._pod_library[pod.ident] = pod

		self._cleanup_trailing_dims()

	def _cleanup_trailing_dims(self):
		for pod in self._pods_idca:
			if pod.shape[-1] == -1:
				pod.trailing_dim = self._pods_idca.shape[-1]

	def set_alternatives(self, altids=None, altnames=None):
		if altids is not None:
			self._master_altids = altids
		if altnames is not None:
			self._master_altnames = altnames

	def alternative_codes(self):
		return self._master_altids

	def alternative_names(self):
		return self._master_altnames

	def alternative_name_dict(self):
		return {c:n for c,n in zip(self._master_altids, self._master_altnames)}

	@property
	def alternatives(self):
		"""A list of 2-tuples, each giving the id and name of an alternative"""
		return [(c,n) for c,n in zip(self._master_altids, self._master_altnames)]

	def vars_idco(self):
		return self._pods_idco.names()

	def vars_idca(self):
		return self._pods_idca.names()

	def _get_dims_of_command(self, cmd_as_utf8):
		from tokenize import tokenize, untokenize, NAME, OP, STRING
		from io import BytesIO
		g = tokenize(BytesIO(cmd_as_utf8).readline)
		# identify needed dims
		dims = 1
		for toknum, tokval, _, _, _ in g:
			if toknum == NAME and (tokval in self._pods_idca.names()):
				dims = 2
				break
		return dims


	def caseindexes(self, selector=None):
		if selector is None:
			selector = self._default_selector
		if selector is None:
			return numpy.arange(self._master_n_cases)
		else:
			return numpy.arange(self._master_n_cases)[selector]

	def array_idco(self, *vars, dtype=numpy.float64, selector=None, strip_nan=True):
		"""Extract a set of idco values into a new array.

		Parameters
		----------
		vars : tuple of str
			A tuple (or other iterable) giving the expressions (often column names, but any valid
			SQLite expression works) to extract as :ref:`idco` format variables.
		selector : None or slice
			If given, use this to slice the caseids used to build
			the array.
		strip_nan : bool
			If True (the default) then NAN values are converted to 0, and INF
			values are converted to the largest magnitude real number representable
			in the selected dtype.

		Other Parameters
		----------------
		dtype : str or dtype
			Describe the data type you would like the output array to adopt, probably
			numpy.int64, numpy.float64, or numpy.bool.

		Returns
		-------
		data : ndarray
			An array with specified dtype, of shape (n_cases,len(vars)).
		"""
		if selector is None:
			selector = self._default_selector
		result = self._pods_idco.get_data_items(vars, selector=selector, dtype=dtype)
		if strip_nan:
			result = numpy.nan_to_num(result)
		return result

	def dataframe_idco(self, *vars, selector=None, **kwargs):
		"""Extract a set of idco values into a new dataframe.

		Parameters
		----------
		vars : tuple of str
			A tuple (or other iterable) giving the expressions (often column names, but any valid
			SQLite expression works) to extract as :ref:`idco` format variables.
		selector : None or slice
			If given, use this to slice the caseids used to build
			the array.
		strip_nan : bool
			If True (the default) then NAN values are converted to 0, and INF
			values are converted to the largest magnitude real number representable
			in the selected dtype.

		Other Parameters
		----------------
		dtype : str or dtype
			Describe the data type you would like the output array to adopt, probably
			numpy.int64, numpy.float64, or numpy.bool.

		Returns
		-------
		data : pandas.DataFrame
			A DataFrame with data of specified dtype
		"""
		if selector is None:
			selector = self._default_selector
		d = self.array_idco(*vars, selector=selector, **kwargs)
		return pandas.DataFrame(
			data=d,
			columns=vars,
			index=self.caseindexes(selector=selector),
		)


	def load_idco(self, *names, arr=None, dtype=numpy.float64, selector=None, strip_nan=True, mask_pattern=0, mask_names=None, log=None, fallback_to_idce=False):
		"""Extract a set of idco values.

		Parameters
		----------
		names : tuple of str
			A tuple (or other iterable) giving the expressions (often column names, but any valid
			SQLite expression works) to extract as :ref:`idco` format variables.
		arr : array-like or None
			The array into which the data shall be loaded.
		dtype : str or dtype
			Describe the data type you would like the output array to adopt, probably
			numpy.int64, numpy.float64, or numpy.bool.
		selector : None or slice
			If given, use this to slice the caseids used to build
			the array.
		strip_nan : bool
			If True (the default) then NAN values are converted to 0, and INF
			values are converted to the largest magnitude real number representable
			in the selected dtype.
		mask_pattern : int, optional
		mask_names : array-like, optional
			An array of int32 of the same shape as `names`.  For each item, if the
			mask_names value bitwise-and the mask_pattern evaluates
			as True, then the data will be skipped. Note the default mask_pattern is 0 so
			even if mask_names is given, everything will be loaded unless the the mask_pattern
			is set to some other value. If mask_pattern is not zero but mask_names is not given,
			the mask_names will be inferred from get_durable_mask values of the component pods.
			This is an optimization tool
			for reloading data that may not have changed (e.g. for logsum generation).

		Other Parameters
		----------------

		Returns
		-------
		data : ndarray
			An array with specified dtype, of shape (n_cases,len(vars)).
		"""
		if selector is None:
			selector = self._default_selector
		try:
			if arr is None:
				arr = numpy.zeros(self.idco.shape_result(selector, names), dtype=dtype)
			else:
				if _sqz(arr.shape) != _sqz(self.idco.shape_result(selector, names)):
					import warnings
					newshape = self.idco.shape_result(selector, names)
					warnings.warn(f"load_idco: injection array is not correctly sized, re-initializing from {arr.shape} to {newshape}")
					raise ValueError
					arr = numpy.zeros(newshape, dtype=dtype)
		except EmptyPodsError:
			if fallback_to_idce:
				raise NotImplementedError
			else:
				return None

		if len(names)==0:
			return arr

		if mask_pattern != 0 and mask_names is None:
			mask_names = self.idco.get_data_masks(names)


		result = self.idco.load_data_items(names, result=arr, selector=selector, dtype=dtype,
												 mask_pattern=mask_pattern, mask_names=mask_names, log=log)
		if strip_nan:
			result = numpy.nan_to_num(result)
		return result

	def load_idce(self, *names, arr=None, dtype=numpy.float64, selector=None, strip_nan=True, mask_pattern=0, mask_names=None, log=None):
		"""Extract a set of idce values, with indexes.

		Parameters
		----------
		names : tuple of str
			A tuple (or other iterable) giving the expressions (often column names, but any valid
			SQLite expression works) to extract as :ref:`idco` format variables.
		arr : array-like or None
			The array into which the data shall be loaded.
		dtype : str or dtype
			Describe the data type you would like the output array to adopt, probably
			numpy.int64, numpy.float64, or numpy.bool.
		selector : None or slice
			If given, use this to slice the caseids used to build
			the array.
		strip_nan : bool
			If True (the default) then NAN values are converted to 0, and INF
			values are converted to the largest magnitude real number representable
			in the selected dtype.
		mask_pattern : int, optional
		mask_names : array-like, optional
			An array of int32 of the same shape as `names`.  For each item, if the
			mask_names value bitwise-and the mask_pattern evaluates
			as True, then the data will be skipped. Note the default mask_pattern is 0 so
			even if mask_names is given, everything will be loaded unless the the mask_pattern
			is set to some other value. If mask_pattern is not zero but mask_names is not given,
			the mask_names will be inferred from get_durable_mask values of the component pods.
			This is an optimization tool
			for reloading data that may not have changed (e.g. for logsum generation).

		Other Parameters
		----------------

		Returns
		-------
		data : idce_arrays
		"""
		if selector is None:
			selector = self._default_selector
		try:
			if arr is None:
				arr = numpy.zeros(self.idce.shape_result(selector, names, use_metashape=True), dtype=dtype)
			else:
				if _sqz(arr.shape) != _sqz(self.idce.shape_result(selector, names, use_metashape=True)):
					import warnings
					newshape = self.idce.shape_result(selector, names)
					warnings.warn(f"load_idce: injection array is not correctly sized, re-initializing from {arr.shape} to {newshape}")
					raise ValueError
					# arr = numpy.zeros(newshape, dtype=dtype)
		except EmptyPodsError:
			return None

		if len(names)==0:
			return None

		if mask_pattern != 0 and mask_names is None:
			mask_names = self.idce.get_data_masks(names)

		caseindexes, altindexes = self.idce.load_casealt_indexes()
		result = self.idce.load_data_items(
			names,
			result=arr,
			selector=selector,
			dtype=dtype,
			mask_pattern=mask_pattern,
			mask_names=mask_names,
			log=log,
			use_metashape=True,
		)
		if strip_nan:
			result = numpy.nan_to_num(result)

		from ..model.idce_tools import idce_arrays
		return idce_arrays(caseindexes, altindexes, result)


	def array_idca(self, *vars, dtype=numpy.float64, selector=None, strip_nan=True):
		"""Extract a set of idca values.

		Parameters
		----------
		vars : tuple of str
			A tuple (or other iterable) giving the expressions (often column names, but any valid
			SQLite expression works) to extract as :ref:`idco` format variables.
		selector : None or slice
			If given, use this to slice the caseids used to build
			the array.
		strip_nan : bool
			If True (the default) then NAN values are converted to 0, and INF
			values are converted to the largest magnitude real number representable
			in the selected dtype.

		Other Parameters
		----------------
		dtype : str or dtype
			Describe the data type you would like the output array to adopt, probably
			numpy.int64, numpy.float64, or numpy.bool.

		Returns
		-------
		data : ndarray
			An array with specified dtype, of shape (n_cases,len(vars)).
		"""
		if selector is None:
			selector = self._default_selector
		result = self._pods_idca.get_data_items(vars, selector=selector, dtype=dtype)
		if strip_nan:
			result = numpy.nan_to_num(result)
		return result

	def dataframe_idca(self, *vars, selector=None, **kwargs):
		"""Extract a set of idca values into a new dataframe.

		Parameters
		----------
		vars : tuple of str
			A tuple (or other iterable) giving the expressions (often column names, but any valid
			SQLite expression works) to extract as :ref:`idca` format variables.
		selector : None or slice
			If given, use this to slice the caseids used to build
			the array.
		strip_nan : bool
			If True (the default) then NAN values are converted to 0, and INF
			values are converted to the largest magnitude real number representable
			in the selected dtype.

		Other Parameters
		----------------
		dtype : str or dtype
			Describe the data type you would like the output array to adopt, probably
			numpy.int64, numpy.float64, or numpy.bool.

		Returns
		-------
		data : pandas.DataFrame
			A DataFrame with data of specified dtype
		"""
		if selector is None:
			selector = self._default_selector
		d = self.array_idca(*vars, selector=selector, **kwargs)
		d = d.reshape(d.shape[0]*d.shape[1], -1)

		rows = pandas.MultiIndex.from_product(
			[self.caseindexes(selector=selector), self.alternative_codes(), ],
			names=['case', 'alt', ]
		)

		return pandas.DataFrame(
			data=d,
			columns=vars,
			index=rows,
		)

	def load_idca(self, *names, arr=None, dtype=numpy.float64, selector=None, strip_nan=True, mask_pattern=0, mask_names=None, log=None, fallback_to_idce=False):
		"""Extract a set of idca values.

		Parameters
		----------
		names : tuple of str
			A tuple (or other iterable) giving the expressions (often column names, but any valid
			SQLite expression works) to extract as :ref:`idco` format variables.
		arr : array-like or None
			The array into which the data shall be loaded.
		dtype : str or dtype
			Describe the data type you would like the output array to adopt if not given, probably
			numpy.int64, numpy.float64, or numpy.bool.
		selector : None or slice
			If given, use this to slice the caseids used to build
			the array.
		strip_nan : bool
			If True (the default) then NAN values are converted to 0, and INF
			values are converted to the largest magnitude real number representable
			in the selected dtype.
		mask_pattern : int, optional
		mask_names : array-like, optional
			An array of int32 of the same shape as `names`.  For each item, if the
			mask_names value bitwise-and the mask_pattern evaluates
			as True, then the data will be skipped. Note the default mask_pattern is 0 so
			even if mask_names is given, everything will be loaded unless the the mask_pattern
			is set to some other value. If mask_pattern is not zero but mask_names is not given,
			the mask_names will be inferred from get_durable_mask values of the component pods.
			This is an optimization tool
			for reloading data that may not have changed (e.g. for logsum generation).

		Returns
		-------
		data : ndarray
			An array with specified dtype, of shape (n_cases,len(vars)).
		"""
		if selector is None:
			selector = self._default_selector
		try:
			if arr is None:
				arr = numpy.zeros(self.idca.shape_result(selector, names), dtype=dtype)
			else:
				out_shape = self.idca.shape_result(selector, names)
				if _sqz(arr.shape) != _sqz(out_shape):
					if (arr.shape[0]!=out_shape[0]) or (arr.shape[1] < out_shape[1]):
						import warnings
						newshape = self.idca.shape_result(selector, names)
						warnings.warn(f"load_idca: injection array is not correctly sized, re-initializing from {arr.shape} to {newshape}")
						raise ValueError
						arr = numpy.zeros(newshape, dtype=dtype)
		except EmptyPodsError:
			if fallback_to_idce:
				temp = self.load_idce(*names, arr=None, dtype=dtype, selector=selector, strip_nan=strip_nan,
									  mask_pattern=mask_pattern, mask_names=mask_names, log=log)
				return temp.as_idca()
			else:
				return None

		if len(names)==0:
			return arr

		if mask_pattern != 0 and mask_names is None:
			mask_names = self.idca.get_data_masks(names)

		try:
			result = self.idca.load_data_items(names, result=arr, selector=selector, dtype=dtype,
												 mask_pattern=mask_pattern, mask_names=mask_names, log=log)
		except NameError:
			if fallback_to_idce:
				temp = self.load_idce(*names, arr=None, dtype=dtype, selector=selector, strip_nan=strip_nan,
									  mask_pattern=mask_pattern, mask_names=mask_names, log=log)
				result[:] = temp.as_idca()
			else:
				raise

		if strip_nan:
			result = numpy.nan_to_num(result)
		return result

	def __repr__(self):
		s = "<larch.DataService>"
		if len(self._pods_idco):
			s += "\n | idco:"
			for dat in self._pods_idco:
				s += "\n |   "
				s += repr(dat).replace("\n","\n |   ")
		if len(self._pods_idca):
			s += "\n | idca:"
			for dat in self._pods_idca:
				s += "\n |   "
				s += repr(dat).replace("\n","\n |   ")
		if len(self._pods_idce):
			s += "\n | idce:"
			for dat in self._pods_idce:
				s += "\n |   "
				s += repr(dat).replace("\n","\n |   ")
		return s

	@property
	def idco(self):
		return self._pods_idco

	@property
	def idce(self):
		return self._pods_idce

	@property
	def idca(self):
		return self._pods_idca

	def idca_reset(self, pull_idco=False):
		self.idca.clear()
		self.idca.n_alts = self.n_alts
		if pull_idco:
			for ident, pod in self._pod_library.items():
				try:
					pod_ca = pod.as_idca( )
				except AttributeError:
					pass
				else:
					self._pods_idca.append(pod_ca)


	@classmethod
	def from_CE_and_SA(cls, ce, sa, *, co=None):
		from .h5 import H5PodCO
		return cls(
			ce,
			co,
			H5PodCO(shape=(ce.shape[0],)),
			altids=sa.altcodes,
			altnames=sa.altnames,
			broadcastable=False,
		)


	def make_dataframes(self, req_data, *, selector=None, float_dtype=numpy.float64):

		if isinstance(req_data, str):
			from ..util import Dict
			import textwrap
			req_data = Dict.load(textwrap.dedent(req_data))

		if 'ca' in req_data:
			df_ca = self.dataframe_idca(*req_data['ca'], dtype=float_dtype, selector=selector)
		else:
			df_ca = None

		if 'co' in req_data:
			df_co = self.dataframe_idco(*req_data['co'], dtype=float_dtype, selector=selector)
		else:
			df_co = None

		if 'choice_ca' in req_data:
			df_ch = self.dataframe_idca(req_data['choice_ca'], dtype=float_dtype, selector=selector)
		elif 'choice_co' in req_data:
			raise NotImplementedError('choice_co')
		elif 'choice_co_code' in req_data:
			raise NotImplementedError('choice_co_code')
		else:
			df_ch = None

		if 'weight_co' in req_data:
			df_wt = self.dataframe_idco(req_data['weight_co'], dtype=float_dtype, selector=selector)
		else:
			df_wt = None

		if 'avail_ca' in req_data:
			df_av = self.dataframe_idca(req_data['avail_ca'], dtype=numpy.int8, selector=selector)
		elif 'avail_co' in req_data:
			raise NotImplementedError('avail_co')
		else:
			df_av = None

		from ..dataframes import DataFrames

		result = DataFrames(
			co=df_co,
			ca=df_ca,
			av=df_av,
			ch=df_ch,
			wt=df_wt,
			alt_codes=self.alternative_codes(),
			alt_names=self.alternative_names(),
		)

		if 'standardize' in req_data and req_data.standardize:
			result.standardize()

		return result


def validate_dataservice(dataservice, req_data):
	"""
	Check if an object is a sufficient dataservice.
	"""

	if isinstance(req_data, str):
		from ..util import Dict
		import textwrap
		req_data = Dict.load(textwrap.dedent(req_data))

	missing_methods = set()

	if 'ca' in req_data:
		if not hasattr(dataservice, 'dataframe_idca'):
			missing_methods.add('dataframe_idca')

	if 'co' in req_data:
		if not hasattr(dataservice, 'dataframe_idco'):
			missing_methods.add('dataframe_idco')

	if 'choice_ca' in req_data:
		if not hasattr(dataservice, 'dataframe_idca'):
			missing_methods.add('dataframe_idca')
	elif 'choice_co' in req_data:
		raise NotImplementedError('choice_co')
	elif 'choice_co_code' in req_data:
		raise NotImplementedError('choice_co_code')

	if 'weight_co' in req_data:
		if not hasattr(dataservice, 'dataframe_idco'):
			missing_methods.add('dataframe_idco')

	if 'avail_ca' in req_data:
		if not hasattr(dataservice, 'dataframe_idca'):
			missing_methods.add('dataframe_idca')
	elif 'avail_co' in req_data:
		raise NotImplementedError('avail_co')

	if len(missing_methods)>0:
		raise ValueError('dataservice is missing '+", ".join(missing_methods))

