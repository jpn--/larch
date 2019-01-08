import re
from .generic import *

class H5PodCA(H5Pod):
	"""
	A HDF5 group node containing :ref:`idca` format data.

	The CA pod is includes a selection of data arrays, where
	each array has rows of values attributable to casewise observations, and columns attributable to
	individual choice alternatives.  The ordering
	of the data is important, and all arrays in an H5PodCA must have the same shape
	and be ordered the same way, so that any group of arrays can be simply concantenated to
	create a 3-d :ref:`idca` array of data.  The same applies for multiple H5PodCA's that are
	associated together in a single H5Data object.
	"""
	def __init__(self, filename=None, mode='a', groupnode=None, *,
	             h5f=None, inmemory=False, temp=False,
	             complevel=1, complib='zlib',
				 ident=None,
				 do_nothing=False,
				 reshape=None,
				 rename_vars=None,
				 shape=None,
				 **kwargs
	             ):
		"""
		Initialized a HDF5 group node containing :ref:`idca` format data.

		The CA pod is includes a selection of data arrays, where
		each array has rows of values attributable to casewise observations, and columns attributable to
		individual choice alternatives.  The ordering
		of the data is important, and all arrays in an H5PodCA must have the same shape
		and be ordered the same way, so that any group of arrays can be simply concantenated to
		create a 3-d :ref:`idca` array of data.  The same applies for multiple H5PodCA's that are
		associated together in a single H5Data object.

		Parameters
		----------
		filename
		mode
		groupnode
		h5f
		inmemory
		temp
		complevel
		complib
		ident
		do_nothing
		reshape : 2-tuple, optional
			Give a new shape for this pod.  If the old shape cannot be coerced into the new shape,
			will raise an error.  This is useful to coerce a dense idco pod (which has a cases by alts in a single
			column, with every case having an identical number of alts) into an idca pod.
		rename_vars : iterable, optional
			A sequence of 2-tuples, giving (pattern, replacement) that will be fed to re.sub.
			For example, give ('^','prefix_') to add prefix to all variable names, or
			('^from_this_name$','to_this_name') to change an exact name from one thing to another.
		kwargs
		"""
		super().__init__(
			filename=filename, mode=mode, groupnode=groupnode,
			h5f=h5f, inmemory=inmemory, temp=temp,
			complevel=complevel, complib=complib,
			do_nothing=do_nothing,
			ident=ident,
			shape=shape,
		)
		for k,v in kwargs.items():
			if isinstance(v, numpy.ndarray):
				self.add_array(k, v)
			else:
				raise TypeError(f"don't know how to manage keyword {k} with value type {type(v)}")

		if reshape is not None:
			self.reshape(reshape)

		if rename_vars is not None:
			for pattern, replacement in rename_vars:
				q = [(re.sub(pattern, replacement, k),k) for k in self._groupnode._v_children.keys()]
				for _to,_from in q:
					if _to != _from:
						self._groupnode._v_children[_from]._f_rename(_to)

	def reshape(self, shape):
		if not isinstance(shape, tuple):
			raise TypeError('reshape must be tuple')
		if len(shape) != 2:
			raise TypeError('reshape must be 2-tuple')
		if shape[0] == -1 and shape[1] > 0:
			shape = (int(numpy.product(self.shape) / shape[1]), shape[1])
		if shape[1] == -1 and shape[0] > 0:
			shape = (shape[0], int(numpy.product(self.shape) / shape[0]))
		if shape[0] * shape[1] != numpy.product(self.shape):
			raise ValueError(f'incompatible reshape {shape} for current shape {self.shape}')
		for k in self._groupnode._v_children.keys():
			self._groupnode._v_children[k].shape = shape
		self.shape = shape

	@property
	def podtype(self):
		return 'idca'


	def flatten(self, flat_filename=None, flat_groupnode=None, skips=(), keeps=(), overwrite=False):
		from .idco import H5PodCO
		import re

		if flat_filename is None:
			flat_filename = "{}_flat{}".format(*os.path.splitext(self.filename))

		if flat_filename == self.filename:
			raise ValueError("cannot overwrite same file (maybe one day)")

		if flat_groupnode is None:
			flat_groupnode = self._groupnode._v_pathname

		result = H5PodCO(flat_filename, 'a' if not overwrite else 'w', flat_groupnode)
		for n in self.names():
			keep_n = True
			for s in skips:
				if re.match(s,n):
					keep_n = False
					break
			if len(keeps):
				k_n = [re.match(k, n) for k in keeps]
				if not numpy.any(k_n):
					keep_n = False
			if not keep_n:
				continue
			arr = self.__getattr__(n)[:]
			flat = arr.reshape(-1)
			try:
				di = self.__getattr__(n)._v_attrs.DICTIONARY
			except AttributeError:
				di = None
			try:
				ti = self.__getattr__(n)._v_attrs.TITLE
			except AttributeError:
				ti = None
			result.add_array(n, flat, title=ti, dictionary=di)
		return result

	@classmethod
	def from_idce(cls, ce_pod, caseindex, altindex, names=None, present=None, **kwargs):

		self = cls(**kwargs)

		if names is None:
			names = set(ce_pod.names())
			names.discard(caseindex)
			names.discard(altindex)
		cx = ce_pod._groupnode._v_children[caseindex][:]
		ax = ce_pod._groupnode._v_children[altindex][:]
		shape = (cx.max()+1, ax.max()+1)
		for n in names:
			n_data = numpy.zeros(shape, ce_pod._groupnode._v_children[n].dtype)
			n_data[cx,ax] = ce_pod._groupnode._v_children[n][:]
			self.add_array(n, n_data)

		if present is not None:
			n_data = numpy.zeros(shape, bool)
			n_data[cx,ax] = True
			self.add_array(present, n_data)

		return self


	@classmethod
	def from_csv(cls, csv_filename, caseindex=None, altindex=None, caselabels=None, altlabels=None, names=None, present=None, ca_kwargs=None, **kwargs):
		"""
		Load a CSV file into a H5PodCA via H5PodCE.

		Parameters
		----------
		csv_filename
		caseindex
		altindex
		names
		present
		kwargs

		Returns
		-------

		"""
		from .idce import H5PodCE

		if caseindex is None and caselabels is None:
			raise ValueError("must give caseindex or caselabels")
		if altindex is None and altlabels is None:
			raise ValueError("must give caseindex or caselabels")

		ce = H5PodCE.from_csv(csv_filename, **kwargs)

		if caseindex is None and caselabels is not None:
			import uuid
			caseindex = str(uuid.uuid1())
			from ....warning import ignore_warnings
			with ignore_warnings():
				ce.create_indexes_from_labels(caseindex, labels_name=caselabels)

		if altindex is None and altlabels is not None:
			import uuid
			altindex = str(uuid.uuid1())
			from ....warning import ignore_warnings
			with ignore_warnings():
				ce.create_indexes_from_labels(altindex, labels_name=altlabels)

		if ca_kwargs is None:
			ca_kwargs = {}

		ca = cls.from_idce(ce, caseindex, altindex, names=names, present=present, **ca_kwargs)
		return ca