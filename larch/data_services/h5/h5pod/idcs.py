
from .idca import *
from ... import _reserved_names_
from ...general import selector_len_for, _sqz_same
from ...service import Pods

#
# class H5PodCS(H5PodCA):
# 	"""
# 	A HDF5 group node containing implicit :ref:`idca` format data created by stacking :ref:`idco` data.
#
# 	The CA pod is includes a selection of data arrays, where
# 	each array has rows of values attributable to casewise observations, and columns attributable to
# 	individual choice alternatives.  The ordering
# 	of the data is important, and all arrays in an H5PodCA must have the same shape
# 	and be ordered the same way, so that any group of arrays can be simply concantenated to
# 	create a 3-d :ref:`idca` array of data.  The same applies for multiple H5PodCA's that are
# 	associated together in a single H5Data object.
# 	"""
# 	def __init__(self, *args, **kwargs):
# 		super().__init__(*args, **kwargs)
# 		# if the _BUNCHES_ sub-group does not exist, create it if possible
# 		from ..h5util import get_or_create_subgroup
# 		b = get_or_create_subgroup(self._groupnode, '_BUNCHES_', skip_on_readonly=True)
# 		if b is None:
# 			self._bunches = {}
# 		else:
# 			from .. import H5Vault
# 			self._bunches = H5Vault(vaultnode=b)
# 		self._altcodes = []
#
# 	@property
# 	def podtype(self):
# 		return 'idcs'
#
# 	def set_bunch(self, bunch_name, bunch_def):
# 		if not isinstance(bunch_def, dict):
# 			raise TypeError('bunch_def must be a dict of alt:varname')
# 		self._bunches[bunch_name] = bunch_def
#
# 	def set_alts(self, alts):
# 		self._altcodes = alts
# 		return self
#
# 	def _get_list_from_bunchname(self, bunchname):
# 		if not isinstance(bunchname, str):
# 			return bunchname
# 		if bunchname not in self._bunches:
# 			raise KeyError(bunchname)
# 		names_dict = self._bunches[bunchname]
# 		result = []
# 		for n in self._altcodes:
# 			if n not in names_dict:
# 				result.append('0')
# 			else:
# 				result.append(names_dict[n])
# 		return result
#
# 	def __contains__(self, item):
# 		if item in self._bunches:
# 			return True
# 		return False
#
# 	def __getitem__(self, item):
# 		if len(self._altcodes) == 0:
# 			raise ValueError("alternatives vector not set")
#
# 		if isinstance(item, tuple) and len(item)>=2 and isinstance(item[-1], slice):
# 			names, slice_ = item[:-1], item[-1]
# 		else:
# 			names = item
# 			slice_ = None
#
# 		# convert a single name string to a one item list
# 		if isinstance(names, str):
# 			names = [names, ]
#
# 		names = [self._get_list_from_bunchname(i) for i in names]
#
# 		# check that all names are actually lists or tuples of the same length
# 		first_len = len(names[0])
# 		for cmd in names[1:]:
# 			assert( len(cmd) == first_len )
#
# 		def slice_len_for(slc, seqlen):
# 			if slc is None:
# 				return seqlen
# 			start, stop, step = slc.indices(seqlen)
# 			return max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)
#
# 		dtype = numpy.float64
#
# 		result = numpy.zeros( [slice_len_for(slice_, self.shape[0]), *self.shape[1:-1], first_len, len(names)], dtype=dtype)
#
# 		for i, cmdgrp in enumerate(names):
# 			for j, cmd in enumerate(cmdgrp):
# 				result[...,j,i] = self._evaluate_single_item(cmd, slice_)
#
# 		return result
#
# 	def _load_into(self, names, slc, result):
# 		if len(self._altcodes) == 0:
# 			raise ValueError("alternatives vector not set")
# 		# convert a single name string to a one item list
# 		if isinstance(names, str):
# 			names = [names, ]
# 		names = [self._get_list_from_bunchname(i) for i in names]
#
# 		# check that all names are actually lists or tuples of the same length
# 		first_len = len(names[0])
# 		for cmd in names[1:]:
# 			assert( len(cmd) == first_len )
#
# 		def slice_len_for(slc, seqlen):
# 			if slc is None:
# 				return seqlen
# 			start, stop, step = slc.indices(seqlen)
# 			return max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)
#
# 		assert (tuple(result.shape) == tuple([slice_len_for(slc, self.shape[0]), *self.shape[1:-1], first_len, len(names)]))
# 		for i, cmdgrp in enumerate(names):
# 			for j, cmd in enumerate(cmdgrp):
# 				result[...,j,i] = self._evaluate_single_item(cmd, slc)
# 		return result
#
# 	@property
# 	def shape(self):
# 		"""The shape of the pod.
#
# 		"""
# 		if 'SHAPE' in self._groupnode._v_attrs:
# 			return tuple(self._groupnode._v_attrs['SHAPE'][:]) + (len(self._altcodes),)
# 		if len(self.names()):
# 			for v in self._groupnode._v_children.values():
# 				try:
# 					found_shape = v.shape
# 				except:
# 					pass
# 				else:
# 					try:
# 						self.shape = found_shape
# 					except:
# 						pass
# 					return tuple(found_shape) + (len(self._altcodes),)
# 		raise NoKnownShape()
#
# 	def __xml__(self):
# 		from pprint import pformat
# 		x = super().__xml__()
# 		tr = x[0].elem('tr')
# 		tr.elem('th', text='bunches', style='vertical-align:top;')
# 		td = tr.elem('td')
# 		t1 = td.elem('table', cls='dictionary')
# 		header = t1.elem('thead').elem('tr')
# 		header.elem('th', text='name')
# 		header.elem('th', text='definition')
# 		for k,v in self._bunches.items():
# 			if k not in _reserved_names_:
# 				tr1 = t1.elem('tr')
# 				tr1.elem('td', text=k)
# 				tr1.elem('td', text=pformat(v))
# 		return x



class H5PodCS(H5PodCA):
	"""
	A HDF5 group node containing implicit :ref:`idca` format data created by stacking :ref:`idco` data.

	The CA pod is includes a selection of data arrays, where
	each array has rows of values attributable to casewise observations, and columns attributable to
	individual choice alternatives.  The ordering
	of the data is important, and all arrays in an H5PodCA must have the same shape
	and be ordered the same way, so that any group of arrays can be simply concantenated to
	create a 3-d :ref:`idca` array of data.  The same applies for multiple H5PodCA's that are
	associated together in a single H5Data object.
	"""
	def __init__(self, podgroup, storage=None, ident=None, alts=None, shape=None, **kwargs):
		if isinstance(podgroup, (list,tuple)) and not isinstance(podgroup, Pods):
			podgroup = Pods(podgroup)
		if not isinstance(podgroup, Pods):
			raise TypeError('podgroup must be Pods')
		super().__init__(do_nothing=True, ident=ident, shape=shape)
		self.podgroup = podgroup
		# if the _BUNCHES_ sub-group does not exist, create it if possible
		from ..h5util import get_or_create_subgroup
		b = get_or_create_subgroup(storage, '_BUNCHES_', skip_on_readonly=True)
		if b is None:
			self._bunches = {}
		else:
			from .. import H5Vault
			self._bunches = H5Vault(vaultnode=b)
		if alts is None:
			self._altcodes = []
		else:
			self._altcodes = alts
		self._groupnode = b
		for k,v in kwargs.items():
			self.set_bunch(k,v)

	@property
	def podtype(self):
		return 'idcs'

	def set_bunch(self, bunch_name, bunch_def):
		if not isinstance(bunch_def, dict):
			raise TypeError('bunch_def must be a dict of alt:varname')
		self._bunches[bunch_name] = bunch_def

	def set_alts(self, alts):
		self._altcodes = alts
		return self

	def _get_list_from_bunchname(self, bunchname):
		if not isinstance(bunchname, str):
			return bunchname
		if bunchname not in self._bunches:
			raise KeyError(bunchname)
		names_dict = self._bunches[bunchname]
		result = []
		for n in self._altcodes:
			if n not in names_dict:
				result.append('0')
			else:
				result.append(names_dict[n])
		return result

	def __contains__(self, item):
		if item in self._bunches:
			return True
		return False

	def names(self):
		return [i for i in self._bunches.keys() if i not in _reserved_names_]

	def __getitem__(self, item):
		if len(self._altcodes) == 0:
			raise ValueError("alternatives vector not set")

		if isinstance(item, tuple) and len(item)>=2 and isinstance(item[-1], slice):
			names, selector = item[:-1], item[-1]
		else:
			names = item
			selector = None

		# convert a single name string to a one item list
		if isinstance(names, str):
			names = [names, ]

		names = [self._get_list_from_bunchname(i) for i in names]

		# check that all names are actually lists or tuples of the same length
		first_len = len(names[0])
		for cmd in names[1:]:
			assert( len(cmd) == first_len )

		dtype = numpy.float64

		result = numpy.zeros( [selector_len_for(selector, self.shape[0]), *self.shape[1:-1], first_len, len(names)], dtype=dtype)

		for i, cmdgrp in enumerate(names):
			for j, cmd in enumerate(cmdgrp):
				# if selector is None:
				# 	result[...,j,i] = self.podgroup[ cmd ].squeeze()
				# else:
				# 	result[...,j,i] = self.podgroup[ cmd, selector ].squeeze()
				self.podgroup.load_data_item(cmd, result[..., j, i], selector)
		return result

	def load_data_item(self, name, result, selector=None):
		if len(self._altcodes) == 0:
			raise ValueError("alternatives vector not set")

		names = self._get_list_from_bunchname(name)

		required_shape = (selector_len_for(selector, self.shape[0]), *self.shape[1:-1], len(names))
		_sqz_same(result.shape, required_shape, comment=f'idcs for {name}')

		for i, cmd in enumerate(names):
			# if selector is None:
			# 	result[...,i] = self.podgroup[ cmd ].squeeze()
			# else:
			# 	# result[...,i] = self.podgroup[ cmd, selector ].squeeze()
			self.podgroup.load_data_item(cmd, result[..., i], selector)
		return result



	@property
	def shape(self):
		"""The shape of the pod.

		"""
		return tuple(self.podgroup.shape) + (len(self._altcodes),)

	def __xml__(self):
		from pprint import pformat
		x = super().__xml__(no_data=True)
		tr = x[0].elem('tr')
		tr.elem('th', text='bunches', style='vertical-align:top;')
		td = tr.elem('td')
		t1 = td.elem('table', cls='dictionary')
		header = t1.elem('thead').elem('tr')
		header.elem('th', text='name')
		header.elem('th', text='definition')
		for k,v in self._bunches.items():
			if k not in _reserved_names_:
				tr1 = t1.elem('tr')
				tr1.elem('td', text=k)
				tr1.elem('td', text=pformat(v))
		return x

	def __repr__(self):
		from pprint import pformat
		from ....util.text_manip import max_len
		s = f"<larch.{self.__class__.__name__}>"
		if self._groupnode is not None:
			s += f"\n |  file: {truncate_path_for_display(self.filename)}"
			if self._groupnode._v_pathname != "/":
				s += f"\n |  node: {self._groupnode._v_pathname}"
			try:
				s += f"\n |  shape: {self.shape}"
			except NoKnownShape:
				pass
		if len(self._bunches):
			s += "\n |  bunches:"
			for k,v in self._bunches.items():
				if k not in _reserved_names_:
					s += f"\n |    {k}: "
					s += f"\n |    {' '*len(k)}  ".join(pformat(v).split('\n'))
		else:
			s += "\n |  bunches: <empty>"
		return s