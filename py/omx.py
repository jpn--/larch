
import tables as _tb
import numpy
from .core import LarchError
from .util import dicta

class OMXBadFormat(LarchError):
	pass

class OMXIncompatibleShape(OMXBadFormat):
	pass


class OMX(_tb.file.File):

	def __init__(self, *arg, complevel=5, complib='zlib', **kwarg):
		if 'filters' in kwarg:
			super().__init__(*arg, **kwarg)
		else:
			super().__init__(*arg, filters=_tb.Filters(complib=complib, complevel=complevel), **kwarg)
		try:
			self.data = self._getOrCreatePath("/data", True)
		except _tb.exceptions.FileModeError:
			raise OMXBadFormat("the '/data' node does not exist and cannot be created")
		try:
			self.lookup = self._getOrCreatePath("/lookup", True)
		except _tb.exceptions.FileModeError:
			raise OMXBadFormat("the '/lookup' node does not exist and cannot be created")
		if 'OMX_VERSION' not in self.root._v_attrs:
			try:
				self.root._v_attrs.OMX_VERSION = "0.2"
			except _tb.exceptions.FileModeError:
				raise OMXBadFormat("the root OMX_VERSION attribute does not exist and cannot be created")
		if 'SHAPE' not in self.root._v_attrs:
			try:
				self.root._v_attrs.SHAPE = numpy.zeros(2, dtype=int)
			except _tb.exceptions.FileModeError:
				raise OMXBadFormat("the root SHAPE attribute does not exist and cannot be created")
		self.rlookup = dicta()
		self.rlookup._helper = self.get_reverse_lookup

	@property
	def shape(self):
		sh = self.root._v_attrs.SHAPE[:]
		return (sh[0],sh[1])

	def add_matrix(self, name, obj, **kwargs):
		if len(obj.shape) != 2:
			raise OMXIncompatibleShape('all omx arrays must have 2 dimensional shape')
		if self.data._v_nchildren>0:
			if obj.shape != self.shape:
				raise OMXIncompatibleShape('this omx has shape {!s} but you want to add {!s}'.format(self.shape, obj.shape))
		if self.data._v_nchildren == 0:
			shp = numpy.empty(2, dtype=int)
			shp[0] = obj.shape[0]
			shp[1] = obj.shape[1]
			self.root._v_attrs.SHAPE = shp
		return self.create_carray(self.data, name, obj=obj, **kwargs)

	def add_lookup(self, name, obj, **kwargs):
		if len(obj.shape) != 1:
			raise OMXIncompatibleShape('all omx lookups must have 1 dimensional shape')
		if self.data._v_nchildren>0:
			if obj.shape[0] not in self.shape:
				raise OMXIncompatibleShape('this omx has shape {!s} but you want to add a lookup with {!s}'.format(self.shape, obj.shape))
		if self.data._v_nchildren == 0:
			raise OMXIncompatibleShape("don't add lookup to omx with no data")
		return self.create_carray(self.lookup, name, obj=obj, **kwargs)

	def get_reverse_lookup(self, name):
		labels = self.lookup._v_children[name][:]
		label_to_i = dict(enumerate(labels))
		self.rlookup[name] = label_to_i
		return label_to_i



