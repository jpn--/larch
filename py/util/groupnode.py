
import tables
import numpy
import types
import pandas

#class GroupNode(tables.group.Group):
#	def __new__(cls, parentnode, name=None, *arg, **kwarg):
#		if name is None:
#			name = parentnode._v_name
#			parentnode = parentnode._v_parent
#		self = parentnode._v_file.get_node(parentnode, name)
#		self.__class__ = GroupNode
#		return self
#	def __init__(self, parentnode, name=None, *arg, **kwarg):
#		if name is None:
#			name = parentnode._v_name
#			parentnode = parentnode._v_parent
#		super().__init__(parentnode, name, *arg, **kwarg)
#	def __getattr__(self, attr):
#		try:
#			return super().__getattr__(attr)
#		except tables.exceptions.NoSuchNodeError as err:
#			from larch.util.text_manip import case_insensitive_close_matches
#			did_you_mean_list = case_insensitive_close_matches(attr, self._v_children.keys())
#			if len(did_you_mean_list)>0:
#				did_you_mean = str(err)+"\nDid you mean {}?".format(" or ".join("'{}'".format(s) for s in did_you_mean_list))
#				raise tables.exceptions.NoSuchNodeError(did_you_mean) from None
#			raise


def _uniques(self, slicer=None, counts=False):
	if isinstance(slicer, (bool,int)) and counts is False:
		counts = bool(slicer)
		slicer = None
	if slicer is None:
		slicer = slice(None)
	if counts:
		x = numpy.unique(self[slicer], return_counts=counts)
		return pandas.Series(x[1],x[0])
	return numpy.unique(self[slicer])

def _pytables_link_dereference(i):
	if isinstance(i, tables.link.ExternalLink):
		i = i()
	if isinstance(i, tables.link.SoftLink):
		i = i.dereference()
	return i


class GroupNode():
	def __init__(self, parentnode, name=None, *arg, **kwarg):
		if name is None:
			name = parentnode._v_name
			parentnode = parentnode._v_parent
		n = _pytables_link_dereference(parentnode._v_file.get_node(parentnode, name))
		super().__setattr__('_v_node', n)
	def __getattr__(self, attr):
		try:
			if attr=='_v_node':
				return super().__getattr__('_v_node')
			x = getattr(self._v_node, attr)
			x.uniques = types.MethodType( _uniques, x )
			return x
		except tables.exceptions.NoSuchNodeError as err:
			from larch.util.text_manip import case_insensitive_close_matches
			did_you_mean_list = case_insensitive_close_matches(attr, self._v_node._v_children.keys())
			if len(did_you_mean_list)>0:
				did_you_mean = str(err)+"\nDid you mean {}?".format(" or ".join("'{}'".format(s) for s in did_you_mean_list))
				raise tables.exceptions.NoSuchNodeError(did_you_mean) from None
			raise
	def __setattr__(self, attr, val):
		return setattr(self._v_node, attr, val)
	def __contains__(self, *arg, **kwarg):
		return self._v_node.__contains__(*arg, **kwarg)
	def __str__(self, *arg, **kwarg):
		return self._v_node.__str__(*arg, **kwarg)
	def __repr__(self, *arg, **kwarg):
		return self._v_node.__repr__(*arg, **kwarg)
	def __getitem__(self, key):
		return self.__getattr__(key)
	def __setitem__(self, key, value):
		return self.__setattr__(key,value)

