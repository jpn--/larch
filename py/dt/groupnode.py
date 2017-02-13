
import tables
import numpy
import types
import pandas
import os
import warnings
import textwrap
from ..util.naming import make_valid_identifier, NotAPythonIdentifier

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
	action = self[slicer]
	len_action = len(action)
	try:
		action = action[~numpy.isnan(action)]
	except TypeError:
		num_nan = 0
	else:
		num_nan = len_action-len(action)
	if counts:
		x = numpy.unique(action, return_counts=counts)
		y = pandas.Series(x[1],x[0])
		if num_nan:
			y[numpy.nan] = num_nan
		return y
	if num_nan:
		numpy.append(action, numpy.nan)
	return numpy.unique(action)



def _pytables_extlink_dereference(extlink, **kwargs):
	"""Dereference extlink.target and return the object.

	You can pass all the arguments supported by the :func:`open_file`
	function (except filename, of course) so as to open the referenced
	external file.
	"""

	filename, target = extlink._get_filename_node()

	if not os.path.isabs(filename):
		# Resolve the external link with respect to the this
		# file's directory.
		base_directory = os.path.dirname(extlink._v_file.filename)
		filename = os.path.join(base_directory, filename)

	if extlink.extfile is None or not extlink.extfile.isopen:
		extlink.extfile = tables.open_file(filename, **kwargs)
	else:
		# XXX: implement better consistency checks
		assert extlink.extfile.filename == filename
		assert extlink.extfile.mode == kwargs.get('mode', 'r')

	return extlink.extfile._get_node(target)




def _pytables_link_dereference(i):
	if isinstance(i, tables.link.ExternalLink):
		i = i()
	if isinstance(i, tables.link.SoftLink):
		try:
			i = i.dereference()
		except tables.NoSuchNodeError:
			root = i._v_file.root
			path_parts = i.target.split('/')[1:]
			g = GroupNode(root)
			for pth in path_parts:
				g = g[pth]
			i = g
	return i






class GroupNodeV1():
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
			if len(attr)<=3 or attr[0]!='_' or attr[2]!='_':
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




def _get_children_including_extern(node):
	kids = set(node._v_children.keys())
	extern_n = 1
	while "_extern_{}".format(extern_n) in kids:
		extern_deref = _pytables_link_dereference(node._v_children["_extern_{}".format(extern_n)])
		kids.remove("_extern_{}".format(extern_n))
		kids |= _get_children_including_extern(extern_deref)
		extern_n += 1
	return kids








class CArrayExt(tables.carray.CArray):
	@property
	def dict(self):
		if 'DICTIONARY' in self._v_attrs:
			return self._v_attrs.DICTIONARY
		return None

	@dict.setter
	def dict(self, val):
		self._v_attrs.DICTIONARY = val

	DICTIONARY = dict
	dictionary = dict

	@property
	def title(self):
		if 'TITLE' in self._v_attrs:
			return self._v_attrs.TITLE
		return None

	@title.setter
	def title(self, val):
		self._v_attrs.TITLE = val

	TITLE = title


	@property
	def description(self):
		"Same as title, except string is reformatted by stripping excess whitespace."
		if 'TITLE' in self._v_attrs:
			return textwrap.fill(self._v_attrs.TITLE, width=78)
		return None

	@description.setter
	def description(self, val):
		self._v_attrs.TITLE = " ".join(str(val).split())

	DESCRIPTION = description
	descrip = description

	def descrip_w(self, width=70, **kwargs):
		return textwrap.fill(self.TITLE, width=width, **kwargs)



class GroupNode():
	def __init__(self, parentnode, name=None, *arg, **kwarg):
		if name is None:
			name = parentnode._v_name
			parentnode = parentnode._v_parent
		n = _pytables_link_dereference(parentnode._v_file.get_node(parentnode, name))
		super().__setattr__('_v_node', n)
	def __getattr__(self, attr):
		if attr=='_v_node':
			return super().__getattr__('_v_node')
		if len(attr)>3 and attr[0]=='_' and attr[2]=='_':
			try:
				return super().__getattr__('_v_node').__getattr__(attr)
			except:
				pass
		extern_did_you_mean_list = set()
		try:
			try:
				x = getattr(self._v_node, attr)
			except tables.exceptions.NoSuchNodeError:
				extern_n = 1
				while True:
					if "_extern_{}".format(extern_n) not in self._v_node:
						raise
					extern_deref = _pytables_link_dereference(self._v_node._v_children["_extern_{}".format(extern_n)])
					if attr in extern_deref:
						x = getattr(extern_deref, attr)
						break
					else:
						extern_did_you_mean_list |= extern_deref._v_children.keys()
					extern_n += 1
		except tables.exceptions.NoSuchNodeError as err:
			from larch.util.text_manip import case_insensitive_close_matches
			did_you_mean_list = case_insensitive_close_matches(attr, self._v_node._v_children.keys()|extern_did_you_mean_list)
			if len(did_you_mean_list)>0:
				did_you_mean = str(err)+"\nDid you mean {}?".format(" or ".join("'{}'".format(s) for s in did_you_mean_list))
				raise tables.exceptions.NoSuchNodeError(did_you_mean) from None
			raise
		x = _pytables_link_dereference(x)
		if len(attr)<=3 or attr[0]!='_' or attr[2]!='_':
			x.uniques = types.MethodType( _uniques, x )
		if isinstance(x, tables.Group):
			x = GroupNode(x)
		if x.__class__ == tables.carray.CArray:
			x.__class__ = CArrayExt
		return x
	def __setattr__(self, attr, val):
		return setattr(self._v_node, attr, val)
	def __contains__(self, arg):
		if self._v_node.__contains__(arg):
			return True
		extern_n = 1
		while "_extern_{}".format(extern_n) in self._v_node._v_children:
			try:
				extern_deref = _pytables_link_dereference(self._v_node._v_children["_extern_{}".format(extern_n)])
			except OSError as err:
				warnings.warn(str(err))
			else:
				if arg in _get_children_including_extern(extern_deref):
					return True
			extern_n += 1
		return False
	def __repr__(self, *arg, **kwarg):
		return "<larch.DT:GroupNode> "+self._v_node._v_pathname+"\n  "+"\n  ".join(sorted(self._v_children_keys_including_extern))
	def __getitem__(self, key):
		if isinstance(key,slice):
			raise TypeError('cannot slice a group (yet)')
		return self.__getattr__(key)
	def __setitem__(self, key, value):
		if isinstance(key,slice):
			raise TypeError('cannot slice a group (yet)')
		return self.__setattr__(key,value)
	
	def add_group_node(self, name):
		return self._v_file.create_group(self._v_node, name)
	
	@property
	def _v_children_keys_including_extern(self):
		return _get_children_including_extern(self)

	def __dir__(self):
		return super().__dir__() + list(_get_children_including_extern(self))

	def add_external_data(self, link):
		"""
		Add a linkage to an external data file.
		
		The method creates a linkage to an external data file without actually
		copying the data into the current file.  Instead it creates an HDF5
		external link node within the current group node, which points to the 
		equivalent group node in the other file (e.g., in the idco group you would
		point to the idco group of the other file, not an individual data item).
		All of the data nodes within the external file will then be automatically
		and (almost) transparently linked into the current file, from the perspective
		of the GroupNode object.
		
		Parameters
		----------
		link : str
			The location of the external linkage.  Must be given as "filename:/path/to/node"
			where node is the target group node (not an individual array).
			
		"""
		if isinstance(link, GroupNode):
			fname = link._v_node._v_file.filename
			if fname == self._v_node._v_file.filename:
				raise TypeError('cannot link to external data in same file, try a local link')
			link = fname +":"+ link._v_node._v_pathname
		if ":/" not in link:
			raise TypeError("must give link as filename:/path/to/node")
		linkfile, linknode = link.split(":/")
		
		# change to a relative path only if the common path has something more than
		# the root dir; sharing only the root dir is a hint that one of the files
		# is a temp file, and the relative path may not be a valid path.
		try:
			if len(os.path.commonpath([linkfile, os.path.dirname( self._v_file.filename )])) > 2:
				linkfile = os.path.relpath(linkfile, os.path.dirname( self._v_file.filename ))
		except ValueError:
			pass

		link = linkfile+":/"+linknode
		extern_n = 1
		while '_extern_{}'.format(extern_n) in self._v_node:
			extern_n += 1
		return self._v_file.create_external_link(self._v_node, '_extern_{}'.format(extern_n), link)


	def add_external_node(self, name, link):
		"""
		Add a linkage to a single external node.
		"""
		if isinstance(link, GroupNode):
			fname = link._v_node._v_file.filename
			if fname == self._v_node._v_file.filename:
				raise TypeError('cannot link to external data in same file, try a local link')
			link = fname +":"+ link._v_node._v_pathname
		return self._v_file.create_external_link(self._v_node, name, link)


	def add_external_omx(self, omx_filename, rowindexnode, prefix="", n_alts=-1, n_lookup=-1, absolute_path=False, local_rowindexnode=None, suppress_identifier_warning=False):
		'''
		Add an external linkage from this group to the values in an OMX file.
		
		For matrix data tables, this will create a virtual idca keyed array, with the rowindexnode giving the
		indexes and the external matrix linked with the values.
		
		For lookups, set n_lookup equal to the length of the lookups and they will be treated effectively the same
		as a data table, except with only one value, creating a virtual idco keyed array.
		
		Alternatively for lookups, set n_alts equal to the length of the lookups and they will be treated as 
		a data table with a single row.  The rowindexnode is ignored in this case, as every row is the same, creating
		a virtual idca keyed array (with just one key).
		
		Parameters
		----------
		omx_filename : str or OMX
			Either the path to an OMX file or the open OMX object.
		rowindexnode : pytables integer-dtype array-type node
			This should be an existing node in the group (or elsewhere in the hdf5 file). It contains the integer 
			index values of the rows that will be linked.  If given as None, matrix data tables will not be linked, and
			matrix lookup tables will only be linked using the n_alts form.
		prefix : str
		n_alts : int
		n_lookup : int 
		absolute_path : bool
		'''
		anything_linked = False
		if not isinstance(omx_filename, str) and hasattr(omx_filename, 'filename'):
			omx_filename = omx_filename.filename		
		if not absolute_path:
			omx_filename = os.path.relpath(omx_filename, os.path.dirname( self._v_file.filename ))
		
		if local_rowindexnode is not None and rowindexnode is None:
			rowindexnode = self[local_rowindexnode]
		if rowindexnode is not None and not rowindexnode._v_isopen:
			raise TypeError('rowindexnode is closed')
		
		def rowindexnode_():
			nonlocal rowindexnode
			if rowindexnode is None:
				return None
			if rowindexnode._v_isopen:
				return rowindexnode
			if local_rowindexnode is not None:
				return self[local_rowindexnode]
			raise TypeError('rowindexnode not stable open')
		
		temp_num = 1
		while 'temp_omx_{}'.format(temp_num) in self._v_file.root._v_children:
			temp_num += 1
		self._v_file.create_external_link(self._v_file.root, 'temp_omx_{}'.format(temp_num), omx_filename+":/")
		temp_omx = lambda: self._v_file.root._v_children['temp_omx_{}'.format(temp_num)]()
		if 'data' in temp_omx() and rowindexnode_() is not None:
			for vname in sorted(temp_omx().data._v_children):
				try:
					vgrp = self._v_file.create_group(self._v_node, prefix+vname)
				except tables.exceptions.NodeError:
					warnings.warn('the name "{}" already exists'.format(prefix+vname))
				else:
					self._v_file.create_soft_link(vgrp, '_index_', rowindexnode_())
					self._v_file.create_external_link(vgrp, '_values_', omx_filename+":/data/"+vname)
					anything_linked = True
		if 'lookup' in temp_omx():
			for lname in sorted(temp_omx().lookup._v_children):
				if temp_omx().lookup._v_children[lname].shape == (n_alts,):
					# This is idca type data.
					# The constructed index is len(caseids) but all zeros.
					# Each values from the index plucks the entire lookup vector.
					# The resulting pseudoarray is shape (nCases,nAlts)
					with warnings.catch_warnings():
						if suppress_identifier_warning:
							warnings.filterwarnings('ignore', category=NotAPythonIdentifier)
						full_lname = make_valid_identifier(prefix+lname)
					try:
						vgrp = self._v_file.create_group(self._v_node, full_lname)
					except tables.exceptions.NodeError:
						warnings.warn('the name "{}" already exists'.format(full_lname))
					else:
						self._v_file.create_carray(vgrp, '_index_', shape=_pytables_link_dereference(self._v_file.root.larch.caseids).shape, atom=tables.Int32Atom()) # zeros
						self._v_file.create_external_link(vgrp, '_values_', omx_filename+":/lookup/"+lname)
						vgrp._v_attrs.transpose_values = True
						anything_linked = True
				if temp_omx().lookup._v_children[lname].shape == (n_lookup,) and rowindexnode is not None:
					# This is idco type data.
					# The provided rowindexnode should be len(caseids)
					# The values from the index pluck single values out of the lookup vector.
					# The resulting pseudoarray is shape (nCases,)
					full_lname = make_valid_identifier(prefix+lname)
					try:
						vgrp = self._v_file.create_group(self._v_node, full_lname)
					except tables.exceptions.NodeError:
						warnings.warn('the name "{}" already exists'.format(full_lname))
					else:
						self._v_file.create_soft_link(vgrp, '_index_', rowindexnode)
						self._v_file.create_external_link(vgrp, '_values_', omx_filename+":/lookup/"+lname)
						anything_linked = True
		if not anything_linked:
			from ..omx import OMX
			try:
				omx_repr = repr(OMX(omx_filename))
				warnings.warn('nothing was linked from file "{}"\n{}'.format(omx_filename, omx_repr), stacklevel=2)
			except OSError:
				warnings.warn('nothing was linked from file "{}"'.format(omx_filename), stacklevel=2)

	@property
	def dict(self):
		if 'DICTIONARY' in self._v_node._v_attrs:
			return self._v_node._v_attrs.DICTIONARY
		return None

	@dict.setter
	def dict(self, val):
		self._v_node._v_attrs.DICTIONARY = val

	DICTIONARY = dict
	dictionary = dict

	@property
	def title(self):
		if 'TITLE' in self._v_node._v_attrs:
			return self._v_node._v_attrs.TITLE
		return None

	@title.setter
	def title(self, val):
		self._v_node._v_attrs.TITLE = val

	TITLE = title
	DESCRIPTION = title
	description = title
	descrip = title




