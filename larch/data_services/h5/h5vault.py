
import tables as tb
import os
from pathlib import Path
from collections.abc import MutableMapping
import cloudpickle
import pickle
from .h5util import get_or_create_group

class H5Vault(MutableMapping):

	def __init__(self, filename=None, mode='a', vaultnode=None, *,
	             h5f=None, inmemory=False, temp=False,
	             complevel=1, complib='zlib',
	             ):

		if isinstance(filename, H5Vault):
			# Copy / Re-Class contructor
			x = filename
			self._vaultnode = x._vaultnode
			self._h5f_own = False
			return

		if isinstance(filename, tb.group.Group) and vaultnode is None:
			# Called with just a group node, use it
			vaultnode = filename
			filename = None

		if isinstance(filename, (str,Path)):
			filename = os.fspath(filename)
			if vaultnode is None:
				vaultnode = "_VAULT_"

		if isinstance(vaultnode, tb.group.Group):
			# Use the existing info from this group node, ignore all other inputs
			self._vaultnode = vaultnode
			filename = vaultnode._v_file.filename
			mode = vaultnode._v_file.mode
			self._h5f_own = False

		elif isinstance(vaultnode, str):

			# apply expanduser to filename to allow for home-folder based filenames
			if isinstance(filename,str):
				filename = os.path.expanduser(filename)

			if filename is None:
				temp = True
				from ..util.temporaryfile import TemporaryFile
				self._TemporaryFile = TemporaryFile(suffix='.h5d')
				filename = self._TemporaryFile.name

			if h5f is not None:
				self._h5f_own = False
				# self._vaultnode = self._h5f.get_node(vaultnode)
				top = self._h5f.get_node('/')
				self._vaultnode = get_or_create_group(self._h5f, top, vaultnode, skip_on_readonly=True)
			else:
				kwd = {}
				if inmemory or temp:
					kwd['driver']="H5FD_CORE"
				if temp:
					kwd['driver_core_backing_store']=0
				if complevel is not None:
					kwd['filters']=tb.Filters(complib=complib, complevel=complevel)
				self._h5f_obj = tb.open_file(filename, mode, **kwd)
				self._h5f_own = True
				top = self._h5f_obj.get_node('/')
				self._vaultnode = get_or_create_group(self._h5f_obj, top, vaultnode, skip_on_readonly=True)
		else:
			raise ValueError('must give groupnode as `str` or `tables.group.Group`')

	@property
	def _h5f(self):
		return self._vaultnode._v_file

	def __contains__(self, name):
		if not isinstance(name, str):
			raise TypeError(f'vault keys must be str not {type(name)}')
		name = name.replace('.','_')
		if name in self._vaultnode._v_children:
			return True
		return False

	def __setitem__(self, name, value):
		if not isinstance(name, str):
			raise TypeError('H5Vault requires str keys')
		name = name.replace('.','_')
		if name not in self._vaultnode._v_children:
			vault_bin = self._h5f.create_vlarray(self._vaultnode, name, tb.ObjectAtom())
		else:
			vault_bin = self._vaultnode._v_children[name]
		vault_bin.append(cloudpickle.dumps(value))

	def _get_item_or_older(self, name, index=-1):
		name_ = name.replace('.','_')
		if name_ not in self._vaultnode._v_children:
			raise KeyError(name_+' not in vault')
		else:
			vault_bin = self._vaultnode._v_children[name_]
		return pickle.loads(vault_bin[index])

	def __getitem__(self, name):
		return self._get_item_or_older(name)

	def keys(self):
		return self._vaultnode._v_children.keys()

	def wipe(self, regex='.*'):
		import re
		names = [name for name in self.keys() if re.search(regex,name)]
		for name in names:
			self._vaultnode._v_file.remove_node(self._vaultnode, name)

	def __delitem__(self, key):
		self.wipe(key)

	def __iter__(self):
		for k in self.keys():
			yield k

	def __len__(self):
		return len(self._vaultnode._v_children.keys())


