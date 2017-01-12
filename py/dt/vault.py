

from . import _tb


def in_vault(self, name):
	vault = self.get_or_create_group(self.h5top, 'vault')
	name = name.replace('.','_')
	if name in vault:
		return True
	if 'stack.' in name:
		try:
			return 'stack' in self.idca._v_children[name[6:]]._v_attrs
		except:
			pass
	return False

def to_vault(self, name, value):
	vault = self.get_or_create_group(self.h5top, 'vault')
	name = name.replace('.','_')
	if name not in vault:
		vault_bin = self.h5f.create_vlarray(vault, name, _tb.ObjectAtom())
	else:
		vault_bin = vault._v_children[name]
	vault_bin.append(value)

def from_vault(self, name, index=-1):
	vault = self.get_or_create_group(self.h5top, 'vault')
	name_ = name.replace('.','_')
	if name_ not in vault:
		# not in vault, check for attrib...
		if 'stack.' in name:
			try:
				return self.idca._v_children[name[6:]]._v_attrs.stack
			except:
				pass
		raise KeyError(name_+' not in vault')
	else:
		vault_bin = vault._v_children[name_]
	return vault_bin[index]

def vault_keys(self):
	vault = self.get_or_create_group(self.h5top, 'vault')
	return vault._v_children.keys()

def wipe_vault(self, regex='.*'):
	vault = self.get_or_create_group(self.h5top, 'vault')
	import re
	names = [name for name in self.vault_keys() if re.search(regex,name)]
	for name in names:
		vault._v_file.remove_node(vault, name)

def del_vault(self):
	if 'vault' in self.h5top:
		self.h5top._v_file.remove_node(self.h5top, 'vault')
