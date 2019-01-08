
import tables as tb

def get_or_create_group(h5file, where, name=None, title='', filters=None, createparents=False, skip_on_readonly=False):
	if "/" in name:
		names = name.split('/')
		where += '/' + '/'.join(names[:-1])
		name = names[-1]
	try:
		return h5file.get_node(where, name=name)
	except tb.NoSuchNodeError:
		if name is not None:
			try:
				return h5file.create_group(where, name, title=title, filters=filters, createparents=createparents)
			except tb.FileModeError:
				if skip_on_readonly:
					return
		else:
			raise



def get_or_create_subgroup(parentgroup, name, title='', filters=None, skip_on_readonly=False):
	from .h5pod.generic import H5Pod
	if isinstance(parentgroup, H5Pod):
		parentgroup = parentgroup._groupnode
	if parentgroup is None and skip_on_readonly:
		return None
	h5file = parentgroup._v_file
	try:
		return h5file.get_node(parentgroup, name=name)
	except tb.NoSuchNodeError:
		if name is not None:
			try:
				return h5file.create_group(parentgroup, name, title=title, filters=filters, createparents=False)
			except tb.FileModeError:
				if skip_on_readonly:
					return
		else:
			raise
