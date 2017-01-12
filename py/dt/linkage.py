import re
from . import DT
from . import _pytables_link_dereference, _tb, numpy


def DTx(filename=None, *, caseids=None, alts=None, **kwargs):
	"""Build a new DT with externally linked data.
	
	Parameters
	----------
	filename : str or None
		The name of the new DT file to create.  If None, a temporary file is created.
	idco{n} : str
		A file path to a DT file containing idco variables to link.  `n` can be any number.
		If the same variable name appears multiple times, the highest numbered source file
		is the one that survives.
		Must be passed as a keyword argument.
	idca{n} : str
		A file path to a DT file containing idca variables to link.  `n` can be any number.
		If the same variable name appears multiple times, the highest numbered source file
		is the one that survives.
		Must be passed as a keyword argument.
	
	Notes
	-----
	Every parameter other than `filename` must be passed as a keyword argument.
	"""
	dt_init_kwargs = {}
	idco_kwargs = {}
	idca_kwargs = {}
	
	if isinstance(caseids, DT):
		_fname = caseids.h5f.filename
		caseids.close()
		caseids = _fname
	if isinstance(alts, DT):
		_fname = alts.h5f.filename
		alts.close()
		alts = _fname
	
	for kwd,kwarg in kwargs.items():
		if re.match('idco[0-9]*$',kwd):
			if isinstance(kwarg, DT):
				_fname = kwarg.h5f.filename
				kwarg.close()
				idco_kwargs[kwd] = _fname
			else:
				idco_kwargs[kwd] = kwarg
		elif re.match('idca[0-9]*$',kwd):
			if isinstance(kwarg, DT):
				_fname = kwarg.h5f.filename
				kwarg.close()
				idca_kwargs[kwd] = _fname
			else:
				idca_kwargs[kwd] = kwarg
		else:
			dt_init_kwargs[kwd] = kwarg

	if len(idco_kwargs)==0 and len(idca_kwargs)==0:
		raise TypeError('at least one idca or idco source must be given')

	d = DT(filename, **dt_init_kwargs)
	got_caseids = False
	got_alts = False
	if caseids is not None:
		d.remove_node_if_exists(d.h5top, 'caseids')
		if ":/" not in caseids:
			tag_caseids = caseids + ":/larch/caseids"
		else:
			tag_caseids = caseids
		d.create_external_link(d.h5top, 'caseids', tag_caseids)
		got_caseids = True

	def swap_alts(tag_alts):
		try:
			d.alts.altids._f_rename('altids_pending_delete')
		except _tb.exceptions.NoSuchNodeError:
			pass
		try:
			d.alts.names._f_rename('names_pending_delete')
		except _tb.exceptions.NoSuchNodeError:
			pass
		d.alts.add_external_data(tag_alts)
		if 'names' in d.alts:
			d.remove_node_if_exists(d.alts._v_node, 'names_pending_delete')
		if 'altids' in d.alts:
			d.remove_node_if_exists(d.alts._v_node, 'altids_pending_delete')
		return True

	if alts is not None:
		if ":/" not in alts:
			tag_alts = alts + ":/larch/alts"
		else:
			tag_alts = alts
		got_alts = swap_alts(tag_alts)

	for idca_kw in sorted(idca_kwargs):
		idca = idca_kwargs[idca_kw]
		if idca is not None:
			if ":/" not in idca:
				tag_idca = idca + ":/larch/idca"
				tag_caseids = idca + ":/larch/caseids"
				tag_alts = idca + ":/larch/alts"
			else:
				tag_idca = idca
				tag_caseids = None
				tag_alts = None
			newnode = _pytables_link_dereference(d.idca.add_external_data(tag_idca))
			for subnodename in newnode._v_children:
				subnode = newnode._v_children[subnodename]
				if isinstance(subnode, _tb.group.Group) and 'stack' in subnode._v_attrs:
					localnewnode = d.idca.add_group_node(subnodename)
					localnewnode._v_attrs['stack'] = subnode._v_attrs['stack']
			if not got_caseids and tag_caseids is not None:
				d.remove_node_if_exists(d.h5top, 'caseids')
				d.create_external_link(d.h5top, 'caseids', tag_caseids)
				got_caseids = True
			if not got_alts and tag_alts is not None:
				got_alts = swap_alts(tag_alts)
	for idco_kw in sorted(idco_kwargs):
		idco = idco_kwargs[idco_kw]
		if idco is not None:
			if ":/" not in idco:
				tag_idco = idco + ":/larch/idco"
				tag_caseids = idco + ":/larch/caseids"
				tag_alts = idco + ":/larch/alts"
			else:
				tag_idco = idco
				tag_caseids = None
				tag_alts = None
			newnode = _pytables_link_dereference(d.idco.add_external_data(tag_idco))
			if not got_caseids and tag_caseids is not None:
				d.remove_node_if_exists(d.h5top, 'caseids')
				d.create_external_link(d.h5top, 'caseids', tag_caseids)
				got_caseids = True
			if not got_alts and tag_alts is not None:
				got_alts = swap_alts(tag_alts)
	return d

def DTL(source):
	return DTx(None, idco=source, idca=source)
