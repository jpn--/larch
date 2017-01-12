
from . import _pytables_link_dereference, _tb, numpy


class DT_idco_stack_manager:

	def __init__(self, parent, stacktype):
		self.parent = parent
		self.stacktype = stacktype

	def _check(self):
		def isinstance_(obj, things):
			obj = _pytables_link_dereference(obj)
#			try:
#				obj = obj.dereference()
#			except AttributeError:
#				pass
			return isinstance(obj, things)
		if isinstance_(self.parent.idca[self.stacktype], _tb.Array):
			raise TypeError('The {} is an array, not a stack.'.format(self.stacktype))
		if not isinstance_(self.parent.idca[self.stacktype], (_tb.Group,GroupNode)):
			raise TypeError('The {} stack is not set up.'.format(self.stacktype))

	def _make_zeros(self):
		def isinstance_(obj, things):
			obj = _pytables_link_dereference(obj)
#			try:
#				obj = obj.dereference()
#			except AttributeError:
#				pass
			return isinstance(obj, things)
		try:
			if isinstance_(self.parent.idca[self.stacktype], _tb.Array):
				self.parent.h5f.remove_node(self.parent.idca._v_node, self.stacktype)
		except (_tb.exceptions.NoSuchNodeError, KeyError):
			pass
		# create new group if it does not exist
		try:
			self.parent.h5f.create_group(self.parent.idca._v_node, self.stacktype)
		except _tb.exceptions.NodeError:
			pass
		if 'stack' not in self.parent.idca[self.stacktype]._v_attrs:
			##self.parent.idca[self.stacktype]._v_attrs.stack = ["0"]*self.parent.nAlts()
			self._stackdef_vault = ["0"]*self.parent.nAlts()


	def __call__(self, *cols, varname=None):
		"""Set up the :ref:`idca` stack data array from :ref:`idco` variables.
		
		The choice array needs to be in :ref:`idca` format. If
		your data isn't in that format, it's still easy to create the correct
		availability array by stacking together the appropriate :ref:`idco` columns.
		This command simplifies that process.
		
		Parameters
		----------
		cols : tuple of str
			The names of the :ref:`idco` expressions that represent availability. 
			They should be given in exactly the same order as they appear in the
			alternative codes array.
		varname : str or None
			The name of the new :ref:`idca` variable to create. Defaults to None.
			
		Raises
		------
		_tb.exceptions.NodeError
			If a variable of the name given by `varname` already exists.
		NameError
			If the expression contains a name that cannot be evaluated from within
			the existing :ref:`idco` data.
		TypeError
			If the wrong number of cols arguments is provided; it must exactly match the
			number of alternatives.
		"""
		if len(cols)==1 and len(cols[0])==self.parent.nAlts():
			cols = cols[0]
		if len(cols) != self.parent.nAlts():
			raise TypeError('the input columns must exactly match the alternatives, you gave {} but there are {} alternatives'.format(len(cols), self.parent.nAlts()))
		# Raise an exception when a col is invalid
		self.parent.multi_check_co(cols)
		cols = list(cols)
		if varname is None:
			try:
				self.parent.h5f.remove_node(self.parent.idca._v_node, self.stacktype)
			except _tb.exceptions.NoSuchNodeError:
				pass
			self.parent.h5f.create_group(self.parent.idca._v_node, self.stacktype)
			##self.parent.idca[self.stacktype]._v_attrs.stack = cols
			self._stackdef_vault = cols
		else:
			ch = self.parent.array_idco(*cols, dtype=numpy.float64)
			self.parent.new_idca(varname, ch)
			try:
				self.parent.h5f.remove_node(self.parent.idca._v_node, self.stacktype)
			except _tb.exceptions.NoSuchNodeError:
				pass
			self.parent.h5f.create_soft_link(self.parent.idca._v_node, self.stacktype, target=self.parent.idca._v_pathname+'/'+varname)

	def __getitem__(self, key):
		self._check()
		slotarray = numpy.where(self.parent._alternative_codes()==key)[0]
		if len(slotarray) == 1:
			##return self.parent.idca[self.stacktype]._v_attrs.stack[slotarray[0]]
			return self._stackdef_vault[slotarray[0]]
		
		else:
			raise KeyError("key {} not found".format(key) )

	def __setitem__(self, key, value):
		slotarray = numpy.where(self.parent._alternative_codes()==key)[0]
		if len(slotarray) == 1:
			if self.stacktype not in self.parent.idca:
				self._make_zeros()
			if 'stack' not in self.parent.idca[self.stacktype]._v_attrs and not self.parent.in_vault('stack.'+self.stacktype):
				self._make_zeros()
			##tempobj = self.parent.idca[self.stacktype]._v_attrs.stack
			tempobj = self._stackdef_vault
			tempobj[slotarray[0]] = value
			##self.parent.idca[self.stacktype]._v_attrs.stack = tempobj
			self._stackdef_vault = tempobj
		else:
			raise KeyError("key {} not found".format(key) )

	def __repr__(self):
		self._check()
		s = "<stack_idco: {}>".format(self.stacktype)
		for n,altid in enumerate(self.parent._alternative_codes()):
			s += "\n  {}: {!r}".format(altid, self[altid])
		return s

	@property
	def _stackdef_vault(self):
		return self.parent.from_vault('stack.'+self.stacktype)

	@_stackdef_vault.setter
	def _stackdef_vault(self, value):
		self.parent.to_vault('stack.'+self.stacktype, value)
	
