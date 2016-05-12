


class Category():
	'''Defines categories of parameters to be used in report generation.
	
	Parameters
	----------
	name : str
		A descriptive category name that will be used to label the category 
		in a report.
	members : tuple
		The various members of this category, which can be parameter names 
		(as `str`), or other [sub-]`Category` objects.
	'''
	def __init__(self, name, *members):
		self.name = name
		if len(members)==1 and isinstance(members[0],(list,tuple)):
			self.members = members[0]
		else:
			self.members = members
	def complete_members(self):
		x = []
		for p in self.members:
			if isinstance(p,(category,rename)):
				x += p.complete_members()
			else:
				x += [p,]
		return x


category = Category  # legacy

class Rename():
	'''Defines an alternate display name for parameters to be used in report generation.
	
	Often the names of parameters actually used in the estimation process are 
	abbreviated or arcane, especially if some aspect of the source data or legacy
	models are older and not compatible with longer and more descriptive names.
	But modern report generation can be enhanced by using those longer and more 
	descriptive names.  This function allows a descriptive name to be attached to
	one or more parameters; attaching the same name to different parameters allows
	those parameters to be linked together and appear on the same line of multi-model
	reports.	
	
	Parameters
	----------
	name : str
		A descriptive name that will be used to label the parameter
		in a report.
	members : tuple
		The various members of this category, which should be parameter names.
	'''
	def __init__(self, name, *members):
		self.name = name
		if len(members)==1 and isinstance(members[0],(list,tuple)):
			self.members = members[0]
		else:
			self.members = members
	def find_in(self, m):
		if self not in m:
			raise LarchError("%s not in model",self.name)
		if self.name in m:
			return self.name
		for i in self.members:
			if i in m:
				return i
	def complete_members(self):
		x = [self.name,]
		for p in self.members:
			if isinstance(p,(category,rename)):
				x += p.complete_members()
			else:
				x += [p,]
		return x
	def __str__(self):
		return self.name

rename = Rename   # legacy




class pmath():
	def __init__(self, name, *, default=None):
		self._p = name
		self._fmt = "{}"
		self._name = ""
		self._default_value = default
	def value(self,m):
		if self._p in m.alias_names():
			als = m.alias(self._p)
			return m.metaparameter(self._p).value
		if self._p in m:
			return m[self._p].value
		if self._default_value is not None:
			return self._default_value
		raise LarchError("parameter {} not in model".format(self._p))
	def name(self, n):
		self._name = n
		return self
	def fmt(self,format):
		self._fmt = format
		return self
	def str(self,m):
		return self._fmt.format(self.value(m))
	def getname(self):
		return self._name
	def valid(self,m):
		if self._p in m:
			return True
		return False
	def __add__(self, other):
		return _param_add(self,other)
	def __radd__(self, other):
		return _param_add(other,self)
	def __sub__(self, other):
		return _param_subtract(self,other)
	def __rsub__(self, other):
		return _param_subtract(other,self)
	def __mul__(self, other):
		return _param_multiply(self,other)
	def __rmul__(self, other):
		return _param_multiply(other,self)
	def __truediv__(self, other):
		return _param_divide(self,other)
	def __rtruediv__(self, other):
		return _param_divide(other,self)
	def __neg__(self):
		return _param_negate(self)

class _param_add(pmath):
	def __init__(self,left,right):
		self._p = None
		self._left = left
		self._right = right
		self._fmt = "{}"
		self._name = ""
	def value(self,m):
		if isinstance(self._left, pmath):
			x = self._left.value(m)
		else:
			x = self._left
		if isinstance(self._right, pmath):
			x += self._right.value(m)
		else:
			x += self._right
		return x
	def valid(self,m):
		if isinstance(self._left, pmath):
			x = self._left.valid(m)
		else:
			x = True
		if isinstance(self._right, pmath):
			x &= self._right.valid(m)
		return x

class _param_subtract(pmath):
	def __init__(self,left,right):
		self._p = None
		self._left = left
		self._right = right
		self._fmt = "{}"
		self._name = ""
	def value(self,m):
		if isinstance(self._left, pmath):
			x = self._left.value(m)
		else:
			x = self._left
		if isinstance(self._right, pmath):
			x -= self._right.value(m)
		else:
			x -= self._right
		return x
	def valid(self,m):
		if isinstance(self._left, pmath):
			x = self._left.valid(m)
		else:
			x = True
		if isinstance(self._right, pmath):
			x &= self._right.valid(m)
		return x


class _param_multiply(pmath):
	def __init__(self,left,right):
		self._p = None
		self._left = left
		self._right = right
		self._fmt = "{}"
		self._name = ""
	def value(self,m):
		if isinstance(self._left, pmath):
			x = self._left.value(m)
		else:
			x = self._left
		if isinstance(self._right, pmath):
			x *= self._right.value(m)
		else:
			x *= self._right
		return x
	def valid(self,m):
		if isinstance(self._left, pmath):
			x = self._left.valid(m)
		else:
			x = True
		if isinstance(self._right, pmath):
			x &= self._right.valid(m)
		return x


class _param_divide(pmath):
	def __init__(self,left,right):
		self._p = None
		self._left = left
		self._right = right
		self._fmt = "{}"
		self._name = ""
	def value(self,m):
		if isinstance(self._left, pmath):
			x = self._left.value(m)
		else:
			x = self._left
		try:
			if isinstance(self._right, pmath):
				x /= self._right.value(m)
			else:
				x /= self._right
		except ZeroDivisionError:
			return float('NaN')
		return x
	def valid(self,m):
		if isinstance(self._left, pmath):
			x = self._left.valid(m)
		else:
			x = True
		if isinstance(self._right, pmath):
			x &= self._right.valid(m)
		return x


class _param_negate(pmath):
	def __init__(self,orig):
		self._p = None
		self._orig = orig
		self._fmt = "{}"
		self._name = ""
	def value(self,m):
		if isinstance(self._orig, pmath):
			return -self._orig.value(m)
		else:
			return -self._orig
	def valid(self,m):
		if isinstance(self._orig, pmath):
			return self._orig.valid(m)
		else:
			return True

