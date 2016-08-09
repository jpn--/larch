

class dicta(dict):
	'''Dictionary with attribute access.'''
	def __getattr__(self, name):
		try:
			return self[name]
		except KeyError:
			if '_helper' in self:
				return self['_helper'](name)
			raise AttributeError(name)
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__
	def __repr__(self):
		if self.keys():
			m = max(map(len, list(self.keys()))) + 1
			return '\n'.join([k.rjust(m) + ': ' + repr(v) for k, v in self.items()])
		else:
			return self.__class__.__name__ + "()"

class dictal(dict):
	'''Dictionary with attribute access and lowercase str keys.'''
	def __init__(self, *args, **kwargs):
		super().__init__()
		for arg in args:
			for key,val in arg.items():
				self[key] = val
		for key,val in kwargs.items():
			self[key] = val
	def __setitem__(self, key, val):
		if not isinstance(key,str):
			raise TypeError("dictal keys must be strings")
		return super().__setitem__(key.casefold(), val)
	def __getitem__(self, key):
		if not isinstance(key,str):
			raise TypeError("dictal keys must be strings")
		return super().__getitem__(key.casefold())
	def __delitem__(self, key):
		if not isinstance(key,str):
			raise TypeError("dictal keys must be strings")
		return super().__delitem__(key.casefold())
	def __getattr__(self, name):
		try:
			return self[name]
		except KeyError:
			if '_helper' in self:
				return self['_helper'](name)
			raise AttributeError(name)
	__setattr__ = __setitem__
	__delattr__ = __delitem__
	def __repr__(self):
		if self.keys():
			m = max(map(len, list(self.keys()))) + 1
			return '\n'.join([k.rjust(m) + ': ' + repr(v) for k, v in self.items()])
		else:
			return self.__class__.__name__ + "()"






class function_cache(dict):
	def __getitem__(self, key):
		if key not in self:
			self[key] = z = dicta()
			return z
		return super().__getitem__(key)





class quickdot(dict):
	'''Autoexpanding dictionary with attribute access.'''
	def __getattr__(self, name):
		try:
			return self[name]
		except KeyError:
			if '_helper' in self:
				return self['_helper'](name)
			raise AttributeError(name)
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__
	def __repr__(self):
		if self.keys():
			m = max(map(len, list(self.keys()))) + 1
			return '\n'.join(['┣'+k.rjust(m) + ': ' + repr(v).replace('\n','\n┃'+' '*(m+2)) for k, v in self.items()])
		else:
			return self.__class__.__name__ + "()"
	def __getitem__(self, key):
		if key not in self:
			self[key] = z = quickdot()
			return z
		return super().__getitem__(key)
