

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
