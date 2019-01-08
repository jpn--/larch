
from itertools import chain
_RaiseKeyError = object() # singleton for no-default behavior

class SignalDict(dict):  # dicts take a mapping or iterable as their optional first argument
	__slots__ = ('_SignalDict__getitem_callback',
				 '_SignalDict__setitem_callback',
				 '_SignalDict__delitem_callback',) # no __dict__ - that would be redundant
	@staticmethod # because this doesn't make sense as a global function.
	def _process_args(mapping=(), **kwargs):
		if hasattr(mapping, 'items'):
			mapping = getattr(mapping, 'items')()
		return ((k, v) for k, v in chain(mapping, getattr(kwargs, 'items')()))
	def __init__(self, getitem_callback, setitem_callback, delitem_callback, mapping=(), **kwargs):
		self.__getitem_callback = getitem_callback
		self.__setitem_callback = setitem_callback
		self.__delitem_callback = delitem_callback
		super().__init__(self._process_args(mapping, **kwargs))
	def __getitem__(self, k):
		ret = super().__getitem__(k)
		self.__getitem_callback(k)
		return ret
	def __setitem__(self, k, v):
		ret = super().__setitem__(k, v)
		self.__setitem_callback(k,v)
		return ret
	def __delitem__(self, k):
		ret = super().__delitem__(k)
		self.__delitem_callback(k)
		return ret
	def get(self, k, default=None):
		return super().get(k, default)
	def setdefault(self, k, default=None):
		return super().setdefault(k, default)
	def pop(self, k, v=_RaiseKeyError):
		if v is _RaiseKeyError:
			return super().pop(k)
		return super().pop(k, v)
	def update(self, mapping=(), **kwargs):
		super().update(self._process_args(mapping, **kwargs))
	def __contains__(self, k):
		return super().__contains__(k)
	def copy(self): # don't delegate w/ super - dict.copy() -> dict :(
		return type(self)(self)
	def __repr__(self):
		return '{0}({1})'.format(type(self).__name__, super().__repr__())
