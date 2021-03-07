
class TouchNotify():
	"""
	A mixin to execute a notification callback any time a
	mutate-possible command is called.
	"""

	def __init__(self, *args, touch_callback_name='mangle', **kwargs):
		self._touch_name = touch_callback_name
		self._touch = lambda: None
		super().__init__(*args, **kwargs)

	def __set_name__(self, owner, name):
		# self : compiledmethod
		# owner : parent class that will have `self` as a member
		# name : the name of the attribute that `self` will be
		self.public_name = name
		self.private_name = '_touchable_' + name

	def __get__(self, obj, objtype=None):
		# self : TouchNotify
		# obj : instance of parent class that has `self` as a member, or None
		# objtype : class of `obj`
		result = getattr(obj, self.private_name, None)
		if result is not None:
			result.set_touch_callback(getattr(obj, self._touch_name, lambda: None))
		return result

	def __set__(self, obj, value):
		# self : TouchNotify
		# obj : instance of parent class that has `self` as a member
		# value : the new value that is trying to be assigned
		if not (isinstance(value, type(self)) or value is None):
			value = type(self)(value)
		setattr(obj, self.private_name, value)

	def __delete__(self, obj):
		# self : TouchNotify
		# obj : instance of parent class that has `self` as a member
		setattr(obj, self.private_name, None)

	def get_touch_callback(self):
		return self._touch

	def set_touch_callback(self, callback):
		if callable(callback):
			self._touch = callback
		elif callback is None:
			self._touch = lambda: None
		else:
			raise TypeError('callback must be callable')

	def del_touch_callback(self):
		self._touch = lambda: None

	def touch(self):
		self._touch()

	def __setitem__(self, key, value):
		ret = super().__setitem__(key, value)
		try:
			self.__touch()
		except AttributeError:
			pass # on init, may not yet have _touch
		return ret

	def __setattr__(self, key, value):
		ret = super().__setattr__(key, value)
		try:
			self.__touch()
		except AttributeError:
			pass # on init, may not yet have _touch
		return ret

	def __delitem__(self, key):
		ret = super().__delitem__(key)
		self._touch()
		return ret

	def __delattr__(self, key):
		ret = super().__delattr__(key)
		self._touch()
		return ret

	def setdefault(self, *args, **kwargs):
		ret = super().setdefault(*args, **kwargs)
		self._touch()
		return ret

	def append(self, *args, **kwargs):
		ret = super().append(*args, **kwargs)
		self._touch()
		return ret

	def extend(self, *args, **kwargs):
		ret = super().extend(*args, **kwargs)
		self._touch()
		return ret

	def update(self, *args, **kwargs):
		ret = super().update(*args, **kwargs)
		self._touch()
		return ret

	def insert(self, *args, **kwargs):
		ret = super().insert(*args, **kwargs)
		self._touch()
		return ret

	def remove(self, *args, **kwargs):
		ret = super().remove(*args, **kwargs)
		self._touch()
		return ret

	def pop(self, *args, **kwargs):
		ret = super().pop(*args, **kwargs)
		self._touch()
		return ret

	def popitem(self, *args, **kwargs):
		ret = super().popitem(*args, **kwargs)
		self._touch()
		return ret

	def clear(self, *args, **kwargs):
		ret = super().clear(*args, **kwargs)
		self._touch()
		return ret

	def __call__(self, *args, **kwargs):
		ret = super().__call__(*args, **kwargs)
		self._touch()
		return ret

	def __iadd__(self, other):
		ret = super().__iadd__(other)
		self._touch()
		return ret

	def __isub__(self, other):
		ret = super().__isub__(other)
		self._touch()
		return ret

	def __imul__(self, other):
		ret = super().__imul__(other)
		self._touch()
		return ret

	def __idiv__(self, other):
		ret = super().__idiv__(other)
		self._touch()
		return ret

	def __iand__(self, other):
		ret = super().__iand__(other)
		self._touch()
		return ret

	def __ior__(self, other):
		ret = super().__ior__(other)
		self._touch()
		return ret

	def __ixor__(self, other):
		ret = super().__ixor__(other)
		self._touch()
		return ret

	def __getstate__(self):
		return {}

	def __setstate__(self, state):
		self._touch = lambda: None
