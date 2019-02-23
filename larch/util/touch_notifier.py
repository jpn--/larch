
class TouchNotify():
	"""
	A mixin to execute a notification callback any time a
	mutate-possible command is called.
	"""

	def __init__(self, *args, touch_callback=None, **kwargs):
		self.__touch_callback = touch_callback if callable(touch_callback) else lambda: None
		super().__init__(*args, **kwargs)

	def get_touch_callback(self):
		try:
			return self.__touch_callback
		except AttributeError:
			return lambda: None

	def set_touch_callback(self, callback):
		if callable(callback):
			self.__touch_callback = callback
		elif callback is None:
			self.__touch_callback = lambda: None
		else:
			raise TypeError('callback must be callable')

	def del_touch_callback(self):
		self.__touch_callback = lambda: None

	def touch(self):
		self.__touch_callback()

	def __setitem__(self, key, value):
		ret = super().__setitem__(key, value)
		try:
			self.__touch_callback()
		except AttributeError:
			pass
		return ret

	def __setattr__(self, key, value):
		ret = super().__setattr__(key, value)
		try:
			self.__touch_callback()
		except AttributeError:
			pass
		return ret

	def __delitem__(self, key):
		ret = super().__delitem__(key)
		self.__touch_callback()
		return ret

	def __delattr__(self, key):
		ret = super().__delattr__(key)
		self.__touch_callback()
		return ret

	def setdefault(self, *args, **kwargs):
		ret = super().setdefault(*args, **kwargs)
		self.__touch_callback()
		return ret

	def append(self, *args, **kwargs):
		ret = super().append(*args, **kwargs)
		self.__touch_callback()
		return ret

	def extend(self, *args, **kwargs):
		ret = super().extend(*args, **kwargs)
		self.__touch_callback()
		return ret

	def update(self, *args, **kwargs):
		ret = super().update(*args, **kwargs)
		self.__touch_callback()
		return ret

	def insert(self, *args, **kwargs):
		ret = super().insert(*args, **kwargs)
		self.__touch_callback()
		return ret

	def remove(self, *args, **kwargs):
		ret = super().remove(*args, **kwargs)
		self.__touch_callback()
		return ret

	def pop(self, *args, **kwargs):
		ret = super().pop(*args, **kwargs)
		self.__touch_callback()
		return ret

	def popitem(self, *args, **kwargs):
		ret = super().popitem(*args, **kwargs)
		self.__touch_callback()
		return ret

	def clear(self, *args, **kwargs):
		ret = super().clear(*args, **kwargs)
		self.__touch_callback()
		return ret

	def __call__(self, *args, **kwargs):
		ret = super().__call__(*args, **kwargs)
		self.__touch_callback()
		return ret

	def __iadd__(self, other):
		ret = super().__iadd__(other)
		self.__touch_callback()
		return ret

	def __isub__(self, other):
		ret = super().__isub__(other)
		self.__touch_callback()
		return ret

	def __imul__(self, other):
		ret = super().__imul__(other)
		self.__touch_callback()
		return ret

	def __idiv__(self, other):
		ret = super().__idiv__(other)
		self.__touch_callback()
		return ret

	def __iand__(self, other):
		ret = super().__iand__(other)
		self.__touch_callback()
		return ret

	def __ior__(self, other):
		ret = super().__ior__(other)
		self.__touch_callback()
		return ret

	def __ixor__(self, other):
		ret = super().__ixor__(other)
		self.__touch_callback()
		return ret

	def __getstate__(self):
		return {}

	def __setstate__(self, state):
		self.__touch_callback = lambda: None

#
# class EditableList:
#
# 	def __init__(self, content=None, touched=None):
# 		self._data = [] if content is None else list(content)
# 		if touched is None:
# 			self._touched = lambda: print("TOUCH")
# 		else:
# 			self._touched = touch
#
# 	def __getitem__(self, position):
# 		return self._data[position]
#
# 	def __setitem__(self, position, value):
# 		self._data[position] = value
# 		self._touched()
#
# 	def append(self, value):
# 		self._data.append(value)
# 		self._touched()
#
#
# # this is our descriptor object
# class Bar(object):
# 	def __init__(self):
# 		self.values = EditableList()
#
# 	def __get__(self, instance, owner):
# 		print("returned from descriptor object", type(instance), type(owner), instance.its_me())
# 		return self.values
#
# 	def __set__(self, instance, values):
# 		print("set in descriptor object", type(instance), type(values), instance.its_me())
# 		self.values = EditableList(values)
# 		self.values._touched = instance.mangle
#
# 	def __delete__(self, instance):
# 		print("deleted in descriptor object", type(instance))
# 		del self.values
# 		self.values = EditableList()
#
# 	def __set_name__(self, owner, name):
# 		print("setname in descriptor object", type(owner), owner, name, type(self))
# 		self.name = name
# 		self.internal_name = '_' + name
#
#
# class Foo(object):
# 	bar = Bar()
# 	bax = 888.888
#
# 	def __init__(self):
# 		self.bar = []
#
# 	def mangle(self):
# 		print("MANGLED")
#
# 	def its_me(self):
# 		return repr(self.bax)