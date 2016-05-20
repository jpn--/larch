


class shadow_manager:

	def __init__(self, model):
		self.model = model

	def __getitem__(self, key):
		return self.model.alias(key)

	def __setitem__(self, key, value):
		from .roles import _param_multiply, _param_divide
		if isinstance(value, _param_multiply):
			if isinstance(value._left, (int, float)):
				number = value._left
				param = value._right
			else:
				number = value._right
				param = value._left
			if not isinstance(param, str):
				raise TypeError("shadow parameters must be a linear transform of a parameter")
			return self.model.alias(key, param, number)
		if isinstance(value, _param_divide):
			if isinstance(value._right, (int, float)):
				number = 1.0/value._right
				param = value._left
			else:
				raise TypeError("shadow parameters must be a linear transform of a parameter")
			if not isinstance(param, str):
				raise TypeError("shadow parameters must be a linear transform of a parameter")
			return self.model.alias(key, param, number)
		if isinstance(value, str):
			return self.model.alias(key, value, 1.0)

	def __repr__(self):
		return "<larch.shadow_manager {!r}>".format(self.model.alias_names())

	def __str__(self):
		return "<larch.shadow_manager {!s}>".format(self.model.alias_names())