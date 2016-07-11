

class shadow_manager:

	def __init__(self, model):
		self._model = model

	def __getitem__(self, key):
		return self._model.alias(key)

	def __setitem__(self, key, value):
		from . import ModelParameter
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
			return self._model.alias(key, param, number)
		if isinstance(value, _param_divide):
			if isinstance(value._right, (int, float)):
				number = 1.0/value._right
				param = value._left
			else:
				raise TypeError("shadow parameters must be a linear transform of a parameter")
			if not isinstance(param, str):
				raise TypeError("shadow parameters must be a linear transform of a parameter")
			return self._model.alias(key, param, number)
		if isinstance(value, str):
			return self._model.alias(key, value, 1.0)
		if isinstance(value, ModelParameter):
			return self._model.alias(key, value.name, 1.0)
		raise TypeError("cannot assign {} to shadow_parameter".format(str(type(value))))

	def __repr__(self):
		return "<larch.shadow_manager {!r}>".format(self._model.alias_names())

	def __str__(self):
		return "<larch.shadow_manager {!s}>".format(self._model.alias_names())

	def __getattr__(self, key):
		if key=='_model':
			return self.__dict__['_model']
		return self.__getitem__(key)

	def __setattr__(self, key, value):
		if key=='_model':
			self.__dict__['_model'] = value
		else:
			self.__setitem__(key,value)



class metaparameter_manager:

	def __init__(self, model):
		self._model = model

	def __getitem__(self, key):
		if isinstance(key,int):
			return self._model[key]
		if key in self._model.parameter_names():
			return self._model.parameter(key)
		if key in self._model.alias_names():
			return self._model.alias(key)
		raise KeyError('{} not found'.format(key))

	def __setitem__(self, key, value):
		raise TypeError("cannot assign to metaparameter".format(str(type(value))))

	def __repr__(self):
		return "<larch.metaparameter_manager>"

	def __str__(self):
		return "<larch.metaparameter_manager>"

	def __getattr__(self, key):
		if key=='_model':
			return self.__dict__['_model']
		return self.__getitem__(key)

	def __setattr__(self, key, value):
		if key=='_model':
			self.__dict__['_model'] = value
		else:
			self.__setitem__(key,value)

	def __call__(self, key):
		return self.__getitem__(key)
