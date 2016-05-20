
from .core import ModelParameter

class ParameterManager:
	"""Manages parameters for a :class:`Model`.	"""


	def __init__(self, model):
		self.model = model

	def __call__(self, *arg, **kwarg):
		return self.model._parameter_(*arg, **kwarg)

	def __getitem__(self, key):
		if isinstance(key,int):
			return self.model[key]
		return self.model.parameter(key)

	def __setitem__(self, key, val):
		from .roles import _param_multiply, _param_divide
		if isinstance(val, ModelParameter):
			return self.model.parameter(key, value=val.value, null_value=val.null_value,
										initial_value=val.initial_value,
										max=val.max_value, min=val.min_value,
										holdfast=val.holdfast)
		if isinstance(val, (int,float)):
			return self.model.parameter(key, value=val, null_value=val, initial_value=val)
		if isinstance(val, _param_multiply):
			if isinstance(val._left, (int, float)):
				number = val._left
				param = val._right
			else:
				number = val._right
				param = val._left
			if not isinstance(param, str):
				raise TypeError("shadow parameters must be a linear transform of a parameter")
			return self.model.alias(key, param, number)
		if isinstance(val, _param_divide):
			if isinstance(val._right, (int, float)):
				number = 1.0/val._right
				param = val._left
			else:
				raise TypeError("shadow parameters must be a linear transform of a parameter")
			if not isinstance(param, str):
				raise TypeError("shadow parameters must be a linear transform of a parameter")
			return self.model.alias(key, param, number)
		if isinstance(val, str):
			return self.model.alias(key, value, 1.0)

	def __repr__(self):
		ret = "═"*80
		ret += "\nlarch.ParameterManager for {0}".format(self.model.title)
		ret += ("\n────┬"+"─"*75)
		for n,name in enumerate(self.model.parameter_names()):
			ret += "\n{:> 3d} │ {}".format(n,name)
		aliases = self.model.alias_names()
		if len(aliases):
			ret += "\n────┼"+"─"*75
			for n,name in enumerate(self.model.alias_names()):
				ret += "\n    │ {!s}".format(self.model.alias(name))
		ret += "\n════╧"+"═"*75
		return ret

	def __str__(self):
		return repr(self)


