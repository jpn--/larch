import numpy
import pandas
from larch.core import LinearFunction

def _optional_index(y):
	if y is not None:
		return pandas.Index(y)

class ParameterCollection():
	def __init__(self, names, altindex,
				 utility_ca=None, utility_co=None):
		self._altindex = pandas.Index( altindex )
		# self._coef_utility_ca = numpy.asanyarray( param_ca, dtype=numpy.float64)
		# self._coef_utility_co = numpy.asanyarray( param_co, dtype=numpy.float64)

		self._utility_co_functions = utility_co or {}
		self._utility_ca_function  = utility_ca or LinearFunction()

		nameset = set(names)
		u_co_dataset = set()


		for altcode, linear_function in utility_co.items():
			for component in linear_function:
				nameset.add(str(component.param))
				u_co_dataset.add(str(component.data))
		self._u_co_varindex = pandas.Index( u_co_dataset )

		self._u_ca_varindex = pandas.Index( str(component.data) for component in self._utility_ca_function )

		for component in self._utility_ca_function:
			nameset.add(str(component.param))

		self._coef_utility_co = numpy.zeros( [len(self._u_co_varindex), len(self._altindex)], dtype=numpy.float64)
		self._coef_utility_ca = numpy.zeros( len(self._u_ca_varindex), dtype=numpy.float64)

		self._parameter_update_scheme = {}
		for n,component in enumerate(self._utility_ca_function):
			if str(component.param) not in self._parameter_update_scheme:
				self._parameter_update_scheme[str(component.param)] = []
			self._parameter_update_scheme[str(component.param)].append( ('_coef_utility_ca', (n,)) )

		for altcode, linear_function in utility_co.items():
			for component in linear_function:
				if str(component.param) not in self._parameter_update_scheme:
					self._parameter_update_scheme[str(component.param)] = []
				self._parameter_update_scheme[str(component.param)].append( ('_coef_utility_co', (self._u_co_varindex.get_loc(str(component.data)), self._altindex.get_loc(altcode) )) )

		for name in nameset:
			if name not in names:
				names.append(name)

		self._master = pandas.Series(index=names, dtype=numpy.float64)

	def set_value(self, name, value):
		self._master[name] = value
		if name in self._parameter_update_scheme:
			schemes = self._parameter_update_scheme[name]
			for member, location in schemes:
				self.__getattribute__(member)[location] = value

	def get_value(self, name):
		return self._master[name]

	# 'mu_motor', value=0.7257824244230557
	# ModelParameter('mu_nonmotor', value=0.7689340538871795)
