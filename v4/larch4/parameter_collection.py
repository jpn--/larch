import numpy
import pandas


class ParameterCollection():
	def __init__(self, altindex, param_ca, utility_ca_index, param_co, utility_co_index):
		self._altindex = pandas.Index( altindex )
		self._u_ca_varindex = pandas.Index( utility_ca_index )
		self._u_co_varindex = pandas.Index( utility_co_index )
		self._coef_utility_ca = numpy.asanyarray( param_ca, dtype=numpy.float64)
		self._coef_utility_co = numpy.asanyarray( param_co, dtype=numpy.float64)

	def get_value(self, name):
		if name is 'mu_motor':
			return 0.7257824244230557
		elif name is 'mu_nonmotor':
			return 0.7689340538871795
		return 0.5

	# 'mu_motor', value=0.7257824244230557
	# ModelParameter('mu_nonmotor', value=0.7689340538871795)
