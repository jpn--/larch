import numpy
import pandas
from .linalg.gemm import dgemm
from .linalg.gemv import dgemv


def _calculate_linear_product(params, data_ca, data_co, result_array, alpha_ca=1.0, alpha_co=1.0):
	c, a, v1 = data_ca.shape
	c2, v2 = data_co.shape
	assert (c == c2)
	assert (c == result_array.shape[0])
	assert (a <= result_array.shape[1])
	data_ca_ = data_ca.view()
	data_ca_.shape = (c * a, v1)
	result_array_ = result_array[:,:a].view()
	result_array_.shape = (c * a,)
	dgemv(
		alpha=alpha_ca,
		a=data_ca_,
		x=params._coef_utility_ca,
		beta=0,
		y=result_array_)
	dgemm(
		alpha=alpha_co,
		a=data_co,
		b=params._coef_utility_co,
		beta=1.0,
		c=result_array)


class DataCollection():
	def __init__(self, caseindex, altindex, utility_ca_index, utility_co_index, utility_ca_data, utility_co_data, avail):
		self._caseindex = pandas.Index( caseindex )
		self._altindex = pandas.Index( altindex )
		self._u_ca_varindex = pandas.Index( utility_ca_index )
		self._u_co_varindex = pandas.Index( utility_co_index )
		self._u_ca = numpy.asanyarray(utility_ca_data, dtype=numpy.float64) # shape = (C,A,Vca)
		self._u_co = numpy.asanyarray(utility_co_data, dtype=numpy.float64) # shape = (C,Vco)
		self._avail = numpy.asanyarray(avail, dtype=bool)                   # shape = (C,A)

	def _calculate_exp_utility_elemental(self, params, result_array=None):
		if result_array is None:
			result_array = numpy.empty([len(self._caseindex), len(self._altindex)])
		_calculate_linear_product(params, self._u_ca, self._u_co, result_array)
		numpy.exp(result_array[:,:len(self._altindex)], where=self._avail, out=result_array[:,:len(self._altindex)])
		result_array[:, :len(self._altindex)][~self._avail] = 0
		return result_array



