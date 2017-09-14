import numpy
import pandas
from .linalg.gemm import dgemm
from .linalg.gemv import dgemv
from .math.elementwise import sum_of_elementwise_product

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
		x=params.coef_utility_ca,
		beta=0,
		y=result_array_)
	dgemm(
		alpha=alpha_co,
		a=data_co,
		b=params.coef_utility_co,
		beta=1.0,
		c=result_array)


def _optional_array(x, **kwargs):
	if x is not None:
		return numpy.asanyarray(x, **kwargs)
	return None

def _optional_arg(*args):
	for x in args:
		if x is not None:
			return x
	return None

class DataCollection():
	def __init__(self, caseindex, altindex,
				 utility_ca_index=None,
				 utility_co_index=None,
				 quantity_ca_index=None,
				 utility_ca_data=None,
				 utility_co_data=None,
				 quantity_ca_data=None,
				 avail_data=None,
				 choice_ca_data=None,
				 source=None,
	             param_collect=None):
		self._source = source
		self._caseindex = pandas.Index( caseindex )
		self._altindex = pandas.Index( altindex )
		self._u_ca_varindex = pandas.Index( _optional_arg(utility_ca_index,[]) )
		self._u_co_varindex = pandas.Index( _optional_arg(utility_co_index ,[]) )
		self._q_ca_varindex = pandas.Index( _optional_arg(quantity_ca_index ,[]) )
		self._u_ca = _optional_array(utility_ca_data, dtype=numpy.float64) # shape = (C,A,Vca)
		self._u_co = _optional_array(utility_co_data, dtype=numpy.float64) # shape = (C,Vco)
		self._q_ca = _optional_array(quantity_ca_data, dtype=numpy.float64) # shape = (C,A,Qca)
		self._avail = _optional_array(avail_data, dtype=bool)                   # shape = (C,A)
		self._choice_ca = _optional_array(choice_ca_data, dtype=numpy.float64) # shape = (C,A)

	@property
	def n_cases(self):
		return len(self._caseindex)

	@property
	def n_alts(self):
		return len(self._altindex)

	def load_data(self, source=None):
		if source is None:
			source = self._source
		if source is None:
			raise ValueError('no data source known')
		if self._u_ca is None:
			self._u_ca = source.array_idca(*(self._u_ca_varindex))
		if self._u_co is None:
			self._u_co = source.array_idco(*(self._u_co_varindex))
		if self._q_ca is None:
			self._q_ca = source.array_idca(*(self._q_ca_varindex))
		if self._avail is None:
			self._avail = source.array_avail().squeeze()
		if self._choice_ca is None:
			self._choice_ca = source.array_choice().squeeze()

	def _calculate_exp_utility_elemental(self, params, result_array=None):
		if result_array is None:
			result_array = numpy.empty([len(self._caseindex), len(self._altindex)])
		_calculate_linear_product(params, self._u_ca, self._u_co, result_array)
		numpy.exp(result_array[:,:len(self._altindex)], where=self._avail, out=result_array[:,:len(self._altindex)])
		result_array[:, :len(self._altindex)][~self._avail] = 0
		return result_array

	def _calculate_log_like(self, logprob):
		return sum_of_elementwise_product(logprob, self._choice_ca)

