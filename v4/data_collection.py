import numpy
import pandas
import scipy
import scipy.linalg.blas


class ParameterCollection():

	def __init__(self, altindex, param_ca, param_ca_names, param_co, param_co_names):
		self._coef_utility_ca = pandas.Series(
			data=param_ca if param_ca is not None else numpy.zeros(len(param_ca_names)),
			index=param_ca_names,
			dtype=numpy.float64
			)    # shape = (V)
		self._coef_utility_co = pandas.DataFrame(
			dtype=numpy.float64,
			index=param_co_names,
			columns=altindex,
			data=param_co if param_co is not None else numpy.zeros([len(param_co_names), len(altindex)])
		) # shape = (V,A)
	



class DataCollection():

	def __init__(self, caseindex, altindex, data_ca, data_ca_names, data_co, data_co_names, avail):
		self.index = caseindex
		self.utility_ca = pandas.Panel(
			data=data_ca, 
			items=self.index, 
			major_axis=altindex, 
			minor_axis=data_ca_names, 
			dtype=numpy.float64
			)	# shape = (C,A,V)
		self.utility_co = pandas.DataFrame(
			data=data_co, 
			index=self.index, 
			columns=data_co_names, 
			dtype=numpy.float64
			) 	# shape = (C,V)
		self.avail = pandas.DataFrame(
			data=avail, 
			index=self.index, 
			columns=altindex, 
			dtype=numpy.bool
			) 	# shape = (C,A)
		
	def _calculate_linear_product(self, params, result_frame, alpha_ca=1.0, alpha_co=1.0):
		c,a,v1 = self.utility_ca.shape
		c2,v2 = self.utility_co.shape
		assert( c==c2 )
		print(result_frame.values)
		print(result_frame.values.flags)
		scipy.linalg.blas.dgemv(
			alpha=alpha_ca, 
			a=self.utility_ca.values.reshape(c*a,v1), 
			x=params._coef_utility_ca.values, 
			beta=0, 
			y=result_frame.values,    ### Need to transpose
			overwrite_y=1)
		print(result_frame.values)
		print(result_frame.values.flags)
		scipy.linalg.blas.dgemm(
			alpha=alpha_co, 
			a=self.utility_co.values, 
			b=params._coef_utility_co.values, 
			beta=1.0, 
			c=result_frame.values,   # This is correct, but the transpose fix above could muck it up
			overwrite_c=1)
		print(result_frame.values)
		print(result_frame.values.flags)
		
		
		return result_frame
		
		
		
		
		



if __name__=='__main__':
	import larch
	m = larch.Model.Example()
	m.provision()
	c = m.df.caseids()
	a = m.df.altids()
	dc = DataCollection(
		c,
		a,
		m.data.UtilityCA,
		m.needs()['UtilityCA'].get_variables(),
		m.data.UtilityCO,
		m.needs()['UtilityCO'].get_variables(),
		m.data.Avail.squeeze()
	)
	
	zer = numpy.zeros([len(c),len(a)])
	#zer = None
	u = pandas.DataFrame(data=zer, index=c, columns=a, dtype=numpy.float64)
	
	coef_ca = numpy.array([[[-0.05133987]],

       [[-0.0049205 ]]]).squeeze()
	
	coef_co = numpy.array([[[  0.00000000e+00],
        [ -2.17805758e+00],
        [ -3.72514509e+00],
        [ -6.70978857e-01],
        [ -2.37636315e+00],
        [ -2.06847334e-01]],

       [[  0.00000000e+00],
        [ -2.16997114e-03],
        [  3.57474000e-04],
        [ -5.28629571e-03],
        [ -1.28083368e-02],
        [ -9.68624390e-03]]]).squeeze()
        
        
	coef_ca *= 0
	coef_co *= 0
# 	coef_co[0,1] = 0.1
# 	coef_co[1,3] = 0.1
	coef_ca[1] = 0.1
	p = ParameterCollection(a, coef_ca, m.needs()['UtilityCA'].get_variables(), coef_co, m.needs()['UtilityCO'].get_variables() )
	
	dc._calculate_linear_product(p, u)
	
# 	u.values[~dc.avail.values] = 0
# 	numpy.exp(u, out=u.values, where=dc.avail.values)
# 	uT = u.values.T
# 	uT /= uT.sum(0)
	
	
	