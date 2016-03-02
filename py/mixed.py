
import numpy
from scipy.optimize import minimize as _minimize

class NormalMixedModel:
	def __init__(self, kernel, random_parameters, ndraws=500, seed=None):
		super().__init__()
		self.ndraws = ndraws
		self._rand_par_screen = numpy.zeros(len(kernel), dtype=bool)
		for rp in random_parameters:
			if isinstance(rp,str) and rp not in kernel:
				raise KeyError("cannot find parameter {}".format(rp))
			if isinstance(rp,int) and (rp >= len(kernel) or rp < 0):
				raise KeyError("cannot find parameter {}".format(rp))
			self._rand_par_screen[kernel[rp]._get_index()] = True
		self._rand_par_values = numpy.empty([ndraws, len(kernel)], dtype=numpy.float64)
		self._rand_par_values[:,:] = kernel.parameter_array[None,:]
		if seed is not None:
			numpy.random.seed(seed)
		self._rand_draws = numpy.random.normal(size=[ndraws, self.nrand()])
		self.kernel = kernel
		self.kernel.setUp()
		self.choleski = numpy.zeros([self.nrand(),self.nrand()], dtype=numpy.float64)
		self.simulated_probabity = numpy.zeros([kernel.nCases(), kernel.nAlts()])

	def nrand(self):
		return int(self._rand_par_screen.sum())

	def __len__(self):
		return len(self.kernel) - self.nrand() + int(self.nrand()*(self.nrand()+1)/2)

	def simulate_probabity(self, *args):
		self._set_parameter_values(*args)
		self.simulated_probabity[:,:] = 0
#		self._rand_par_values[:,:] = self.kernel.parameter_array[None,:]
#		self._rand_par_values[:,self._rand_par_screen] = numpy.dot(self._rand_draws,self.choleski)
		for draw in range(self.ndraws):
			self.simulated_probabity += self.kernel.probability( self._rand_par_values[draw,:] )[:,:self.kernel.nAlts()]
		self.simulated_probabity /= self.ndraws

	def loglike(self, *args):
		self.simulate_probabity(*args)
		ll = self.kernel.log_likelihood_from_prob(self.simulated_probabity)
		if len(args)>0:
			print("LL[{!s}]={}".format(args[0],ll))
		else:
			print("LL[{!s}]={}".format(self.parameter_values(),ll))
		return ll

	def negative_loglike(self, *args):
		return -self.loglike(*args)

	def _set_parameter_values(self, *args):
		if len(args)>0:
			border = self._border()
			self.kernel.parameter_array[~self._rand_par_screen] = args[0][:border]
			self.choleski[numpy.triu_indices(self.nrand())] = args[0][border:]
			self._rand_par_values[:,self._rand_par_screen] = numpy.dot(self._rand_draws,self.choleski)
			self._rand_par_values[:,~self._rand_par_screen] = args[0][:border]

	def _border(self):
		return len(self.kernel)-self.nrand()

	def parameter_values(self, *args):
		if len(args):
			return self._set_parameter_values(*args)
		cholindex = numpy.triu_indices(self.nrand())
		np = len(self.kernel) -self.nrand() + len(cholindex[0])
		v = numpy.empty(np)
		v[:len(self.kernel)][~self._rand_par_screen] = self.kernel.parameter_array[~self._rand_par_screen]
		v[len(self.kernel)-self.nrand():] = self.choleski[cholindex]
		return v

	def maximize_loglike(self, **kwargs):
		return _minimize(self.negative_loglike, self.parameter_values(), **kwargs)

#	def d_loglike_mnl(self, *args):
#		n_utilco_param_shape = (len(self.kernel.utility.co.needs()), self.kernel.nAlts())
#		n_utilca_params = len(self.kernel.utility.ca.needs())
#		d_util_ca = numpy.empty(self.simulated_probabity.shape + (n_utilca_params,))
#		d_util_co = numpy.empty(self.simulated_probabity.shape + n_utilco_param_shape )
#		d_util_ca_root = numpy.empty([self.kernel.nCases(), n_utilca_params])
#		d_util_co_root = numpy.empty((self.kernel.nCases(), )+n_utilco_param_shape)
#		d_pr_ca = numpy.empty(self.simulated_probabity.shape + (n_utilca_params,))
#		d_pr_co = numpy.empty(self.simulated_probabity.shape + n_utilco_param_shape )
#		self._set_parameter_values(*args)
#		data_co = self.kernel.Data("UtilityCO")
#		data_ca = self.kernel.Data("UtilityCA")
#		coef_co = self.kernel.Coef("UtilityCO")
#		coef_ca = self.kernel.Coef("UtilityCA")
#		for anum in range(self.kernel.nAlts()):
#			d_util_ca[:,anum,:] = data_ca[:,anum,:]
#			d_util_co[:,anum,:,:] = 0
#			d_util_co[:,anum,:,anum] = data_co[:,:]
#		d_util_ca_root[:,:] =
#		for vnum, vname in enumerate(m.utility.co.needs()):
#			for anum in range(self.kernel.nAlts()):
#				d_pr_co[:,anum,:] = pr[:,anum,None] * data_co[:,:] + pr[:,anum,None] * pr[:,anum,None] * data_co[:,:]
#			data_co[]


	def d_prob(self):
		border = self._border()
		dPr = numpy.zeros([self.kernel.nCases(), self.kernel.nAlts(), len(self)])
		
		for draw in range(self.ndraws):
			dPr_draw = numpy.zeros([self.kernel.nCases(), self.kernel.nAlts(), len(self)])
			self.kernel.parameter_array[:] = self._rand_par_values[draw,:]
			self.kernel.probability()
			temp = self.kernel._ngev_d_prob()
			temp2 = temp[:,:self.kernel.nAlts(),self._rand_par_screen]
			dPr[:,:,:border] += temp[:,:self.kernel.nAlts(),~self._rand_par_screen]
			dPr_draw[:,:,:border] += temp[:,:self.kernel.nAlts(),~self._rand_par_screen]
			temp3 = temp2[:,:,numpy.triu_indices(self.nrand())[1]] * self._rand_draws[draw, numpy.triu_indices(self.nrand())[0]][None,None,:]
			dPr[:,:,border:] += temp3  ## maybe 1 not 0
			dPr_draw[:,:,border:] += temp3  ## maybe 1 not 0
		dPr /= self.ndraws
		return dPr

	def d_ll(self):
		self.simulate_probabity()
		dPr = self.d_prob()
		dLL = numpy.zeros(len(self))
		for c in range(self.kernel.nCases()):
			for a in range(self.kernel.nAlts()):
				if self.simulated_probabity[c,a]>0 and self.kernel.Data("Choice")[c,a]:
					dLL += self.kernel.Data("Choice")[c,a]/self.simulated_probabity[c,a] * dPr[c,a,:]
		return dLL
			