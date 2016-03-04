
import numpy
from scipy.optimize import minimize as _minimize
from .logging import getLogger as _getLogger

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
		self.kernel.teardown()
		self.kernel._force_feed(3)
		self.kernel.setUp()
		self.choleski = numpy.zeros([self.nrand(),self.nrand()], dtype=numpy.float64)
		self.simulated_probability = numpy.zeros([kernel.nCases(), kernel.nAlts()])

	def nrand(self):
		return int(self._rand_par_screen.sum())

	def __len__(self):
		return len(self.kernel) - self.nrand() + int(self.nrand()*(self.nrand()+1)/2)

	def simulate_probability(self, *args):
		self._set_parameter_values(*args)
		self.simulated_probability[:,:] = 0
		for draw in range(self.ndraws):
			self.simulated_probability += self.kernel.probability( self._rand_par_values[draw,:] )[:,:self.kernel.nAlts()]
		self.simulated_probability /= self.ndraws

	def loglike(self, *args):
		self.simulate_probability(*args)
		ll = self.kernel.log_likelihood_from_prob(self.simulated_probability)
		if len(args)>0:
			_getLogger().critical("LL [{!s}]={}".format(args[0],ll))
		else:
			_getLogger().critical("LL [{!s}]={}".format(self.parameter_values(),ll))
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
		return _minimize(self.negative_loglike, self.parameter_values(), jac=self.negative_d_loglike, **kwargs)


	def d_prob(self):
		border = self._border()
		dPr = numpy.zeros([self.kernel.nCases(), self.kernel.nAlts(), len(self)])
		self.simulated_probability[:,:] = 0
		for draw in range(self.ndraws):
			self.kernel.parameter_array[:] = self._rand_par_values[draw,:]
			self.simulated_probability += self.kernel.probability()[:,:self.kernel.nAlts()]
			temp = self.kernel._ngev_d_prob()
			temp2 = temp[:,:self.kernel.nAlts(),self._rand_par_screen]
			dPr[:,:,:border] += temp[:,:self.kernel.nAlts(),~self._rand_par_screen]
			temp3 = temp2[:,:,numpy.triu_indices(self.nrand())[1]] * self._rand_draws[draw, numpy.triu_indices(self.nrand())[0]][None,None,:]
			dPr[:,:,border:] += temp3
		dPr /= self.ndraws
		self.simulated_probability /= self.ndraws
		return dPr

	def d_loglike(self, *args):
		self._set_parameter_values(*args)
		dPr = self.d_prob()
		factor = self.kernel.Data("Choice")/self.simulated_probability[:,:,None]
		factor[ ~numpy.isfinite(factor) ] = 0
		if len(args)>0:
			_getLogger().critical("dLL[{!s}]".format(args[0],))
		else:
			_getLogger().critical("dLL[{!s}]".format(self.parameter_values(),))
		return numpy.sum(factor*dPr, axis=(0,1))

	def negative_d_loglike(self, *args):
		return -self.d_loglike(*args)
