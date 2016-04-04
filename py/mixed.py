
import numpy
from scipy.optimize import minimize as _minimize
from .logging import getLogger as _getLogger
from .util.attribute_dict import function_cache

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
		self._cached_results = function_cache()

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

	def _mixed_loglike(self, *args):
		self.simulate_probability(*args)
		ll = self.kernel.log_likelihood_from_prob(self.simulated_probability)
		if len(args)>0:
			_getLogger().info("LL [{!s}]={}".format(args[0],ll))
		else:
			_getLogger().info("LL [{!s}]={}".format(self.parameter_values(),ll))
		return ll

	def loglike(self, *args, cached=True, holdfast_unmask=0, blp_contraction_threshold=1e-8):
		if len(args)>0:
			self.parameter_values(args[0], holdfast_unmask)
		cache_mask = numpy.asarray(self.parameter_values())
		cache_mask[:len(self.kernel.parameter_holdfast_array)][self.kernel.parameter_holdfast_array==2] = 0
		if cached:
			try:
				ll= self._cached_results[cache_mask.tobytes()].loglike
#				print("LL",ll,"(cached)","<-",self.parameter_values())
				return ll
			except (KeyError, AttributeError):
				pass
		if hasattr(self,'blp_shares_map') and hasattr(self,'logmarketshares') and blp_contraction_threshold is not None:
			# BLP contraction
			delta_norm = 1e9
			delta_norm_prev = 1e10
			while delta_norm > blp_contraction_threshold and delta_norm<delta_norm_prev*.99:
				delta_norm_prev = delta_norm
				self.simulate_probability(self.parameter_values())
				pr_sum = (self.simulated_probability*self.kernel.Data("Weight")).sum(0)
				pr_sum /= pr_sum.sum()
				delta = self.logmarketshares - numpy.log(pr_sum)
				self.kernel.parameter_array[self.blp_shares_map] += delta
				self._set_parameter_values()
				delta_norm = numpy.sum(delta**2)
#				print(self.parameter_values(),"delta_norm",delta_norm)
			# remean shocks
			mean_shock = numpy.mean(self.kernel.parameter_array[self.blp_shares_map])
			self.kernel.parameter_array[self.blp_shares_map] -= mean_shock
			self._set_parameter_values()
		else:
			self.simulate_probability()
		# otherwise not cached (or not correctly) so calculate anew
		ll = self.kernel.log_likelihood_from_prob(self.simulated_probability)
		if numpy.isnan(ll):
			ll = -numpy.inf
		if isinstance(self._cached_results, function_cache):
			self._cached_results[cache_mask.tobytes()].loglike = ll
#		print("LL",ll,"<-",self.parameter_values())
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
		else:
			self._rand_par_values[:,~self._rand_par_screen] = self.kernel.parameter_array[~self._rand_par_screen]

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

	def maximize_loglike(self, method='SLSQP', **kwargs):
		return _minimize(self.negative_loglike, self.parameter_values(), jac=self.negative_d_loglike, method=method, **kwargs)


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
			_getLogger().info("dLL[{!s}]".format(args[0],))
		else:
			_getLogger().info("dLL[{!s}]".format(self.parameter_values(),))
		ret = numpy.sum(factor*dPr, axis=(0,1))
#		print(ret)
		return ret

	def negative_d_loglike(self, *args):
		return -self.d_loglike(*args)

	def __str__(self):
		s = "<larch.mixed.NormalMixedModel> Temporary Report\n"
		s += "="*60+"\n"
		for name,value in zip(self.kernel.parameter_names(output_type=numpy.array)[~self._rand_par_screen], self.kernel.parameter_array[~self._rand_par_screen]):
			s += "{:25s}\t{: 0.6g}\n".format(name,value)
		for index, value in enumerate(self.choleski[numpy.triu_indices(self.nrand())]):
			s += "Choleski_{:<16}\t{: 0.6g}\n".format(index,value)
		s += "="*60
		return s

	def setup_blp_contraction(self, shares_map, log_marketshares=None):
		assert( len(self.kernel.alternative_codes()) == len(shares_map) )
		self.blp_shares_map = shares_map
		for i in shares_map:
			self.kernel[i].holdfast = 2
		if log_marketshares is None:
			ch = self.kernel.Data("Choice")
			shares = ch.sum(0)
			shares /= shares.sum()
			self.logmarketshares = numpy.log(shares).squeeze()
		else:
			self.logmarketshares = log_marketshares

