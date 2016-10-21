from itertools import count
import numpy
from .naming import parenthize



def smoothed_piecewise_linear(basevar, breaks, smoothness=1):
	from ..roles import X
	outs = [
		X(basevar),
	]
	
	if isinstance(smoothness, (int,float)):
		smoothness = numpy.full_like( breaks, smoothness )
	else:
		smoothness = numpy.asarray(smoothness)

	smoothness = 1.0/smoothness
	smoothness[~numpy.isfinite(smoothness)] = 0
	
	def maker(var, loc, smoo):
		if smoo==0:
			return X("fmax(0,{0}-{1})".format(var, loc), descrip=var+" (@{})".format(loc))
		if smoo==1:
			return X("(logaddexp(0,{0}-{1}))".format(var, loc), descrip=var+" (@{}~{})".format(loc,smoo))
		return X("(logaddexp(0,{2}*({0}-{1})))/{2}".format(var, loc, smoo), descrip=var+" (@{}~{})".format(loc,smoo))


	outs += [
		maker(basevar, loc, s) for loc,s in zip(breaks, smoothness)
	]
	return outs



def piecewise_linear_function(basevar, breaks, smoothness=1, baseparam=None):
	"""Smoothed piecewise linear function with marginal breakpoints."""
	from ..roles import P, X
	from ..core import LinearFunction
	Xs = smoothed_piecewise_linear(basevar, breaks, smoothness)
	if baseparam is None:
		baseparam = basevar
	Ps = [P(baseparam),]
	Ps += [ P("{}_{}".format(baseparam,b)) for b in breaks ]
	f = LinearFunction()
	for x,p in zip(Xs,Ps):
		f += x * p
	f._dimlabel=basevar
	return f

def gross_piecewise_linear_function(basevar, breaks, baseparam=None):
	"""Smoothed piecewise linear function with marginal breakpoints."""
	if baseparam is None:
		baseparam = basevar
	from ..roles import P, X
	from ..core import LinearFunction
	Xs = []
	Ps = []
	prev_b = 0
	for b in breaks:
		Xs += [X("fmin( {1}-{2} , fmax(0,{0}-{2}))".format(basevar, b, prev_b), descrip=basevar+" ({}-{})".format(prev_b,b))]
		Ps += [P("{0}_{2}_{1}".format(baseparam, b, prev_b))]
		prev_b = b
	Xs += [X("fmax(0,{0}-{1})".format(basevar, prev_b), descrip=basevar+" ({}+)".format(prev_b))]
	Ps += [P("{0}_{1}_up".format(baseparam, prev_b))]
	f = LinearFunction()
	for x,p in zip(Xs,Ps):
		f += x * p
	f._dimlabel=basevar
	return f


def polynomial_linear_function(basevar, powers, baseparam=None, invertpower=False, scaling=1):
	from ..roles import P, X
	from ..core import LinearFunction
	if invertpower:
		if scaling==1:
			Xs = [ X(basevar) if pwr==1 else X("({})**(1/{})".format(basevar,pwr)) for pwr in powers ]
		else:
			Xs = [ X(basevar)/scaling if pwr==1 else X("({})**(1/{})".format(basevar,pwr))/(scaling**(1/pwr)) for pwr in powers ]
	else:
		if scaling==1:
			Xs = [ X(basevar) if pwr==1 else X("({})**{}".format(basevar,pwr)) for pwr in powers ]
		else:
			Xs = [ X(basevar)/scaling if pwr==1 else X("({})**{}".format(basevar,pwr))/(scaling**pwr) for pwr in powers ]
	Ps = [ P(baseparam) if pwr==1 else P("{}_{}".format(baseparam,pwr)) for pwr in powers ]
	f = LinearFunction()
	for x,p in zip(Xs,Ps):
		f += x * p
	f._dimlabel=basevar
	return f


def log_and_linear_function(basevar, baseparam=None):
	from ..roles import P, X
	f = P(baseparam)*X(basevar) + P("log{}P1".format(baseparam))*X('log1p({})'.format(basevar))
	f._dimlabel=basevar
	return f

def log_and_piecewise_linear_function(basevar, breaks, smoothness=1, baseparam=None):
	from ..roles import P, X
	f = piecewise_linear_function(basevar, breaks, smoothness, baseparam) + P("log{}P1".format(baseparam))*X('log1p({})'.format(basevar))
	f._dimlabel=basevar
	return f

def log_and_gross_piecewise_linear_function(basevar, breaks, baseparam=None):
	from ..roles import P, X
	f = gross_piecewise_linear_function(basevar, breaks, baseparam) + P("log{}P1".format(baseparam))*X('log1p({})'.format(basevar))
	f._dimlabel=basevar
	return f



def _LinearFunction_evaluate(self, dataspace, model):
	if len(self)>0:
		i = self[0]
		y = i.data.eval(**dataspace) * i.param.default_value(0).value(model)
	for i in self[1:]:
		y += i.data.eval(**dataspace) * i.param.default_value(0).value(model)
	return y



#def smoothed_piecewise_linear(basevar, breaks):
#	from ..roles import X
#	outlen = len(breaks)-1
#	if outlen<2:
#		raise TypeError('there must be at least 3 values in breaks (min, cut, max)')
#
#	locs = []
#	scales = []
#
#	for low,mid,high in zip(breaks[:-2],breaks[1:-1],breaks[2:]):
#		if low>=mid or mid>=high:
#			raise TypeError('breaks must be strictly increasing')
#		locs.append(mid)
#		scales.append(numpy.sqrt((mid-low)*(high-mid))/4)
#
#	outs = [
#		X(basevar),
#	]
#
#	outs += [
#		X("({0})/(1+exp((-({0})+{1})/{2}))".format(basevar, loc, scal), descrip=basevar+" (Delay{})".format(n+1)) for loc,scal,n in zip(locs,scales,count())
#	]
#
#	return outs
#
#
#def smoothed_piecewise_linear_regular(basevar, breaks, scale):
#	from ..roles import X
#	outs = [
#		X(basevar),
#	]
#	outs += [
#		X("({0})/(1+exp((-({0})+{1})/{2}))".format(basevar, loc, scale), descrip=basevar+" (Delay{})".format(n+1)) for loc,n in zip(breaks,count())
#	]
#	return outs
#
#

def piecewise_decay(basevar, levels):
	from ..roles import X
	for lev in levels:
		if float(lev) <= 0:
			raise ValueError("piecewise_decay levels cannot include nonpositive values")
	return [X("exp(-{}/{})".format(parenthize(basevar, True), parenthize(lev,True)), descrip=basevar+" (Decay{})".format(lev)) for lev in levels]


def piecewise_decay_function(basevar, levels, keep_linear=False, baseparam=None):
	from ..roles import P, X
	from ..core import LinearFunction
	Xs = piecewise_decay(basevar, levels)
	if baseparam is None:
		baseparam = basevar
	if keep_linear:
		Ps = [P(baseparam),]
		Xs = [X(basevar),] + Xs
	else:
		Ps = []
	Ps += [ P("{}_{}".format(baseparam,b)) for b in levels ]
	f = LinearFunction()
	for x,p in zip(Xs,Ps):
		f += x * p
	f._dimlabel=basevar
	return f




