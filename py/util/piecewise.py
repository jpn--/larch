from itertools import count
import numpy
from .naming import parenthize



def smoothing_blocks(basevar, turns, smoothness):
	prev_t, prev_s = None,None
	if isinstance(smoothness, (int,float)) or len(smoothness)==1 or smoothness is None:
		smoothness = numpy.full_like(turns, smoothness)
	Xs = []
	for t,si in zip(turns, smoothness):
		if si==0:
			if prev_t is None:
				y = "1.0*({0}<{2})".format(basevar, prev_t, t, prev_s, s)
			else:
				y = "({0}>{1}) * ({0}<{2})".format(basevar, prev_t, t, prev_s, s)
		else:
			s = 1/si
			if s==1:
				s_txt = ""
			else:
				s_txt = "{}*".format(s)
			if prev_s==1:
				ps_txt = ""
			else:
				ps_txt = "{}*".format(prev_s)

			if prev_t is None:
				y = "(1/(1)-(1/(1+exp(-{4}({0}-{2})))))".format(basevar, prev_t, t, ps_txt, s_txt)
			else:
				y = "(1/(1+exp(-{3}({0}-{1})))-(1/(1+exp(-{4}({0}-{2})))))".format(basevar, prev_t, t, ps_txt, s_txt)
		Xs.append(y)
		prev_t, prev_s = t,s
	
	if s==1:
		s_txt = ""
	else:
		s_txt = "{}*".format(s)
	if prev_s==1:
		ps_txt = ""
	else:
		ps_txt = "{}*".format(prev_s)
	y = "(1/(1+exp(-{3}({0}-{1}))))".format(basevar, prev_t, t, ps_txt, s_txt)
	Xs.append(y)
	return Xs



def smoothed_piecewise_function(basevar, blockvar, turns, smoothness, baseparam=None):
	if baseparam is None:
		baseparam = basevar
	from ..roles import P, X
	from ..core import LinearFunction
	Xs = [X(basevar)*X(x) for x in smoothing_blocks(blockvar, turns, smoothness)]
	Ps = []
	prev_b = None
	for b in turns:
		if prev_b is None:
			Ps += [P("{0}_up_to_{1}".format(baseparam, b))]
		else:
			Ps += [P("{0}_{2}_{1}".format(baseparam, b, prev_b))]
		prev_b = b
	Ps += [P("{0}_{1}_and_up".format(baseparam, prev_b))]
	f = LinearFunction()
	for x,p in zip(Xs,Ps):
		f += x * p
	f._x_ident=blockvar
	return f



def smoothed_piecewise_linear(basevar, breaks, smoothness=1):
	from ..roles import X
	outs = [
		X(basevar),
	]
	
	if isinstance(smoothness, (int,float)):
		smoothness = numpy.full_like( breaks, smoothness )
	else:
		smoothness = numpy.asarray(smoothness)

	while len(breaks) > len(smoothness):
		smoothness = numpy.resize(smoothness,len(smoothness)+1)
		smoothness[-1] = smoothness[-2]

	smoothness = 1.0/smoothness
	smoothness[~numpy.isfinite(smoothness)] = 0
	
	def maker(var, loc, smoo):
		if smoo==0:
			return X("fmax(0,{0}-{1})".format(var, loc), descrip=var+" (@{})".format(loc))
		if smoo==1:
			return X("(logaddexp(0,{0}-{1}))".format(var, loc), descrip=var+" (@{}~{})".format(loc,numpy.round(smoo,5)))
		return X("(logaddexp(0,{2}*({0}-{1})))/{2}".format(var, loc, smoo), descrip=var+" (@{}~{})".format(loc,numpy.round(smoo,5)))


	outs += [
		maker(basevar, loc, s) for loc,s in zip(breaks, smoothness)
	]
	return outs


def smoothed_piecewise_logarithmic(basevar, breaks, smoothness=1):
	breaks_l = [numpy.log1p(b) for b in breaks]
	basevar_l = "log1p({})".format(basevar)

	from ..roles import X
	outs = [
		X(basevar_l),
	]
	
	if isinstance(smoothness, (int,float)):
		smoothness = numpy.full_like( breaks, smoothness )
	else:
		smoothness = numpy.asarray(smoothness)

	smoothness = 1.0/smoothness
	smoothness[~numpy.isfinite(smoothness)] = 0
	
	def maker(var, loc, smoo, loc_label):
		if smoo==0:
			return X("fmax(0,{0}-{1})".format(var, loc), descrip=var+" (@{})".format(loc_label))
		if smoo==1:
			return X("(logaddexp(0,{0}-{1}))".format(var, loc), descrip=var+" (@{}~{})".format(loc_label,numpy.round(smoo,5)))
		return X("(logaddexp(0,{2}*({0}-{1})))/{2}".format(var, loc, smoo), descrip=var+" (@{}~{})".format(loc_label,numpy.round(smoo,5)))

	outs += [
		maker(basevar_l, loc, s, loclabel) for loc,s,loclabel in zip(breaks_l, smoothness, breaks)
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
	f._x_ident=basevar
	return f

def piecewise_logarithmic_function(basevar, breaks, smoothness=1, baseparam=None):
	"""Smoothed piecewise logarithmic function with marginal breakpoints."""
	from ..roles import P, X
	from ..core import LinearFunction
	Xs = smoothed_piecewise_logarithmic(basevar, breaks, smoothness)
	if baseparam is None:
		baseparam = basevar
	Ps = [P(baseparam),]
	Ps += [ P("{}_{}".format(baseparam,b)) for b in breaks ]
	f = LinearFunction()
	for x,p in zip(Xs,Ps):
		f += x * p
	f._x_ident=basevar
	return f


def gross_piecewise_linear_function(basevar, breaks, smoothness=1, baseparam=None):
	"""Smoothed piecewise linear function with gross breakpoints."""
	if smoothness==0 or smoothness is None:
		return nonsmooth_gross_piecewise_linear_function(basevar, breaks, baseparam)
	from ..roles import P, X
	from ..core import LinearFunction
	Xs = smoothed_piecewise_linear(basevar, breaks, smoothness)
	if baseparam is None:
		baseparam = basevar
	Ps = []
	prev_b = None
	for b in breaks:
		if prev_b is None:
			Ps += [ P("{}_up_to_{}".format(baseparam,b)) ]
		else:
			Ps += [ P("{}_{}_{}".format(baseparam,prev_b,b)) ]
		prev_b = b
	Ps += [P("{}_{}_and_up".format(baseparam,prev_b))]
	f = LinearFunction()
	n = len(Xs)-1
	for x1,x2,p in zip(Xs[:-1],Xs[1:],Ps):
		f += (x1-x2) * p
	f += Xs[-1] * Ps[-1]
	f._x_ident=basevar
	return f


def gross_piecewise_logarithmic_function(basevar, breaks, smoothness=1, baseparam=None):
	"""Smoothed piecewise linear function with gross breakpoints."""
	if smoothness==0 or smoothness is None:
		return nonsmooth_gross_piecewise_logarithmic_function(basevar, breaks, baseparam)
	from ..roles import P, X
	from ..core import LinearFunction
	Xs = smoothed_piecewise_logarithmic(basevar, breaks, smoothness)
	if baseparam is None:
		baseparam = basevar
	Ps = []
	prev_b = None
	for b in breaks:
		if prev_b is None:
			Ps += [ P("{}_up_to_{}".format(baseparam,b)) ]
		else:
			Ps += [ P("{}_{}_{}".format(baseparam,prev_b,b)) ]
		prev_b = b
	Ps += [P("{}_{}_and_up".format(baseparam,prev_b))]
	f = LinearFunction()
	n = len(Xs)-1
	for x1,x2,p in zip(Xs[:-1],Xs[1:],Ps):
		f += (x1-x2) * p
	f += Xs[-1] * Ps[-1]
	f._x_ident=basevar
	return f









def piecewise_linear_function_with_log_tail(basevar, breaks, smoothness=1, baseparam=None, tail=None):
	"""Smoothed piecewise linear function with marginal breakpoints and a log tail."""
	if baseparam is None:
		baseparam = basevar
	f = piecewise_linear_function(basevar, breaks, smoothness, baseparam)
	if tail is None:
		tail=breaks[-1]
	from ..roles import P, X
	f += P('log{0}P1_over{1}'.format(baseparam, tail)) * X('log1p({0})*({0}>{1})'.format(basevar, tail))
	f._x_ident=basevar
	return f


def nonsmooth_gross_piecewise_linear_function(basevar, breaks, baseparam=None):
	"""Piecewise linear function with gross breakpoints.
	
	The gross break points version of a piecewise curve is written such that
	each piece's coefficient represents the gross (entire) slope, not the
	marginal change in slope relative to the previous segment.  This makes 
	it so the t-statistics do not reflect the signifigance of change.  On
	the other hand, constraints for making the slope globally positive or 
	negative are simpler to construct.
	
	Parameters
	----------
	basevar : str
		The variable to use as the base variable.
	breaks : sequence of floats
		The break points of the piecewise linear function. Note that 
		if N breaks are given, there will be N+1 segments.
	baseparam : str or None
		The base parameter name.  If not given, `basevar` is used.

	Returns
	-------
	LinearFunction
		The piecewise linear function.

	Examples
	--------
	>>> from larch.util.piecewise import gross_piecewise_linear_function
	>>> f = gross_piecewise_linear_function('Aaa', [1,2,3])
	>>> print(f)
	  = P('Aaa_up_to_1') * X('fmin( 1 , Aaa)')
	  + P('Aaa_1_2') * X('fmin( 2-1 , fmax(0,Aaa-1))')
	  + P('Aaa_2_3') * X('fmin( 3-2 , fmax(0,Aaa-2))')
	  + P('Aaa_3_and_up') * X('fmax(0,Aaa-3)')
	>>> f2 = gross_piecewise_linear_function('Aaa', [1,20,300], baseparam='ParName')
	>>> print(f2)
	  = P('ParName_up_to_1') * X('fmin( 1-0 , fmax(0,Aaa-0))')
	  + P('ParName_1_20') * X('fmin( 20-1 , fmax(0,Aaa-1))')
	  + P('ParName_20_300') * X('fmin( 300-20 , fmax(0,Aaa-20))')
	  + P('ParName_300_and_up') * X('fmax(0,Aaa-300)')
	"""
	if baseparam is None:
		baseparam = basevar
	from ..roles import P, X
	from ..core import LinearFunction
	Xs = []
	Ps = []
	prev_b = None
	for b in breaks:
		if prev_b is None:
			Xs += [X("fmin( {1} , {0})".format(basevar, b), descrip=basevar+" (up to {})".format(b))]
			Ps += [P("{0}_up_to_{1}".format(baseparam, b))]
		else:
			Xs += [X("fmin( {1}-{2} , fmax(0,{0}-{2}))".format(basevar, b, prev_b), descrip=basevar+" ({}-{})".format(prev_b,b))]
			Ps += [P("{0}_{2}_{1}".format(baseparam, b, prev_b))]
		prev_b = b
	Xs += [X("fmax(0,{0}-{1})".format(basevar, prev_b), descrip=basevar+" ({}+)".format(prev_b))]
	Ps += [P("{0}_{1}_and_up".format(baseparam, prev_b))]
	f = LinearFunction()
	for x,p in zip(Xs,Ps):
		f += x * p
	f._x_ident=basevar
	return f


def nonsmooth_gross_piecewise_logarithmic_function(basevar, breaks, baseparam=None):
	"""Piecewise logarithmic function with gross breakpoints.
	
	The gross break points version of a piecewise curve is written such that
	each piece's coefficient represents the gross (entire) slope, not the
	marginal change in slope relative to the previous segment.  This makes 
	it so the t-statistics do not reflect the signifigance of change.  On
	the other hand, constraints for making the slope globally positive or 
	negative are simpler to construct.
	
	Parameters
	----------
	basevar : str
		The variable to use as the base variable.
	breaks : sequence of floats
		The break points of the piecewise linear function. Note that 
		if N breaks are given, there will be N+1 segments.
	baseparam : str or None
		The base parameter name.  If not given, `basevar` is used.

	Returns
	-------
	LinearFunction
		The piecewise logarithmic function.

	Examples
	--------
	>>> from larch.util.piecewise import gross_piecewise_linear_function
	>>> f = gross_piecewise_linear_function('Aaa', [1,2,3])
	>>> print(f)
	  = P('Aaa_up_to_1') * X('fmin( 1 , Aaa)')
	  + P('Aaa_1_2') * X('fmin( 2-1 , fmax(0,Aaa-1))')
	  + P('Aaa_2_3') * X('fmin( 3-2 , fmax(0,Aaa-2))')
	  + P('Aaa_3_and_up') * X('fmax(0,Aaa-3)')
	>>> f2 = gross_piecewise_linear_function('Aaa', [1,20,300], baseparam='ParName')
	>>> print(f2)
	  = P('ParName_up_to_1') * X('fmin( 1-0 , fmax(0,Aaa-0))')
	  + P('ParName_1_20') * X('fmin( 20-1 , fmax(0,Aaa-1))')
	  + P('ParName_20_300') * X('fmin( 300-20 , fmax(0,Aaa-20))')
	  + P('ParName_300_and_up') * X('fmax(0,Aaa-300)')
	"""
	if baseparam is None:
		baseparam = basevar
	from ..roles import P, X
	from ..core import LinearFunction
	Xs = []
	Ps = []
	prev_b = None
	for b in breaks:
		if prev_b is None:
			Xs += [X("fmin( {1} , log1p({0}))".format(basevar, numpy.log1p(b)), descrip=basevar+" (up to {})".format(b))]
			Ps += [P("{0}_up_to_{1}".format(baseparam, b))]
		else:
			Xs += [X("fmin( {1}-{2} , fmax(0,log1p({0})-{2}))".format(basevar, numpy.log1p(b), numpy.log1p(prev_b)), descrip=basevar+" ({}-{})".format(prev_b,b))]
			Ps += [P("{0}_{2}_{1}".format(baseparam, b, prev_b))]
		prev_b = b
	Xs += [X("fmax(0,{0}-{1})".format(basevar, numpy.log1p(prev_b)), descrip=basevar+" ({}+)".format(prev_b))]
	Ps += [P("{0}_{1}_and_up".format(baseparam, prev_b))]
	f = LinearFunction()
	for x,p in zip(Xs,Ps):
		f += x * p
	f._x_ident=basevar
	return f



def gross_piecewise_linear_function_with_log_tail(basevar, breaks, baseparam=None):
	"""Piecewise linear function with gross breakpoints and a log tail."""
	if baseparam is None:
		baseparam = basevar
	f = gross_piecewise_linear_function(basevar, breaks, baseparam)
	from ..roles import P, X
	f = sum(f[:-1]) + P('log{0}P1_over{1}'.format(baseparam, breaks[-1])) * X('log1p({0})*({0}>{1})'.format(basevar, breaks[-1]))
	f._x_ident=basevar
	return f


def polynomial_linear_function(basevar, powers, baseparam=None, invertpower=False, scaling=1):
	"""
	Create a polynomial LinearFunction.
	
	The resulting function looks like :math:`{ \\beta }_1 X^{n_1} + { \\beta }_2 X^{n_2} + { \\beta }_3 X^{n_3} + ...`.
	
	Note that the function is polynomial only in the data, and the powers are given explicitly.
	The resulting function is still a linear-in-parameters function of data (and transformed data),
	and not actually a non-linear function of parameters that will be estimated.
	
	Parameters
	----------
	basevar : str
		The variable to use as the base variable.
	powers : iterable of numbers
		The powers of the polynomial.  Can be integer or floats.  Explicitly include 1 
		to include a linear component.
	baseparam : str or None
		The base parameter name.  If not given, `basevar` is used.
	invertpower : bool
		If True, the inverse (i.e., 1/N) of the `powers` is used.
	scaling : float
		A scaling term.  The `basevar` is divided by this term before the power is taken,
		which can help prevent high power terms becoming unreasonably large.

	Returns
	-------
	LinearFunction
		The polynomial linear function.

	Examples
	--------
	>>> from larch.util.piecewise import polynomial_linear_function
	>>> f = polynomial_linear_function('Aaa', [1,2,3])
	>>> print(f)
	  = P('Aaa') * X('Aaa')
	  + P('Aaa_2') * X('(Aaa)**2')
	  + P('Aaa_3') * X('(Aaa)**3')
	>>> f2 = polynomial_linear_function('Aaa', [1,2,3], invertpower=True)
	>>> print(f2)
	  = P('Aaa') * X('Aaa')
	  + P('Aaa_2') * X('(Aaa)**(1/2)')
	  + P('Aaa_3') * X('(Aaa)**(1/3)')
	
	"""
	from ..roles import P, X
	from ..core import LinearFunction
	if baseparam is None:
		baseparam = basevar
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
	f._x_ident=basevar
	return f


def log_and_linear_function(basevar, baseparam=None):
	"""
	Create a two term LinearFunction for log plus linear.
	
	The resulting function looks like :math:`{ \\beta }_1 X + { \\beta }_2 \log ( 1 + X )`.
	
	Parameters
	----------
	basevar : str
		The variable to use as the base variable.
	baseparam : str or None
		The base parameter name.  If not given, `basevar` is used.

	Returns
	-------
	LinearFunction
		The two term linear function.

	Examples
	--------
	>>> from larch.util.piecewise import log_and_linear_function
	>>> f = log_and_linear_function('Aaa')
	>>> print(f)
	  = P('Aaa') * X('Aaa')
	  + P('logAaaP1') * X('log1p(Aaa)')
	>>> f2 = log_and_linear_function('Aaa','Bbb')
	>>> print(f2)
	  = P('Bbb') * X('Aaa')
	  + P('logBbbP1') * X('log1p(Aaa)')
	
	"""
	from ..roles import P, X
	if baseparam is None:
		baseparam = basevar
	f = P(baseparam)*X(basevar) + P("log{}P1".format(baseparam))*X('log1p({})'.format(basevar))
	f._x_ident=basevar
	return f

def log_and_piecewise_linear_function(basevar, breaks, smoothness=1, baseparam=None):
	from ..roles import P, X
	f = piecewise_linear_function(basevar, breaks, smoothness, baseparam) + P("log{}P1".format(baseparam))*X('log1p({})'.format(basevar))
	f._x_ident=basevar
	return f

def log_and_gross_piecewise_linear_function(basevar, breaks, baseparam=None):
	from ..roles import P, X
	f = gross_piecewise_linear_function(basevar, breaks=breaks, baseparam=baseparam) + P("log{}P1".format(baseparam))*X('log1p({})'.format(basevar))
	f._x_ident=basevar
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
	f._x_ident=basevar
	return f




