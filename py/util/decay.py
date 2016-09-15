from itertools import count
import numpy
from .naming import parenthize



def smoothed_piecewise_linear(basevar, breaks, smoothness=1):
	from ..roles import X
	outs = [
		X(basevar),
	]
	if smoothness==0:
		outs += [
			X("fmax(0,{0}-{1})".format(basevar, loc), descrip=basevar+" (@{})".format(loc,smoothness)) for loc in breaks
		]
		return outs
	
	
	s = 1.0/smoothness
	outs += [
#		X("(log(1+exp({2}*({0}-{1}))))/{2}".format(basevar, loc, s), descrip=basevar+" (@{}~{})".format(loc,smoothness)) for loc in breaks
		X("(logaddexp(0,{2}*({0}-{1})))/{2}".format(basevar, loc, s), descrip=basevar+" (@{}~{})".format(loc,smoothness)) for loc in breaks
	]
	return outs


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
