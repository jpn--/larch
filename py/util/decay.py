from itertools import count
import numpy




def decay_transform(basevar, breaks):
	from ..roles import X
	outlen = len(breaks)-1
	if outlen<2:
		raise TypeError('there must be at least 3 values in breaks (min, cut, max)')

	locs = []
	scales = []

	for low,mid,high in zip(breaks[:-2],breaks[1:-1],breaks[2:]):
		if low>=mid or mid>=high:
			raise TypeError('breaks must be strictly increasing')
		locs.append(mid)
		scales.append(numpy.sqrt((mid-low)*(high-mid))/4)

	outs = [
		X(basevar),
	]

	outs += [
		X("({0})/(1+exp((-({0})+{1})/{2}))".format(basevar, loc, scal), descrip=basevar+" (Delay{})".format(n+1)) for loc,scal,n in zip(locs,scales,count())
	]

	return outs