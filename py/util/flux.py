
from math import log10


def flux(should_be, actually_is):
	difference = abs(should_be - actually_is);
	magnitude  = (abs(should_be)+abs(actually_is)) / 2 ;
	if magnitude:
		return float(difference)/float(magnitude);
	return 0.0;



def flux_mag(should_be, actually_is, format="{:< 12.2g}"):
	try:
		flx = log10(flux(should_be, actually_is))
	except ValueError:
		return format.replace('g','s').replace('f','s').replace(' ','').format('NA')
	else:
		if flx < -6:
			return format.replace('g','s').replace('f','s').replace(' ','').format('ok')
		else:
			return format.format(flx)