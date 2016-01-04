
from numpy import log10


def flux(should_be, actually_is):
	difference = abs(should_be - actually_is)
	if difference==0:
		return log10(0.0)
	magnitude  = max(abs(should_be),abs(actually_is))
	if magnitude:
		return log10(float(difference)/float(magnitude))
	return 0.0


def flux_mag(should_be, actually_is, format="{:< 12.2g}"):
	try:
		flx = flux(should_be, actually_is)
	except ValueError:
		return format.replace('g','s').replace('f','s').replace(' ','').format('NA')
	else:
		if flx < -6:
			return "ok: "+format.format(flx)
			#return format.replace('g','s').replace('f','s').replace(' ','').format('ok')
		else:
			return "    "+format.format(flx)