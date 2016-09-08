

def decay_transform(basevar, outlen, maxvalue, style='gentle'):
	from ..roles import X
	from itertools import count
	if outlen==4:
		if style=='gentle':
			locs = [0.175*maxvalue, 0.4*maxvalue, 0.675*maxvalue]
			scales = [0.13125*maxvalue, 0.2*maxvalue, 0.16875*maxvalue]
		elif style=='sharp':
			locs = [0.1925*maxvalue, 0.44*maxvalue, 0.695*maxvalue]
			scales = [0.09625*maxvalue, 0.11*maxvalue, 0.11815*maxvalue]
		else:
			raise TypeError('style must be gentle or sharp')
	elif outlen==3:
		if style=='gentle':
			locs = [0.23*maxvalue, 0.555*maxvalue, ]
			scales = [0.1725*maxvalue, 0.2775*maxvalue, ]
		elif style=='sharp':
			raise TypeError('sharp not configured for 3 breaks')
		else:
			raise TypeError('style must be gentle or sharp')
	elif outlen==2:
		if style=='gentle':
			locs = [0.365*maxvalue, ]
			scales = [0.27375*maxvalue, ]
		elif style=='sharp':
			raise TypeError('sharp not configured for 2 breaks')
		else:
			raise TypeError('style must be gentle or sharp')
	else:
		raise TypeError('invalid number of breaks, must be 2, 3, or 4')

	outs = [
		X(basevar),
	]
	
	outs += [
		X("({0})*exp(-exp(((-({0})+{1})/{2})))".format(basevar, loc, scal), descrip=basevar+" (Delay{})".format(n+1)) for loc,scal,n in zip(locs,scales,count())
	]

	return outs