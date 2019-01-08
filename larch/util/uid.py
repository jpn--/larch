_uidn = 0

def uid():
	global _uidn
	_uidn += 1
	return "rx{}".format(_uidn)
