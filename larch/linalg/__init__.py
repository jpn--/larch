

def general_inverse(a):
	"""Find the matrix inverse if possible, otherwise find the pseudo-inverse."""
	import numpy
	if not numpy.isfinite(a).all():
		raise ValueError("nonfinite values in array")
	try:
		eig = numpy.linalg.eigvalsh(a)
		if numpy.min(numpy.abs(eig)) < 0.001:
			raise numpy.linalg.linalg.LinAlgError()
		x = numpy.linalg.inv(a)
	except numpy.linalg.linalg.LinAlgError:
		import scipy.linalg
		x = scipy.linalg.pinvh(a)
	return x


