

def general_inverse(a, small_eig=0.0001):
	"""Find the matrix inverse if possible, otherwise find the pseudo-inverse."""
	import numpy
	if not numpy.isfinite(a).all():
		raise ValueError("nonfinite values in array")
	min_eig = None
	try:
		eig = numpy.linalg.eigvalsh(a)
		min_eig = numpy.min(numpy.abs(eig))
		if min_eig < small_eig:
			raise numpy.linalg.linalg.LinAlgError()
		x = numpy.linalg.inv(a)
	except numpy.linalg.linalg.LinAlgError:
		if min_eig is not None and min_eig < small_eig:
			import warnings
			warnings.warn(f"minimum eig {min_eig} in general_inverse")
		import scipy.linalg
		x = scipy.linalg.pinvh(a)
	return x


