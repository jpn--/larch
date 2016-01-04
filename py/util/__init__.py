from .orderedset import OrderedSet
from .filemanager import *
from .temporaryfile import TemporaryFile, TemporaryHtml




def magic3(dtype=None):
	import numpy
	if dtype is None:
		dtype=numpy.float64
	lo_shu = numpy.asarray([[4,9,2],[3,5,7],[8,1,6]], dtype=dtype)
	return lo_shu