from .orderedset import OrderedSet
from .filemanager import *
from .temporaryfile import TemporaryFile, TemporaryHtml
from .persistent import stored_dict
from .attribute_dict import dicta, quickdot

from . import categorize
from . import piecewise

allowed_math = ('log', 'exp', 'log1p', 'logaddexp')


def magic3(dtype=None):
	import numpy
	if dtype is None:
		dtype=numpy.float64
	lo_shu = numpy.asarray([[4,9,2],[3,5,7],[8,1,6]], dtype=dtype)
	return lo_shu



import platform
import os
computer = os.path.splitext(platform.node())[0]


mplstyle_filepath = os.path.join( os.path.split(__file__)[0], 'larch.mplstyle' )



