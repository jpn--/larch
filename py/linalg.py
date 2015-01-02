#
#  Copyright 2007-2015 Jeffrey Newman
#
#  This file is part of Larch.
#
#  Larch is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Larch is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Larch.  If not, see <http://www.gnu.org/licenses/>.
#

import numpy.linalg
import scipy.linalg
from .array import pack
from . import logging
_Skr = logging.getScriber("linalg")

def general_inverse(a):
	"""Find the matrix inverse if possible, otherwise find the pseudo-inverse."""
	#_Skr.log(5,"call:general_inverse")
	try:
		try:
			x = numpy.linalg.inv(a)
			#_Skr.log(5,"normal matrix inverse calculated")
		except numpy.linalg.linalg.LinAlgError:
			#x = numpy.linalg.pinv(a) # slower than scipy's pinvh for symmetric matrices
			x = scipy.linalg.pinvh(a)
			_Skr.log(5,"matrix pseudo-inverse calculated")
	except:
		print("error in general_inverse")
		raise
	#_Skr.log(5,"ending:general_inverse")
	return pack(x)

def matrix_inverse(a):
	return pack(numpy.linalg.inv(a))
