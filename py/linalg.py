#
#  Copyright 2007-2016 Jeffrey Newman
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
			eig = numpy.linalg.eigvalsh(a)
			if numpy.min(numpy.abs(eig)) < 0.001:
				raise numpy.linalg.linalg.LinAlgError()
			x = numpy.linalg.inv(a)
			#_Skr.log(5,"normal matrix inverse calculated")
		except numpy.linalg.linalg.LinAlgError:
			#x = numpy.linalg.pinv(a) # slower than scipy's pinvh for symmetric matrices
			x = scipy.linalg.pinvh(a)
			_Skr.log(5,"matrix pseudo-inverse calculated")
	except:
		#print("error in general_inverse")
		raise
	#_Skr.log(5,"ending:general_inverse")
	return pack(x)

def matrix_inverse(a):
	return pack(numpy.linalg.inv(a))



def possible_overspecification(a, holdfast_vector=None):
	ret = []
	if holdfast_vector is None:
		eigenvalues,eigenvectors = numpy.linalg.eigh(a)
		for i in range(len(eigenvalues)):
			if numpy.abs(eigenvalues[i]) < 0.001:
				v = eigenvectors[:,i]
				v = numpy.round(v,7)
				ret.append( (eigenvalues[i], numpy.where(v)[0])  )
	else:
		a_packed = a[~holdfast_vector.astype(bool),:][:,~holdfast_vector.astype(bool)]
		eigenvalues_packed,eigenvectors_packed = numpy.linalg.eigh(a_packed)
#		eigenvalues_unpacked = numpy.ones(a.shape[0])
#		eigenvectors_unpacked = numpy.zeros(a.shape)
#		eigenvalues_unpacked[~holdfast_vector.astype(bool)] = eigenvalues_packed
#		eigenvectors_unpacked[~holdfast_vector.astype(bool),:][:,~holdfast_vector.astype(bool)] = eigenvectors_packed
		for i in range(len(eigenvalues_packed)):
			if numpy.abs(eigenvalues_packed[i]) < 0.001:
				v = eigenvectors_packed[:,i]
				v = numpy.round(v,7)
				v_unpacked = numpy.zeros(a.shape[0])
				v_unpacked[~holdfast_vector.astype(bool)] = v
				ret.append( (eigenvalues_packed[i], numpy.where(v_unpacked)[0])  )
	return ret

