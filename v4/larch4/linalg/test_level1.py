from .level1 import daxpy, dcopy

import numpy


def test_axpy_copy():
	a = numpy.asarray(numpy.arange(5), dtype=float, order='C')
	b = numpy.asarray(numpy.arange(5,10), dtype=float, order='C')

	daxpy(0.1,a,b)

	assert( b[0]==5.0 )
	assert( b[1]==6.1 )
	assert( b[2]==7.2 )
	assert( b[3]==8.3 )
	assert( b[4]==9.4 )

	dcopy(a,b)

	assert( b[0]==0. )
	assert( b[1]==1. )
	assert( b[2]==2. )
	assert( b[3]==3. )
	assert( b[4]==4. )
