from .gemv import dgemv

import numpy    

def test_gemv_overwrite():
	a = numpy.asarray(numpy.arange(25).reshape((5,5)),dtype=float, order= 'C')
	b = numpy.asarray(numpy.arange(5),dtype=float, order= 'C')
	
	def pick(aa,bb):
		c = numpy.ones((10,), float, order="C")
		dgemv(1.0,aa,bb,0,c)
		c_ = numpy.dot(aa,bb)
		assert (c[:c_.shape[0]]==c_).all()
		assert (c[c_.shape[0]:]==1).all()

	pick(a,b)	
	pick(a.T,b)
	pick(a[1:4,1:3],b[2:4])
	pick(a.T[1:4,1:3],b[2:4])

def test_gemv_addon():
	a = numpy.asarray(numpy.arange(25).reshape((5,5)),dtype=float, order= 'C')
	b = numpy.asarray(numpy.arange(5),dtype=float, order= 'C')

	def pick(aa,bb):
		c = numpy.ones((10,), float, order="C")
		dgemv(1.0,aa,bb,1,c)
		c_ = numpy.dot(aa,bb) + 1
		assert (c[:c_.shape[0]]==c_).all()
		assert (c[c_.shape[0]:]==1).all()

	pick(a,b)
	pick(a.T,b)
	pick(a[1:4,1:3],b[2:4])
	pick(a.T[1:4,1:3],b[2:4])

def test_gemv_offset():
	a = numpy.asarray(numpy.arange(25).reshape((5,5)),dtype=float, order= 'C')
	b = numpy.asarray(numpy.arange(5),dtype=float, order= 'C')

	def pick(aa,bb):
		c = numpy.ones((10,), float, order="C")
		cc = c[1:]
		dgemv(1.0,aa,bb,1,cc)
		c_ = numpy.dot(aa,bb) + 1
		assert (c[1:c_.shape[0]+1]==c_).all()
		assert (c[c_.shape[0]+1:]==1).all()
		assert (c[0]==1)

	pick(a,b)
	pick(a.T,b)
	pick(a[1:4,1:3],b[2:4])
	pick(a.T[1:4,1:3],b[2:4])

