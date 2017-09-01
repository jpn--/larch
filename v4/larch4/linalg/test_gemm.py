from .gemm import dgemm

import numpy    

def test_gemm_overwrite():
	a = numpy.asarray(numpy.arange(25).reshape((5,5)),dtype=float, order= 'C')
	b = numpy.asarray(numpy.arange(25).reshape((5,5)),dtype=float, order= 'C')
	
	def pick(aa,bb):
		c = numpy.ones((7,7), float, order="C")
		dgemm(1.0,aa,bb,0,c)
		c_ = numpy.dot(aa,bb)
		assert (c[:c_.shape[0],:c_.shape[1]]==c_).all()
		assert (c[c_.shape[0]:,:]==1).all()
		assert (c[:,c_.shape[1]:]==1).all()
	
	pick(a,b)	
	pick(a.T,b.T)	
	pick(a,b.T)	
	pick(a.T,b)	
	pick(a[1:4,:],b[:,2:4])
	pick(a.T[1:4,:],b[:,2:4])
	pick(a[1:4,:],b.T[:,2:4])
	pick(a.T[1:4,:],b.T[:,2:4])
	
def test_gemm_addon():
	a = numpy.asarray(numpy.arange(25).reshape((5,5)),dtype=float, order= 'C')
	b = numpy.asarray(numpy.arange(25).reshape((5,5)),dtype=float, order= 'C')
	
	def pick(aa,bb):
		c = numpy.ones((7,7), float, order="C")
		dgemm(1.0,aa,bb,1,c)
		c_ = numpy.dot(aa,bb) + 1
		assert (c[:c_.shape[0],:c_.shape[1]]==c_).all()
		assert (c[c_.shape[0]:,:]==1).all()
		assert (c[:,c_.shape[1]:]==1).all()
	
	pick(a,b)	
	pick(a.T,b.T)	
	pick(a,b.T)	
	pick(a.T,b)	
	pick(a[1:4,:],b[:,2:4])
	pick(a.T[1:4,:],b[:,2:4])
	pick(a[1:4,:],b.T[:,2:4])
	pick(a.T[1:4,:],b.T[:,2:4])
	
def test_gemm_offset():
	a = numpy.asarray(numpy.arange(25).reshape((5,5)),dtype=float, order= 'C')
	b = numpy.asarray(numpy.arange(25).reshape((5,5)),dtype=float, order= 'C')
	
	def pick(aa,bb):
		c = numpy.ones((7,7), float, order="C")
		cc = c[1:,1:]
		dgemm(1.0,aa,bb,0,cc)
		c_ = numpy.dot(aa,bb)
		assert (c[1:c_.shape[0]+1,1:c_.shape[1]+1]==c_).all()
		assert (c[c_.shape[0]+1:,:]==1).all()
		assert (c[:,c_.shape[1]+1:]==1).all()
		assert (c[0,:]==1).all()
		assert (c[:,0]==1).all()
	
	pick(a,b)	
	pick(a.T,b.T)	
	pick(a,b.T)	
	pick(a.T,b)	
	pick(a[1:4,:],b[:,2:4])
	pick(a.T[1:4,:],b[:,2:4])
	pick(a[1:4,:],b.T[:,2:4])
	pick(a.T[1:4,:],b.T[:,2:4])

def test_gemm_T():
	a = numpy.asarray(numpy.arange(25).reshape((5,5)),dtype=float, order= 'C')
	b = numpy.asarray(numpy.arange(25).reshape((5,5)),dtype=float, order= 'C')
	
	def pick(aa,bb):
		c = numpy.ones((7,7), float, order="C")
		cc = c.T
		dgemm(1.0,aa,bb,0,cc)
		c_ = numpy.dot(aa,bb)
		assert (c.T[:c_.shape[0],:c_.shape[1]]==c_).all()
		assert (c.T[c_.shape[0]:,:]==1).all()
		assert (c.T[:,c_.shape[1]:]==1).all()
	
	pick(a,b)	
	pick(a.T,b.T)	
	pick(a,b.T)	
	pick(a.T,b)	
	pick(a[1:4,:],b[:,2:4])
	pick(a.T[1:4,:],b[:,2:4])
	pick(a[1:4,:],b.T[:,2:4])
	pick(a.T[1:4,:],b.T[:,2:4])
