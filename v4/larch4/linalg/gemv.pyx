from scipy.linalg.cython_blas cimport dgemv as _dgemv
cimport numpy as np
cimport cython

@cython.boundscheck(False)
cdef void _dgemv_strider(
	double[:,:] a, int lda,
	double[:] x, int incx,
	double[:] y, int incy,
	double alpha= 1.0,
	double beta= 0.0,
	int trans_a=0
):
	cdef int m, n
	cdef char trans_a_

	n = a.shape[1]
	m = a.shape[0]

	if trans_a:
		trans_a_ = b'T'
	else:
		trans_a_ = b'N'

	_dgemv(&trans_a_, &m, &n, &alpha, &a[0,0], &lda, &x[0], &incx, &beta, &y[0], &incy)
	
	
def _isSorted(x, reverse=False): 
	import operator
	my_operator = operator.ge if reverse else operator.le
	return all(my_operator(x[i], x[i + 1])
			   for i in range(len(x) - 1))


def _fortran_check(z):
	if z.flags.c_contiguous:
		#print("FC-C+",label)
		return z.T, 1 #'T'
	elif z.flags.f_contiguous:
		#print("FC-F+",label)
		return z, 0 # 'N'
	elif _isSorted(z.strides): # f-not-contiguous
		#print("FC-F-",label)
		return z, 0 # 'N'
	else: # c-not-contiguous
		#print("FC-C-",label)
		return z.T, 1 #'T'
		

def dgemv(alpha,a,x,beta,y):
	incy = y.strides[0] // y.dtype.itemsize
	incx = x.strides[0] // x.dtype.itemsize
	a, trans_a = _fortran_check(a)
	lda = int(a.strides[1] / a.dtype.itemsize)
	_dgemv_strider(a,lda,x,incx,y,incy,alpha,beta,trans_a)
		