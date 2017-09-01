from scipy.linalg.cython_blas cimport dgemm as _dgemm
cimport numpy as np
cimport cython

cpdef _dot_strider(
	double[:,:] a, int lda,
	double[:,:] b, int ldb,
	double[:,:] c, int ldc,
	double alpha= 1.0,
	double beta= 0.0,
	int trans_a=0, 
	int trans_b=0
):
	cdef int m, n, k, i
	cdef char trans_a_, trans_b_
	
	if trans_a:
		k = a.shape[0]
		m = a.shape[1]
		trans_a_ = b'T'
	else:
		k = a.shape[1]
		m = a.shape[0]
		trans_a_ = b'N'

	if trans_b:
		n = b.shape[0]
		trans_b_ = b'T'
	else:
		n = b.shape[1]
		trans_b_ = b'N'

	#print('m,n,k',m,n,k)

	_dgemm(&trans_a_, &trans_b_, &m, &n, &k, &alpha, &a[0,0], &lda, &b[0,0], &ldb, &beta, &c[0,0], &ldc)
	
	
def _isSorted(x, reverse=False): 
	import operator
	my_operator = operator.ge if reverse else operator.le
	return all(my_operator(x[i], x[i + 1])
			   for i in range(len(x) - 1))


def _fortran_check(z, label):
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
		

def dgemm(alpha,a,b,beta,c):
	c, trans_c = _fortran_check(c, "MAT-C")
	ldc = int(c.strides[1] / c.dtype.itemsize)
	if trans_c:
		a, trans_a = _fortran_check(a.T, "MAT-At")
		b, trans_b = _fortran_check(b.T, "MAT-Bt")
		lda = int(a.strides[1] / a.dtype.itemsize)
		ldb = int(b.strides[1] / b.dtype.itemsize)
		#print("b...",b.strides,ldb,b.shape,'T' if trans_b else 'N')
		#print("a...",a.strides,lda,a.shape,'T' if trans_a else 'N')
		return _dot_strider(b,ldb,a,lda,c,ldc,alpha,beta,trans_b,trans_a)
	else:	 
		a, trans_a = _fortran_check(a, "MAT-A")
		b, trans_b = _fortran_check(b, "MAT-B")
		lda = int(a.strides[1] / a.dtype.itemsize)
		ldb = int(b.strides[1] / b.dtype.itemsize)
		#print("a...",a.strides,lda,a.shape,'T' if trans_a else 'N')
		#print("b...",b.strides,ldb,b.shape,'T' if trans_b else 'N')
		return _dot_strider(a,lda,b,ldb,c,ldc,alpha,beta,trans_a,trans_b)
		