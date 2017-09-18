from scipy.linalg.cython_blas cimport daxpy as _daxpy
from scipy.linalg.cython_blas cimport dcopy as _dcopy
cimport cython

@cython.boundscheck(False)
cdef _daxpy_strider(
	double a,
	double[:] x, int incx,
	double[:] y, int incy
):
	cdef int n = x.shape[0]
	_daxpy(&n, &a, &x[0], &incx, &y[0], &incy)


cdef void daxpy_memview(
	double a,
	double[:] x,
	double[:] y,
):
	cdef int n = x.shape[0]
	cdef int incx = x.strides[0]
	cdef int incy = y.strides[0]
	_daxpy(&n, &a, &x[0], &incx, &y[0], &incy)



def daxpy(a,x,y):
	cdef int incy = y.strides[0] // y.dtype.itemsize
	cdef int incx = x.strides[0] // x.dtype.itemsize
	return _daxpy_strider(a,x,incx,y,incy)



@cython.boundscheck(False)
cdef _dcopy_strider(
	double[:] x, int incx,
	double[:] y, int incy
):
	cdef int n = x.shape[0]
	_dcopy(&n, &x[0], &incx, &y[0], &incy)


def dcopy(x,y):
	cdef int incy = y.strides[0] // y.dtype.itemsize
	cdef int incx = x.strides[0] // x.dtype.itemsize
	return _dcopy_strider(x,incx,y,incy)
