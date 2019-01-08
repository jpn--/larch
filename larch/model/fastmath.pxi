

# from libc.math cimport exp, log, pow
# from numpy.math cimport expf, logf

cdef extern from "fastexp.h" nogil:
	float fastexp (float p)
	float fasterexp (float p)

cdef extern from "fastlog.h" nogil:
	float fastlog (float p)
	float fasterlog (float p)

cdef inline float f_exp(float x) nogil:
	return fastexp(x)

cdef inline float f_log(float x) nogil:
	return fastlog(x)



cdef inline double exp_fast2(double x):
	x = 1.0 + x / 1024
	x *= x
	x *= x
	x *= x
	x *= x
	x *= x
	x *= x
	x *= x
	x *= x
	x *= x
	x *= x
	return x

cdef inline double exp_fast3(double x):
	x = 1.0 + x / 256.0
	x *= x
	x *= x
	x *= x
	x *= x
	x *= x
	x *= x
	x *= x
	x *= x
	return x
