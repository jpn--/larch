


cdef class CBlocker0:

	cdef public double[:]   meta

	cdef public double[:]   util_ca
	cdef public double[:,:] util_co
	cdef public double[:]   quant_ca
	cdef public double[:]   logsums
	cdef public double[:]   allocs
	cdef public double[:]   sizemults

	cpdef void initialize(self)

cdef class CBlocker1:

	cdef public double[:,:]   meta

	cdef public double[:,:]   util_ca
	cdef public double[:,:,:] util_co
	cdef public double[:,:]   quant_ca
	cdef public double[:,:]   logsums
	cdef public double[:,:]   allocs
	cdef public double[:,:]   sizemults

	cpdef void initialize(self)