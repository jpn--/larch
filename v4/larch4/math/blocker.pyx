
import numpy


cdef class CBlocker0:

	def __init__(self, blocker):
		self.meta = blocker.outer
		if len(blocker.inners)>0:
			self.util_ca = blocker.inners[0]
		if len(blocker.inners)>1:
			self.util_co = blocker.inners[1]
		if len(blocker.inners)>2:
			self.quant_ca = blocker.inners[2]
		if len(blocker.inners)>3:
			self.logsums = blocker.inners[3]
		if len(blocker.inners)>4:
			self.allocs = blocker.inners[4]
		if len(blocker.inners)>5:
			self.sizemults = blocker.inners[5]

	cdef void initialize(self) nogil:
		cdef int i
		for i in range(self.meta.shape[0]):
			self.meta[i] = 0

cdef class CBlocker1:

	def __init__(self, blocker):
		self.meta = blocker.outer
		if len(blocker.inners)>0:
			self.util_ca = blocker.inners[0]
		if len(blocker.inners)>1:
			self.util_co = blocker.inners[1]
		if len(blocker.inners)>2:
			self.quant_ca = blocker.inners[2]
		if len(blocker.inners)>3:
			self.logsums = blocker.inners[3]
		if len(blocker.inners)>4:
			self.allocs = blocker.inners[4]
		if len(blocker.inners)>5:
			self.sizemults = blocker.inners[5]

	cdef void initialize(self) nogil:
		cdef int i,j
		for i in range(self.meta.shape[0]):
			for j in range(self.meta.shape[1]):
				self.meta[i,j] = 0

def check_cblocker(b):
	cdef CBlocker0 j = CBlocker0(b)