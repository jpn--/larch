# cython: profile=True

from libc.math cimport pow, exp, log
cimport numpy as np
cimport cython
import numpy
from numpy.math cimport INFINITY

from cython.parallel cimport prange


@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _exp_inplace_1(
		double[:] arr,
) nogil:
	cdef int len_i = arr.shape[0]
	cdef int i
	for i in prange(len_i, schedule='static'):
		arr[i] = exp(arr[i])


@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _log_inplace_1(
		double[:] arr,
) nogil:
	cdef int len_i = arr.shape[0]
	cdef int i
	for i in prange(len_i, schedule='static'):
		arr[i] = log(arr[i])

def exp_inplace_1(double[:] arr):
	with nogil:
		_exp_inplace_1( arr )

def log_inplace_1(double[:] arr):
	with nogil:
		_log_inplace_1( arr )



@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _exp_inplace_2(
		double[:,:] arr,
) nogil:
	cdef int len_i = arr.shape[0]
	cdef int len_j = arr.shape[1]
	cdef int i,j
	for i in prange(len_i, schedule='static'):
		for j in range(len_j):
			arr[i,j] = exp(arr[i,j])


@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _log_inplace_2(
		double[:,:] arr,
) nogil:
	cdef int len_i = arr.shape[0]
	cdef int len_j = arr.shape[1]
	cdef int i,j
	for i in prange(len_i, schedule='static'):
		for j in range(len_j):
			arr[i,j] = log(arr[i,j])

def exp_inplace_2(double[:,:] arr):
	with nogil:
		_exp_inplace_2( arr )

def log_inplace_2(double[:,:] arr):
	with nogil:
		_log_inplace_2( arr )





@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef void _util_of_nest(
	double[:,:] V_elemental,
	double[:,:] V_nests,
	int[:] child_slots,
	int n_children, # = child_slots.shape[0]
	int parent_slot,
	double MU,
	int n_threads
):
	cdef int ci, i, j
	cdef int len_j = V_elemental.shape[0]
	cdef int n_elementals = V_elemental.shape[1]
	cdef int parent_slot_in_nests = parent_slot-n_elementals

	cdef double old_shifter=0
	cdef double new_shifter=0
	cdef double z

	cdef double cutoff = 500
	cdef double* ptr

	if MU==1.0:
		# optimized for speed, otherwise generic version is technically correct
		for j in prange(len_j, schedule='static', nogil=True, num_threads=n_threads):
			V_nests[j,parent_slot_in_nests] = 0
			for i in range(n_children):
				ci = child_slots[i]
				if ci < n_elementals:
					V_nests[j,parent_slot_in_nests] += exp(V_elemental[j,ci])
				else:
					V_nests[j,parent_slot_in_nests] += exp(V_nests[j,ci-n_elementals])
			V_nests[j,parent_slot_in_nests] = log(V_nests[j,parent_slot_in_nests])
	elif MU==0:
		for j in prange(len_j, schedule='static', nogil=True, num_threads=n_threads):
			V_nests[j,parent_slot_in_nests] = 0
			for i in range(n_children):
				ci = child_slots[i]
				if ci < n_elementals:
					if V_nests[j,parent_slot_in_nests] < V_elemental[j,ci]:
						V_nests[j,parent_slot_in_nests] = V_elemental[j,ci]
				else:
					if V_nests[j,parent_slot_in_nests] < V_nests[j,ci-n_elementals]:
						V_nests[j,parent_slot_in_nests] = V_nests[j,ci-n_elementals]
	# else: # no shifters
	# 	for j in prange(len_j, schedule='static', nogil=True, num_threads=n_threads):
	# 		ptr = &V_nests[j,parent_slot_in_nests]
	# 		ptr[0] = 0
	# 		for i in range(n_children):
	# 			ci = child_slots[i]
	# 			if ci < n_elementals:
	# 				z = (V_elemental[j,ci])
	# 			else:
	# 				z = (V_nests[j,ci-n_elementals])
	# 			if z != -INFINITY:
	# 				ptr[0] += exp(z/MU)
	# 		ptr[0] = log(V_nests[j,parent_slot_in_nests])*MU
	else: # with shifters
		for j in prange(len_j, schedule='static', nogil=True, num_threads=n_threads):
			V_nests[j,parent_slot_in_nests] = 0
			old_shifter=0
			new_shifter=0
			for i in range(n_children):
				ci = child_slots[i]
				if ci < n_elementals:
					z = (V_elemental[j,ci] / MU) - old_shifter
				else:
					z = (V_nests[j,ci-n_elementals] / MU) - old_shifter
				if z > cutoff:
					new_shifter = (z - cutoff) + old_shifter
					z = cutoff
					#V_nests[j,parent_slot_in_nests] = exp(log(V_nests[j,parent_slot_in_nests]) - (new_shifter-old_shifter))
					#V_nests[j,parent_slot_in_nests] = exp(log(V_nests[j,parent_slot_in_nests]) + (old_shifter-new_shifter))
					V_nests[j,parent_slot_in_nests] *= exp(old_shifter-new_shifter)
					old_shifter = new_shifter
				V_nests[j,parent_slot_in_nests] += exp(z)
			V_nests[j,parent_slot_in_nests] = (log(V_nests[j,parent_slot_in_nests])+old_shifter)*MU



@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _exp_util_of_nest(
	double[:,:] expU_elemental,
	double[:,:] expU_nests,
	int[:] child_slots,
	int n_children, # = child_slots.shape[0]
	int parent_slot,
	double MU
):
	cdef int ci, i, j
	cdef int len_j = expU_elemental.shape[0]
	cdef int n_elementals = expU_elemental.shape[1]
	cdef int parent_slot_in_nests = parent_slot-n_elementals

	if MU==1.0:
		# optimized for speed, otherwise generic version is technically correct
		for j in prange(len_j, nogil=True):
			expU_nests[j,parent_slot-n_elementals] = 0
			for i in range(n_children):
				ci = child_slots[i]
				if ci < n_elementals:
					expU_nests[j,parent_slot_in_nests] += expU_elemental[j,ci]
				else:
					expU_nests[j,parent_slot_in_nests] += expU_nests[j,ci-n_elementals]
	elif MU==0:
		for j in prange(len_j, nogil=True):
			expU_nests[j,parent_slot-n_elementals] = 0
			for i in range(n_children):
				ci = child_slots[i]
				if ci < n_elementals:
					if expU_nests[j,parent_slot_in_nests] < expU_elemental[j,ci]:
						expU_nests[j,parent_slot_in_nests] = expU_elemental[j,ci]
				else:
					if expU_nests[j,parent_slot_in_nests] < expU_nests[j,ci-n_elementals]:
						expU_nests[j,parent_slot_in_nests] = expU_nests[j,ci-n_elementals]
	else:
		for j in prange(len_j, nogil=True):
			expU_nests[j,parent_slot-n_elementals] = 0
			for i in range(n_children):
				ci = child_slots[i]
				if ci < n_elementals:
					expU_nests[j,parent_slot_in_nests] += pow(expU_elemental[j,ci],(1/MU))
				else:
					expU_nests[j,parent_slot_in_nests] += pow(expU_nests[j,ci-n_elementals],(1/MU))
			expU_nests[j,parent_slot_in_nests] = pow(expU_nests[j,parent_slot_in_nests],MU)





@cython.boundscheck(False)
@cython.cdivision(True)
cpdef _util_of_nest_with_checker(
	double[:,:] U_elemental,
	double[:,:] U_nests,
	int[:] child_slots,
	int n_children, # = child_slots.shape[0]
	int parent_slot,
	double MU,
	int n_threads
):
	cdef int n_elementals = U_elemental.shape[1]
	cdef int parent_slot_in_nests = parent_slot-n_elementals
	if MU==0:
		raise ZeroDivisionError("zero MU not yet implemented")
	if parent_slot_in_nests < 0:
		raise ValueError('parent slot is in the elementals')
	_util_of_nest( U_elemental, U_nests, child_slots, n_children, parent_slot, MU, n_threads)



@cython.boundscheck(False)
@cython.cdivision(True)
cpdef _exp_util_of_nest_with_checker(
	double[:,:] expU_elemental,
	double[:,:] expU_nests,
	int[:] child_slots,
	int n_children, # = child_slots.shape[0]
	int parent_slot,
	double MU
):
	cdef int n_elementals = expU_elemental.shape[1]
	cdef int parent_slot_in_nests = parent_slot-n_elementals
	if MU==0:
		raise ZeroDivisionError("zero MU not yet implemented")
	if parent_slot_in_nests < 0:
		raise ValueError('parent slot is in the elementals')
	_exp_util_of_nest( expU_elemental, expU_nests, child_slots, n_children, parent_slot, MU)


def util_of_nests(
		double[:,:] U_elemental,
		double[:,:] U_nests,
		graph,
		params,
		n_threads = 1
	):
	cdef int[:] child_slots
	cdef double mu
	for node in graph.topological_sorted_no_elementals:
		child_slots = graph.successor_slots(node)
		if 'parameter' in graph.node[node]:
			mu = params.get_value(graph.node[node]['parameter'])
		else:
			mu = 1.0
		_util_of_nest_with_checker( U_elemental, U_nests, child_slots, child_slots.shape[0], graph.standard_slot_map[node], mu, n_threads )



def exp_util_of_nests(
		double[:,:] expU_elemental,
		double[:,:] expU_nests,
		graph,
		params
	):
	cdef int[:] child_slots
	cdef double mu
	for node in graph.topological_sorted_no_elementals:
		child_slots = graph.successor_slots(node)
		if 'parameter' in graph.node[node]:
			mu = params.get_value(graph.node[node]['parameter'])
		else:
			mu = 1.0
		_exp_util_of_nest_with_checker( expU_elemental, expU_nests, child_slots, child_slots.shape[0], graph.standard_slot_map[node], mu )

