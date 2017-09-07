

from libc.math cimport pow, exp, log
cimport numpy as np
cimport cython
import numpy

from cython.parallel cimport prange

@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _util_of_nest(
	double[:,:] V_elemental,
	double[:,:] V_nests,
	int[:] child_slots,
	int n_children, # = child_slots.shape[0]
	int parent_slot,
	double MU
):
	cdef int ci, i, j
	cdef int len_j = V_elemental.shape[0]
	cdef int n_elementals = V_elemental.shape[1]
	cdef int parent_slot_in_nests = parent_slot-n_elementals

	cdef double old_shifter=0
	cdef double new_shifter=0
	cdef double z

	cdef double cutoff = 500

	if MU==1.0:
		# optimized for speed, otherwise generic version is technically correct
		for j in prange(len_j, nogil=True):
			V_nests[j,parent_slot-n_elementals] = 0
			for i in range(n_children):
				ci = child_slots[i]
				if ci < n_elementals:
					V_nests[j,parent_slot_in_nests] += exp(V_elemental[j,ci])
				else:
					V_nests[j,parent_slot_in_nests] += exp(V_nests[j,ci-n_elementals])
	elif MU==0:
		for j in prange(len_j, nogil=True):
			V_nests[j,parent_slot-n_elementals] = 0
			for i in range(n_children):
				ci = child_slots[i]
				if ci < n_elementals:
					if V_nests[j,parent_slot_in_nests] < V_elemental[j,ci]:
						V_nests[j,parent_slot_in_nests] = V_elemental[j,ci]
				else:
					if V_nests[j,parent_slot_in_nests] < V_nests[j,ci-n_elementals]:
						V_nests[j,parent_slot_in_nests] = V_nests[j,ci-n_elementals]
	else:
		for j in prange(len_j, nogil=True):
			V_nests[j,parent_slot-n_elementals] = 0
			old_shifter=0
			new_shifter=0
			for i in range(n_children):
				ci = child_slots[i]
				if ci < n_elementals:
					z = (V_elemental[j,ci] / MU) + old_shifter
				else:
					z = (V_nests[j,ci-n_elementals] / MU) + old_shifter
				if z > cutoff:
					new_shifter = cutoff - z
					z = cutoff
					V_nests[j,parent_slot_in_nests] *= exp(new_shifter-old_shifter)
					old_shifter = new_shifter
				V_nests[j,parent_slot_in_nests] += exp(z)
			V_nests[j,parent_slot_in_nests] = log(V_nests[j,parent_slot_in_nests])*MU - old_shifter



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
	double MU
):
	cdef int n_elementals = U_elemental.shape[1]
	cdef int parent_slot_in_nests = parent_slot-n_elementals
	if MU==0:
		raise ZeroDivisionError("zero MU not yet implemented")
	if parent_slot_in_nests < 0:
		raise ValueError('parent slot is in the elementals')
	_util_of_nest( U_elemental, U_nests, child_slots, n_children, parent_slot, MU)



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
		_exp_util_of_nest_with_checker( U_elemental, U_nests, child_slots, child_slots.shape[0], graph.standard_slot_map[node], mu )



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

