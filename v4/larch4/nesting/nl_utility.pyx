

from libc.math cimport pow
cimport numpy as np
cimport cython
import numpy

# from cython.parallel cimport prange

@cython.boundscheck(False)
@cython.cdivision(True)
cdef _exp_util_of_nest(
	double[:,:] expU_elemental,
	double[:,:] expU_nests,
	int[:] child_slots,
	int n_children, # = child_slots.shape[0]
	int parent_slot,
	double MU
):
	cdef int ci, i, j, len_j, n_elementals, parent_slot_in_nests
	len_j = expU_elemental.shape[0]
	n_elementals = expU_elemental.shape[1]
	parent_slot_in_nests = parent_slot-n_elementals
	if MU==0:
		raise ZeroDivisionError("zero MU not yet implemented")
	if parent_slot_in_nests < 0:
		raise ValueError('parent slot is in the elementals')
	#for j in prange(len_j, nogil=True):
	for j in range(len_j):
		expU_nests[j,parent_slot-n_elementals] = 0
		for i in range(n_children):
			ci = child_slots[i]
			if ci < n_elementals:
				expU_nests[j,parent_slot_in_nests] += pow(expU_elemental[j,ci],(1/MU))
			else:
				expU_nests[j,parent_slot_in_nests] += pow(expU_nests[j,ci-n_elementals],(1/MU))
	for j in range(len_j):
		expU_nests[j,parent_slot_in_nests] = pow(expU_nests[j,parent_slot_in_nests],MU)




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
		_exp_util_of_nest( expU_elemental, expU_nests, child_slots, child_slots.shape[0], graph.standard_slot_map[node], mu )