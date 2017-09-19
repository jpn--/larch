# cython: profile=True


from libc.math cimport pow, log
from numpy.math cimport INFINITY
cimport numpy as np
cimport cython
import numpy
import networkx as nx

from cython.parallel cimport prange


@cython.boundscheck(False)
@cython.cdivision(True)
cdef conditional_logprob_from_nest_exp_util(
		const double[:,:] expU_elemental,
		const double[:,:] expU_nests,
		int[:] child_slots,
		int n_children,                       # = child_slots.shape[0]
		int parent_slot,
		double MU,
		double[:,:] conditional_probability,  # output, shape = (n_cases, n_children)
	):

	cdef int ci, i, j
	cdef int len_j = expU_elemental.shape[0]
	cdef int n_elementals = expU_elemental.shape[1]
	cdef int parent_slot_in_nests = parent_slot-n_elementals

	for j in prange(len_j, nogil=True):
		for i in range(n_children):
			ci = child_slots[i]
			if ci < n_elementals:
				if expU_nests[j,parent_slot_in_nests] == 0 or expU_elemental[j,ci] == 0:
					conditional_probability[j,i] = -INFINITY
				else:
					conditional_probability[j,i] = log(expU_elemental[j,ci]         /expU_nests[j,parent_slot_in_nests])/MU
			else:
				if expU_nests[j,parent_slot_in_nests] == 0 or expU_nests[j,ci-n_elementals] == 0:
					conditional_probability[j,i] = -INFINITY
				else:
					conditional_probability[j,i] = log(expU_nests[j,ci-n_elementals]/expU_nests[j,parent_slot_in_nests])/MU


# expU_nest = 	( ∑j exp( Vj )**(1/mu) )**mu
#
#
#          exp( Vi / mu )        exp(Vi) ** (1/mu)       exp( Vi )**(1/mu)      exp( Vi )
# cp = 	 -----------------  =  --------------------  =  ------------------- = (-----------)**(1/mu)
#         ∑j exp(Vj / mu)       ∑j exp(Vj)**(1/mu)       expU_nest**(1/mu)      expU_nest




def conditional_logprob_from_tree(
		double[:,:] expU_elemental,
		double[:,:] expU_nests,
		graph,
		params,
		conditional_logprobability_dict
	):
	cdef int[:] child_slots
	cdef double mu
	for node in graph.topological_sorted_no_elementals:
		child_slots = graph.successor_slots(node)
		if 'parameter' in graph.node[node]:
			mu = params.get_value(graph.node[node]['parameter'])
		else:
			mu = 1.0
		conditional_logprob_from_nest_exp_util( expU_elemental, expU_nests, child_slots, child_slots.shape[0], graph.standard_slot_map[node], mu, conditional_logprobability_dict[node] )


def elemental_logprob_from_conditional_logprob(
		conditional_logprobability_dict,
		graph,
		logprobability
):
	for upnestcode in conditional_logprobability_dict.keys():
		for dnslot, dncode in enumerate(graph.successors_iter(upnestcode)):
			for elemental_code in graph.elemental_descendants_iter(dncode):
				elemental_slot = graph.standard_slot_map[elemental_code]
				logprobability[:,elemental_slot] += conditional_logprobability_dict[upnestcode][:,dnslot]




def total_prob_from_log_conditional_prob(
		log_conditional_probability,
		graph,
		total_probability,  # shape = (cases, nodes)
):
	total_probability[:,-1] = 1.0
	total_probability[:,:-1] = 0.0

	ups, dns, first_visits = graph.edge_slot_arrays()
	for e in range(graph.n_edges):
		e_ = graph.n_edges-e-1
		total_probability[:,dns[e_]] += numpy.exp(log_conditional_probability[:,e_]) * total_probability[:,ups[e_]]




@cython.boundscheck(False)
@cython.cdivision(True)
cdef conditional_logprob_from_nest_util(
		const double[:,:] U_elemental,
		const double[:,:] U_nests,
		int[:] child_slots,
		int n_children,                       # = child_slots.shape[0]
		int parent_slot,
		double MU,
		double[:,:] conditional_logprob,  # output, shape = (n_cases, n_children)
	):

	cdef int ci, i, j
	cdef int len_j = U_elemental.shape[0]
	cdef int n_elementals = U_elemental.shape[1]
	cdef int parent_slot_in_nests = parent_slot-n_elementals

	for j in prange(len_j, nogil=True):
		for i in range(n_children):
			ci = child_slots[i]
			if ci < n_elementals:
				if U_nests[j,parent_slot_in_nests] == -INFINITY or U_elemental[j,ci] == -INFINITY:
					conditional_logprob[j,i] = -INFINITY
				else:
					conditional_logprob[j,i] = (U_elemental[j,ci] - U_nests[j,parent_slot_in_nests])/MU
			else:
				if U_nests[j,parent_slot_in_nests] == -INFINITY or U_nests[j,ci-n_elementals] == -INFINITY:
					conditional_logprob[j,i] = -INFINITY
				else:
					conditional_logprob[j,i] = (U_nests[j,ci-n_elementals]-U_nests[j,parent_slot_in_nests])/MU


# U_nest = 	mu*log( ∑j exp( Vj / mu ) )
#
#
#          exp( Vi / mu )      exp( Vi / mu )
# cp = 	 -----------------  = ----------------  = exp( (Vi - U_nest) / mu )
#         ∑j exp(Vj / mu)     exp(U_nest / mu)
#
# log(cp) = ( (Vi - U_nest) / mu )

def conditional_logprob_from_tree_util(
		double[:,:] U_elemental,
		double[:,:] U_nests,
		graph,
		params,
		conditional_logprobability_dict
	):
	cdef int[:] child_slots
	cdef double mu
	for node in graph.topological_sorted_no_elementals:
		child_slots = graph.successor_slots(node)
		if 'parameter' in graph.node[node]:
			mu = params.get_value(graph.node[node]['parameter'])
		else:
			mu = 1.0
		conditional_logprob_from_nest_util( U_elemental, U_nests, child_slots, child_slots.shape[0], graph.standard_slot_map[node], mu, conditional_logprobability_dict[node] )
