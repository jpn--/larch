

from numpy.math cimport INFINITY
from ..linalg.level1 cimport daxpy_memview
from libc.math cimport pow, exp, log
from scipy.linalg.cython_blas cimport daxpy as _daxpy


cpdef case_dUtility_dFusedParameters(
		int c,

		int n_cases,
		int n_elementals,
		int n_nests,
		int n_edges,

		double[:,:] log_prob_elementals,  # shape = [cases,alts]
		double[:,:] util_elementals,      # shape = [cases,alts]
		double[:,:] util_nests,           # shape = [cases,nests]

		double[:] coef_u_ca,               # shape = (vars_ca)
		double[:,:] coef_u_co,             # shape = (vars_co,alts)
		double[:] coef_q_ca,               # shape = (qvars_ca)
		double[:] coef_mu,                 # shape = (nests)

		double[:,:] d_util_meta,           # shape = (nodes,n_meta_coef)  refs same memory as the d_util_coef_* arrays...
		double[:,:] d_util_coef_u_ca,      # shape = (nodes,vars_ca)
		double[:,:,:] d_util_coef_u_co,    # shape = (nodes,vars_co,alts)
		double[:,:] d_util_coef_q_ca,      # shape = (nodes,qvars_ca)
		double[:,:] d_util_coef_mu,        # shape = (nodes,nests)

		double[:,:,:] data_u_ca,           # shape = (cases,alts,vars_ca)
		double[:,:] data_u_co,             # shape = (cases,vars_co)

		double[:,:] log_conditional_prob,      # shape = (cases,edges)

		int[:] edge_child_slot,            # shape = (edges)
		int[:] edge_parent_slot,           # shape = (edges)
		int[:] first_visit_to_child,       # shape = (edges)
):
	cdef:
		int e, a, nest_slot, dn, up, up_n, n, incx, j
		double cond_prob

	for e in range(n_edges):
		# 'e' is iterated over all the links in the network

		dn = edge_child_slot[e]
		up = edge_parent_slot[e]
		up_n = up - n_elementals

		if log_conditional_prob[c,e] < -1.797e+308: # i.e. -INF
			continue

		# First, we calculate the effect of various parameters on the utility
		# of 'a' directly. For elemental alternatives, this means beta parameters.
		# For other nodes, only mu has a direct effect

		# BETA for SELF (elemental alternatives)
		if dn < n_elementals and first_visit_to_child[e]>0:
			d_util_coef_u_ca[dn,:] = data_u_ca[c,dn,:]
			d_util_coef_u_co[dn,:,dn] = data_u_co[c,:]

		# MU for SELF (adjust the kiddies contributions)
		if dn >= n_elementals and first_visit_to_child[e]>0:
			d_util_coef_mu[dn,dn-n_elementals] += util_nests[c,dn-n_elementals]
			d_util_coef_mu[dn,dn-n_elementals] /= coef_mu[dn-n_elementals]

		# MU for Parent (non-competitive edge)
		cond_prob = exp(log_conditional_prob[c,e])
		if dn < n_elementals:
			d_util_coef_mu[up, up_n] -= cond_prob * util_elementals[c,dn]
		else:
			d_util_coef_mu[up, up_n] -= cond_prob * util_nests[c,dn-n_elementals]

		# Finally, roll up secondary effects on parents
		if cond_prob>0:
			# daxpy_memview(cond_prob,d_util_meta[dn,:],d_util_meta[up,:])
			# # n = d_util_meta.shape[1]
			# # incx = d_util_meta.strides[1]
			# # _daxpy(&n, &cond_prob, &d_util_meta[dn,0], &incx, &d_util_meta[up,0], &incx)
			for j in range(d_util_meta.shape[1]):
				d_util_meta[up,j] += d_util_meta[dn,j] * cond_prob




cpdef case_dProbability_dFusedParameters(
		int c,

		int n_cases,
		int n_elementals,
		int n_nests,
		int n_edges,
		int n_meta_coef,

		double[:,:] util_elementals,      # shape = [cases,alts]
		double[:,:] util_nests,           # shape = [cases,nests]

		double[:] coef_u_ca,               # shape = (vars_ca)
		double[:,:] coef_u_co,             # shape = (vars_co,alts)
		double[:] coef_q_ca,               # shape = (qvars_ca)
		double[:] coef_mu,                 # shape = (nests)

		double[:,:] d_util_meta,           # shape = (nodes,n_meta_coef)  refs same memory as the d_util_coef_* arrays...
		double[:,:] d_util_coef_u_ca,      # shape = (nodes,vars_ca)
		double[:,:,:] d_util_coef_u_co,    # shape = (nodes,vars_co,alts)
		double[:,:] d_util_coef_q_ca,      # shape = (nodes,qvars_ca)
		double[:,:] d_util_coef_mu,        # shape = (nodes,nests)

		double[:,:] log_conditional_prob,  # shape = (cases,edges)
		double[:,:] total_probability,     # shape = (cases,nodes)

		double[:,:  ] d_prob_meta,         # shape = (nodes,n_meta_coef)  refs same memory as the d_prob_coef_* arrays...
		double[:,:  ] d_prob_coef_u_ca,    # shape = (nodes,vars_ca)
		double[:,:,:] d_prob_coef_u_co,    # shape = (nodes,vars_co,alts)
		double[:,:  ] d_prob_coef_q_ca,    # shape = (nodes,qvars_ca)
		double[:,:  ] d_prob_coef_mu,      # shape = (nodes,nests)

		int[:] edge_child_slot,            # shape = (edges)
		int[:] edge_parent_slot,           # shape = (edges)
		int[:] first_visit_to_child,       # shape = (edges)

		# Workspace arrays, not input or output but must be pre-allocated

		double[:] scratch,              # shape = (n_meta_coef,)  refs same memory as the scratch_coef_* arrays...
		double[:] scratch_coef_mu,      # shape = (nests,)

):
	cdef int i,j, e, e_rev
	cdef int dn_slot, up_slot, up_slot_in_nests, dn_slot_in_nests
	cdef double multiplier

	for i in range(d_prob_meta.shape[0]):
		for j in range(d_prob_meta.shape[1]):
			d_prob_meta[i,j] = 0

	for e_rev in range(n_edges):
		e = n_edges-e_rev-1

		dn_slot = edge_child_slot[e]
		up_slot = edge_parent_slot[e]
		up_slot_in_nests = up_slot-n_elementals
		dn_slot_in_nests = dn_slot-n_elementals

		for i in range(n_meta_coef):
			scratch[i] = d_util_meta[dn_slot,i] - d_util_meta[up_slot,i]

		if 0:
			# for competitive edges, adjust phi
			# scratch += X_Phi()[edge] - sum over competes(Alloc[compete]*X_Phi[compete])
			pass # TODO, assume no competitive edges for now
		else:
			# adjust Mu for hierarchical structure, noncompete
			if dn_slot_in_nests < 0:
				if util_elementals[c,dn_slot] != -INFINITY:
					scratch_coef_mu[up_slot_in_nests] += (util_nests[c,up_slot_in_nests] - util_elementals[c,dn_slot]) / coef_mu[up_slot_in_nests]
			else:
				if util_nests[c,dn_slot_in_nests] != -INFINITY:
					scratch_coef_mu[up_slot_in_nests] += (util_nests[c,up_slot_in_nests] - util_nests[c,dn_slot_in_nests]) / coef_mu[up_slot_in_nests]

		multiplier = total_probability[c,up_slot]/coef_mu[up_slot_in_nests]
		for i in range(n_meta_coef):
			scratch[i] *= multiplier
			scratch[i] += d_prob_meta[up_slot, i]
			d_prob_meta[dn_slot, i] += scratch[i] * exp(log_conditional_prob[c,e])






cpdef case_dLogLike_dFusedParameters(
		int c,
		int n_meta_coef,

		double[:,:] total_probability,     # shape = (cases,nodes)

		double[:,:  ] d_prob_meta,         # shape = (nodes,n_meta_coef)  refs same memory as the d_prob_coef_* arrays...
		double[:,:  ] d_prob_coef_u_ca,    # shape = (nodes,vars_ca)
		double[:,:,:] d_prob_coef_u_co,    # shape = (nodes,vars_co,alts)
		double[:,:  ] d_prob_coef_q_ca,    # shape = (nodes,qvars_ca)
		double[:,:  ] d_prob_coef_mu,      # shape = (nodes,nests)

		double[:  ] d_LL_meta,           # shape = (n_meta_coef)  refs same memory as the d_LL_coef_* arrays...
		double[:  ] d_LL_coef_u_ca,      # shape = (vars_ca)
		double[:,:] d_LL_coef_u_co,      # shape = (vars_co,alts)
		double[:  ] d_LL_coef_q_ca,      # shape = (qvars_ca)
		double[:  ] d_LL_coef_mu,        # shape = (nests)

		double[:,:] choices,               # shape = (cases,nodes|elementals)

	):
	cdef int a, i

	for i in range(n_meta_coef):
		d_LL_meta[i] = 0

	for a in range(choices.shape[1]):
		if choices[c,a]>0:
			if total_probability[c,a]>0:
				for i in range(n_meta_coef):
					d_LL_meta[i] += d_prob_meta[a,i] * choices[c,a]/total_probability[c,a]
			else:
				pass # TODO: raise ZeroProbWhenChosen






from ..linalg.contiguous_group import Blocker
import numpy


def _d_loglike_single_case(
		int c,
		p,       # Model
		dU,      # Blocker
		dP,      # Blocker
		scratch, # Blocker
		dLL,     # Blocker
		int[:] edge_child_slot,            # shape = (edges)
		int[:] edge_parent_slot,           # shape = (edges)
		int[:] first_visit_to_child,       # shape = (edges)
):
	dU.outer[:] = 0
	dP.outer[:] = 0
	scratch.outer[:] = 0
	dLL.outer[:] = 0

	case_dUtility_dFusedParameters(
		c, # int c,

		p.data.n_cases,  # int n_cases,
		p.data.n_alts,   # int n_elementals,
		len(p.graph) - p.data.n_alts, #  int n_nests,
		p.graph.n_edges, # int n_edges,

		p.work.log_prob,        # double[:,:] log_prob_elementals,  # shape = [cases,alts]
		p.work.util_elementals, # double[:,:] util_elementals,      # shape = [cases,alts]
		p.work.util_nests,      # double[:,:] util_nests,           # shape = [cases,nests]

		p.coef_utility_ca,  # double[:] coef_u_ca,               # shape = (vars_ca)
		p.coef_utility_co,  # double[:,:] coef_u_co,             # shape = (vars_co,alts)
		p.coef_quantity_ca, # double[:] coef_q_ca,               # shape = (qvars_ca)
		p.coef_logsums,     # double[:] coef_mu,                 # shape = (nests)

		dU.outer,     # double[:,:] d_util_meta,           # shape = (nodes,n_meta_coef)  refs same memory as the d_util_coef_* arrays...
		dU.inners[0], # double[:,:] d_util_coef_u_ca,      # shape = (nodes,vars_ca)
		dU.inners[1], # double[:,:,:] d_util_coef_u_co,    # shape = (nodes,vars_co,alts)
		dU.inners[2], # double[:,:] d_util_coef_q_ca,      # shape = (nodes,qvars_ca)
		dU.inners[3], # double[:,:] d_util_coef_mu,        # shape = (nodes,nests)

		p.data._u_ca, # double[:,:,:] data_u_ca,           # shape = (cases,alts,vars_ca)
		p.data._u_co, # double[:,:] data_u_co,             # shape = (cases,vars_co)

		p.work.log_conditional_probability, # double[:,:] conditional_prob,      # shape = (cases,edges)

		edge_child_slot,      # int[:] edge_child_slot,            # shape = (edges)
		edge_parent_slot,     # int[:] edge_parent_slot,           # shape = (edges)
		first_visit_to_child, # int[:] first_visit_to_child,       # shape = (edges)
	)
	case_dProbability_dFusedParameters(
		c,
		p.data.n_cases,  # int n_cases,
		p.data.n_alts,  # int n_elementals,
		len(p.graph) - p.data.n_alts,  # int n_nests,
		p.graph.n_edges,  # int n_edges,

		dU.outer.shape[1], # int n_meta_coef,

	    p.work.util_elementals,  # double[:,:] util_elementals,      # shape = [cases,alts]
	    p.work.util_nests,       # double[:,:] util_nests,           # shape = [cases,nests]

	    p.coef_utility_ca,   # double[:] coef_u_ca,               # shape = (vars_ca)
	    p.coef_utility_co,   # double[:,:] coef_u_co,             # shape = (vars_co,alts)
	    p.coef_quantity_ca,  # double[:] coef_q_ca,               # shape = (qvars_ca)
	    p.coef_logsums,      # double[:] coef_mu,                 # shape = (nests)

	    dU.outer,      # double[:,:] d_util_meta,           # shape = (nodes,n_meta_coef)  refs same memory as the d_util_coef_* arrays...
	    dU.inners[0],  # double[:,:] d_util_coef_u_ca,      # shape = (nodes,vars_ca)
	    dU.inners[1],  # double[:,:,:] d_util_coef_u_co,    # shape = (nodes,vars_co,alts)
	    dU.inners[2],  # double[:,:] d_util_coef_q_ca,      # shape = (nodes,qvars_ca)
	    dU.inners[3],  # double[:,:] d_util_coef_mu,        # shape = (nodes,nests)

	    p.work.log_conditional_probability,  # double[:,:] conditional_prob,      # shape = (cases,edges)
	    p.work.total_probability,       # shape = (nodes)

	    dP.outer,  # double[:,:] d_util_meta,           # shape = (nodes,n_meta_coef)  refs same memory as the d_util_coef_* arrays...
	    dP.inners[0],  # double[:,:] d_util_coef_u_ca,      # shape = (nodes,vars_ca)
	    dP.inners[1],  # double[:,:,:] d_util_coef_u_co,    # shape = (nodes,vars_co,alts)
	    dP.inners[2],  # double[:,:] d_util_coef_q_ca,      # shape = (nodes,qvars_ca)
	    dP.inners[3],  # double[:,:] d_util_coef_mu,        # shape = (nodes,nests)

		edge_child_slot,      # int[:] edge_child_slot,            # shape = (edges)
		edge_parent_slot,     # int[:] edge_parent_slot,           # shape = (edges)
		first_visit_to_child, # int[:] first_visit_to_child,       # shape = (edges)

	    # Workspace arrays, not input or output but must be pre-allocated

		scratch.outer, # double[:] scratch,              # shape = (n_meta_coef,)
		scratch.inners[3], #double[:] scratch_coef_mu,      # shape = (nests,)
	)

	case_dLogLike_dFusedParameters(
		c,
		dU.outer.shape[1],  # int n_meta_coef,

		p.work.total_probability,  # shape = (nodes)

		dP.outer,      # double[:,:] d_util_meta,           # shape = (nodes,n_meta_coef)  refs same memory as...
		dP.inners[0],  # double[:,:] d_util_coef_u_ca,      # shape = (nodes,vars_ca)
		dP.inners[1],  # double[:,:,:] d_util_coef_u_co,    # shape = (nodes,vars_co,alts)
		dP.inners[2],  # double[:,:] d_util_coef_q_ca,      # shape = (nodes,qvars_ca)
		dP.inners[3],  # double[:,:] d_util_coef_mu,        # shape = (nodes,nests)

		dLL.outer,      # double[:,:] d_util_meta,           # shape = (nodes,n_meta_coef)  refs same memory as...
		dLL.inners[0],  # double[:,:] d_util_coef_u_ca,      # shape = (nodes,vars_ca)
		dLL.inners[1],  # double[:,:,:] d_util_coef_u_co,    # shape = (nodes,vars_co,alts)
		dLL.inners[2],  # double[:,:] d_util_coef_q_ca,      # shape = (nodes,qvars_ca)
		dLL.inners[3],  # double[:,:] d_util_coef_mu,        # shape = (nodes,nests)

		p.data._choice_ca,  #double[:,:] choices,               # shape = (cases,nodes|elementals)

	)

	return p.push_to_parameterlike(dLL)





def _d_loglike_casewise(
		p, # Model
):

	cdef int c
	cdef int n_cases = p.data.n_cases

	edge_slot_arrays = p.graph.edge_slot_arrays()
	dU = Blocker([len(p.graph)], [
		p.coef_utility_ca.shape,
		p.coef_utility_co.shape,
		p.coef_quantity_ca.shape,
		p.coef_logsums.shape,
	])

	dP = Blocker([len(p.graph)], [
		p.coef_utility_ca.shape,
		p.coef_utility_co.shape,
		p.coef_quantity_ca.shape,
		p.coef_logsums.shape,
	])

	scratch = Blocker([], [
		p.coef_utility_ca.shape,
		p.coef_utility_co.shape,
		p.coef_quantity_ca.shape,
		p.coef_logsums.shape,
	])

	dLL = Blocker([], [
		p.coef_utility_ca.shape,
		p.coef_utility_co.shape,
		p.coef_quantity_ca.shape,
		p.coef_logsums.shape,
	])

	dLLc = numpy.zeros([n_cases, len(p.frame)])

	for c in range(n_cases):
		dLLc[c,:] = _d_loglike_single_case(
			c,
			p,       # Model
			dU,      # Blocker
			dP,      # Blocker
			scratch, # Blocker
			dLL,     # Blocker
			edge_slot_arrays[1], # shape = (edges)
		    edge_slot_arrays[0], # shape = (edges)
		    edge_slot_arrays[2], # shape = (edges)
		)
	return dLLc


def _d_loglike(
		p, # Model
):

	cdef int c
	cdef int n_cases = p.data.n_cases

	edge_slot_arrays = p.graph.edge_slot_arrays()
	dU = Blocker([len(p.graph)], [
		p.coef_utility_ca.shape,
		p.coef_utility_co.shape,
		p.coef_quantity_ca.shape,
		p.coef_logsums.shape,
	])

	dP = Blocker([len(p.graph)], [
		p.coef_utility_ca.shape,
		p.coef_utility_co.shape,
		p.coef_quantity_ca.shape,
		p.coef_logsums.shape,
	])

	scratch = Blocker([], [
		p.coef_utility_ca.shape,
		p.coef_utility_co.shape,
		p.coef_quantity_ca.shape,
		p.coef_logsums.shape,
	])

	dLL = Blocker([], [
		p.coef_utility_ca.shape,
		p.coef_utility_co.shape,
		p.coef_quantity_ca.shape,
		p.coef_logsums.shape,
	])

	dLL_ = numpy.zeros([len(p.frame)])
	bhhh_ = numpy.zeros([len(p.frame),len(p.frame)])

	for c in range(n_cases):
		this_case = _d_loglike_single_case(
			c,
			p,       # Model
			dU,      # Blocker
			dP,      # Blocker
			scratch, # Blocker
			dLL,     # Blocker
			edge_slot_arrays[1], # shape = (edges)
		    edge_slot_arrays[0], # shape = (edges)
		    edge_slot_arrays[2], # shape = (edges)
		)
		dLL_[:] += this_case
		bhhh_[:,:] += numpy.outer(this_case,this_case) # candidate for DSYR from blas or DSPR for packed
	return dLL_, bhhh_
