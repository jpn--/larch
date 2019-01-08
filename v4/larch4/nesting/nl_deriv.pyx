# cython: profile=True


from numpy.math cimport INFINITY
from ..linalg.level1 cimport daxpy_memview
from libc.math cimport pow, exp, log
from scipy.linalg.cython_blas cimport daxpy as _daxpy
from cython.parallel cimport prange
from ..math.blocker cimport CBlocker0, CBlocker1


cdef void case_dUtility_dFusedParameters(
		int c,

		int n_cases,
		int n_elementals,
		int n_nests,
		int n_edges,

		double[:,:] log_prob_elementals,  # shape = [cases,alts]
		double[:,:] util_elementals,      # shape = [cases,alts]
		double[:,:] util_nests,           # shape = [cases,nests]

		CBlocker0 coef,
		CBlocker1 d_util,

		double[:,:,:] data_u_ca,           # shape = (cases,alts,vars_ca)
		double[:,:] data_u_co,             # shape = (cases,vars_co)

		double[:,:] log_conditional_prob,      # shape = (cases,edges)

		int[:] edge_child_slot,            # shape = (edges)
		int[:] edge_parent_slot,           # shape = (edges)
		int[:] first_visit_to_child,       # shape = (edges)
) nogil:
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
			d_util.util_ca[dn,:] = data_u_ca[c,dn,:]
			d_util.util_co[dn,:,dn] = data_u_co[c,:]

		# MU for SELF (adjust the kiddies contributions)
		if dn >= n_elementals and first_visit_to_child[e]>0:
			d_util.logsums[dn,dn-n_elementals] += util_nests[c,dn-n_elementals]
			d_util.logsums[dn,dn-n_elementals] /= coef.logsums[dn-n_elementals]

		# MU for Parent (non-competitive edge)
		cond_prob = exp(log_conditional_prob[c,e])
		if dn < n_elementals:
			d_util.logsums[up, up_n] -= cond_prob * util_elementals[c,dn]
		else:
			d_util.logsums[up, up_n] -= cond_prob * util_nests[c,dn-n_elementals]

		# Finally, roll up secondary effects on parents
		if cond_prob>0:
			# daxpy_memview(cond_prob,d_util_meta[dn,:],d_util_meta[up,:])
			# # n = d_util_meta.shape[1]
			# # incx = d_util_meta.strides[1]
			# # _daxpy(&n, &cond_prob, &d_util_meta[dn,0], &incx, &d_util_meta[up,0], &incx)
			for j in range(d_util.meta.shape[1]):
				d_util.meta[up,j] += d_util.meta[dn,j] * cond_prob




cdef void case_dProbability_dFusedParameters(
		int c,

		int n_cases,
		int n_elementals,
		int n_nests,
		int n_edges,
		int n_meta_coef,

		double[:,:] util_elementals,      # shape = [cases,alts]
		double[:,:] util_nests,           # shape = [cases,nests]

		CBlocker0 coef,

		CBlocker1 d_util,

		double[:,:] log_conditional_prob,  # shape = (cases,edges)
		double[:,:] total_probability,     # shape = (cases,nodes)

		CBlocker1 d_prob,

		int[:] edge_child_slot,            # shape = (edges)
		int[:] edge_parent_slot,           # shape = (edges)
		int[:] first_visit_to_child,       # shape = (edges)

		# Workspace arrays, not input or output but must be pre-allocated

		CBlocker0 scratch,              # shape = (n_meta_coef,)  refs same memory as the scratch_coef_* arrays...
		# double[:] scratch_coef_mu,      # shape = (nests,)

) nogil:
	cdef int i,j, e, e_rev
	cdef int dn_slot, up_slot, up_slot_in_nests, dn_slot_in_nests
	cdef double multiplier

	for i in range(d_prob.meta.shape[0]):
		for j in range(d_prob.meta.shape[1]):
			d_prob.meta[i,j] = 0

	for e_rev in range(n_edges):
		e = n_edges-e_rev-1

		dn_slot = edge_child_slot[e]
		up_slot = edge_parent_slot[e]
		up_slot_in_nests = up_slot-n_elementals
		dn_slot_in_nests = dn_slot-n_elementals

		for i in range(n_meta_coef):
			scratch.meta[i] = d_util.meta[dn_slot,i] - d_util.meta[up_slot,i]

		if 0:
			# for competitive edges, adjust phi
			# scratch += X_Phi()[edge] - sum over competes(Alloc[compete]*X_Phi[compete])
			pass # TODO, assume no competitive edges for now
		else:
			# adjust Mu for hierarchical structure, noncompete
			if dn_slot_in_nests < 0:
				if util_elementals[c,dn_slot] != -INFINITY:
					scratch.logsums[up_slot_in_nests] += (util_nests[c,up_slot_in_nests] - util_elementals[c,dn_slot]) / coef.logsums[up_slot_in_nests]
			else:
				if util_nests[c,dn_slot_in_nests] != -INFINITY:
					scratch.logsums[up_slot_in_nests] += (util_nests[c,up_slot_in_nests] - util_nests[c,dn_slot_in_nests]) / coef.logsums[up_slot_in_nests]

		multiplier = total_probability[c,up_slot]/coef.logsums[up_slot_in_nests]
		for i in range(n_meta_coef):
			scratch.meta[i] *= multiplier
			scratch.meta[i] += d_prob.meta[up_slot, i]
			d_prob.meta[dn_slot, i] += scratch.meta[i] * exp(log_conditional_prob[c,e])






cdef void case_dLogLike_dFusedParameters(
		int c,
		int n_meta_coef,

		double[:,:] total_probability,     # shape = (cases,nodes)

		CBlocker1 d_prob,         # shape = (nodes,n_meta_coef)
		CBlocker0 d_LL,           # shape = (n_meta_coef)

		double[:,:] choices,               # shape = (cases,nodes|elementals)

	) nogil:
	cdef int a, i

	for i in range(n_meta_coef):
		d_LL.meta[i] = 0

	for a in range(choices.shape[1]):
		if choices[c,a]>0:
			if total_probability[c,a]>0:
				for i in range(n_meta_coef):
					d_LL.meta[i] += d_prob.meta[a,i] * choices[c,a]/total_probability[c,a]
			else:
				pass # TODO: raise ZeroProbWhenChosen






from ..linalg.contiguous_group import Blocker
import numpy


cdef void _d_loglike_single_case(
		int c,

		# p,                 # Model
		int n_cases,                           # p.data.n_cases,
		int n_alts,                            # p.data.n_alts,
		int n_nests,                           # len(p.graph) - p.data.n_alts,
		int n_edges,                           # p.graph.n_edges,
		double[:,:]   log_prob,                # p.work.log_prob,
		double[:,:]   util_elementals,         # p.work.util_elementals,
		double[:,:]   util_nests,              # p.work.util_nests,
		CBlocker0     coeffs,                  # CBlocker0(p._coef_block),
		double[:,:,:] data_util_ca,            # p.data._u_ca,
		double[:,:]   data_util_co,            # p.data._u_co,
		double[:,:]   log_conditional_prob,    # p.work.log_conditional_probability,
		double[:,:]   total_probability,       # p.work.total_probability,
		double[:,:]   choices,                 # p.data._choice_ca,

		CBlocker1 dU,      # Blocker
		CBlocker1 dP,      # Blocker
		CBlocker0 scratch, # Blocker
		CBlocker0 dLL,     # Blocker
		int[:] edge_child_slot,            # shape = (edges)
		int[:] edge_parent_slot,           # shape = (edges)
		int[:] first_visit_to_child,       # shape = (edges)
) nogil:
	dU.initialize()
	dP.initialize()
	scratch.initialize()
	dLL.initialize()

	case_dUtility_dFusedParameters(
		c, # int c,

		n_cases,  # int n_cases,
		n_alts,   # int n_elementals,
		n_nests,  #  int n_nests,
		n_edges,  # int n_edges,

		log_prob,        # double[:,:] log_prob_elementals,  # shape = [cases,alts]
		util_elementals, # double[:,:] util_elementals,      # shape = [cases,alts]
		util_nests,      # double[:,:] util_nests,           # shape = [cases,nests]

		coeffs,        # CBlocker0    # shape = (n_meta_coef)

		dU,  # CBlocker1    # shape = (nodes,n_meta_coef)

		data_util_ca, # double[:,:,:] data_u_ca,           # shape = (cases,alts,vars_ca)
		data_util_co, # double[:,:] data_u_co,             # shape = (cases,vars_co)

		log_conditional_prob, # double[:,:] conditional_prob,      # shape = (cases,edges)

		edge_child_slot,      # int[:] edge_child_slot,            # shape = (edges)
		edge_parent_slot,     # int[:] edge_parent_slot,           # shape = (edges)
		first_visit_to_child, # int[:] first_visit_to_child,       # shape = (edges)
	)
	case_dProbability_dFusedParameters(
		c,
		n_cases,  # int n_cases,
		n_alts,  # int n_elementals,
		n_nests,  # int n_nests,
		n_edges,  # int n_edges,

		dU.meta.shape[1], # int n_meta_coef,

	    util_elementals,  # double[:,:] util_elementals,      # shape = [cases,alts]
	    util_nests,       # double[:,:] util_nests,           # shape = [cases,nests]

		coeffs,

	    dU,      # CBlocker1           # shape = (nodes,n_meta_coef)

	    log_conditional_prob,  # double[:,:] conditional_prob,      # shape = (cases,edges)
	    total_probability,       # shape = (nodes)

		dP,       # CBlocker1           # shape = (nodes,n_meta_coef)

		edge_child_slot,      # int[:] edge_child_slot,            # shape = (edges)
		edge_parent_slot,     # int[:] edge_parent_slot,           # shape = (edges)
		first_visit_to_child, # int[:] first_visit_to_child,       # shape = (edges)

	    # Workspace arrays, not input or output but must be pre-allocated

		scratch, # CBlocker0,              # shape = (n_meta_coef,)
	)

	case_dLogLike_dFusedParameters(
		c,
		dU.meta.shape[1],  # int n_meta_coef,

		total_probability,  # shape = (nodes)

		dP,            # CBlocker1                          # shape = (nodes,n_meta_coef)  refs same memory as...
		dLL,           # CBlocker0                          # shape = (n_meta_coef)

		choices,  #double[:,:] choices,               # shape = (cases,nodes|elementals)

	)





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
		_d_loglike_single_case(
			c,
			# Model
			p.data.n_cases,
			p.data.n_alts,
			len(p.graph) - p.data.n_alts,
			p.graph.n_edges,
			p.work.log_prob,
			p.work.util_elementals,
			p.work.util_nests,
			CBlocker0(p._coef_block),
			p.data._u_ca,
			p.data._u_co,
			p.work.log_conditional_probability,
			p.work.total_probability,
			p.data._choice_ca,

			dU,      # Blocker
			dP,      # Blocker
			scratch, # Blocker
			dLL,     # Blocker
			edge_slot_arrays[1], # shape = (edges)
		    edge_slot_arrays[0], # shape = (edges)
		    edge_slot_arrays[2], # shape = (edges)
		)
		dLLc[c,:] = p.push_to_parameterlike( dLL.outer )
	return dLLc


def _d_loglike(
		p, # Model
):

	cdef int c, i, j

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

	dLL_cum = Blocker([], [
		p.coef_utility_ca.shape,
		p.coef_utility_co.shape,
		p.coef_quantity_ca.shape,
		p.coef_logsums.shape,
	])

	dLL_cum_c = CBlocker0(dLL_cum)

	bhhh_cum = numpy.zeros([dLL.size,dLL.size])


	cdef:
		int           n_cases                 = p.data.n_cases
		int           n_alts                  = p.data.n_alts
		int           n_nests                 = len(p.graph) - p.data.n_alts
		int           n_edges                 = p.graph.n_edges
		double[:,:]   log_prob                = p.work.log_prob
		double[:,:]   util_elementals         = p.work.util_elementals
		double[:,:]   util_nests              = p.work.util_nests
		CBlocker0     coeffs                  = CBlocker0(p._coef_block)
		double[:,:,:] data_util_ca            = p.data._u_ca
		double[:,:]   data_util_co            = p.data._u_co
		double[:,:]   log_conditional_prob    = p.work.log_conditional_probability
		double[:,:]   total_probability       = p.work.total_probability
		double[:,:]   choices                 = p.data._choice_ca

		CBlocker1 _c_dU      = CBlocker1(dU     )
		CBlocker1 _c_dP      = CBlocker1(dP     )
		CBlocker0 _c_scratch = CBlocker0(scratch)
		CBlocker0 _c_dLL     = CBlocker0(dLL    )
		double[:,:] bhhh_cum_c = bhhh_cum

		int[:] edge_slot_arrays1 = edge_slot_arrays[1], # shape = (edges)
		int[:] edge_slot_arrays0 = edge_slot_arrays[0], # shape = (edges)
		int[:] edge_slot_arrays2 = edge_slot_arrays[2], # shape = (edges)


	# for c in prange(n_cases, nogil=True):
	for c in range(n_cases):
		_d_loglike_single_case(
			c,
			# Model
			n_cases              ,
			n_alts               ,
			n_nests              ,
			n_edges              ,
			log_prob             ,
			util_elementals      ,
			util_nests           ,
			coeffs               ,
			data_util_ca         ,
			data_util_co         ,
			log_conditional_prob ,
			total_probability    ,
			choices              ,

			_c_dU      , # Blocker
			_c_dP      , # Blocker
			_c_scratch , # Blocker
			_c_dLL     , # Blocker
			edge_slot_arrays1, # shape = (edges)
		    edge_slot_arrays0, # shape = (edges)
		    edge_slot_arrays2, # shape = (edges)
		)
		for i in range(_c_dLL.meta.shape[0]):
			dLL_cum_c.meta[i] += _c_dLL.meta[i]
			for j in range(_c_dLL.meta.shape[0]):
				bhhh_cum_c[i,j] += _c_dLL.meta[i]*_c_dLL.meta[j] # candidate for DSYR from blas or DSPR for packed

	dLL_cum_params = p.push_to_parameterlike(dLL_cum)

	return dLL_cum_params, bhhh_cum
