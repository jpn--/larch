# cython: language_level=3

from ..general_precision cimport *


cdef class TreeStructure:

	cdef:
		int             n_nodes
		int             n_elementals
		l4_float_t[:]   model_mu_param_values     # [n_nodes]
		l4_float_t[:,:] model_alpha_param_values  # [n_nodes, n_nodes]
		int[:]          model_mu_param_slots      # [n_nodes]

		int           n_edges
		int[:]        edge_dn              # [n_edges]
		int[:]        edge_up              # [n_edges]
		l4_float_t[:] edge_alpha_values    # [n_edges]
		l4_float_t[:] edge_logalpha_values # [n_edges]
		int[:]        first_edge_for_up    # [n_nodes] index of first edge where this node is the up
		int[:]        n_edges_for_up       # [n_nodes] n edge where this node is the up
