# cython: language_level=3, embedsignature=True

import numpy

cdef class TreeStructure:

	def __init__(self, model, graph):
		self.n_nodes = len(graph)
		self.n_elementals = graph.n_elementals()
		mu, muslots, alpha, up, dn, num, start, val = graph._get_simple_mu_and_alpha(model)
		self.model_mu_param_values = mu        # [n_nodes]
		self.model_mu_param_slots = muslots    # [n_nodes]
		self.model_alpha_param_values = alpha  # [n_nodes,n_nodes]
		self.n_edges               = dn.shape[0]    #
		self.edge_dn               = dn             # [n_edges]
		self.edge_up               = up             # [n_edges]
		self.edge_alpha_values     = val            # [n_edges]
		self.edge_logalpha_values  = numpy.log(val) # [n_edges]
		self.first_edge_for_up     = start          # [n_nodes] index of first edge where this node is the up
		self.n_edges_for_up        = num            # [n_nodes] n edge where this node is the up
