
import numpy

class WorkspaceCollection():

	def __init__(self, data_coll, parameter_coll, graph=None):
		self.util_elementals = numpy.zeros([data_coll.n_cases, data_coll.n_alts])
		self.log_prob = numpy.zeros([data_coll.n_cases, data_coll.n_alts])

		if graph:
			self.util_nests = numpy.zeros([data_coll.n_cases, len(graph) - data_coll.n_alts])
			self.log_conditional_probability = numpy.zeros([data_coll.n_cases, graph.n_edges])
			self.log_conditional_prob_dict = {}
			n = 0
			ups, dns, fis = graph.edge_slot_arrays()
			while n < graph.n_edges:
				code = graph.standard_sort[ups[n]]
				degree = graph.out_degree(code)
				self.log_conditional_prob_dict[code] = self.log_conditional_probability[:, n:n + degree]
				n += degree
