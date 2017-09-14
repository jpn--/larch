
import numpy

class WorkspaceCollection():

	def __init__(self, data_coll, parameter_coll, graph=None):
		self.util_elementals = numpy.zeros([data_coll.n_cases, data_coll.n_alts])
		self.log_prob = numpy.zeros([data_coll.n_cases, data_coll.n_alts])

		if graph:
			self.util_nests = numpy.zeros([data_coll.n_cases, len(graph) - data_coll.n_alts])
			self.log_conditional_prob = {
				code: numpy.zeros([data_coll.n_cases, code_out_degree])
				for code, code_out_degree in graph.out_degree_iter()
				if code_out_degree > 0
			}