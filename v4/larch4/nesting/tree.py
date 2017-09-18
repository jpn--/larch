import networkx as nx
from collections import OrderedDict
import numpy
from ..util.touch_notifier import TouchNotify

class NestingTree(TouchNotify,nx.DiGraph):

	node_dict_factory = OrderedDict
	adjlist_dict_factory = OrderedDict

	def __init__(self, *arg, root_id=0, **kwarg):
		if len(arg) and isinstance(arg[0], NestingTree):
			super().__init__(*arg, **kwarg)
			self._root_id = arg[0]._root_id
		else:
			super().__init__(*arg, **kwarg)
			self._root_id = root_id
		if self._root_id not in self.node:
			self.add_node(root_id, name='', root=True)
		self._clear_caches()

	def _clear_caches(self):
		self._topological_sorted = None
		self._topological_sorted_no_elementals = None
		self._standard_sort = None
		self._standard_slot_map = None
		self._predecessor_slots = {}
		self._successor_slots = {}
		self.touch()

	def add_edge(self, u, v, *arg, **kwarg):
		if 'implied' not in kwarg:
			drops = []
			for u_,v_,imp_ in self.in_edges_iter(nbunch=[v], data='implied'):
				if imp_:
					drops.append([u_,v_])
			for d in drops:
				self._remove_edge_no_implied(*d)
		self._clear_caches()
		return super().add_edge(u, v, *arg, **kwarg)

	def _remove_edge_no_implied(self, u, v, *arg, **kwarg):
		result = super().remove_edge(u, v)
		self._clear_caches()
		return result

	def remove_edge(self, u, v, *arg, **kwarg):
		result = super().remove_edge(u, v)
		if self.in_degree(v)==0 and v!=self._root_id:
			self.add_edge(self._root_id, v, implied=True)
		self._clear_caches()
		return result

	def add_node(self, code, *arg, children=(), parent=None, **kwarg):
		result = super().add_node(code, *arg, **kwarg)
		for child in children:
			self.add_edge(code, child)
		if parent is not None:
			self.add_edge(parent, code)
		else:
			if self.in_degree(code)==0 and code!=self._root_id:
				self.add_edge(self._root_id, code, implied=True)
		self._clear_caches()
		return result

	def add_nodes(self, codes, *arg, parent=None, **kwarg):
		for code in codes:
			self.add_node(code, *arg, parent=parent, **kwarg)

	@property
	def topological_sorted(self):
		if self._topological_sorted is not None:
			# use cached sort if available
			return self._topological_sorted
		self._topological_sorted = nx.topological_sort(self,reverse=True)
		return self._topological_sorted

	@property
	def topological_sorted_no_elementals(self):
		if self._topological_sorted_no_elementals is not None:
			# use cached  if available
			return self._topological_sorted_no_elementals
		self._topological_sorted_no_elementals = self.topological_sorted.copy()
		for code, out_degree in self.out_degree_iter():
			if not out_degree:
				self._topological_sorted_no_elementals.remove(code)
		return self._topological_sorted_no_elementals

	@property
	def standard_sort(self):
		if self._standard_sort is not None:
			# use cached if available
			return self._standard_sort
		self._standard_sort = self.elementals() + self.topological_sorted_no_elementals
		return self._standard_sort

	@property
	def standard_slot_map(self):
		if self._standard_slot_map is not None:
			# use cached if available
			return self._standard_slot_map
		self._standard_slot_map = {i:n for n,i in enumerate(self.standard_sort)}
		return self._standard_slot_map

	def predecessor_slots(self, code):
		if code in self._predecessor_slots:
			return self._predecessor_slots[code]
		s = numpy.empty(self.in_degree(code), dtype=numpy.int32)
		for n,i in enumerate( self.predecessors_iter(code) ):
			s[n] = self.standard_slot_map[i]
		self._predecessor_slots[code] = s
		return s

	def successor_slots(self, code):
		if code in self._successor_slots:
			return self._successor_slots[code]
		s = numpy.empty(self.out_degree(code), dtype=numpy.int32)
		for n,i in enumerate( self.successors_iter(code) ):
			s[n] = self.standard_slot_map[i]
		self._successor_slots[code] = s
		return s

	def elementals_iter(self):
		for code, out_degree in self.out_degree_iter():
			if not out_degree:
				yield code

	def elementals(self):
		return [i for i in self.elementals_iter()]

	def elemental_descendants_iter(self, code):
		if not self.out_degree(code):
			yield code
			return
		all_d = nx.descendants(self, code)
		for dcode, dout_degree in self.out_degree_iter(all_d):
			if not dout_degree:
				yield dcode

	def elemental_descendants(self, code):
		return [i for i in self.elemental_descendants_iter(code)]

	@property
	def n_edges(self):
		return sum(len(v) for v in self.edge.values())

	def edge_slot_arrays(self):
		s = self.n_edges
		up = numpy.zeros(s, dtype=numpy.int32)
		dn = numpy.zeros(s, dtype=numpy.int32)
		first_visit = numpy.zeros(s, dtype=numpy.int32)
		n = s
		first_visit_found = set()
		for upcode in reversed(self.standard_sort):
			upslot = self.standard_slot_map[upcode]
			for dnslot in reversed(self.successor_slots(upcode)):
				n -= 1
				up[n] = upslot
				dn[n] = dnslot
		for n in range(s):
			if dn[n] not in first_visit_found:
				first_visit[n] = 1
				first_visit_found.add(dn[n])
		return up, dn, first_visit
