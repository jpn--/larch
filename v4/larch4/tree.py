import networkx as nx



class NestingTree(nx.DiGraph):

	def __init__(self, *arg, root_id=0, **kwarg):
		super().__init__(*arg, **kwarg)
		self._root_id = root_id
		if self._root_id not in self.node:
			self.add_node(root_id, name='', root=True)

	def add_edge(self, u, v, *arg, **kwarg):
		drops = []
		for u_,v_,imp_ in self.in_edges_iter(nbunch=[v], data='implied'):
			if imp_:
				drops.append([u_,v_])
		for d in drops:
			self.remove_edge(*d)
		return super().add_edge(u, v, *arg, **kwarg)

	def remove_edge(self, u, v, *arg, **kwarg):
		result = super().remove_edge(u, v)
		if self.in_degree(v)==0 and v!=self._root_id:
			G.add_edge(self._root_id, v, implied=True)
		return result

	def add_node(self, code, *arg, children=(), parent=None, **kwarg):
		result = super().add_node(*arg, **kwarg)
		for child in children:
			self.add_edge(code, child)
		if self.out_degree(code)==0 and code!=self._root_id:
			self.add_edge(code, self._root_id, implied=True)
		return result

