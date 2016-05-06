
from ..util.pmath import category, pmath, rename
from ..core import LarchError, ParameterAlias
from io import StringIO
import numpy

def _tex_text(x):
	return x.replace('#','\#').replace('$','\$').replace('&','\&').replace('_','\_').replace('^','\^')



class PdfModelReporter():

	def pdf_nesting_tree(self,**format):
		existing_format_keys = list(format.keys())
		for key in existing_format_keys:
			if key.upper()!=key: format[key.upper()] = format[key]
		if 'GRAPHWIDTH' not in format: format['GRAPHWIDTH'] = 6.5
		if 'GRAPHHEIGHT' not in format: format['GRAPHHEIGHT'] = 4
		if 'UNAVAILABLE' not in format: format['UNAVAILABLE'] = True
		if 'HIDECODES' not in format: format['HIDECODES'] = False
		import pygraphviz as viz
		from io import BytesIO
		G=viz.AGraph(name='Tree',directed=True,size="{GRAPHWIDTH},{GRAPHHEIGHT}".format(**format))
		for n,name in self.alternatives().items():
			if format['HIDECODES']:
				G.add_node(n, label='<{1}>'.format(n,name))
			else:
				G.add_node(n, label='<{1} <FONT COLOR="#999999">({0})</FONT>>'.format(n,name))
		eG = G.add_subgraph(name='cluster_elemental', nbunch=self.alternative_codes(), color='#cccccc', bgcolor='#eeeeee',
					   label='Elemental Alternatives', labelloc='b', style='rounded,solid')
		unavailable_nodes = set()
		if format['UNAVAILABLE']==True or format['UNAVAILABLE']=='HIDE':
			if self.is_provisioned():
				try:
					for n, ncode in enumerate(self.alternative_codes()):
#						print("AVCHEK1",ncode,'-->',numpy.sum(self.Data('Avail'),axis=0)[n,0])
						if numpy.sum(self.Data('Avail'),axis=0)[n,0]==0: unavailable_nodes.add(ncode)
				except: raise
			if self.db is None:
				legible_avail = False
			else:
				legible_avail = not isinstance(self.db.queries.avail, str)
			if legible_avail:
				for ncode,navail in self.db.queries.avail.items():
					try:
#						print("AVCHEK2",ncode,'-->',navail)
						if navail=='0': unavailable_nodes.add(ncode)
					except: raise
			if format['UNAVAILABLE']==True:
				eG.add_subgraph(name='cluster_elemental_unavailable', nbunch=unavailable_nodes, color='#bbbbbb', bgcolor='#dddddd',
							   label='Unavailable Alternatives', labelloc='b', style='rounded,solid')
			if format['UNAVAILABLE']=='HIDE':
				for unavailable_node in unavailable_nodes:
					G.delete_node(unavailable_node)
		G.add_node(self.root_id, label="Root")
		for n in self.node.nodes():
			if self.node[n]._altname==self.node[n].param:
				if format['HIDECODES']:
					G.add_node(n, label='<{1}>'.format(n,self.node[n]._altname,self.node[n].param))
				else:
					G.add_node(n, label='<{1} <FONT COLOR="#999999">({0})</FONT>>'.format(n,self.node[n]._altname,self.node[n].param))
			else:
				if format['HIDECODES']:
					G.add_node(n, label='<{1}<BR/>µ<SUB>{2}</SUB>>'.format(n,self.node[n]._altname,self.node[n].param))
				else:
					G.add_node(n, label='<{1} <FONT COLOR="#999999">({0})</FONT><BR/>µ<SUB>{2}</SUB>>'.format(n,self.node[n]._altname,self.node[n].param))
		up_nodes = set()
		down_nodes = set()
		for i,j in self.link.links():
			G.add_edge(i,j)
			down_nodes.add(j)
			up_nodes.add(i)
		all_nodes = set(self.alternative_codes()) | up_nodes | down_nodes
		for j in all_nodes-down_nodes-unavailable_nodes:
			G.add_edge(self.root_id,j)
		pyg_imgdata = BytesIO()
		G.draw(pyg_imgdata, format='pdf', prog='dot')
		return pyg_imgdata


