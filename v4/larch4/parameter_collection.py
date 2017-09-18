import numpy
import pandas
from .roles import LinearFunction, LinearComponent, ParameterRef, DictOfLinearFunction
from .util import SignalDict

def _optional_index(y):
	if y is not None:
		return pandas.Index(y)





def _empty_parameter_frame(names, nullvalue=0, initvalue=0):
	return pandas.DataFrame(index=names, data=dict(
			value = numpy.full(len(names), fill_value=initvalue, dtype=numpy.float64),
			minimum = numpy.full(len(names), fill_value=-numpy.inf, dtype=numpy.float64),
			maximum = numpy.full(len(names), fill_value= numpy.inf, dtype=numpy.float64),
			nullvalue = numpy.full(len(names), fill_value=nullvalue, dtype=numpy.float64),
			initvalue = numpy.full(len(names), fill_value=initvalue, dtype=numpy.float64),
			holdfast = numpy.zeros(len(names), dtype=numpy.int8)
		), columns=['value', 'initvalue', 'nullvalue', 'minimum', 'maximum', 'holdfast'])


class ParameterCollection():

	def __init__(self, names, altindex,
					utility_ca=None,
					utility_co=None,
					quantity_ca=None,
					graph=None,
	):
		self._altindex = pandas.Index( altindex )

		self._utility_co_functions = DictOfLinearFunction(utility_co, touch_callback=self.mangle, alts_validator=self._check_alternative)
		self._utility_ca_function  = LinearFunction(utility_ca, touch_callback=self.mangle)

		self._quantity_ca_function  = LinearFunction(quantity_ca, touch_callback=self.mangle)

		self._graph = graph if graph is not None else self._mnl_graph()

		if names is None:
			names = ()

		self.frame = _empty_parameter_frame(names)
		self._scan_all_ensure_names()
		self._parameter_update_scheme = {}
		self.mangle()

	def _check_alternative(self, a):
		if a in self._altindex:
			return True
		return False


	def _scan_all_ensure_names(self):
		self._scan_utility_ensure_names()
		self._scan_quantity_ensure_names()
		self._scan_logsums_ensure_names()

	def _scan_utility_ensure_names(self):
		nameset = set()
		u_co_dataset = set()

		for altcode, linear_function in self._utility_co_functions.items():
			for component in linear_function:
				nameset.add(str(component.param))
				u_co_dataset.add(str(component.data))
		self._u_co_varindex = pandas.Index( u_co_dataset )

		self._u_ca_varindex = pandas.Index( str(component.data) for component in self._utility_ca_function )
		for component in self._utility_ca_function:
			nameset.add(str(component.param))

		self._ensure_names(nameset)


	def _scan_quantity_ensure_names(self):
		nameset = set()

		self._q_ca_varindex = pandas.Index( str(component.data) for component in self._quantity_ca_function )
		for component in self._q_ca_varindex:
			nameset.add(str(component.param))

		self._ensure_names(nameset)

	def _scan_logsums_ensure_names(self):
		nameset = set()

		self._q_ca_varindex = pandas.Index( str(component.data) for component in self._quantity_ca_function )
		for nodecode in self.graph.topological_sorted_no_elementals:
			if nodecode != self.graph._root_id:
				param_name = str(self.graph.node[nodecode]['parameter'])
				nameset.add(param_name)

		self._ensure_names(nameset)

	def _ensure_names(self, names, **kwargs):
		existing_names = set(self.frame.index)
		nameset = set(names)
		missing_names = nameset - existing_names
		if missing_names:
			self.frame = self.frame.append(_empty_parameter_frame([n for n in names if (n in missing_names)], **kwargs), verify_integrity=True)


	def mangle(self, *args, **kwargs):
		self._mangled = True

	def unmangle(self):
		if self._mangled:
			self._scan_all_ensure_names()
			self._initialize_derived_util_coef_arrays()
			self._mangled = False

	@property
	def coef_utility_co(self):
		self.unmangle()
		return self._coef_utility_co

	@property
	def coef_utility_ca(self):
		self.unmangle()
		return self._coef_utility_ca

	@property
	def coef_quantity_ca(self):
		self.unmangle()
		return self._coef_quantity_ca

	@property
	def coef_logsums(self):
		self.unmangle()
		return self._coef_logsums

	@property
	def coef_block(self):
		self.unmangle()
		return self._coef_block.outer

	@property
	def utility_ca_vars(self):
		self.unmangle()
		return self._u_ca_varindex

	@property
	def utility_co_vars(self):
		self.unmangle()
		return self._u_co_varindex

	@property
	def quantity_ca_vars(self):
		self.unmangle()
		return self._q_ca_varindex

	def _initialize_derived_util_coef_arrays(self):
		from .linalg.contiguous_group import Blocker
		n_logsum_params = len(self.graph)-len(self._altindex)
		self._coef_block = Blocker(
			[], [
				[len(self._u_ca_varindex)],                      # utility ca
				[len(self._u_co_varindex), len(self._altindex)], # utility co
				[len(self._q_ca_varindex)],                      # quantity ca
				[n_logsum_params],                               # logsums
			], dtype=numpy.float64
		)

		self._coef_utility_ca = self._coef_block.inners[0]
		self._coef_utility_co = self._coef_block.inners[1]
		self._coef_quantity_ca = self._coef_block.inners[2]
		self._coef_logsums = self._coef_block.inners[3]
		self._coef_logsums[:] = 1.0

		self._parameter_update_scheme = {}
		self._parameter_recall_scheme = {
			'_coef_utility_ca':[],
			'_coef_utility_co':[],
			'_coef_quantity_ca':[],
			'_coef_logsums':[],
		}

		for n,component in enumerate(self._utility_ca_function):
			param_name = str(component.param)
			if param_name not in self._parameter_update_scheme:
				self._parameter_update_scheme[param_name] = []
			self._parameter_update_scheme[param_name].append( ('_coef_utility_ca', (n,)) )
			self._parameter_recall_scheme['_coef_utility_ca'].append( ((n,), self.frame.index.get_loc(param_name)) )

		for altcode, linear_function in self._utility_co_functions.items():
			for component in linear_function:
				param_name = str(component.param)
				if param_name not in self._parameter_update_scheme:
					self._parameter_update_scheme[param_name] = []
				coord = (self._u_co_varindex.get_loc(str(component.data)), self._altindex.get_loc(altcode) )
				self._parameter_update_scheme[param_name].append( ('_coef_utility_co', coord) )
				self._parameter_recall_scheme['_coef_utility_co'].append( (coord, self.frame.index.get_loc(param_name)) )

		for n,component in enumerate(self._quantity_ca_function):
			param_name = str(component.param)
			if param_name not in self._parameter_update_scheme:
				self._parameter_update_scheme[param_name] = []
			self._parameter_update_scheme[param_name].append( ('_coef_quantity_ca', (n,)) )
			self._parameter_recall_scheme['_coef_quantity_ca'].append(((n,), self.frame.index.get_loc(param_name)))

		for n,nestcode in enumerate(self.graph.topological_sorted_no_elementals):
			node_dict = self.graph.node[nestcode]
			try:
				param_name = str(node_dict['parameter'])
			except KeyError:
				if self.graph._root_id != nestcode:
					raise
			else:
				if param_name not in self._parameter_update_scheme:
					self._parameter_update_scheme[param_name] = []
				self._parameter_update_scheme[param_name].append( ('_coef_logsums', (n,)) )
				self._parameter_recall_scheme['_coef_logsums'].append(((n,), self.frame.index.get_loc(param_name)))

		self._refresh_derived_arrays()

	def _refresh_derived_arrays(self):
		for name in self._parameter_update_scheme:
			value = self.frame.loc[name,'value']
			schemes = self._parameter_update_scheme[name]
			for member, location in schemes:
				self.__getattribute__(member)[location] = value

	def push_to_parameterlike(self, meta):
		result = numpy.zeros( tuple(meta.shape_prefix)+(len(self.frame),), dtype=meta.dtype )
		for i in self._parameter_recall_scheme['_coef_utility_ca']:
			result[...,i[1]] += meta.inners[0][...,i[0][0]]
		for i in self._parameter_recall_scheme['_coef_utility_co']:
			result[...,i[1]] += meta.inners[1][...,i[0][0],i[0][1]]
		for i in self._parameter_recall_scheme['_coef_quantity_ca']:
			result[...,i[1]] += meta.inners[2][...,i[0][0]]
		for i in self._parameter_recall_scheme['_coef_logsums']:
			result[...,i[1]] += meta.inners[3][...,i[0][0]]
		return result


	def set_value(self, name, value):
		self.frame.loc[name,'value'] = value
		if name in self._parameter_update_scheme:
			schemes = self._parameter_update_scheme[name]
			for member, location in schemes:
				self.__getattribute__(member)[location] = value

	def get_value(self, name):
		return self.frame.loc[name,'value']

	def __getitem__(self, name):
		return self.frame.loc[name,:]

	def set_values(self, values):
		if len(values) != len(self.frame):
			raise ValueError(f'gave {len(values)} values, needs to be exactly {len(self.frame)} values')
		self.frame.loc[:,'value'] = values[:]
		for name in self._parameter_update_scheme:
			schemes = self._parameter_update_scheme[name]
			value = self.get_value(name)
			for member, location in schemes:
				self.__getattribute__(member)[location] = value

	@property
	def utility_ca(self):
		return LinearFunction() + self._utility_ca_function

	@utility_ca.setter
	def utility_ca(self, value):
		if isinstance(value, (LinearComponent, ParameterRef)):
			value = LinearFunction() + value
		if not isinstance(value, LinearFunction):
			raise TypeError('needs LinearFunction')
		self._utility_ca_function = value
		self._utility_ca_function.set_touch_callback(self.mangle)
		self.mangle()

	@property
	def utility_co(self):
		return self._utility_co_functions

	@utility_co.setter
	def utility_co(self, value):
		if not isinstance(value, (dict, DictOfLinearFunction)):
			raise TypeError('needs [dict] of {key:LinearFunction}')
		value = value.copy()
		for k in value.keys():
			if isinstance(value[k], (LinearComponent, ParameterRef)):
				value[k] = LinearFunction() + value[k]
			if not isinstance(value[k], LinearFunction):
				raise TypeError('needs dict of {key:[LinearFunction]}')
		self._utility_co_functions = DictOfLinearFunction(value, touch_callback=self.mangle, alts_validator=self._check_alternative)

	@property
	def graph(self):
		return self._graph

	def _set_graph(self, value):
		from .nesting.tree import NestingTree
		self._graph = NestingTree(value)
		self._graph.set_touch_callback(self.mangle)
		self.mangle()

	@graph.setter
	def graph(self, value):
		return self._set_graph(value)

	def _mnl_graph(self):
		from .nesting.tree import NestingTree
		root_id = 0
		while root_id in self._altindex:
			root_id += 1
		t = NestingTree(root_id=root_id)
		t.add_nodes(self._altindex)
		return t

