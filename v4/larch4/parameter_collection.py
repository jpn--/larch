import numpy
import pandas
from larch.core import LinearFunction, LinearComponent
from larch.roles import ParameterRef
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
				 utility_co=None):
		self._altindex = pandas.Index( altindex )

		self._utility_co_functions = SignalDict(self.mangle,self.mangle,self.mangle,utility_co or {})
		self._utility_ca_function  = utility_ca or LinearFunction()

		self.frame = _empty_parameter_frame(names)
		self._scan_utility_ensure_names()
		self._parameter_update_scheme = {}
		self.mangle()

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
			self._scan_utility_ensure_names()
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
	def utility_ca_vars(self):
		self.unmangle()
		return self._u_ca_varindex

	@property
	def utility_co_vars(self):
		self.unmangle()
		return self._u_co_varindex

	def _initialize_derived_util_coef_arrays(self):

		self._coef_utility_co = numpy.zeros( [len(self._u_co_varindex), len(self._altindex)], dtype=numpy.float64)
		self._coef_utility_ca = numpy.zeros( len(self._u_ca_varindex), dtype=numpy.float64)

		self._parameter_update_scheme = {}

		for n,component in enumerate(self._utility_ca_function):
			if str(component.param) not in self._parameter_update_scheme:
				self._parameter_update_scheme[str(component.param)] = []
			self._parameter_update_scheme[str(component.param)].append( ('_coef_utility_ca', (n,)) )

		for altcode, linear_function in self._utility_co_functions.items():
			for component in linear_function:
				if str(component.param) not in self._parameter_update_scheme:
					self._parameter_update_scheme[str(component.param)] = []
				self._parameter_update_scheme[str(component.param)].append( ('_coef_utility_co', (self._u_co_varindex.get_loc(str(component.data)), self._altindex.get_loc(altcode) )) )

		self._refresh_derived_arrays()

	def _refresh_derived_arrays(self):
		for name in self._parameter_update_scheme:
			value = self.frame.loc[name,'value']
			schemes = self._parameter_update_scheme[name]
			for member, location in schemes:
				self.__getattribute__(member)[location] = value

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
		self.mangle()

	@property
	def utility_co(self):
		return self._utility_co_functions

	@utility_co.setter
	def utility_co(self, value):
		if not isinstance(value, (dict, SignalDict)):
			raise TypeError('needs [dict] of {key:LinearFunction}')
		value = value.copy()
		for k in value.keys():
			if isinstance(value[k], (LinearComponent, ParameterRef)):
				value[k] = LinearFunction() + value[k]
			if not isinstance(value[k], LinearFunction):
				raise TypeError('needs dict of {key:[LinearFunction]}')
		self._utility_co_functions = SignalDict(self.mangle, self.mangle, self.mangle, value or {})




