# cython: language_level=3, embedsignature=True

import re as _re
import keyword as _keyword
import numpy as _numpy
import sys

_ParameterRef_C_repr_txt = "P"
_DataRef_repr_txt = 'X'


cdef class UnicodeRef_C(unicode):

	pass

cdef class Ref_Gen:

	def __init__(self, kind):
		self._kind = kind

	def __getattr__(self, key):
		return self._kind(key)

	def __call__(self, arg):
		return self._kind(str(arg))

	def __getitem__(self, arg):
		return self._kind(str(arg))


P = Ref_Gen(ParameterRef_C)
X = Ref_Gen(DataRef_C)

cdef class ParameterRef_C(UnicodeRef_C):

	def __repr__(self):
		if _re.match("[_A-Za-z][_a-zA-Z0-9]*$", self) and not _keyword.iskeyword(self):
			return "{}.{}".format(_ParameterRef_C_repr_txt, self)
		else:
			return "{}('{}')".format(_ParameterRef_C_repr_txt, self)

	def __eq__(self, other):
		if isinstance(other, str) and not isinstance(other, DataRef_C):
			if str(self) == str(other):
				return True
		return False

	def __ne__(self, other):
		return not self.__eq__(other)

	def __hash__(self):
		return super().__hash__()

	def __pos__(self):
		return self

	def __add__(self, other):
		if other == 0:
			return self
		if isinstance(other, (ParameterRef_C, LinearComponent_C, LinearFunction_C)):
			return LinearComponent_C(param=str(self), data="1") + other
		# return _param_add(self, other)
		raise NotImplementedError(f"<{self.__class__.__name__}> + <{other.__class__.__name__}>")

	def __radd__(self, other):
		if other == 0:
			return self
		if isinstance(other, (ParameterRef_C, LinearComponent_C, LinearFunction_C)):
			return other + LinearComponent_C(param=str(self), data="1")
		# return _param_add(other, self)
		raise NotImplementedError(f"<{other.__class__.__name__}> + <{self.__class__.__name__}>")

	def __sub__(self, other):
		if other == 0:
			return self
		if isinstance(other, (ParameterRef_C, LinearComponent_C, LinearFunction_C)):
			return LinearComponent_C(param=str(self), data="1") - other
		#return _param_subtract(self, other)
		raise NotImplementedError(f"<{self.__class__.__name__}> - <{other.__class__.__name__}>")

	def __rsub__(self, other):
		if other == 0:
			return self
		if isinstance(other, (ParameterRef_C, LinearComponent_C, LinearFunction_C)):
			return other - LinearComponent_C(param=str(self), data="1")
		#return _param_subtract(other, self)
		raise NotImplementedError(f"<{other.__class__.__name__}> - <{self.__class__.__name__}>")

	def __mul__(self, DataRef_C other):
		return LinearComponent_C(str(self), str(other))


cdef class DataRef_C(UnicodeRef_C):

	def __repr__(self):
		if _re.match("[_A-Za-z][_a-zA-Z0-9]*$", self) and not _keyword.iskeyword(self):
			return "{}.{}".format(_DataRef_repr_txt, self)
		else:
			return "{}('{}')".format(_DataRef_repr_txt, self)

	def __eq__(self, other):
		if isinstance(other, str) and not isinstance(other, ParameterRef_C):
			if str(self) == str(other):
				return True
		return False

	def __ne__(self, other):
		return not self.__eq__(other)

	def __hash__(self):
		return super().__hash__()

	def __pos__(self):
		return self

	def __mul__(self, other):
		if isinstance(other, ParameterRef_C):
			return LinearComponent_C(str(other), str(self))
		raise NotImplementedError(f"<{self.__class__.__name__}> * <{other.__class__.__name__}>")



cdef class LinearComponent_C:

	def __init__(self, unicode param, unicode data='1', l4_float_t scale=1):
		self._param = param
		self._data = data
		self._scale = scale

	@property
	def param(self):
		return ParameterRef_C(self._param)

	@property
	def data(self):
		return DataRef_C(self._data)

	@property
	def scale(self):
		return self._scale

	def __pos__(self):
		return self

	def __repr__(self):
		try:
			if self.scale == 1.0:
				try:
					data_is_1 = (float(self.data) == 1)
				except:
					data_is_1 = False
				if data_is_1:
					return f"({self.param!r})"
				else:
					return f"({self.param!r} * {self.data!r})"
			return f"({self.param!r} * {self.scale} * {self.data!r})"
		except AttributeError:
			return f"<{self.__class__.__name__} {id(self)} with error>"

	def __add__(self, other):
		if other == ():
			return self
		elif isinstance(other, LinearComponent_C):
			return LinearFunction_C([self, other])
		elif isinstance(other, LinearFunction_C):
			return LinearFunction_C([self, *other])
		elif other == 0:
			return self
		elif isinstance(other, ParameterRef_C):
			return LinearFunction_C([self, LinearComponent_C(param=other)])
		else:
			raise NotImplementedError()

	def __radd__(self, other):
		if other == 0:
			return self
		return other.__add__(self)

	def __mul__(self, other):
		if isinstance(other, (int, float, )):
			return self.__class__(
				param=str(self.param),
				data=str(self.data),
				scale=self.scale * other,
			)
		elif isinstance(other, (DataRef_C, )):
			return self.__class__(
				param=str(self.param),
				data=str(self.data * other),
				scale=self.scale,
			)
		else:
			raise TypeError(f'unsupported operand type(s) for {self.__class__.__name__}: {type(other)}')

	def __rmul__(self, other):
		if isinstance(other, (int, float, )):
			return self.__class__(
				param=str(self.param),
				data=str(self.data),
				scale=self.scale * other,
			)
		elif isinstance(other, (DataRef_C, )):
			return self.__class__(
				param=str(self.param),
				data=str(self.data * other),
				scale=self.scale,
			)
		else:
			raise TypeError(f'unsupported operand type(s) for {self.__class__.__name__}: {type(other)}')

	def __iter__(self):
		return iter(LinearFunction_C([self]))

	def __eq__(self, other):
		if isinstance(other, LinearFunction_C) and len(other) == 1:
			other = other[0]
		if not isinstance(other, LinearComponent_C):
			return False
		if self.param != other.param:
			return False
		if self.data != other.data:
			return False
		if self.scale != other.scale:
			return False
		return True

	def __xml__(self, exponentiate_parameter=False, resolve_parameters=None):
		from xmle import Elem
		x = Elem('div')
		# x << tooltipped_style()

		if resolve_parameters is not None:
			if exponentiate_parameter:
				plabel = resolve_parameters.pvalue(self.param, default_value="This is a Parameter")
				if not isinstance(plabel, str):
					plabel = "exp({0:.3g})={1:.3g}".format(plabel, _numpy.exp(plabel))
			else:
				plabel = "{0:.3g}".format(resolve_parameters.pvalue(self.param, default_value="This is a Parameter"))
		else:
			plabel = "This is a Parameter"


		data_tail = " * "
		try:
			if float(self.data)==1:
				data_tail = ""
		except:
			pass

		if self.scale == 1.0:
			if exponentiate_parameter:
				x.elem('span', tail="exp(")
				a_p = x.elem('div', attrib={'class':'tooltipped'}, text=repr(self.param), tail=")"+data_tail)
				a_p.elem('span', attrib={'class':'tooltiptext'}, text=plabel)
			else:
				a_p = x.elem('div', attrib={'class':'tooltipped'}, text=repr(self.param), tail=data_tail)
				a_p.elem('span', attrib={'class':'tooltiptext'}, text=plabel)
		else:
			if exponentiate_parameter:
				x.elem('span', tail="exp(")
				a_p = x.elem('div', attrib={'class':'tooltipped'}, text=repr(self.param), tail=f" * {self.scale}){data_tail}")
				a_p.elem('span', attrib={'class':'tooltiptext'}, text=plabel)
			else:
				a_p = x.elem('div', attrib={'class':'tooltipped'}, text=repr(self.param), tail=f" * {self.scale}{data_tail}")
				a_p.elem('span', attrib={'class':'tooltiptext'}, text=plabel)
		if data_tail == " * ":
			a_x = x.elem('div', attrib={'class':'tooltipped'}, text=repr(self.data))
			a_x.elem('span', attrib={'class':'tooltiptext'}, text="This is Data")
		return x

	def _repr_html_(self):
		return self.__xml__().tostring()

	def _evaluate(self, p_getter, x_namespace=None, **kwargs):
		return self.scale * p_getter(str(self.param)) * self.data.eval(namespace=x_namespace, **kwargs)

	def __copy__(self):
		return self.__class__(
			param=str(self.param),
			data=str(self._data),
			scale=self.scale,
		)



def _try_mangle(instance):
	try:
		instance.mangle()
	except AttributeError as err:
		print(f"No Mangle L: {err}")

def _try_mangle_h(instance_holder):
	try:
		instance_holder._instance.mangle()
	except AttributeError as err:
		print(f"No Mangle L2: {err}")


cdef class LinearFunction_C:

	def __init__(self, init=None):
		self._func = list()
		self._instance = None
		if init is not None:
			for i in init:
				if isinstance(i, LinearComponent_C):
					self._func.append(i)
				else:
					raise TypeError(f'members of {self.__class__.__name__} must be LinearComponent_C')

	def set_instance(self, instance):
		self._instance = instance

	def __fresh(self, instance):
		newself = LinearFunction_C()
		newself._instance = instance
		setattr(instance, self.private_name, newself)
		return newself

	def __get__(self, instance, owner):

		cdef LinearFunction_C newself

		if instance is None:
			return self
		try:
			newself = getattr(instance, self.private_name)
		except AttributeError:
			newself = self.__fresh(instance)
		if newself is None:
			newself = self.__fresh(instance)
		return newself

	def __set__(self, instance, values):

		cdef LinearFunction_C newself

		try:
			newself = getattr(instance, self.private_name)
		except AttributeError:
			newself = self.__fresh(instance)
		if newself is None:
			newself = self.__fresh(instance)
		newself.__init__(values)
		_try_mangle_h(newself)

	def __delete__(self, instance):

		cdef LinearFunction_C newself

		try:
			newself = getattr(instance, self.private_name)
		except AttributeError:
			newself = self.__fresh(instance)
		newself.__init__()
		newself._instance = instance
		_try_mangle_h(newself)

	def __set_name__(self, owner, name):
		self.name = name
		self.private_name = "_"+name

	def __getitem__(self, item):
		return self._func[item]

	def __setitem__(self, key, LinearComponent_C value not None):
		self._func[key] = value
		_try_mangle(self._instance)

	def __delitem__(self, key):
		del self._func[key]
		_try_mangle(self._instance)

	def __len__(self):
		return len(self._func)

	def insert(self, index, LinearComponent_C value not None):
		self._func.insert(index, value)
		_try_mangle(self._instance)

	def append(self, LinearComponent_C value not None):
		self._func.append(value)
		_try_mangle(self._instance)

	def extend(self, values):
		for v in values:
			if not isinstance(v, LinearComponent_C):
				raise TypeError("cannot add type {} to LinearFunction".format(type(v)))
		self._func.extend(values)
		_try_mangle(self._instance)

	def __add__(self, other):
		if other == ():
			return self
		if isinstance(other, LinearFunction_C):
			return self.__class__(*self._func, *other._func)
		if isinstance(other, ParameterRef_C):
			other = LinearComponent_C(param=other)
		if isinstance(other, LinearComponent_C):
			result = self.__class__(self)
			result.append(other)
			return result
		raise TypeError("cannot add type {} to LinearFunction".format(type(other)))

	def __iadd__(self, other):
		if isinstance(other, ParameterRef_C):
			other = LinearComponent_C(param=other)
		if other == ():
			return self
		elif isinstance(other, LinearFunction_C):
			self._func.extend(other)
			_try_mangle(self._instance)
		elif isinstance(other, LinearComponent_C):
			self.append(other)
		else:
			raise TypeError("cannot add type {} to LinearFunction".format(type(other)))
		return self

	def __radd__(self, other):
		return LinearFunction_C([*other, *self])

	def __pos__(self):
		return self

	def __mul__(self, other):
		trial = LinearFunction_C()
		for component in self:
			trial.append(component * other)
		return trial

	def __rmul__(self, other):
		trial = LinearFunction_C()
		for component in self:
			trial += other * component
		return trial

	def __contains__(self, val):
		if isinstance(val, ParameterRef_C):
			for i in self:
				if i.param == val:
					return True
			return False
		if isinstance(val, DataRef_C):
			for i in self:
				if i.data == val:
					return True
			return False
		raise TypeError("the searched for content must be of type ParameterRef_C or DataRef")

	def _index_of(self, val):
		if isinstance(val, ParameterRef_C):
			for n, i in enumerate(self):
				if i.param == val:
					return n
			raise KeyError('ParameterRef_C not found')
		if isinstance(val, DataRef_C):
			for n, i in enumerate(self):
				if i.data == val:
					return n
			raise KeyError('DataRef not found')
		raise TypeError("the searched for content must be of type ParameterRef_C or DataRef")

	def reformat_param(self, container=None, pattern=None, repl=None, **kwargs):
		"""
		Transform all the parameters in the LinearFunction.

		Parameters
		----------
		container : str
			A format string, into which the previous parameters are formatted.
			Use this to append things to the parameter names.
		pattern : str
		repl : str
			Passed to `re.sub` with each existing parameter as the base string
			to be searched.

		Examples
		--------
		>>> from larch.roles import P,X
		>>> f = P.InVehTime * X.IVTT + P.OutOfVehTime * X.OVTT
		>>> f1 = f.reformat_param('{}_Suffix')
		>>> str(f1)
		'(P.InVehTime_Suffix * X.IVTT) + (P.OutOfVehTime_Suffix * X.OVTT)'
		>>> f2 = f.reformat_param(pattern='(Veh)', repl='Vehicle')
		>>> str(f2)
		'(P.InVehicleTime * X.IVTT) + (P.OutOfVehicleTime * X.OVTT)'

		"""
		import re
		r = self.__class__()
		for i in self:
			if pattern is None:
				param = i.param
			else:
				if repl is None:
					raise TypeError('must give repl with pattern')
				param = re.sub(pattern, repl, i.param, **kwargs)
			if container is None:
				container = '{}'
			r += LinearComponent_C(data=i.data, param=container.format(param), scale=i.scale)
		return r

	def reformat_data(self, container=None, pattern=None, repl=None, **kwargs):
		"""
		Transform all the data in the LinearFunction.

		Parameters
		----------
		container : str
			A format string, into which the previous data strings are formatted.
			Use this to apply common global transforms to the data.
		pattern : str
		repl : str
			Passed to `re.sub` with each existing data string as the base string
			to be searched.
		"""
		import re
		r = self.__class__()
		for i in self:
			if pattern is None:
				data = i.data
			else:
				if repl is None:
					raise TypeError('must give repl with pattern')
				data = re.sub(pattern, repl, i.data, **kwargs)
			if container is None:
				container = '{}'
			r += LinearComponent_C(data=container.format(data), param=i.param, scale=i.scale)
		return r

	def __code__(self):
		return " + ".join(f"({repr(i)})" for i in self)

	def __eq__(self, other):
		if not isinstance(other, LinearFunction_C):
			return False
		if len(self) != len(other):
			return False
		for i, j in zip(self, other):
			if i != j: return False
		return True

	def __repr__(self):
		if len(self):
			result = " + ".join(repr(i) for i in self)
			if len(result)<80:
				return result
			else:
				return "  "+result.replace(" + ","\n+ ")
		return f"<Empty {self.__class__.__name__}>"

	def __xml__(self, linebreaks=False, lineprefix="", exponentiate_parameters=False, resolve_parameters=None):
		from xmle import Elem
		x = Elem('div', attrib={'class':'LinearFunc'})
		for n,i in enumerate(self):
			ix_ = i.__xml__(exponentiate_parameter=exponentiate_parameters, resolve_parameters=resolve_parameters).getchildren()
			if linebreaks:
				if n>0 or lineprefix:
					ix_.insert(0,Elem('br', tail = lineprefix+" + "))
			else:
				if n < len(self)-1:
					if ix_[-1].tail is None:
						ix_[-1].tail = " + "
					else:
						ix_[-1].tail += " + "
			for ii in ix_:
				x << ii
		if len(self) == 0:
			x << Elem('span', text=repr(self))
		return x

	def _repr_html_(self):
		return self.__xml__().tostring()

	def data(self, cls=None):
		if cls is None:
			return [_.data for _ in self]
		else:
			return [cls(_.data) for _ in self]

	def _evaluate(self, p_getter, x_namespace=None, **more_x_namespace):
		if hasattr(p_getter,'pvalue') and callable(p_getter.pvalue):
			p_getter = p_getter.pvalue
		return sum(j._evaluate(p_getter, x_namespace=x_namespace, **more_x_namespace) for j in self)

	def copy(self):
		result = self.__class__(self)
		return result

	def __deepcopy__(self, memodict):
		result = self.__class__()
		import copy
		for i in self:
			result.append(copy.deepcopy(i, memodict))
		return result

	def _linear_plot_2d_data(self, p_getter, x_name, x_min, x_max, n_points=100, **other_namespace):
		import numpy
		if hasattr(self, 'plotting_namespace') and len(other_namespace)==0:
			other_namespace = self.plotting_namespace
		x = numpy.linspace(x_min, x_max, n_points)
		y = self._evaluate(p_getter, {x_name:x}, **other_namespace)
		return x,y

	def linear_plot_2d(self, p_getter, x_name, x_min, x_max, n_points=100, *, xlabel=None, svg=True, header=None, **other_namespace):

		# Delayed evaluation mode...
		if p_getter is None:
			return lambda x: self.linear_plot_2d(x, x_name, x_min, x_max, n_points=n_points, xlabel=xlabel, svg=svg, header=header, **other_namespace)

		# Active evaluation mode...
		from matplotlib import pyplot as plt
		plt.clf()
		x,y = self._linear_plot_2d_data(p_getter, x_name, x_min, x_max, n_points, **other_namespace)
		if hasattr(self, 'plotting_label'):
			plt.plot(x, y, label=self.plotting_label)
		else:
			plt.plot(x, y)
		if xlabel is None:
			plt.xlabel(x_name)
		else:
			plt.xlabel(xlabel)
		plt.tight_layout(pad=0.5)
		from ..util.plotting import plot_as_svg_xhtml
		if svg is True:
			svg = {}
		if svg or svg == {}:
			if header is not None:
				svg['header'] = header
			return plot_as_svg_xhtml(plt, **svg)
		else:
			plt.show()

	def _inplot_linear_plot_2d(self, plt, p_getter, x_name, x_min, x_max, n_points=100, *, xlabel=None, svg=True, header=None, **other_namespace):

		# Delayed evaluation mode...
		if p_getter is None:
			return lambda x: self._inplot_linear_plot_2d(plt, x, x_name, x_min, x_max, n_points=n_points, xlabel=xlabel, svg=svg, header=header, **other_namespace)

		# Active evaluation mode...
		x,y = self._linear_plot_2d_data(p_getter, x_name, x_min, x_max, n_points, **other_namespace)
		if hasattr(self, 'plotting_label'):
			plt.plot(x, y, label=self.plotting_label)
		else:
			plt.plot(x,y)

	def total_ordering_increasing(self):
		from toolz.itertoolz import sliding_window
		snaps = []
		for windows_size in range(2, len(self)-1):
			for sub_p in sliding_window(windows_size, self):
				snaps.append(sub_p[0].param.lessthan(sub_p[-1].param))

		return snaps

	


cdef class DictOfLinearFunction_C:

	def __init__(self, mapping=None, alts_validator=None, **kwargs):
		self._map = {}
		if mapping is None:
			mapping = {}
		for k,v in mapping.items():
			self._map[k] = LinearFunction_C(v)
		for k,v in kwargs.items():
			try:
				self._map[k] = LinearFunction_C(v)
			except:
				print(v)
				print(type(v))
				raise

		self._alts_validator = alts_validator
		self._instance = None

	def __fresh(self, instance):
		cdef DictOfLinearFunction_C newself
		newself = DictOfLinearFunction_C()
		newself._instance = instance
		setattr(instance, self.private_name, newself)
		return newself

	def __get__(self, instance, owner):

		cdef DictOfLinearFunction_C newself

		if instance is None:
			return self
		try:
			newself = getattr(instance, self.private_name)
		except AttributeError:
			newself = self.__fresh(instance)
		if newself is None:
			newself = self.__fresh(instance)
		return newself

	def __set__(self, instance, values):

		cdef DictOfLinearFunction_C newself

		try:
			newself = getattr(instance, self.private_name)
		except AttributeError:
			newself = self.__fresh(instance)
		if newself is None:
			newself = self.__fresh(instance)
		newself.__init__(values)
		_try_mangle_h(newself)

	def __delete__(self, instance):
		self.__set__(instance, None)

	def __set_name__(self, owner, name):
		self.name = name
		self.private_name = "_"+name

	def set_alts_validator(self, av):
		self._alts_validator = av

	def __getitem__(self, k):
		try:
			v = self._map[k]
			v.set_instance(self._instance)
			return v
		except KeyError:
			if self._alts_validator is None or self._alts_validator(k):
				v = self._map[k] = LinearFunction_C()
				v.set_instance(self._instance)
				return v
			else:
				raise

	def __setitem__(self, k, v):
		if isinstance(v, ParameterRef_C):
			v = v * DataRef_C('1')
		if isinstance(v, int) and v==0:
			v = LinearFunction_C()
		elif isinstance(v, LinearComponent_C):
			v = LinearFunction_C([v])
		elif isinstance(v, list):
			v = LinearFunction_C(v)
		elif not isinstance(v, LinearFunction_C):
			raise TypeError(f"only accepts LinearFunction_C values, not {type(v)}")
		if "with error" in repr(v):
			raise ValueError("found error here")
		v.set_instance(self._instance)
		self._map[k] = v
		_try_mangle(self._instance)

	def __delitem__(self, key):
		del self._map[key]
		_try_mangle(self._instance)

	def __iter__(self):
		return iter(self._map.keys())

	def __len__(self):
		return len(self._map)

	def keys(self):
		return self._map.keys()

	def items(self):
		return self._map.items()

	def values(self):
		return self._map.values()

	def copy(self):
		return type(self)(self)

	def __repr__(self):
		return '{0}({1})'.format(type(self).__name__, repr(self._map))

	def __xml__(self):
		from pprint import pformat
		from xmle import Elem
		x = Elem('div')
		t = x.elem('table', style="margin-top:1px;")
		t.elem('caption', text=f"<larch.{self.__class__.__name__}>", style="caption-side:top;text-align:left;font-family:Roboto;font-weight:700;font-style:normal;font-size:100%;padding:0px;")
		if len(self):
			tr = t.elem('tr')
			tr.elem('th', text="alt")
			tr.elem('th', text='formula')
			for k,v in self._map.items():
				tr = t.elem('tr')
				tr.elem('td', text=str(k))
				try:
					v_ = v.__xml__()
				except AttributeError:
					tr.elem('td', text=str(v))
				else:
					tr.elem('td') << v_
		else:
			tr = t.elem('tr')
			tr.elem('td', text="<empty>")
		return x

	def _repr_html_(self):
		return self.__xml__().tostring()


cdef class GenericContainerCy:

	def __init__(self, arg):
		self.ident = arg

	def mangle(self):
		print(":MANGLE:",self.ident)

	def stats(self):
		print("LF:",self.lf)
		print("DLF:",self.dlf)
		print("_LF:",self._lf)
		print("_DLF:",self._dlf)


class GenericContainerPy(GenericContainerCy):

	lf = LinearFunction_C()
	dlf = DictOfLinearFunction_C()

	def __init__(self, arg):
		super().__init__(arg)

	def stats(self):
		print("LF:",self.lf)
		print("DLF:",self.dlf)
		super().stats()

