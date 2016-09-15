import keyword as _keyword
import re as _re
from . import core
from .core import LinearComponent, LinearFunction
from .exceptions import LarchError

from .util.pmath import category as Category
from .util.pmath import rename as Rename
from .util.naming import parenthize

_chi = u"\u03C7"
_beta = u"\u03B2"
_hahook = u"\u04FC"


_ParameterRef_repr_txt = 'P'
_DataRef_repr_txt = 'X'


class Role(type):
	def __getattr__(cls, key):
		try:
			return super().getattr(key)
		except AttributeError:
			return cls(key)

_data_description_catalog = {}


class DataRef(str, metaclass=Role):
	def __new__(cls, name, **kwarg):
		if hasattr(str, name):
			raise NameError("cannot create DataRef with the name of a str method ({})".format(name))
		if name in ('_descrip', 'descrip', '_role'):
			raise NameError("cannot create DataRef with the name ({})".format(name))
		try:
			raise TypeError("debugging") # The getattr here is not really needed?
			return getattr(cls,name)
		except:
			self = super().__new__(cls, name)
			self._role = 'data'
			return self
	def __init__(self, name, descrip=None):
		if descrip is not None:
			_data_description_catalog[self] = descrip
	@property
	def descrip(self):
		if self in _data_description_catalog:
			return _data_description_catalog[self]
		return None
	@descrip.setter
	def descrip(self,value):
		if value is not None:
			_data_description_catalog[self] = value
	def __repr__(self):
		s = super().__str__()
		if _re.match("[_A-Za-z][_a-zA-Z0-9]*$",s) and not _keyword.iskeyword(s):
			return "{}.{}".format(_DataRef_repr_txt,s)
		else:
			return "{}('{}')".format(_DataRef_repr_txt,s)
	def __add__(self, other):
		if isinstance(other, (ParameterRef,LinearComponent,LinearFunction)):
			return LinearComponent(data=str(self), param="CONSTANT") + other
		return DataRef("{}+{}".format(parenthize(self),parenthize(other, True)))
	def __radd__(self, other):
		if isinstance(other, (ParameterRef,LinearComponent,LinearFunction)):
			return other + LinearComponent(data=str(self), param="CONSTANT")
		return DataRef("{}+{}".format(parenthize(other),parenthize(self, True)))
	def __sub__(self, other):
		if isinstance(other, (ParameterRef,LinearComponent,LinearFunction)):
			return LinearComponent(data=str(self), param="CONSTANT") - other
		return DataRef("{}-{}".format(parenthize(self),parenthize(other, True)))
	def __rsub__(self, other):
		if isinstance(other, (ParameterRef,LinearComponent,LinearFunction)):
			return other - LinearComponent(data=str(self), param="CONSTANT")
		return DataRef("{}-{}".format(parenthize(other),parenthize(self, True)))
	def __mul__(self, other):
		if isinstance(other, LinearFunction):
			trial = LinearFunction()
			for component in other:
				trial += self * component
			return trial
		if isinstance(other, LinearComponent):
			return LinearComponent(data=str(self * other.data), param=str(other.param))
		if type(other) is ParameterRef:
			return LinearComponent(data=str(self), param=str(other))
		if type(other) is _param_negate:
			return LinearComponent(data=str(-self), param=str(other._orig))
		if other==0:
			return 0
		return DataRef("{}*{}".format(parenthize(self),parenthize(other, True)))
	def __rmul__(self, other):
		if isinstance(other, LinearFunction):
			raise NotImplementedError
		if isinstance(other, LinearComponent):
			return LinearComponent(data=str(other.data * self), param=str(other.param))
		if type(other) is ParameterRef:
			return LinearComponent(data=str(self), param=str(other))
		if type(other) is _param_negate:
			return LinearComponent(data=str(-self), param=str(other._orig))
		if other==0:
			return 0
		return DataRef("{}*{}".format(parenthize(other),parenthize(self, True)))
	def __truediv__(self, other):
		if isinstance(other, LinearFunction):
			raise NotImplementedError
		if isinstance(other, LinearComponent):
			raise NotImplementedError
		if isinstance(other, ParameterRef):
			raise NotImplementedError
		return DataRef("{}/{}".format(parenthize(self),parenthize(other, True)))
	def __rtruediv__(self, other):
		if isinstance(other, LinearFunction):
			raise NotImplementedError
		if isinstance(other, LinearComponent):
			raise NotImplementedError
		if isinstance(other, ParameterRef):
			raise NotImplementedError
		return DataRef("{}/{}".format(parenthize(other),parenthize(self, True)))
	def __neg__(self):
		return DataRef("-({})".format(self, True))
	def __pos__(self):
		return self
	def __eq__(self, other):
		if isinstance(other, DataRef):
			if repr(other) == repr(self):
				return True
		if isinstance(other, ParameterRef):
			return False
		if isinstance(other, str):
			if other == super().__str__():
				return True
		return False
	def __hash__(self):
		return hash(super().__str__())
	def eval(self, namespace=None, *, globals=None, **more_namespace):
		import numpy
		use_namespace = {'exp':numpy.exp, 'log':numpy.log, 'log1p':numpy.log1p, 'fabs':numpy.fabs, 'sqrt':numpy.sqrt,
						 'absolute':numpy.absolute, 'isnan':numpy.isnan, 'isfinite':numpy.isfinite, 'logaddexp':numpy.logaddexp, 'fmin':numpy.fmin, 'fmax':numpy.fmax}
		if namespace is not None:
			use_namespace.update(namespace)
		use_namespace.update(more_namespace)
		return eval(self, globals, use_namespace)




class ParameterRef(str, metaclass=Role):
	"""
	An abstract reference to a parameter, which may or may not be included in any given model.
	
	Parameters
	----------
	name : str
		The name of the parameter to reference.
	default : numeric or None
		When a targeted model does not include the referenced parameter, use 
		this value for :meth:`value` and :meth:`str`. If given as None, those
		methods will raise an exception.
	fmt : str or None
		The format string to use for :meth:`str`.
	"""
	
	def __new__(cls, name, default=None, fmt=None, default_value=None, format=None):
		if fmt is None and format is not None:
			fmt = format
		if fmt is None:
			fmt = "{}"
		if default is None and default_value is not None:
			default = default_value
		self = super().__new__(cls, name)
		self._fmt = fmt
		self._name = name
		self._role = 'parameter'
		self._default_value = default
		return self
	def __repr__(self):
		s = super().__str__()
		if _re.match("[_A-Za-z][_a-zA-Z0-9]*$",s) and not _keyword.iskeyword(s):
			return "{}.{}".format(_ParameterRef_repr_txt,s)
		else:
			return "{}('{}')".format(_ParameterRef_repr_txt,s)
	def __str__(self):
		return super().__str__()
	def value(self,m):
		"""
		The value of the parameter in a given model.
		
		Parameters
		----------
		m : Model
			The model from which to extract a parameter value.
		
		Raises
		------
		LarchError
			When the model does not contain a parameter with the same
			name as this ParameterRef, and the default_value for this
			ParameterRef is None.
		"""
		try:
			v = m.metaparameter(str(self)).value
		except (LarchError, KeyError):
			if self._default_value is not None:
				v = self._default_value
			else:
				raise
		return v
	def name(self, n):
		"""
		Set or re-set the name of the parameter referenced.
		
		Parameters
		----------
		n : str
			The new name of the parameter.
			
		Returns
		-------
		ParameterRef
			This method returns the self object, to facilitate method chaining.
		"""
		self._name = n
		return self
	def fmt(self,format):
		"""
		Set or re-set the format string for the parameter referenced.
		
		Parameters
		----------
		format : str
			The new format string. This will be used to format the parameter value when calling :meth:`str`.
			
		Returns
		-------
		ParameterRef
			This method returns the self object, to facilitate method chaining.
		"""
		if format is None or format == "":
			self._fmt = "{}"
		else:
			self._fmt = format
		try:
			self._fmt.format(0)
		except:
			raise LarchError("invalid formating string")
		return self
	def default_value(self,val):
		"""
		Set or re-set the default value.
		
		Parameters
		----------
		val : numeric or None
			The new default value. This will be used by :meth:`value` when 
			the parameter is not included in the targeted model. Set to None
			to raise an exception instead.
			
		Returns
		-------
		ParameterRef
			This method returns the self object, to facilitate method chaining.
		"""
		self._default_value = val
		return self
	def strf(self,m, fmt=None):
		"""
		Gives the :meth:`value` of the parameter in a given model as a string.
		
		The string is formated using the :meth:`fmt` string if given. If not
		given, Python's default string formatting is used.
		
		Parameters
		----------
		m : Model
			The model from which to extract a parameter value.
		fmt : str or None
			If not None, a format string which will be used, overriding any
			previous setting by a :meth:`fmt` command.  The override is 
			valid for this method call only, and does not change the 
			format for future calls.
		
		Raises
		------
		KeyError
			When the model does not contain a parameter with the same
			name as this ParameterRef, and the default_value for this
			ParameterRef is None.
		"""
		try:
			if fmt is not None:
				return fmt.format(self.value(m))
			else:
				return self._fmt.format(self.value(m))
		except LarchError as err:
			if "not in model" in str(err):
				return "NA"
			else:
				raise
		except KeyError:
			return "NA"
	def getname(self):
		return self._name
	def valid(self,m):
		"""
		Check if this ParameterRef would give a value for a given model.
		
		Parameters
		----------
		m : Model
			The model from which to extract a parameter value.
		
		Returns
		-------
		bool
			False if the value method would raise an exception, and True otherwise.
		"""
		if str(self) in m:
			return True
		if self._default_value is not None:
			return True
		return False
	def __add__(self, other):
		if other==0:
			return self
		if isinstance(other, (DataRef, LinearComponent, LinearFunction)):
			return LinearComponent(param=str(self),data="1") + other
		return _param_add(self,other)
	def __radd__(self, other):
		if other==0:
			return self
		if isinstance(other, (DataRef, LinearComponent, LinearFunction)):
			return other + LinearComponent(param=str(self),data="1")
		return _param_add(other,self)
	def __sub__(self, other):
		if other==0:
			return self
		if isinstance(other, (DataRef, LinearComponent, LinearFunction)):
			return LinearComponent(param=str(self),data="1") - other
		return _param_subtract(self,other)
	def __rsub__(self, other):
		if other==0:
			return self
		if isinstance(other, (DataRef, LinearComponent, LinearFunction)):
			return other - LinearComponent(param=str(self),data="1")
		return _param_subtract(other,self)
	def __mul__(self, other):
		if isinstance(other, (LinearComponent,LinearFunction)):
			raise TypeError("cannot do ParameterRef * LinearComponent")
		if type(other) is DataRef:
			return LinearComponent(data=str(other), param=str(self))
		return _param_multiply(self,other)
	def __rmul__(self, other):
		if isinstance(other, (LinearComponent,LinearFunction)):
			raise TypeError("cannot do LinearComponent * ParameterRef")
		if type(other) is DataRef:
			return LinearComponent(data=str(other), param=str(self))
		return _param_multiply(other,self)
	def __truediv__(self, other):
		if isinstance(other, DataRef):
			return self * (1/other)
		if isinstance(other, (LinearComponent,LinearFunction)):
			raise TypeError("cannot do ParameterRef / LinearComponent")
		return _param_divide(self,other)
	def __rtruediv__(self, other):
		if isinstance(other, DataRef):
			raise TypeError("cannot do DataRef / ParameterRef")
		if isinstance(other, (LinearComponent,LinearFunction)):
			raise TypeError("cannot do LinearComponent / ParameterRef")
		return _param_divide(other,self)
	def __neg__(self):
		return _param_negate(self)
	def __pos__(self):
		return self
	def __eq__(self, other):
		if isinstance(other, ParameterRef):
			if repr(other) == repr(self):
				return True
		if isinstance(other, DataRef):
			return False
		if isinstance(other, str):
			if other == super().__str__():
				return True
		return False
	def __hash__(self):
		return hash(repr(self))
	def _to_Linear(self):
		return LinearComponent(param=str(self),data="1")

class _param_math_binaryop(ParameterRef):
	def __new__(cls,left,right):
		self = super().__new__(cls, "")
		self._left = left
		self._right = right
		if isinstance(self._left, ParameterRef) and isinstance(self._right, ParameterRef):
			if left._fmt=="{}" and right._fmt!="{}":
				self._fmt = right._fmt
			else:
				self._fmt = left._fmt
		elif isinstance(self._left, ParameterRef):
			self._fmt = left._fmt
		elif isinstance(self._right, ParameterRef):
			self._fmt = right._fmt
		else:
			self._fmt = "{}"
		self._name = ""
		self._role = 'parameter_math'
		return self


class _param_add(_param_math_binaryop):
	def value(self,m):
		if isinstance(self._left, ParameterRef):
			x = self._left.value(m)
		else:
			x = self._left
		if isinstance(self._right, ParameterRef):
			x += self._right.value(m)
		else:
			x += self._right
		return x
	def valid(self,m):
		if isinstance(self._left, ParameterRef):
			x = self._left.valid(m)
		else:
			x = True
		if isinstance(self._right, ParameterRef):
			x &= self._right.valid(m)
		return x
	def __repr__(self):
		return "({} + {})".format(repr(self._left),repr(self._right))
	def _to_Linear(self):
		return self._left._to_Linear() + self._right._to_Linear()
	def __add__(self, other):
		if isinstance(other, (DataRef, LinearComponent, LinearFunction)):
			return self._to_Linear() + other
		return super().__add__(other)

class _param_subtract(_param_math_binaryop):
	def value(self,m):
		if isinstance(self._left, ParameterRef):
			x = self._left.value(m)
		else:
			x = self._left
		if isinstance(self._right, ParameterRef):
			x -= self._right.value(m)
		else:
			x -= self._right
		return x
	def valid(self,m):
		if isinstance(self._left, ParameterRef):
			x = self._left.valid(m)
		else:
			x = True
		if isinstance(self._right, ParameterRef):
			x &= self._right.valid(m)
		return x
	def __repr__(self):
		return "({} - {})".format(repr(self._left),repr(self._right))


class _param_multiply(_param_math_binaryop):
	def value(self,m):
		if isinstance(self._left, ParameterRef):
			x = self._left.value(m)
		else:
			x = self._left
		if isinstance(self._right, ParameterRef):
			x *= self._right.value(m)
		else:
			x *= self._right
		return x
	def valid(self,m):
		if isinstance(self._left, ParameterRef):
			x = self._left.valid(m)
		else:
			x = True
		if isinstance(self._right, ParameterRef):
			x &= self._right.valid(m)
		return x
	def __repr__(self):
		return "({} * {})".format(repr(self._left),repr(self._right))


class _param_divide(_param_math_binaryop):
	def value(self,m):
		if isinstance(self._left, ParameterRef):
			x = self._left.value(m)
		else:
			x = self._left
		try:
			if isinstance(self._right, ParameterRef):
				x /= self._right.value(m)
			else:
				x /= self._right
		except ZeroDivisionError:
			return float('NaN')
		return x
	def valid(self,m):
		if isinstance(self._left, ParameterRef):
			x = self._left.valid(m)
		else:
			x = True
		if isinstance(self._right, ParameterRef):
			x &= self._right.valid(m)
		return x
	def __repr__(self):
		return "({} / {})".format(repr(self._left),repr(self._right))


class _param_negate(ParameterRef):
	def __new__(cls,orig):
		self = super().__new__(cls, "")
		self._orig = orig
		self._fmt = orig._fmt
		self._name = "-({})".format(orig.getname())
		self._role = 'parameter_math'
		return self
	def value(self,m):
		if isinstance(self._orig, ParameterRef):
			return -self._orig.value(m)
		else:
			return -self._orig
	def valid(self,m):
		if isinstance(self._orig, ParameterRef):
			return self._orig.valid(m)
		else:
			return True
	def __repr__(self):
		return "-({})".format(repr(self._orig))


try:
	from numpy import log as _log, exp as _exp
except ImportError:
	from math import log as _log, exp as _exp

def log(x):
	if isinstance(x,DataRef):
		return DataRef("log({})".format(x))
	else:
		return _log(x)

def exp(x):
	if isinstance(x,DataRef):
		return DataRef("exp({})".format(x))
	else:
		return _exp(x)

class CombinedRef(metaclass=Role):
	def __new__(cls, s):
		return LinearComponent(DataRef(s), ParameterRef(s))



globals()[_ParameterRef_repr_txt] = ParameterRef
globals()[_DataRef_repr_txt] = DataRef
globals()[_ParameterRef_repr_txt+_DataRef_repr_txt] = CombinedRef

core.ParameterRef = ParameterRef
core.DataRef = DataRef

P = ParameterRef
X = DataRef
PX = CombinedRef

__all__ = [_ParameterRef_repr_txt, _DataRef_repr_txt, _ParameterRef_repr_txt+_DataRef_repr_txt]

if __name__=='__main__':



	from pprint import pprint
	pprint(globals())

	w1 = ParameterRef("B1")
	w2 = ParameterRef("B2")
	print(repr(w1))
	print(repr(w2))
	print(repr(w1+w2))
	print(repr(w1-w2))
	print(repr(w1*w2))
	print(repr(w1/w2))

	w1 = DataRef("COL1")
	w2 = DataRef("COL2")
	print(repr(w1))
	print(repr(w2))
	print(repr(w1+w2))
	print(repr(w1-w2))
	print(repr(w1*w2))
	print(repr(w1/w2))

