

import numpy

class ArrayError(Exception): pass

class Array(numpy.ndarray):

	def __new__(subtype, shape=[0,], initialize=True, *args, **kwargs):
		if 'dtype' not in kwargs:
			kwargs['dtype']=numpy.float64
		obj = numpy.ndarray.__new__(subtype, shape, *args, **kwargs)
		# initialize stuff here...
		if initialize:
			obj.initialize()
		return obj

	def __array_finalize__(self, obj):
		if obj is None:
			# Explicit constructor
			return
		if type(obj) is Array:
			# View or template of self
			return

	def resize_like(self, other):
		if other.size3()==1:
			if other.size2()==1:
				self.resize([other.size1(),], refcheck=False)
			else:
				self.resize([other.size1(),other.size2(),], refcheck=False)
		else:
			self.resize([other.size1(), other.size2(), other.size3()], refcheck=False)

	def isNaN(self):
		return numpy.all(numpy.isnan(self))

	def initialize(self):
		from .core import ndarray_init
		ndarray_init(self)

	def exp(self):
		from .core import ndarray_exp
		ndarray_exp(self)

	def log(self):
		from .core import ndarray_log
		ndarray_log(self)

def ArrayBool(*args, **kwargs):
	if 'dtype' not in kwargs:
		kwargs['dtype']=numpy.bool
	return Array(*args, **kwargs)

def pack(a, subclass=Array):
	return numpy.ascontiguousarray(a).view(subclass)


class SymmetricArray(Array):

	def __new__(subtype, shape=[0,], initialize=True, *args, **kwargs):
		if len(shape)>2: raise ArrayError("shape of SymmetricArray must be two dimensions, asking %i"%len(shape))
		if len(shape)==1: shape = [shape[0], shape[0]]
		if shape[0]!=shape[1]: raise ArrayError("shape of SymmetricArray must be square")
		if 'dtype' not in kwargs:
			kwargs['dtype']=numpy.float64
		obj = numpy.ndarray.__new__(subtype, shape, *args, **kwargs)
		# initialize stuff here...
		if initialize:
			obj.initialize()
		return obj

	def __setitem__(self, ij, value):
		numpy.ndarray.__setitem__(self, (ij[0], ij[1]), value)
		numpy.ndarray.__setitem__(self, (ij[1], ij[0]), value)

	def __array_finalize__(self, obj):
		#if len(self.shape)!=2: raise ArrayError("shape of SymmetricArray must be two dimensions, this array has %i dimensions"%len(self.shape))
		#if self.shape[0]!=self.shape[1]: raise ArrayError("shape of SymmetricArray must be square")
		#if len(self.shape)==2 and self.shape[0]==self.shape[1]:
		#	self.use_upper_triangle()
		if obj is None:
			# Explicit constructor
			return
		if type(obj) is Array:
			# View or template of self
			return

	def inv(self):
		from .linalg import general_inverse
		return general_inverse(self)

	def flats(self):
		i = pack(self)
		u,s,v=numpy.linalg.svd(i)
		sx = numpy.round(s,5)
		return v.T[:,sx==0]

	def sharps(self):
		i = pack(self)
		u,s,v=numpy.linalg.svd(i)
		return v.T[:,s>1e5]

	def use_upper_triangle(self):
		from .core import SymmetricArray_use_upper_triangle
		SymmetricArray_use_upper_triangle(self)




class ldarray(Array):
	def __new__(subtype, shape=[0,], initialize=True, vars=[], *args, **kwargs):
		if isinstance(shape, numpy.ndarray):
			if (2<=len(shape.shape)<=3):
				if len(vars) != shape.shape[-1]:
					raise ArrayError("vars must give names for all vars, have %i vars but name %i vars"%(shape.shape[-1],len(vars)))
				obj = numpy.asarray(shape).view(ldarray)
				obj.vars = vars
				return obj
		if not (2<=len(shape)<=3): raise ArrayError("shape of ldarray must be 2 or 3 dimensions, asking %i"%len(shape))
		if len(vars) != shape[-1]: raise ArrayError("vars must give names for all vars, have %i vars but name %i vars"%(shape[-1],len(vars)))
		if 'dtype' not in kwargs:
			kwargs['dtype']=numpy.float64
		obj = numpy.ndarray.__new__(subtype, shape, *args, **kwargs)
		# initialize stuff here...
		if initialize:
			obj.initialize()
		obj.vars = vars
		return obj

	def __array_finalize__(self, obj):
		if obj is None:
			# Explicit constructor
			return
		if type(obj) is ldarray:
			# View or template of self
			# set vars here?
			self.vars = None
			# self.vars = getattr(obj, 'vars', [])
			# if len(self.shape) != len(obj.shape):
			#	raise ArrayError("ldarray does not support slicing that changes dimensions (%i to %i)"%(len(obj.shape), len(self.shape),))
			# if self.shape[-1] != obj.shape[-1]:
			#	raise ArrayError("ldarray does not support slicing that changes number of vars (%i to %i)"%(obj.shape[-1], self.shape[-1],))
			return
	
