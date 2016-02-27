

import numpy

class ArrayError(Exception): pass

class Array(numpy.ndarray):

	def __new__(subtype, shape=[0,], initialize=True, vars=None, *args, **kwargs):
		if 'dtype' not in kwargs:
			kwargs['dtype']=numpy.float64
		if isinstance(shape, numpy.ndarray):
			if vars is not None:
				if (len(shape.shape)==1 and len(vars)!=1):
					raise ArrayError("Array is vector but length of name vector is not 1 (it is {})".format(len(vars)))
				if (2<=len(shape.shape)<=3) and len(vars) != shape.shape[-1]:
					raise ArrayError("Array has {} vars but names {} vars".format(shape.shape[-1], len(vars)))
			obj = numpy.asarray(shape).view(Array)
			obj.vars = vars
			return obj
		obj = numpy.ndarray.__new__(subtype, shape, *args, **kwargs)
		# initialize stuff here...
		if initialize:
			obj.initialize()
		if vars is not None:
			if len(obj.shape) == 1 and len(vars)!=1:
				raise ArrayError("Array is vector but length of name vector is not 1")
			if len(obj.shape) > 1 and len(vars)!=obj.shape[-1]:
				raise ArrayError("Array has {} vars but names {} vars".format(obj.shape[-1], len(vars)))
			obj.vars = vars
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







#from scipy.linalg.blas import dgemv, dgemm





# for possible future use...
#class DataArrayBundle:
#
#	def __init__(self, model=None, name="Utility"):
#		self.idco = None
#		self.idca = None
#		self.idce = None
#		self.name = name
#		self.model = model
#		self._nameset = tuple("{}{}".format(name,x) for x in ("CA","CO"))
#
#	def _get_needed_data(self, dataprovider):
#		big_needs = self.model.needs()
#		local_needs = {n: big_needs[n] for n in self._nameset if n in big_needs}
#		bucket = dataprovider.provision(local_needs)
#		if self._nameset[0] in bucket:
#			self.idca = bucket[self._nameset[0]]
#		if self._nameset[1] in bucket:
#			self.idco = bucket[self._nameset[1]]
#
#	def linear_result(self, outarray, initializer = 0.0):
#		outarrayshape = outarray.shape
#		while len(outarrayshape)<2:
#			outarrayshape = outarrayshape + (1,)
#		if self.idca is not None:
#			outarray.shape = (outarrayshape[0]*outarrayshape[1],)
#			self_idca_shape = self.idca.shape
#			self.idca.shape = (self.idca.shape[0]*self.idca.shape[1], *self.idca.shape[2:])
#			assert( outarray.shape == self.idca.shape[:1] )
#			zz = dgemv(1.0,self.idca, self.model.Coef(self._nameset[0]), initializer,outarray, overwrite_y=1)
#			print("AA")
#			print("explicit result")
#			print(zz)
#			print("input result")
#			print(outarray)
#			initializer = 1.0
#			self.idca.shape = self_idca_shape
#			outarray.shape = self.idca.shape[:2]
#		if self.idco is not None:
#			coef_co = self.model.Coef(self._nameset[1])
#			print(outarray.shape,"outarray.shape")
#			print(( self.idco.shape[0], coef_co.shape[1]),"( self.idco.shape[0], coef_co.shape[1])")
#			assert( outarray.shape == ( self.idco.shape[0], coef_co.shape[1]) )
#			zz = dgemm(1.0,self.idco, coef_co, initializer,outarray, overwrite_c=1)
#			print("BB")
#			print("explicit result")
#			print(zz)
#			print("input result")
#			print(outarray)
#			initializer = 1.0
#		return outarray



