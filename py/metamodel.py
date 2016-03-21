
from .core import autoindex_string
from . import Model
import numpy, itertools
from .linalg import general_inverse

class MetaModel(Model):

	def __init__(self):
		super().__init__()
		self.sub_model = {}
		self.sub_weight = {}
		self.total_weight = 0
		self.sub_ncases = {}
		self.total_ncases = 0

	@property
	def scale(self):
		return (1.0*self.total_weight/self.total_ncases)

	def tearDown(self):
		pass

	def setUp(self, *args):
		pass


	def loglike_null(self):
		return self.loglike(self.parameter_null_values_array)



	def loglike(self, *args, cached=False):
		fun = 0
		if len(args)>0:
			self.parameter_values(args[0])
		meta_parameter_values = self.parameter_values()
		for key,m in self.sub_model.items():
			for value, name in zip(meta_parameter_values, self.parameter_names()):
				if name in m:
					m.parameter(name).value = value
			m_fun = m.loglike(m.parameter_values(), cached=cached)
#			print(key,"m_fun",m_fun)
			fun += m_fun
#		print("   fun",fun)
		return float(fun)

	def negative_loglike(self, *args):
		return -self.loglike(*args)

	def d_loglike(self, *args): 
		if len(args)>0:
			self.parameter_values(args[0])
		meta_parameter_values = self.parameter_values()
		d_ll = numpy.zeros(len(meta_parameter_values))
		for key,m in self.sub_model.items():
			if self.sub_ncases[key] == 0:
				continue
			mapping = {}
			for value, name, slot in zip(meta_parameter_values, self.parameter_names(), itertools.count()):
				if name in m:
					m.parameter(name).value = value
					mapping[m.parameter_index(name)] = slot
			m_d_ll = m.d_loglike_nocache()
			for local_slot, global_slot in mapping.items():
				d_ll[global_slot] += m_d_ll[local_slot]
#		print("d_ll", d_ll)
		return d_ll

	def negative_d_loglike(self, *args):
		return -self.d_loglike(*args)

	def finite_diff_d_loglike(self, v=None):
		from .array import Array
		g = Array([len(self)])
		if v is None:
			v = numpy.asarray(self.parameter_values())
		else:
			assert(len(v)==len(self))
			v = numpy.asarray(v)
		for n in range(len(self)):
			jiggle = v[n] * 1e-5 if v[n] else 1e-5
			v1 = v.copy()
			v1[n] += jiggle
			g[n] = self.loglike(v1)
			v2 = v.copy()
			v2[n] -= jiggle
			g[n] -= self.loglike(v2)
#			print("n=",n,"   g[n]=",g[n], "   jiggle=",jiggle, "   grad=",g[n] / (-2*jiggle))
			g[n] /= (-2*jiggle)
#			print("n=",n,"   g![n]=",g[n], )
		return g



	def d2_loglike(self, *args):
		if len(args)>0:
			self.parameter_values(args[0])
		meta_parameter_values = self.parameter_values()
		d2_ll = numpy.zeros([len(meta_parameter_values),len(meta_parameter_values)])
		for key,m in self.sub_model.items():
			if self.sub_ncases[key] == 0:
				continue
			mapping = {}
			for value, name, slot in zip(meta_parameter_values, self.parameter_names(), itertools.count()):
				if name in m:
					m.parameter(name).value = value
					mapping[m.parameter_index(name)] = slot
			m.freshen()
			m_d2_ll = m.d2_loglike()
			for local_row, global_row in mapping.items():
				for local_col, global_col in mapping.items():
					d2_ll[global_row,global_col] += m_d2_ll[local_row,local_col]
		return d2_ll
 
 
	def bhhh(self, *args):
		if len(args)>0:
			self.parameter_values(args[0])
		meta_parameter_values = self.parameter_values()
		d2_ll = numpy.zeros([len(meta_parameter_values),len(meta_parameter_values)])
		for key,m in self.sub_model.items():
			mapping = {}
			if self.sub_ncases[key] == 0:
				continue
			for value, name, slot in zip(meta_parameter_values, self.parameter_names(), itertools.count()):
				if name in m:
					m.parameter(name).value = value
					mapping[m.parameter_index(name)] = slot
			m.freshen()
			m_d2_ll = m.bhhh()
			for local_row, global_row in mapping.items():
				for local_col, global_col in mapping.items():
					d2_ll[global_row,global_col] += m_d2_ll[local_row,local_col]
		return d2_ll

	def bhhh_tolerance(self, *args):
		if len(args)>0:
			self.parameter_values(args[0])
		meta_parameter_values = self.parameter_values()
		d_ll = self.d_loglike()
		bhhh = self.bhhh()
		ibhhh = general_inverse(bhhh)
		return numpy.dot(numpy.dot(d_ll.T,ibhhh),d_ll)
