

from .metamodel import MetaModel
from .dt import DTL, DT
from .model import Model
from .roles import P,X
import numpy
from .array import pack
import itertools


def correspond_utilityca(m, grand_m=None):
	if grand_m is None:
		grand_m = m
	try:
		return m._correspond_utilityca
	except AttributeError:
		arr = numpy.zeros([len(m.utility.ca), len(grand_m)])
		for uca in range(len(m.utility.ca)):
			slot = grand_m.parameter_index(m.utility.ca[uca].param)
			arr[uca,slot] = 1
		m._correspond_utilityca = arr
		return arr

def correspond_quantity(m, grand_m=None):
	if grand_m is None:
		grand_m = m
	try:
		return m._correspond_quantity
	except AttributeError:
		arr = numpy.zeros([len(m.quantity), len(grand_m)])
		for uca in range(len(m.quantity)):
			slot = grand_m.parameter_index(m.quantity[uca].param)
			arr[uca,slot] = 1
		m._correspond_quantity = arr
		return arr

def correspond_utilityco(m, grand_m=None):
	if grand_m is None:
		grand_m = m
	try:
		return m._correspond_utilityco
	except AttributeError:
		if m.data.utilityco is None:
			m._correspond_utilityco = None
		else:
			arr = numpy.zeros([m.nAlts(), m.data.utilityco.shape[-1], len(grand_m)])
			for aslot,acode in enumerate(m.alternative_codes()):
				if acode in m.utility.co:
					ucoa = m.utility.co[acode]
					for uco in range(len(ucoa)):
						slot = grand_m.parameter_index(ucoa[uco].param)
						uco_slot = m.data.utilityco_varindex(ucoa[uco].data)
						arr[aslot,uco_slot,slot] = 1
			m._correspond_utilityco = arr.reshape(-1, len(grand_m))
		return m._correspond_utilityco


def _d_logsum_d_param_mnl(m, grand_m):
	return _map_submodel_params_to_grand_params(m, grand_m, m.d_logsums())
#	pr = m.work.probability
#	xca = m.data.utilityca
#	if xca is not None:
#		yca = (pr[:,:m.nAlts(),None] * xca).sum(1)
#	else:
#		yca = numpy.zeros([m.nCases(),0])
#	xco = m.data.utilityco
#	if xco is not None:
#		yco = (pr[:,:m.nAlts(),None] * xco[:,None,:])
#		yco = yco.reshape(yco.shape[0], yco.shape[1]*yco.shape[2])
#	else:
#		yco = numpy.zeros([m.nCases(),0])
#	yco_1 = correspond_utilityco(m, grand_m)
#	yca_1 = correspond_utilityca(m, grand_m)
#	q_1 = correspond_quantity(m, grand_m)
#	
#	z = m.data.quantity  # [cases, alts, datacolumns]
#	if z is not None:
#		egam =	m.Coef("QuantityCA").squeeze() #[ datacolumns ]
#		egz = m.work.quantity # [cases, alts]
#		q = (z * egam[None, None, :] * pr[:,:m.nAlts(),None] / egz[:,:,None] ).sum(1) #[ cases, datacolumns ]
#	else:
#		q = numpy.zeros([m.nCases(),0])
#
#	return (
#		+ (numpy.dot(yca,yca_1) if yca_1 is not None else 0)
#		+ (numpy.dot(yco,yco_1) if yco_1 is not None else 0)
#		+ (numpy.dot(q  ,q_1  ) if q_1   is not None else 0)
#	)


def _map_submodel_params_to_grand_params(m, grand_m, values):
	'''
	Push values from local model to grand model.
	
	Parameters
	----------
	m : Model
		local sub model
	grand_m : MetaModel
	values : array 
		with last dim as len(m)
		
	Returns
	-------
	values_ : array
		with last dim as len(grand_m)
	'''
	try:
		m._grand_parameter_map
	except AttributeError:
		arr = numpy.zeros([len(m), len(grand_m)])
		for m_slot, name in enumerate(m.parameter_names()):
			arr[m_slot, grand_m.parameter_index(name)] = 1
		m._grand_parameter_map = arr
	return numpy.dot(values, m._grand_parameter_map)

def _map_grand_params_to_submodel_params(m, grand_m, values):
	'''
	Push values from grand model to local model.
	
	Parameters
	----------
	m : Model
		local sub model
	grand_m : MetaModel
	values : array 
		with last dim as len(grand_m)
		
	Returns
	-------
	values_ : array
		with last dim as len(m)
	'''
	try:
		m._grand_parameter_map
	except AttributeError:
		arr = numpy.zeros([len(m), len(grand_m)])
		for m_slot, name in enumerate(m.parameter_names()):
			arr[m_slot, grand_m.parameter_index(name)] = 1
		m._grand_parameter_map = arr
	return numpy.dot(values, m._grand_parameter_map.T)



class NNNL(MetaModel):

	### Currently only allows one level NNNL,

	def __init__(self, base_model):
		super().__init__()
		base_model._parameter_inclusion_check()
		self.base_model = base_model
		self.df = base_model.df
		try:
			_nCases = self.df.nCases()
		except AttributeError:
			_nCases = 1
		g = base_model.graph
		for i in base_model.nest.nodes():
			alts = g.successors(i)
			d = DT.DummyWithAlts(altcodes=alts, nCases=_nCases)
			self.sub_model[i] = Model(d)
			self.sub_model[i].utility.ca = base_model.utility.ca
			self.sub_model[i].quantity = base_model.quantity
			if base_model.quantity_scale:
				self.sub_model[i].quantity_scale = base_model.quantity_scale
			for a in alts:
				try:
					x = base_model.utility.co[a]
				except IndexError:
					pass
				else:
					self.sub_model[i].utility.co[a] = x
		# Root
		base_nodes = self.base_model.nest.nodes()
		if not base_nodes:
			base_nodes = (1,)
		d = DT.DummyWithAlts(nCases=_nCases, altcodes=base_nodes)
		m = self.sub_model[self.root_id] = Model(d)
		for i in base_model.nest.nodes():
			m.utility.co[i] = base_model.nest[i].param * X(str(base_model.nest[i]._altcode))
		if len(base_model):
			for p in base_model.parameter:
				self.parameter[p.name] = p

	def _specific_warning_notes(self):
		w = "WARNING: This is a non-normalized nested logit (NNNL) model. You must adjust parameter estimates accordingly."
		#import warnings
		#warnings.warn(w, stacklevel=3)
		return w

	def setUp(self, *args, **kwargs):
		self.sub_weight = {}
		self.total_weight = 0
		self._ncases = 0
		for seg_descrip,submodel in self.sub_model.items():
			if self.logger():
				self.logger().log(30, "setUp:{!s}".format(seg_descrip))
			if seg_descrip==self.root_id:

				self.sub_model[seg_descrip].setUp(False)
				self.sub_model[seg_descrip].provision({
					"UtilityCO": numpy.zeros([self.df.nCases(),self.sub_model[seg_descrip].nAlts()]),
					"Avail": numpy.zeros([self.df.nCases(),self.sub_model[seg_descrip].nAlts(),1], dtype=bool),
					"Choice": numpy.zeros([self.df.nCases(),self.sub_model[seg_descrip].nAlts(),1]),
					"Weight": numpy.ones([self.df.nCases(),1])
				})
			
			else:
				
				submodel.setUp(False, False, False, False)
				needs = submodel.needs()
				
				keepalts = numpy.in1d(self.df._alternative_codes(), submodel.df._alternative_codes())
				# keepalts is a bool array for masking
				
				prov = {}
				if 'UtilityCA' in needs:
					prov["UtilityCA"] = pack(self.df.array_idca(*needs['UtilityCA'].get_variables())[:,keepalts,:])
				if 'UtilityCO' in needs:
					prov["UtilityCO"] = self.df.array_idco(*needs['UtilityCO'].get_variables())
				prov["Weight"] = self.df.array_weight()
				prov["Choice"] = pack(self.df.array_choice()[:,keepalts,:])
				prov["Avail"] = pack(self.df.array_avail()[:,keepalts,:])
				if 'QuantityCA' in needs:
					prov["QuantityCA"] = pack(self.df.array_idca(*needs['QuantityCA'].get_variables())[:,keepalts,:])
				
				submodel.provision(prov)
				
			if submodel.df.nCases()==0:
				submodel._ncases = 0
				submodel.sub_weight[seg_descrip] = 0
				continue
			self.sub_weight[seg_descrip] = partwgt = submodel.Data("Weight").sum()
			self.total_weight += partwgt
			self._ncases = partncase = submodel.nCases()
		# circle back to update choice and avail of upper level
		m0 =self.sub_model[self.root_id]
		self.m0slots = { subkey:slot for slot,subkey in enumerate(m0.alternative_codes()) }
		for key in m0.alternative_codes():
			m0.dataedit.avail[:,self.m0slots[key]] = (self.sub_model[key].data.avail.sum(1)>0)
			m0.dataedit.choice[:,self.m0slots[key]] = (self.sub_model[key].data.choice * self.sub_model[key].data.avail).sum(1)
			# set lower level models to feed logsum up
			self.sub_model[key].top_logsums_out = m0.dataedit.utilityco[:,self.m0slots[key]]
		self._setUp_NNNL_host(self._ncases)

	def loglike(self, *args, cached=False):
		fun = 0
		if len(args)>0:
			self.parameter_values(args[0])
		meta_parameter_values = self.parameter_values()
		m0 = self.sub_model[self.root_id]
		for key,m in self.sub_model.items():
			for value, name in zip(meta_parameter_values, self.parameter_names()):
				if name in m:
					m.parameter(name).value = value
			if key==self.root_id:
				pass
			else:
				m_fun = m.loglike(m.parameter_values(), cached=cached)
#				print("LL",key,"=",m_fun)
				fun += m_fun
				# push logsums to upper level
				# m0.dataedit.utilityco[:,self.m0slots[key]] = m.logsums() # skip now, happens automatically
		# circle back to root, which must be last
		m0.dataedit.utilityco[numpy.isneginf(m0.dataedit.utilityco)] = -1.79769313e+308
		m_fun = m0.loglike(m0.parameter_values(), cached=cached)
#		print("LL",self.root_id,"=",m_fun)
		fun += m_fun
		if self.logger():
			self.logger().log(30, "NNNL.LL={} <- {}".format(str(fun),str(self.parameter_array)))
		return float(fun)

	def loglike_casewise(self, *args, cached=False):
		return sum(m.loglike_casewise() for key,m in self.sub_model.items())

	def loglike_casewise_many(self, *args, cached=False):
		x = {}
		for key,m in self.sub_model.items():
			x[key] = m.loglike_casewise()
		return x
	

	def d_loglike_many(self, *args, cached=False):
		x = {}
		for key,m in self.sub_model.items():
			x[key] = _map_submodel_params_to_grand_params(m, self, m.d_loglike_nocache())
		return x
	
	def d_loglike_casewise_many(self, *args, cached=False):
		x = {}
		for key,m in self.sub_model.items():
			x[key] = _map_submodel_params_to_grand_params(m, self, m.d_loglike_casewise())
		return x

	def d_logsums_many(self, *args, cached=False):
		x = {}
		for key,m in self.sub_model.items():
			x[key] = _d_logsum_d_param_mnl(m, self)
		return x


	def probability_roll_up(self):
		if self.work.probability.shape[1] != self.df.nAlts() or self.work.probability.shape[0] != self.df.nCases():
			self._setUp_NNNL_host(self._ncases or self.df.nCases())
		m0 = self.sub_model[self.root_id]
		for slot,nestcode in enumerate(m0.alternative_codes()):
			t = m0.work.probability[:,slot]
			for subslot, altcode in enumerate(self.sub_model[nestcode].alternative_codes()):
				topslot = int(self.df._alternative_slot(altcode))
				self.work.probability[:,topslot] = self.sub_model[nestcode].work.probability[:,subslot] * t

	def d_loglike(self, *args, cached=False):
		"""
		Calculate the vector of first partial derivative w.r.t. the parameters.
		
		Parameters
		----------
		cached : bool
			Ignored in this version.
		"""
		if len(args)>0:
			self.parameter_values(args[0])
		meta_parameter_values = self.parameter_values()
		d_ll = numpy.zeros(len(meta_parameter_values))
		for key,m in self.sub_model.items():
			for value, name in zip(meta_parameter_values, self.parameter_names()):
				if name in m:
					m.parameter(name).value = value

		# deriv with respect to child models
		m0 = self.sub_model[self.root_id]
		for key in m0.alternative_codes():
			m = self.sub_model[key]
			d_ll += _map_submodel_params_to_grand_params(m, self, m.d_loglike(cached=cached))

		m0.dataedit.utilityco[numpy.isneginf(m0.dataedit.utilityco)] = -1.79769313e+308
		d_ll += _map_submodel_params_to_grand_params(m0, self, m0.d_loglike(cached=cached))

		# deriv as it changes the logsums
		for slot,nestcode in enumerate(m0.alternative_codes()):
			t = (m0.data.choice[:,slot,0]-m0.work.probability[:,slot])[:,None] * _d_logsum_d_param_mnl(self.sub_model[nestcode], self)
			mu_name = self.base_model.nest[nestcode].param
			mu = self.parameter_array[ self.parameter_index(mu_name) ]
			d_ll += t.sum(0) * mu

		holds = (self.parameter_holdfast_array!=0)
		if numpy.any(holds):
			d_ll[holds] = 0

		if self.logger():
			self.logger().log(30, "NNNL.dLL={} <- {}".format(str(d_ll),str(self.parameter_array)))
		return d_ll

	def d_loglike_casewise(self, *args, cached=False):
		"""
		Calculate the vector of first partial derivative w.r.t. the parameters.
		
		Parameters
		----------
		cached : bool
			Ignored in this version.
		"""
		if len(args)>0:
			self.parameter_values(args[0])
		meta_parameter_values = self.parameter_values()
		m0 = self.sub_model[self.root_id]
		d_ll = numpy.zeros([m0.nCases(), len(meta_parameter_values)])

		# deriv with respect to child models
		for key,m in self.sub_model.items():
			d_ll += _map_submodel_params_to_grand_params(m, self, -m.d_loglike_casewise())

		# deriv as it changes the logsums
		for slot,nestcode in enumerate(m0.alternative_codes()):
			t = (m0.data.choice[:,slot,0]-m0.work.probability[:,slot])[:,None] * _d_logsum_d_param_mnl(self.sub_model[nestcode], self)
			mu_name = self.base_model.nest[nestcode].param
			mu = self.parameter_array[ self.parameter_index(mu_name) ]
			d_ll += t * mu

		holds = (self.parameter_holdfast_array!=0)
		if numpy.any(holds):
			d_ll[:,holds] = 0

		return d_ll




	def d2_loglike(self, *args, finite_grad=False):
		z = self.finite_diff_hessian(*args, out=self.hessian_matrix, finite_grad=finite_grad)
		if self.hessian_matrix is not z:
			self.hessian_matrix = z
		return z

	def negative_d2_loglike(self, *args, finite_grad=False):
		z = numpy.copy(self.finite_diff_hessian(*args, out=self.hessian_matrix, finite_grad=finite_grad))
		z *= -1
		return z

	def bhhh(self, *args):
		if len(args)>0:
			self.parameter_values(args[0])
		dc = self.d_loglike_casewise()
		return numpy.dot(dc.T,dc)

	def logger(self, *args, **kwargs):
		if args==(1,):
			args = ('NNNL',)
		if args==(True,):
			args = ('NNNL',)
		return super().logger(*args, **kwargs)

	def maximize_loglike(self, *args, **kwargs):
		result = super().maximize_loglike(*args, **kwargs)
		self.probability_roll_up()
		return result


	# Utility Specification Summary
	def xhtml_utilityspec(self,**format):
		
		from .util.xhtml import XML_Builder
		NonBreakSpace = " "

		if len(self.base_model.utility.co)==0: return self.base_model.xhtml_utilityspec_ca_only(**format)
		existing_format_keys = list(format.keys())
		for key in existing_format_keys:
			if key.upper()!=key: format[key.upper()] = format[key]
		if 'PARAM' not in format: format['PARAM'] = '0.4g'
		
		def bracketize(s):
			s = s.strip()
			if len(s)<3: return s
			if s[0]=="(" and s[-1]==")": return s
			if "=" in s: return "({})".format(s)
			if "+" in s: return "({})".format(s)
			if "-" in s: return "({})".format(s)
			if " in " in s.casefold(): return "({})".format(s)
			return s
		
		x = XML_Builder("div", {'class':"utilityspec larch_art"})
		x.h2("Utility Specification", anchor=1, attrib={'class':'larch_art_xhtml'})
		
		for resolved in (True, False):
			if resolved:
				headline = "Resolved Utility"
			else:
				headline = "Formulaic Utility"
			x.h3(headline, anchor=1, attrib={'class':'larch_art_xhtml'})
			
			with x.block("table", {'class':'floatinghead'}):
				with x.thead_:
					with x.tr_:
						x.th('Code')
						x.th('Alternative')
						x.th(headline)
				with x.tbody_:
					for altcode,altname in self.base_model.alternatives().items():
						with x.tr_:
							x.td(str(altcode))
							x.td(str(altname))
							x.start("td")
							
							first_thing = True
							
							def add_util_component(beta, resolved, x, first_thing):
								if resolved:
									beta_val = "{:{PARAM}}".format(self.metaparameter(beta.param).value, **format).strip()
									if not first_thing:
										x.simple("br")
										x.data("+ {}".format(beta_val).replace("+ -","- "))
									else: # is first thing
										if "-" == beta_val[0]:
											x.data(beta_val.replace("-","- "))
										else:
											x.data(NonBreakSpace*2)
											x.data(beta_val)
									first_thing = False
								else:
									if not first_thing:
										x.simple("br")
										x.data("+ ")
									else:
										x.data(NonBreakSpace*2)
									first_thing = False
									x.start('a', {'class':'parameter_reference', 'href':'#param{}'.format(beta.param.replace("#","_hash_"))})
									x.data(beta.param)
									x.end('a')
									if beta.multiplier != 1.0:
										x.data("*"+str(beta.multiplier))
								try:
									beta_data_value = float(beta.data)
									if beta_data_value==1.0:
										beta_data_value=""
									else:
										beta_data_value="*"+str(bracketize(beta_data_value))
								except:
									beta_data_value = "*"+str(bracketize(beta.data))
								x.data(beta_data_value)
								return x, first_thing

							
							for beta in self.base_model.utility.ca:
								x, first_thing = add_util_component(beta, resolved, x, first_thing)
							if altcode in self.base_model.utility.co:
								for beta in self.base_model.utility.co[altcode]:
									x, first_thing = add_util_component(beta, resolved, x, first_thing)
							

							x.end("td")
				
					G = self.base_model.networkx_digraph()
					if len(G.node)>len(self.base_model.alternative_codes())+1:
						with x.tr_:
							x.th('Code')
							x.th('Nest')
							x.th(headline)
						for altcode in self.base_model.nodes_ascending_order(exclude_elementals=True):
							if altcode==self.base_model.root_id:
								altname = 'ROOT'
								mu_name = '1'
							else:
								altname = self.base_model.nest[altcode]._altname
								mu_name = self.base_model.nest[altcode].param
							try:
								skip_mu = (float(mu_name)==1)
							except ValueError:
								skip_mu = False
							with x.tr_:
								x.td(str(altcode))
								x.td(str(altname))
								x.start("td")
								if not skip_mu:
									if resolved:
										beta_val = "{:{PARAM}}".format(self.metaparameter(mu_name).value, **format)
										x.data(beta_val)
									else:
										x.start('a', {'class':'parameter_reference', 'href':'#param{}'.format(mu_name.replace("#","_hash_"))})
										x.data(mu_name)
										x.end('a')
									x.data(" * ")
								x.data("log(")
								for i,successorcode in enumerate(G.successors(altcode)):
									if i>0: x.data("+")
									successorname = G.node[successorcode]['name']
									x.data(" exp(Utility[{}]".format(successorname))
									x.data(") ")
								
								x.data(")")
								x.end("td")
		return x.close()



	# Probability Specification Summary
	def xhtml_probabilityspec(self,**format):
		
		from .util.xhtml import XML_Builder
		
		existing_format_keys = list(format.keys())
		for key in existing_format_keys:
			if key.upper()!=key: format[key.upper()] = format[key]
		if 'PARAM' not in format: format['PARAM'] = '0.4g'
		
		x = XML_Builder("div", {'class':"probabilityspec larch_art"})
		x.h2("Probability Specification", anchor=1, attrib={'class':'larch_art_xhtml'})
		G = self.networkx_digraph()

		for resolved in (True, False):
			if resolved:
				headline = "Resolved Probability"
			else:
				headline = "Formulaic Probability"
			x.h3(headline, anchor=1, attrib={'class':'larch_art_xhtml'})
		
			with x.block("table", {'class':'floatinghead'}):
				with x.thead_:
					with x.tr_:
						x.th('Code')
						x.th('Alternative')
						x.th(headline)
				with x.tbody_:
					for altcode,altname in self.alternatives().items():
						with x.tr_:
							x.td(str(altcode))
							x.td(str(altname))
							x.start("td")
							
							curr = altcode
							if G.in_degree(curr) > 1:
								raise LarchError('xhtml_probabilityspec is not compatible with non-NL models')
							pred = G.predecessors(curr)[0]
							
							curr_name = G.node[curr]['name']
							pred_name = G.node[pred]['name']
							mu_name = '1' if pred==self.root_id else self.nest[pred].param
							try:
								skip_mu = (float(mu_name)==1)
							except ValueError:
								skip_mu = False
							
							def add_part():
								x.data("exp(Utility[{}]".format(curr_name))
#								if not skip_mu:
#									x.data("/")
#									if resolved:
#										beta_val = "{:{PARAM}}".format(self.metaparameter(mu_name).value, **format)
#										x.data(beta_val)
#									else:
#										x.start('a', {'class':'parameter_reference', 'href':'#param{}'.format(mu_name.replace("#","_hash_"))})
#										x.data(mu_name)
#										x.end('a')
								x.data(")/")
								x.data("exp(Utility[{}]".format(pred_name))
#								if not skip_mu:
#									x.data("/")
#									if resolved:
#										beta_val = "{:{PARAM}}".format(self.metaparameter(mu_name).value, **format)
#										x.data(beta_val)
#									else:
#										x.start('a', {'class':'parameter_reference', 'href':'#param{}'.format(mu_name.replace("#","_hash_"))})
#										x.data(mu_name)
#										x.end('a')
								x.data(")")
							add_part()

							while pred != self.root_id:
								curr = pred
								pred = G.predecessors(curr)[0]
								curr_name = G.node[curr]['name']
								pred_name = G.node[pred]['name']
								mu_name = '1' if pred==self.root_id else self.nest[pred].param
								try:
									skip_mu = (float(mu_name)==1)
								except ValueError:
									skip_mu = False
								x.data(" * ")
								add_part()
							x.end("td")
				
		return x.close()


	def xhtml_nesting_tree(self,*arg,**kwarg):
		return self.base_model.xhtml_nesting_tree(*arg,**kwarg)
	
	def xhtml_nesting_tree_textonly(self,*arg,**kwarg):
		return self.base_model.xhtml_nesting_tree_textonly(*arg,**kwarg)

