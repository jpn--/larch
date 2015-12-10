
from .core import Model2, LarchError, _core, ParameterAlias
from .array import SymmetricArray
from .utilities import category, pmath, rename
import numpy
import os
from .util.xhtml import XHTML, XML_Builder
import math
from .model_reporter import ModelReporter


class MetaParameter():
	def __init__(self, name, value, under, initial_value=None):
		self._name = name
		self._value = value
		self.under = under
		self._initial_value = initial_value
	@property
	def name(self):
		return self._name
	@property
	def value(self):
		return self._value
	@property
	def initial_value(self):
		return self._initial_value




class Model(Model2, ModelReporter):

	from .util.roll import roll

	def dir(self):
		for f in dir(self):
			print(" ",f)

	def parameter_wide(self, name):
		try:
			return self.alias(name)
		except LarchError:
			return self.parameter(name)

	def metaparameter(self, name):
		try:
			x = self.alias(name)
		except LarchError:
			if name not in self.parameter_names():
				raise LarchError("cannot find '{}' in model".format(name))
			x = self.parameter(name)
			return MetaParameter(name, x.value, x, x.initial_value)
		else:
			return MetaParameter(name, self.metaparameter(x.refers_to).value * x.multiplier, x, self.metaparameter(x.refers_to).initial_value* x.multiplier)

	def param_sum(self,*arg):
		value = 0
		found_any = False
		for p in arg:
			if isinstance(p,str) and p in self:
				value += self.metaparameter(p).value
				found_any = True
			elif isinstance(p,(int,float)):
				value += p
		if not found_any:
			raise LarchError("no parameters with any of these names: {}".format(str(arg)))
		return value

	def param_product(self,*arg):
		value = 1
		found_any = False
		for p in arg:
			if isinstance(p,str) and p in self:
				value *= self.metaparameter(p).value
				found_any = True
			elif isinstance(p,(int,float)):
				value *= p
		if not found_any:
			raise LarchError("no parameters with any of these names: {}".format(str(arg)))
		return value

	def param_ratio(self, numerator, denominator):
		if isinstance(numerator,str):
			if numerator in self:
				value = self.metaparameter(numerator).value
			else:
				raise LarchError("numerator {} not found".format(numerator))
		elif isinstance(numerator,(int,float)):
			value = numerator
		if isinstance(denominator,str):
			if denominator in self:
				value /= self.metaparameter(denominator).value
			else:
				raise LarchError("denominator {} not found".format(denominator))
		elif isinstance(denominator,(int,float)):
			value /= denominator
		return value

	def _set_nest(self, *args):
		_core.Model2_nest_set(self, *args)
		self.freshen()
	_nest_doc = """\
	A function-like object mapping node codes to names and parameters.
	
	This can be called as if it was a normal method of :class:`Model`.
	It also is an object that acts like a dict with integer keys 
	representing the node code numbers and :class:`larch.core.LinearComponent`
	values.
	
	Parameters
	----------
	id : int
		The code number of the nest. Must be unique to this nest among the 
		set of all nests and all elemental alternatives.
	name : str or None
		The name of the nest. This name is used in various reports.
		It can be any string but generally something short and descriptive
		is useful. If None, the name is set to "nest_{id}".
	parameter : str or None
		The name of the parameter to associate with this nest.  If None, 
		the `name` is used.
		
	Returns
	-------
	:class:`larch.core.LinearComponent`
		The component object for the designated node
		
	Notes
	-----
	Earlier versions of this software required node code numbers to be non-negative.  
	They can now be any 64 bit signed integer.
	
	Because the id and name are distinct data types, Larch can detect (and silently allow) when
	they are transposed (i.e. with `name` given before `id`).
	"""
	nest = property(_core.Model2_nest_get, _set_nest, None, _nest_doc)
	node = property(_core.Model2_nest_get, _set_nest, None, "an alias for :attr:`nest`")
	
	
	def _set_link(self, *args):
		_core.Model2_link_set(self, *args)
		self.freshen()
	_link_doc = """\
	A function-like object defining links between network nodes.
	
	Parameters
	----------
	up_id : int
		The code number of the upstream (i.e. closer to the root node) node on the link.
		This should never be an elemental alternative.
	down_id : int
		The code number of the downstream node on the link. This can be an elemental
		alternative.
	"""
	link = property(_core.Model2_link_get, _set_link, None, _link_doc)
	edge = property(_core.Model2_link_get, _set_link, None, "an alias for :attr:`link`")
	
	def alternatives(self):
		return {code:name for code,name in zip(self.alternative_codes(),self.alternative_names())}

	def _set_rootcode(self, *args):
		_core.Model2__set_root_cellcode(self, *args)
		#self.freshen()
	_rootcode_doc = """\
	The root_id is the code number for the root node in a nested logit or
	network GEV model. The default value for the root_id is 0. It is important
	that the root_id be different from the code for every elemental alternative
	and intermediate nesting node. If it is convenient for one of the elemental
	alternatives or one of the intermediate nesting nodes to have a code number
	of 0 (e.g., for a binary logit model where the choices are yes and no),
	then this value can be changed to some other integer.
	"""
	root_id = property(_core.Model2__get_root_cellcode, _set_rootcode, None, _rootcode_doc)

	def get_data_pointer(self):
		return self._ref_to_db

	db = property(get_data_pointer, Model2.change_data_pointer, Model2.delete_data_pointer)

	def load(self, filename="@@@", *, echo=False):
		if filename=="@@@" and isinstance(self,str):
			filename = self
			self = Model()
		inf = numpy.inf
		nan = numpy.nan
		if (len(filename)>5 and filename[-5:]=='.html') or (len(filename)>6 and filename[-6:]=='.xhtml'):
			from html.parser import HTMLParser
			class LarchHTMLParser_ModelLoader(HTMLParser):
				def handle_starttag(subself, tag, attrs):
					if tag=='meta':
						use = False
						for attrname,attrval in attrs:
							if attrname=='name' and attrval=='pymodel':
								use = True
						if use:
							for attrname,attrval in attrs:
								if attrname=='content':
									self.loads(attrval, use_base64=True, echo=echo)
			parser = LarchHTMLParser_ModelLoader()
			with open(filename) as f:
				parser.feed(f.read())
			self.loaded_from = filename
			return self
		else:
			with open(filename) as f:
				code = compile(f.read(), filename, 'exec')
				exec(code)
			self.loaded_from = filename
			return self

	def loads(self, content="@@@", *, use_base64=False, echo=False):
		if content=="@@@" and isinstance(self,(str,bytes)):
			content = self
			self = Model()
		inf = numpy.inf
		nan = numpy.nan
		if use_base64:
			import base64
			content = base64.standard_b64decode(content)
		if isinstance(content, bytes):
			import zlib
			try:
				content = zlib.decompress(content)
			except zlib.error:
				pass
			import pickle
			try:
				content = pickle.loads(content)
			except pickle.UnpicklingError:
				pass
		if isinstance(content, str):
			if echo: print(content)
			code = compile(content, "<string>", 'exec')
			exec(code)
		else:
			raise LarchError("error in loading")
		return self

	def save(self, filename, overwrite=False, spool=True, report=False, report_cats=['title','params','LL','latest','utilitydata','data','notes']):
		if filename is None:
			import io
			filemaker = lambda: io.StringIO()
		else:
			if os.path.exists(filename) and not overwrite and not spool:
				raise IOError("file {0} already exists".format(filename))
			if os.path.exists(filename) and not overwrite and spool:
				filename, filename_ext = os.path.splitext(filename)
				n = 1
				while os.path.exists("{} ({}){}".format(filename,n,filename_ext)):
					n += 1
				filename = "{} ({}){}".format(filename,n,filename_ext)
			filename, filename_ext = os.path.splitext(filename)
			if filename_ext=="":
				filename_ext = ".py"
			filename = filename+filename_ext
			filemaker = lambda: open(filename, 'w')
		with filemaker() as f:
			if report:
				f.write(self.report(lineprefix="#\t", cats=report_cats))
				f.write("\n\n\n")
			import time
			f.write("# saved at %s"%time.strftime("%I:%M:%S %p %Z"))
			f.write(" on %s\n"%time.strftime("%d %b %Y"))
			f.write(self.save_buffer())
			blank_attr = set(dir(Model()))
			blank_attr.remove('descriptions')
			aliens_found = False
			for a in dir(self):
				if a not in blank_attr:
					if isinstance(getattr(self,a),(int,float)):
						f.write("\n")
						f.write("self.{} = {}\n".format(a,getattr(self,a)))
					elif isinstance(getattr(self,a),(str,)):
						f.write("\n")
						f.write("self.{} = {!r}\n".format(a,getattr(self,a)))
					else:
						if not aliens_found:
							import pickle
							f.write("import pickle\n")
							aliens_found = True
						try:
							p_obj = pickle.dumps(getattr(self,a))
							f.write("\n")
							f.write("self.{} = pickle.loads({})\n".format(a,p_obj))
						except pickle.PickleError:
							f.write("\n")
							f.write("self.{} = 'unpicklable object'\n".format(a,p_obj))
			try:
				return f.getvalue()
			except AttributeError:
				return

	def saves(self):
		"Return a string representing the saved model"
		return self.save(None)

	def __getstate__(self):
		import pickle, zlib
		return zlib.compress(pickle.dumps(self.save(None)))

	def __setstate__(self, state):
		import pickle, zlib
		self.__init__()
		self.loads( pickle.loads(zlib.decompress(state)) )

	def copy(self, other="@@@"):
		if other=="@@@" and isinstance(self,Model):
			other = self
			self = Model()
		if not isinstance(other,Model):
			raise IOError("the object to copy from must be a larch.Model")
		inf = numpy.inf
		nan = numpy.nan
		code = compile(other.save_buffer(), "model_to_copy", 'exec')
		exec(code)
		return self

	def recall(self, nCases=None):
		if nCases is not None:
			self._nCases_recall = nCases

	def __utility_get(self):
		return _core.Model2_utility_get(self)

	def __utility_set(self,x):
		return _core.Model2_utility_set(self,x)

	utility = property(__utility_get, __utility_set)

	def note(self, comment):
		if not hasattr(self,"notes"): self.notes = []
		def _append_note(x):
			x = "{}".format(x).replace("\n"," -- ")
			if x not in self.notes:
				self.notes += [x,]
		if isinstance(comment,(list,tuple)):
			for eachcomment in comment: _append_note(eachcomment)
		else:
			_append_note(comment)

	def networkx_digraph(self):
		try:
			import networkx as nx
		except ImportError:
			import warnings
			warnings.warn("networkx module not installed, unable to build network graph")
			raise
		G = nx.DiGraph()
		G.add_node(self.root_id, name='ROOT')
		for i in self.nest.nodes():
			G.add_node(i, name=self.nest[i]._altname)
		for icode,iname in self.alternatives().items():
			G.add_node(icode, name=iname)
		for i,j in self.link.links():
			G.add_edge(i,j)
		for n in G.nodes():
			if n!=self.root_id and G.in_degree(n)==0:
				G.add_edge(self.root_id,n)
		return G

	def nodes_descending_order(self):
		discovered = []
		discovered.append(self.root_id)
		pending = set()
		pending.add(self.root_id)
		attic = set()
		attic.add(self.root_id)
		G = self.networkx_digraph()
		predecessors = {i:set(G.predecessors(i)) for i in G.nodes()}
		while len(pending)>0:
			n = pending.pop()
			for s in G.successors(n):
				if s not in attic and predecessors[s] <= attic:
					pending.add(s)
					discovered.append(s)
					attic.add(s)
		return discovered

	def nodes_ascending_order(self, exclude_elementals=False):
		discovered = []
		if not exclude_elementals:
			discovered.extend(self.alternative_codes())
		pending = set(self.alternative_codes())
		basement = set(self.alternative_codes())
		G = self.networkx_digraph()
		successors = {i:set(G.successors(i)) for i in G.nodes()}
		while len(pending)>0:
			n = pending.pop()
			for s in G.predecessors(n):
				if s not in basement and successors[s] <= basement:
					pending.add(s)
					discovered.append(s)
					basement.add(s)
		return discovered

	def new_node(self, nest_name=None, param_name="", **kwargs):
		"""Generate a new nest with a new unique code.
		
		Parameters
		----------
		id : int
			The code number of the nest. Must be unique to this nest among the 
			set of all nests and all elemental alternatives.
		nest_name : str or None
			The name of the nest. This name is used in various reports.
			It can be any string but generally something short and descriptive
			is useful. If None, the name is set to "nest_{id}".
		param_name : str
			The name of the parameter to associate with this nest.  If not given,
			or given as an empty string, the `nest_name` is used.

		Returns
		-------
		int
			The code for the newly created nest.
		
		Notes
		-----
		Other keyword parameters are passed through to the nest creation function.
			
		"""
		if len(self.node.nodes())>0:
			max_node = max(self.node.nodes())
		else:
			max_node = 0
		newcode = max(max_node,max(self.alternative_codes()),self.root_id)+1
		self.node(newcode, nest_name, param_name=param_name, **kwargs)
		return newcode
	
	new_nest = new_node

	def report_(self, **kwargs):
		with XHTML('temp', quickhead=self, **kwargs) as f:
			f << self.report(cats='*', style='xml')



	def __str__(self):
		return self.report('txt')



	def stats_utility_co_sqlite(self, where=None):
		"""
		Generate a dataframe of descriptive statistics on the model idco data read from SQLite.
		If the where argument is given, it is used as a filter on the larch_idco table.
		"""
		import pandas
		keys = set()
		db = self._ref_to_db
		stats = None
		for u in self.utility.co:
			if u.data in keys:
				continue
			else:
				keys.add(u.data)
			qry = """
				SELECT
				'{0}' AS DATA,
				min({0}) AS MINIMUM,
				max({0}) AS MAXIMUM,
				avg({0}) AS MEAN,
				stdev({0}) AS STDEV
				FROM {1}
				""".format(u.data, self.db.tbl_idco())
			if where:
				qry += " WHERE {}".format(where)
			s = db.dataframe(qry)
			s = s.set_index('DATA')
			if stats is None:
				stats = s
			else:
				stats = pandas.concat([stats,s])
		return stats

	def stats_utility_co(self):
		"""
		Generate a set of descriptive statistics (mean,stdev,mins,maxs,nonzeros,
		positives,negatives,zeros,mean of nonzero values) on the model's idco data as loaded. Uses weights
		if available.
		"""
		x = self.Data("UtilityCO")
		if bool((self.Data("Weight")!=1).any()):
			w = self.Data("Weight").flatten()
			mean = numpy.average(x, axis=0, weights=w)
			variance = numpy.average((x-mean)**2, axis=0, weights=w)
			stdev = numpy.sqrt(variance)
		else:
			mean = numpy.mean(x,0)
			stdev = numpy.std(x,0)
		mins = numpy.amin(x,0)
		maxs = numpy.amax(x,0)
		nonzer = tuple(numpy.count_nonzero(x[:,i]) for i in range(x.shape[1]))
		pos = tuple(int(numpy.sum(x[:,i]>0)) for i in range(x.shape[1]))
		neg = tuple(int(numpy.sum(x[:,i]<0)) for i in range(x.shape[1]))
		zer = tuple(x[:,i].size-numpy.count_nonzero(x[:,i]) for i in range(x.shape[1]))
		sumx = numpy.sum(x,0)
		mean_nonzer = sumx / numpy.array(nonzer)
		return (mean,stdev,mins,maxs,nonzer,pos,neg,zer,mean_nonzer)
		
	def stats_utility_ca_chosen_unchosen(self):
		"""
		Generate a set of descriptive statistics (mean,stdev,mins,maxs,nonzeros,
		positives,negatives,zeros,mean of nonzero values) on the model's idca data as loaded. Uses weights
		if available.
		"""
		
		x = self.Data("UtilityCA")
		ch = self.Data("Choice")
		
		x_chosen = x[ch.astype(bool).reshape(ch.shape[0],ch.shape[1])]
		x_unchosen = x[~ch.astype(bool).reshape(ch.shape[0],ch.shape[1])]
		
		def _compute(xxx):
			mean_ = numpy.mean(xxx,0)
			stdev_ = numpy.std(xxx,0)
			mins_ = numpy.amin(xxx,0)
			maxs_ = numpy.amax(xxx,0)
			nonzer_ = tuple(numpy.count_nonzero(xxx[:,i]) for i in range(xxx.shape[1]))
			pos_ = tuple(int(numpy.sum(xxx[:,i]>0)) for i in range(xxx.shape[1]))
			neg_ = tuple(int(numpy.sum(xxx[:,i]<0)) for i in range(xxx.shape[1]))
			zer_ = tuple(xxx[:,i].size-numpy.count_nonzero(xxx[:,i]) for i in range(xxx.shape[1]))
			sumx_ = numpy.sum(xxx,0)
			mean_nonzer_ = sumx_ / numpy.array(nonzer_)
			return (mean_,stdev_,mins_,maxs_,nonzer_,pos_,neg_,zer_,mean_nonzer_)
		
		return _compute(x_chosen), _compute(x_unchosen),


	def parameter_names(self, output_type=list):
		x = []
		for n,p in enumerate(self._get_parameter()):
			x.append(p['name'])
		if output_type is not list:
			x = output_type(x)
		return x

	def reconstruct_covariance(self):
		s = len(self)
		from .array import SymmetricArray
		x = SymmetricArray([s])
		names = self.parameter_names()
		for n,p in enumerate(self._get_parameter()):
			for j in range(n+1):
				if names[j] in p['covariance']:
					x[n,j] = p['covariance'][names[j]]
		return x

	def reconstruct_robust_covariance(self):
		s = len(self)
		from .array import SymmetricArray
		x = SymmetricArray([s])
		names = self.parameter_names()
		for n,p in enumerate(self._get_parameter()):
			for j in range(n+1):
				if names[j] in p['robust_covariance']:
					x[n,j] = p['robust_covariance'][names[j]]
		return x

	def hessian(self, recalc=False):
		"The hessian matrix at the converged point of the latest estimation"
		if recalc:
			self.loglike()
			self.d2_loglike()
		return self.hessian_matrix

	def covariance(self, recalc=False):
		"The inverse of the hessian matrix at the converged point of the latest estimation"
		return self.covariance_matrix().view(SymmetricArray)

	def robust_covariance(self, recalc=False):
		"The sandwich estimator at the converged point of the latest estimation"
		return self.robust_covariance_matrix()

	def parameter_holdfast_mask(self):
		mask = numpy.ones([len(self),],dtype=bool)
		for n,p in enumerate(self._get_parameter()):
			if p['holdfast']:
				mask[n] = 0
		return mask

	def parameter_holdfast_release(self):
		for n,p in enumerate(self._get_parameter()):
			p['holdfast'] = False
	
	def parameter_holdfast_mask_restore(self, mask):
		for n,p in enumerate(self._get_parameter()):
			p['holdfast'] = mask[n]

	def rank_check(self, apply_correction=True, zero_correction=False):
		"""
		Check if the model is over-specified.
		"""
		locks = set()
		h = self.hessian(True)
		names = self.parameter_names(numpy.array)
		mask = self.parameter_holdfast_mask()
		h_masked = h[mask,:][:,mask]
		while h_masked.flats().shape[1]:
			bads = numpy.flatnonzero(numpy.round(h_masked.flats()[:,0], 5))
			fixit = bads.flat[0]
			locks.add(names[mask][fixit])
			self.parameter(names[mask][fixit], holdfast=True)
			if zero_correction:
				self.parameter(names[mask][fixit], value=self.parameter(names[mask][fixit]).initial_value)
			mask = self.parameter_holdfast_mask()
			h_masked = h[mask,:][:,mask]
		self.teardown()
		if not apply_correction:
			for l in locks:
				self.parameter(l, holdfast=False)
		return locks

	def parameter_reset_to_initial_values(self):
		for n,p in enumerate(self._get_parameter()):
			p['value'] = p['initial_value']

	def estimate_constants_only(self, repair='-'):
		db = self._ref_to_db
		alts = db.alternatives()
		m = Model(db)
		for a in alts[1:]:
			m.utility.co('1',a[0],a[1])
		m.provision()
		clashes = numpy.nonzero( numpy.logical_and(m.Data("Choice"), ~m.Data("Avail")) )
		n_clashes = len(clashes[0])
		if n_clashes>0:
			m.clash = clashes
			if repair == '+':
				for i in zip(*clashes):
					m.DataEdit("Avail")[i] = True
					print("REPAIR + ",i)
			if repair == '-':
				for i in zip(*clashes):
					m.DataEdit("Choice")[i] = 0
					print("REPAIR - ",i)
			else:
				raise LarchError("Model has {} cases where the chosen alternative is unavailable".format(n_clashes))
		m.estimate()
		self._set_estimation_statistics(log_like_constants=m.LL())
		return m

	def estimate_nil_model(self):
		db = self._ref_to_db
		alts = db.alternatives()
		m = Model(db)
		for a in alts[1:]:
			m.utility.co('0',a[0],a[1])
		m.estimate()
		self._set_estimation_statistics(log_like_nil=m.LL())

	def negative_loglike_(self, x):
		y = self.negative_loglike(x)
		if numpy.isnan(y):
			y = numpy.inf
			print("negative_loglike_ is NAN")
		print("negative_loglike:",x,"->",y)
		return y

	def d_loglike_(self, x):
		y = self.d_loglike(x)
		#if not hasattr(self,"first_grad"):
		#	y *= 0.0000001
		#	self.first_grad = 1
		print("d_loglike:",x,"->",y)
		return y

	def gradient_check(self):
		try:
			if self.is_provisioned()<=0:
				self.provision()
		except LarchError:
			self.provision()
		self.loglike()
		_fin_diff = self.option.force_finite_diff_grad
		_force_recalculate = self.option.force_recalculate
		self.option.force_recalculate = True
		self.option.force_finite_diff_grad = False
		a_grad = self.d_loglike()
		self.option.force_finite_diff_grad = True
		fd_grad = self.d_loglike()
		self.option.force_finite_diff_grad = _fin_diff
		self.option.force_recalculate = _force_recalculate
		namelen = max(len(n) for n in self.parameter_names())
		namelen = max(namelen,9)
		from .util.flux import flux_mag
		from math import log10
		print("{1:<{0}s}\t{2:12s}\t{3:12s}\t{4:12s}\t{5:12s}".format(namelen,'Parameter','Value','Analytic','FiniteDiff','Flux'))
		for name,val,a,fd in zip(self.parameter_names(),self.parameter_values(), a_grad, fd_grad):
			print("{1:<{0}s}\t{2:< 12.6g}\t{3:< 12.6g}\t{4:< 12.6g}\t{5:12s}".format(namelen,name,val,a,fd,flux_mag(fd,a)))

	def loglike_c(self):
		return self._get_estimation_statistics()[0]['log_like_constants']

	def estimate_scipy(self, method='Nelder-Mead', basinhopping=False, constraints=(), **kwargs):
		import scipy.optimize
		import datetime
		starttime = (datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds()
		if basinhopping:
			cb = lambda x, f, accept: print("{} <- {} ({})".format(f,",".join(str(i) for i in x), 'accepted' if accept else 'not accepted'))
			ret = scipy.optimize.basinhopping(
				self.negative_loglike,   # objective function
				self.parameter_values(), # initial values
				minimizer_kwargs=dict(method=method,args=(),jac=self.d_loglike,
										hess=None, hessp=None, bounds=None,
										constraints=constraints, tol=None, callback=None,),
				disp=True,
				callback=cb,
				**kwargs)
		else:
			ret = scipy.optimize.minimize(
				self.negative_loglike,   # objective function
				self.parameter_values(), # initial values
				args=(),
				method=method,
				jac=False, #? self.d_loglike,
				hess=None, hessp=None, bounds=None, constraints=constraints, tol=None, callback=print,
				options=dict(disp=True))
		endtime = (datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds()
		startfrac, startwhole = math.modf(starttime)
		endfrac, endwhole = math.modf(endtime)
		startfrac *= 1000000
		endfrac *= 1000000
		try:
			s = ret.success
		except AttributeError:
			s = False
			try:
				if basinhopping and 'success' in ret.message[0]:
					s = True
			except:
				pass
		if s or True:
			self.parameter_values(ret.x)
			self._set_estimation_statistics( -(ret.fun) )
			self._set_estimation_run_statistics(int(startwhole),int(startfrac),
												int(endwhole),int(endfrac),
												ret.nit,ret.message)
		return ret

	def analyze(self, reportfile=None, css=None, repair=None, est_args=None, est_tight=None, *arg, **kwargs):
		if reportfile is not None:
			htmlfile = XHTML(reportfile, *arg, **kwargs)
			
			xhead = XML_Builder("head")
			if self.title != '':
				xhead.title(self.title)
			xhead.style()
			if css is None:
				css = """
				.error_report {color:red; font-family:monospace;}
				table {border-collapse:collapse;}
				table, th, td {border: 1px solid #999999; padding:2px; font-family:monospace;}
				.statistics_bridge {text-align:center;}
				a.parameter_reference {font-style: italic; text-decoration: none}
				.strut2 {min-width:2in}
				"""
			xhead.data(css.replace('\n',' ').replace('\t',' '))
			xhead.end_style()
			htmlfile << xhead
		else:
			from .util import xhtml
			htmlfile = xhtml.Elem('div')

		if self.is_provisioned(False)<=0:
			self.provision()
		qc = self.db.queries.quality_check()
		
		clashes = numpy.nonzero( numpy.logical_and(self.Data("Choice"), ~self.Data("Avail")) )
		n_clashes = len(clashes[0])
		if n_clashes>0:
			self.clash = clashes
			if repair == '+':
				for i in zip(*clashes):
					self.DataEdit("Avail")[i] = True
				self.note("Model had {} cases where the chosen alternative is unavailable, these have been repaired by making it available".format(n_clashes))
			if repair == '-':
				for i in zip(*clashes):
					self.DataEdit("Choice")[i] = 0
				self.note("Model had {} cases where the chosen alternative is unavailable, these have been repaired by making it not chosen".format(n_clashes))
			else:
				raise LarchError("Model has {} cases where the chosen alternative is unavailable".format(n_clashes))
		
		if len(qc):
			self.note(qc)
		
		try:
			if est_tight is not None:
				self.estimate_tight(est_tight)
			elif est_args is None:
				self.estimate()
			else:
				self.estimate(est_args)
		except KeyboardInterrupt:
			pass

		htmlfile << self.report('*', style='xml')
		
		if reportfile is not None:
			try:
				htmlfile.dump()
				htmlfile._f.view()
			except AttributeError:
				pass
			return htmlfile.root
		else:
			return htmlfile


	def utility_full_constants(self):
		"Add a complete set of alternative specific constants"
		for code, name in self.db.alternatives()[1:]:
			self.utility.co("1",code,name)

	def __contains__(self, x):
		if x is None:
			return False
		if isinstance(x,category):
			for i in x.members:
				if i in self: return True
			return False
		if isinstance(x,pmath):
			return x.valid(self)
		from .roles import ParameterRef
		if isinstance(x,ParameterRef):
			return x.valid(self)
		if isinstance(x,rename):
			found = []
			if x.name in self:
				found.append(x.name)
			for i in x.members:
				if i in self:
					found.append(i)
			if len(found)==0:
				return False
			elif len(found)==1:
				return True
			else:
				raise LarchError("model contains "+(" and ".join(found)))
		return super().__contains__(x)

	def __getitem__(self, x):
		if isinstance(x,rename):
			x = x.find_in(self)
		return super().__getitem__(x)

	def __setitem__(self, x, val):
		if isinstance(x,rename):
			x = x.find_in(self)
		return super().__setitem__(x, val)


class _AllInTheFamily():

	def __init__(self, this, func):
		self.this_ = this
		self.func_ = func

	def __call__(self, *args, **kwargs):
		return [self.func_(i,*args, **kwargs) for i in self.this_]

	def list(self):
		if isinstance(self.func_, property):
			return [self.func_.fget(i) for i in self.this_]
		else:
			return [getattr(i, self.func_) for i in self.this_]

	def __getattr__(self, attr):
		if attr[-1]=="_":
			return super().__getattr__(attr)
		if isinstance(self.func_, property):
			return [getattr(self.func_.fget(i), attr) for i in self.this_]
		else:
			return [getattr(i, attr) for i in self.this_]

	def __setattr__(self, attr, value):
		if attr[-1]=="_":
			return super().__setattr__(attr,value)
		try:
			iterator = iter(value)
		except TypeError:
			multi = False
		else:
			multi = True if len(value)==len(self.this_) else False
		if multi:
			if isinstance(self.func_, property):
				for i,v in zip(self.this_, value):
					setattr(self.func_.fget(i), attr, v)
			else:
				for i,v in zip(self.this_, value):
					setattr(i, attr, v)
		else:
			if isinstance(self.func_, property):
				for i in self.this_:
					setattr(self.func_.fget(i), attr, value)
			else:
				for i in self.this_:
					setattr(i, attr, value)

class ModelFamily(list):

	def __init__(self, *args, **kwargs):
		self._name_map = {}
		list.__init__(self)
		for arg in args:
			self.add(arg)
		for key, arg in kwargs.items():
			self.add(arg, key)

	def add(self, arg, name=None):
		if isinstance(arg, (str,bytes)):
			try:
				self.append(Model.loads(arg))
			except LarchError:
				raise TypeError("family members must be Model objects (or loadable string or bytes)")
		elif isinstance(arg, Model):
			self.append(arg)
		else:
			raise TypeError("family members must be Model objects (or loadable string or bytes)")
		if name is not None:
			if isinstance(name, str):
				self._name_map[name] = len(self)-1
			else:
				raise TypeError("family names must be strings")

	def load(self, file, name=None):
		self.add(Model.load(file), name)

	def __getitem__(self, key):
		if isinstance(key, str):
			return self[self._name_map[key]]
		else:
			return super().__getitem__(key)

	def __setitem__(self,key,value):
		if isinstance(key, str):
			if key in self._name_map:
				slot = self._name_map[key]
				self[slot] = value
			else:
				self.add(value, key)
		else:
			super().__setitem__(key,value)

	def __contains__(self, key):
		return key in self._name_map

	def replicate(self, key, newkey=None):
		source = self[key]
		m = Model.copy(source)
		m.db = source.db
		self.add(m, newkey)
		return m

	def spawn(self, newkey=None):
		"Create a blank model using the same data"
		m = Model(self.db)
		self.add(m, newkey)
		return m

	def _set_db(self, db):
		for i in self:
			i.db = db

	def _get_db(self):
		for i in self:
			if i.db is not None:
				return i.db

	def _del_db(self):
		for i in self:
			del i.db

	db = property(_get_db, _set_db, _del_db)

	def constants_only_model(self, est=True, logger=False):
		m = self.spawn("constants_only")
		m.utility_full_constants()
		m.option.calc_std_errors = False
		m.logger(logger)
		if est:
			m.estimate()

	def all_option(self, opt, value=None):
		if value is None:
			return [i.option[opt] for i in self]
		else:
			for i in self:
				i.option[opt] = value

	def __getattr__(self,attr):
		return _AllInTheFamily(self, getattr(Model,attr))


#	def __getstate__(self):
#		import pickle, zlib
#		mods = [zlib.compress(pickle.dumps(i)) for i in self]
#		return (mods,self._name_map)
#		
#	def __setstate__(self, state):
#		for i in state[0]:
#			self.append(pickle.loads(zlib.decompress(i)))
#		self._name_map = state[1]

