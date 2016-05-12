

from .util.pmath import category
from . import Model
import os
from contextlib import contextmanager
import numpy
import math
from .core import LarchError


from .util.pmath import category as Category
from .util.pmath import rename as Rename


class HTML():
	def __init__(self, filename=None, *, overwrite=False, spool=True, head=None, title=None):
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
				filename_ext = ".html"
			filename = filename+filename_ext
			filemaker = lambda: open(filename, 'w')
		self._f = filemaker()
		self._f.write("<html>\n")
		self._f.write("<head>\n")
		if title:
			self._f.write("<title>{}</title>\n".format(title))
		if head:
			self._f.write(head)
			self._f.write("\n")
		self._f.write("</head>\n<body>\n")

	def dump(self):
		self._f.flush()
		try:
			return self._f.getvalue() + "</body>\n</html>\n"
		except AttributeError:
			return

	def write(self,*arg,**kwarg):
		self._f.write(*arg,**kwarg)

	def close(self,*arg,**kwarg):
		if not self._f.closed:
			self._f.write("</body>\n</html>\n")
			self._f.close(*arg,**kwarg)
	
	def __enter__(self):
		return self

	def __exit__(self, type, value, traceback):
		self.close()

	def smallblock(self,tag,content,cls=None,id=None):
		tcls = "" if cls is None else ' class="{}"'.format(cls)
		tid  = "" if id  is None else ' id="{}"'.format(id)
		self._f.write("<{}{}{}>{}</{}>\n".format(tag,tcls,tid,content,tag))

	h1 = lambda self,content,cls=None,id=None: self.smallblock("h1",content,cls,id)
	h2 = lambda self,content,cls=None,id=None: self.smallblock("h2",content,cls,id)
	h3 = lambda self,content,cls=None,id=None: self.smallblock("h3",content,cls,id)
	h4 = lambda self,content,cls=None,id=None: self.smallblock("h4",content,cls,id)

	bold = lambda self,content,cls=None,id=None: self.smallblock("b",content,cls,id)
	ital = lambda self,content,cls=None,id=None: self.smallblock("i",content,cls,id)

	@contextmanager
	def block(self,tag,cls=None,id=None, content=None):
		self._f.write('<{}'.format(tag))
		if cls is not None:
			self._f.write(' class="{}"'.format(cls))
		if id is not None:
			self._f.write(' id="{}"'.format(id))
		self._f.write('>')
		if content is not None:
			self._f.write(content)
		yield
		self._f.write('</{}>'.format(tag))

	table = lambda self,cls=None,id=None: self.block("table",cls,id)
	tr    = lambda self,cls=None,id=None: self.block("tr",cls,id)
	td = lambda self,content,cls=None,id=None: self.smallblock("td",content,cls,id)
	th = lambda self,content,cls=None,id=None: self.smallblock("th",content,cls,id)



class parameter_category(str): pass





def multireport(models_or_filenames, params=(), ratios=[], *, filename=None, overwrite=False, spool=True, title=None):
	"""
	Generate a combined report on a number of (probably related) models.
	
	Parameters
	----------
	models_or_filenames : iterable
		A list of models, given either as `str` containing a path to a file that
		can be loaded as a :class:`Model`, or pre-loaded :class:`Model` objects.
	params : iterable
		An ordered list of parameters names and/or categories. If given,
		this list will be used to order the resulting table.
	ratios : iterable
		An ordered list of factors to evaluate.
	
	Other Parameters
	----------------
	filename : str
		The file into which to save the multireport
	overwrite : bool
		If `filename` exists, should it be overwritten (default False).
	spool : bool
		If `filename` exists, should the report file be spooled into a 
		similar filename.
	title : str
		An optional title for the report.
	
	"""
	
	
	models = []
	for m in models_or_filenames:
		if isinstance(m, str) and os.path.isfile(m):
			models.append(Model.load(m))
		elif isinstance(m, Model):
			models.append(m)
		else:
			print("Failed to load {}".format(m))

	listed_parameters = set([p for p in params if not isinstance(p,category)])
	for p in params:
		if isinstance(p,category):
			listed_parameters.update( p.complete_members() )
	all_parameters = set()
	for m in models:
		all_parameters.update(m.parameter_names())
	unlisted_parameters = all_parameters - listed_parameters
	

	head = """
	<style>
	table { border-collapse: collapse; font-family: "Helvetica", "Arial", sans-serif; }
	td.dark, th.dark { background-color: #cccccc; padding: 3px; text-align: center; }
	td.light, th.light { background-color: #eeeeee; padding: 3px; text-align: center; }
	td.parameter_category { background-color: #dddddd; font-style: italic; }
	td.table_category { background-color: #333333; color: #FFFFFF; font-weight: 900; }
	td.tstat { font-size: 70%; }
	</style>
	"""

	def param_appears_in_at_least_one_model(p):
		if isinstance(p,category) and len(p.members)==0: return True
		for m in models:
			if p in m: return True
		return False

	with HTML(filename, head=head, overwrite=overwrite, spool=spool, title=title) as f:
		
		if title: f.h1(title)
		
		shades = ['dark','light']
		
		def write_param_row(p):
			if p is None: return
			if param_appears_in_at_least_one_model(p):
				if isinstance(p,category):
					with f.tr(): f.write('<td colspan="{0}" class="parameter_category">{1}</td>'.format(len(models)*2+1,p.name))
					for subp in p.members:
						write_param_row(subp)
				else:
					with f.tr():
						f.write('<td>{}</td>'.format(p))
						shade = 0
						for m in models:
							shade ^= 1
							if p in m:
								m_p = m.parameter_wide(p)
								tstat = m_p.t_stat
								value = m_p.value
								try:
									value = "{:0.6g}".format(value)
								except ValueError:
									value = str(value)
								try:
									tstat = "{:0.3g}".format(tstat)
								except ValueError:
									tstat = str(tstat)
								f.write('<td class="{0}">{1}</td><td class="{0} tstat">({2})</td>'.format(shades[shade],value,tstat))
							else:
								f.write('<td class="{0}">{1}</td><td class="{0} tstat">({1})</td>'.format(shades[shade],"---"))

		def write_factor_row(p):
			if p is None: return
			if param_appears_in_at_least_one_model(p):
				if isinstance(p,category):
					with f.tr(): f.write('<td colspan="{0}" class="parameter_category">{1}</td>'.format(len(models)*2+1,p.name))
					for subp in p.members:
						write_factor_row(subp)
				else:
					with f.tr():
						f.write('<td>{}</td>'.format(p.getname()))
						shade = 0
						for m in models:
							shade ^= 1
							if p in m:
								f.write('<td class="{0}" colspan="2">{1}</td>'.format(shades[shade],p.str(m)))
							else:
								f.write('<td class="{0}" colspan="2">{1}</td>'.format(shades[shade],"---"))

		with f.table():
			with f.block("thead"):
				with f.block("tr"):
					f.write('<th></th>')
					shade = 0
					for m in models:
						shade ^= 1
						if m.title == "Untitled Model" and hasattr(m,"loaded_from"):
							title = m.loaded_from
						else:
							title = m.title
						f.write('<th colspan="2" class="{0}">{1}</th>'.format(shades[shade],title))
			with f.block("tbody"):
				# PARAMETER ESTIMATES
				with f.block("tr"):
					f.write('<td class="table_category">Parameter Estimates</td>')
					shade = 0
					for m in models:
						shade ^= 1
						f.write('<td class="table_category">Estimate</td><td class="table_category tstat">(t-Statistic)</td>'.format(shades[shade]))
				for p in params:
					write_param_row(p)
				if len(params)>0 and len(unlisted_parameters)>0:
					write_param_row(category("Other Parameters"))
				for p in unlisted_parameters:
					write_param_row(p)
				# MODEL STATISTICS
				with f.tr():
					f.write('<td colspan="{0}" class="table_category">Model Statistics</td>'.format(len(models)*2+1))
				with f.tr():
					f.write('<td>Log Likelihood</td>')
					shade = 0
					for m in models:
						shade ^= 1
						f.write('<td colspan="2" class="{0}">{1:0.6g}</td>'.format(shades[shade],m.loglike()))
				with f.tr():
					f.write('<td>Log Likelihood at Constants</td>')
					shade = 0
					for m in models:
						shade ^= 1
						es = m._get_estimation_statistics()
						ll = es[0]['log_like']
						llc = es[0]['log_like_constants']
						if not math.isnan(llc):
							f.write('<td colspan="2" class="{0}">{1:0.6g}</td>'.format(shades[shade],llc))
						else:
							f.write('<td colspan="2" class="{0}">n/a</td>'.format(shades[shade],))
				with f.tr():
					f.write('<td>Log Likelihood at Null Parameters</td>')
					shade = 0
					for m in models:
						shade ^= 1
						es = m._get_estimation_statistics()
						ll = es[0]['log_like']
						llz = es[0]['log_like_null']
						if not math.isnan(llz):
							f.write('<td colspan="2" class="{0}">{1:0.6g}</td>'.format(shades[shade],llz))
						else:
							f.write('<td colspan="2" class="{0}">n/a</td>'.format(shades[shade],))
				with f.tr():
					f.write('<td>Log Likelihood with No Model</td>')
					shade = 0
					for m in models:
						shade ^= 1
						es = m._get_estimation_statistics()
						ll = es[0]['log_like']
						llz = es[0]['log_like_nil']
						if not math.isnan(llz):
							f.write('<td colspan="2" class="{0}">{1:0.6g}</td>'.format(shades[shade],llz))
						else:
							f.write('<td colspan="2" class="{0}">n/a</td>'.format(shades[shade],))
				with f.tr():
					f.write('<td>Rho Squared vs. Constants</td>')
					shade = 0
					for m in models:
						shade ^= 1
						es = m._get_estimation_statistics()
						ll = es[0]['log_like']
						llc = es[0]['log_like_constants']
						if not math.isnan(llc):
							rsc = 1.0-(ll/llc)
							f.write('<td colspan="2" class="{0}">{1:0.4g}</td>'.format(shades[shade],rsc))
						else:
							f.write('<td colspan="2" class="{0}">n/a</td>'.format(shades[shade],))
				with f.tr():
					f.write('<td>Rho Squared vs. Null Parameters</td>')
					shade = 0
					for m in models:
						shade ^= 1
						es = m._get_estimation_statistics()
						ll = es[0]['log_like']
						llz = es[0]['log_like_null']
						if not math.isnan(llz):
							rsz = 1.0-(ll/llz)
							f.write('<td colspan="2" class="{0}">{1:0.4g}</td>'.format(shades[shade],rsz))
						else:
							f.write('<td colspan="2" class="{0}">n/a</td>'.format(shades[shade],))
				with f.tr():
					f.write('<td>Rho Squared vs. No Model</td>')
					shade = 0
					for m in models:
						shade ^= 1
						es = m._get_estimation_statistics()
						ll = es[0]['log_like']
						llz = es[0]['log_like_nil']
						if not math.isnan(llz):
							rsz = 1.0-(ll/llz)
							f.write('<td colspan="2" class="{0}">{1:0.4g}</td>'.format(shades[shade],rsz))
						else:
							f.write('<td colspan="2" class="{0}">n/a</td>'.format(shades[shade],))

						
				# RATIOS
				with f.tr():
					f.write('<td colspan="{0}" class="table_category">Calculated Factors</td>'.format(len(models)*2+1))
				for pm in ratios:
					write_factor_row(pm)
				
				
				
				
#				for name,numerator,denominator,factor,format in ratios:
#					with f.tr():
#						f.write('<td>{}</td>'.format(name))
#						shade = 0
#						for m in models:
#							shade ^= 1
#							try:
#								if isinstance(numerator, tuple):
#									enumerator = m.param_sum(*numerator)
#								elif numerator in m:
#									enumerator = m[numerator].value
#								else: raise LarchError
#								if isinstance(denominator, tuple):
#									edenominator = m.param_sum(*denominator)
#								elif denominator in m:
#									edenominator = m[denominator].value
#								else: raise LarchError
#								i = format.format(numpy.float64(factor)*enumerator/edenominator)
#								f.write('<td colspan="2" class="{0}">{1}</td>'.format(shades[shade],i))
#							except LarchError:
#								f.write('<td colspan="2" class="{0}">---</td>'.format(shades[shade]))
		return f.dump()



