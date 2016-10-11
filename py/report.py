

from .util.pmath import category
from . import Model
import os
from contextlib import contextmanager
import numpy
import math
from .core import LarchError
from .util.xhtml import XML_Builder

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





def multireport(models_or_filenames, params=(), ratios=(), *, filename=None, overwrite=False, spool=True, title=None, model_titles=None):
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
	model_titles : sequence of str
		A sequence of model titles to be used to replace (probably shorten) the model 
		titles used in the report.
	
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
	<link href='https://fonts.googleapis.com/css?family=Roboto:400,300,900,700,500italic|Roboto+Mono:400,300,400italic,700,700italic' rel='stylesheet' type='text/css'>
	<style>
	div.bounding {margin-left:auto; margin-right:auto; padding-left:20px; padding-right:20px; }
	h1 {font-family: "Book Antiqua", Palatino, serif; }
	table { border-collapse: collapse;  font-weight:400; text-align:center;  }
	th {border: 1px solid #999999; font-family: "Roboto", "Helvetica", "Arial", sans-serif; font-size:90%; font-weight:700;}
	table {border: 0; font-family: "Roboto", "Helvetica", "Arial", sans-serif; font-size:90%;}
	td {border: 1px solid #999999; font-family: "Roboto Mono", "Helvetica", "Arial", sans-serif; font-size:90%;}
	td:first-child, th:first-child { text-align:left; padding-left:5px;}
	td.dark, th.dark { background-color: #f0f0f0; padding: 3px; text-align: center; }
	td.light, th.light { background-color: #f8f8f8; padding: 3px; text-align: center; }
	td.parameter_category { background-color: #f4f4f4; font-style: italic; font-family: "Roboto","Helvetica", "Arial", sans-serif; font-weight:500;}
	td.table_category { background-color: #ffffff; font-weight: 900; font-family: "Roboto","Helvetica", "Arial", sans-serif;
		border-left:0; border-right:0; padding-top:20px;
	}
	th.emptyhead {border: 0;}
	tr:first-child > td.table_category { padding-top:5px; }
	td.tstat { font-size: 80%; font-weight: 300;}
	.larch_signature {font-size:80%; font-weight:100; font-style:italic; font-family: "Book Antiqua", Palatino, serif; }
	</style>
	"""

	def param_appears_in_at_least_one_model(p):
		if isinstance(p,category) and len(p.members)==0: return True
		for m in models:
			if p in m: return True
		return False

	with HTML(filename, head=head, overwrite=overwrite, spool=spool, title=title) as f:
		
		f._f.write('<div class="bounding">')
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
								f.write('<td class="{0}" colspan="2">{1}</td>'.format(shades[shade],p.strf(m)))
							else:
								f.write('<td class="{0}" colspan="2">{1}</td>'.format(shades[shade],"---"))

		with f.table():
			with f.block("thead"):
				with f.block("tr"):
					f.write('<th class="emptyhead"></th>')
					shade = 0
					for m_number,m in enumerate(models):
						shade ^= 1
						if m.title == "Untitled Model" and hasattr(m,"loaded_from"):
							title = m.loaded_from
						else:
							title = m.title
						if model_titles is not None:
							try:
								title = model_titles[m_number]
							except:
								pass
						f.write('<th colspan="2" class="{0} reportmodel{2}">{1}</th>'.format(shades[shade],title,m_number))
			with f.block("tbody"):
				# PARAMETER ESTIMATES
				with f.block("tr"):
					f.write('<td class="table_category">Parameter Estimates</td>')
					shade = 0
					for m_number,m in enumerate(models):
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
					for m_number,m in enumerate(models):
						shade ^= 1
						f.write('<td colspan="2" class="{0}">{1:0.6g}</td>'.format(shades[shade],m.loglike()))
				with f.tr():
					f.write('<td>Log Likelihood at Constants</td>')
					shade = 0
					for m_number,m in enumerate(models):
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
					for m_number,m in enumerate(models):
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
					for m_number,m in enumerate(models):
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
					for m_number,m in enumerate(models):
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
					for m_number,m in enumerate(models):
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
					for m_number,m in enumerate(models):
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
				if ratios is not None and len(ratios)>0:
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

		xsign = XML_Builder("div", {'class':'larch_signature'})
		from . import longversion as version
		from .util.img import favicon
		import time
		xsign.start('p')
		xsign.start('img', {'width':"14", 'height':"14", 'src':"data:image/png;base64,{}".format(favicon), 'style':'position:relative;top:2px;' })
		xsign.end('img')
		xsign.data(" Larch {}".format(version))
		xsign.simple('br')
		xsign.data("multireport generated on ")
		xsign.simple('br', attrib={'class':'noprint'})
		xsign.data(time.strftime("%A %d %B %Y "))
		xsign.simple('br', attrib={'class':'noprint'})
		xsign.data(time.strftime("%I:%M:%S %p"))
		xsign.end('p')
		xsign.close()
		f._f.write(xsign.dumps())

		f._f.write('</div>')
		return f.dump()





from .util.xhtml import XHTML, XML_Builder



def multireport_xhtml(models_or_filenames, params=(), ratios=(), *, filename=None,
                      overwrite=False, spool=True, title=None, model_titles=None,
					  css=""):
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
	model_titles : sequence of str
		A sequence of model titles to be used to replace (probably shorten) the model 
		titles used in the report.
	
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
	

	css = """
	@import url(https://fonts.googleapis.com/css?family=Roboto:400,700,500italic|Roboto+Mono:300,400,700,100);

	h1 {font-family: "Book Antiqua", Palatino, serif; }
	table { border-collapse: collapse;  font-weight:400; text-align:center;  }
	th {border: 1px solid #999999; font-family: "Roboto", "Helvetica", "Arial", sans-serif; font-size:90%; font-weight:700;}
	table {border: 0; font-family: "Roboto", "Helvetica", "Arial", sans-serif; font-size:90%;}
	td {border: 1px solid #999999; font-family: "Roboto Mono", "Helvetica", "Arial", sans-serif; font-size:90%;}
	td:first-child, th:first-child { text-align:left; padding-left:5px;}
	td.dark, th.dark { background-color: #ececec; padding: 3px; text-align: center; }
	td.light, th.light { background-color: #f8f8f8; padding: 3px; text-align: center; }
	td.parameter_category { background-color: #f2f2f2; font-style: italic; font-family: "Roboto","Helvetica", "Arial", sans-serif; font-weight:500;}
	td.table_category { background-color: #ffffff; font-weight: 900; font-family: "Roboto","Helvetica", "Arial", sans-serif;
		border-left:0; border-right:0; padding-top:20px;
	}
	th.emptyhead {border: 0;}
	tr:first-child > td.table_category { padding-top:5px; }
	td.tstat { font-size: 70%; font-weight: 100;}
	.larch_signature {font-size:80%; font-weight:100; font-style:italic; font-family: "Book Antiqua", Palatino, serif; }
	"""+css

	def param_appears_in_at_least_one_model(p):
		if isinstance(p,category) and len(p.members)==0: return True
		for m in models:
			if p in m: return True
		return False

	xhtm = XHTML(filename, extra_css=css, overwrite=overwrite, spool=spool)
	xhtm.toc_color = 'night'

	xhtm.title.text = title

	f = XML_Builder()

	if title:
		f.h1(title)
	
	shades = ['light','dark',]
	
	def write_param_row(p):
		if p is None: return
		if param_appears_in_at_least_one_model(p):
			if isinstance(p,category):
				with f.tr_:
					#f.td(p.name, {'colspan':str(len(models)*2+1), 'class':"parameter_category"})
					f.start("td", {'colspan':str(len(models)*2+1), 'class':"parameter_category"})
					f.anchor_auto_toc(p.name, '3')
					f.data(p.name)
					f.end("td")
				for subp in p.members:
					write_param_row(subp)
			else:
				with f.tr_:
					f.td(str(p))
					shade = 0
					for m_number,m in enumerate(models):
						shade ^= 1
						if p in m:
							m_p = m.parameter_wide(p)
							tstat = m_p.t_stat
							value = m_p.value
							try:
								value = "{:0.6g}".format(value)
							except (ValueError, TypeError):
								value = str(value)
							try:
								tstat = "{:0.3g}".format(tstat)
							except (ValueError, TypeError):
								tstat = str(tstat)
							f.td(value, {'class':"{} reportmodel{}".format(shades[shade],m_number)})
							f.td(tstat, {'class':"{} reportmodel{}".format(shades[shade],m_number)})
						else:
							f.td("---", {'class':"{} reportmodel{}".format(shades[shade],m_number)})
							f.td("---", {'class':"{} reportmodel{}".format(shades[shade],m_number)})

	def write_factor_row(p):
		if p is None: return
		if param_appears_in_at_least_one_model(p):
			if isinstance(p,category):
				with f.tr_:
					f.td(p.name, {'colspan':str(len(models)*2+1), 'class':"parameter_category"})
				for subp in p.members:
					write_factor_row(subp)
			else:
				with f.tr_:
					f.td(p.getname())
					shade = 0
					for m_number,m in enumerate(models):
						shade ^= 1
						if p in m:
							f.td(p.strf(m), {'colspan':"2", 'class':"{} reportmodel{}".format(shades[shade],m_number)})
						else:
							f.td("---", {'colspan':"2", 'class':"{} reportmodel{}".format(shades[shade],m_number)})

	f.start('table', {'class':'floatinghead'})

	with f.block("thead"):
		with f.block("tr"):
			f.th("", {'class':"emptyhead"})
			shade = 0
			for m_number,m in enumerate(models):
				shade ^= 1
				if m.title == "Untitled Model" and hasattr(m,"loaded_from"):
					title = m.loaded_from
				else:
					title = m.title
				if model_titles is not None:
					try:
						title = model_titles[m_number]
					except:
						pass
				if hasattr(m,"loaded_from"):
					f.start('th', attrib={'colspan':'2', 'class':"{} reportmodel{}".format(shades[shade],m_number)})
					try:
						href = os.path.basename(m.loaded_from)
					except:
						f.data(title)
					else:
						f.start('a', attrib={'href':href})
						f.data(title)
						f.end('a')
					f.end('th')
				else:
					f.th(title, {'colspan':'2', 'class':"{} reportmodel{}".format(shades[shade],m_number)})
	with f.block("tbody"):
		# PARAMETER ESTIMATES
		with f.block("tr"):
			#f.td("Parameter Estimates", {'class':'table_category'})
			f.start("td",{'class':'table_category'})
			f.anchor_auto_toc("Parameter Estimates", "2")
			f.data("Parameter Estimates")
			f.end("td")
			shade = 0
			for m_number,m in enumerate(models):
				shade ^= 1
				f.td("Estimate",{'class':'table_category reportmodel{}'.format(m_number)})
				f.td("(t\u2011Statistic)",{'class':'table_category tstat reportmodel{}'.format(m_number)}) # non-breaking hyphen
		for p in params:
			write_param_row(p)
		if len(params)>0 and len(unlisted_parameters)>0:
			write_param_row(category("Other Parameters"))
		for p in unlisted_parameters:
			write_param_row(p)
		# MODEL STATISTICS
		with f.tr_:
			#f.td("Model Statistics", {'colspan':str(len(models)*2+1), 'class':'table_category'})
			f.start("td",{'colspan':str(len(models)*2+1), 'class':'table_category'})
			f.anchor_auto_toc("Model Statistics", "2")
			f.data("Model Statistics")
			f.end("td")
		with f.tr_:
			f.td('Log Likelihood')
			shade = 0
			for m_number,m in enumerate(models):
				shade ^= 1
				es = m._get_estimation_statistics()
				ll = es[0]['log_like']
				f.td("{:0.6g}".format(ll), {'colspan':"2", 'class':"{} reportmodel{}".format(shades[shade],m_number)})
		with f.tr_:
			f.td('Log Likelihood at Constants')
			shade = 0
			for m_number,m in enumerate(models):
				shade ^= 1
				es = m._get_estimation_statistics()
				ll = es[0]['log_like']
				llc = es[0]['log_like_constants']
				if not math.isnan(llc):
					f.td('{:0.6g}'.format(llc), {'colspan':"2", 'class':"{} reportmodel{}".format(shades[shade],m_number)})
				else:
					f.td('n/a', {'colspan':"2", 'class':"{} reportmodel{}".format(shades[shade],m_number)})
		with f.tr_:
			f.td('Log Likelihood at Null Parameters')
			shade = 0
			for m_number,m in enumerate(models):
				shade ^= 1
				es = m._get_estimation_statistics()
				ll = es[0]['log_like']
				llz = es[0]['log_like_null']
				if not math.isnan(llz):
					f.td('{:0.6g}'.format(llz), {'colspan':"2", 'class':"{} reportmodel{}".format(shades[shade],m_number)})
				else:
					f.td('n/a', {'colspan':"2", 'class':"{} reportmodel{}".format(shades[shade],m_number)})
		with f.tr_:
			f.td('Log Likelihood with No Model')
			shade = 0
			for m_number,m in enumerate(models):
				shade ^= 1
				es = m._get_estimation_statistics()
				ll = es[0]['log_like']
				llz = es[0]['log_like_nil']
				if not math.isnan(llz):
					f.td('{:0.6g}'.format(llz), {'colspan':"2", 'class':"{} reportmodel{}".format(shades[shade],m_number)})
				else:
					f.td('n/a', {'colspan':"2", 'class':"{} reportmodel{}".format(shades[shade],m_number)})
		with f.tr_:
			f.td('Rho Squared vs. Constants')
			shade = 0
			for m_number,m in enumerate(models):
				shade ^= 1
				es = m._get_estimation_statistics()
				ll = es[0]['log_like']
				llc = es[0]['log_like_constants']
				if not math.isnan(llc):
					rsc = 1.0-(ll/llc)
					f.td('{:0.4g}'.format(rsc), {'colspan':"2", 'class':"{} reportmodel{}".format(shades[shade],m_number)})
				else:
					f.td('n/a', {'colspan':"2", 'class':"{} reportmodel{}".format(shades[shade],m_number)})
		with f.tr_:
			f.td('Rho Squared vs. Null Parameters')
			shade = 0
			for m_number,m in enumerate(models):
				shade ^= 1
				es = m._get_estimation_statistics()
				ll = es[0]['log_like']
				llz = es[0]['log_like_null']
				if not math.isnan(llz):
					rsz = 1.0-(ll/llz)
					f.td('{:0.4g}'.format(rsz), {'colspan':"2", 'class':"{} reportmodel{}".format(shades[shade],m_number)})
				else:
					f.td('n/a', {'colspan':"2", 'class':"{} reportmodel{}".format(shades[shade],m_number)})
		with f.tr_:
			f.td('Rho Squared vs. No Model')
			shade = 0
			for m_number,m in enumerate(models):
				shade ^= 1
				es = m._get_estimation_statistics()
				ll = es[0]['log_like']
				llz = es[0]['log_like_nil']
				if not math.isnan(llz):
					rsz = 1.0-(ll/llz)
					f.td('{:0.4g}'.format(rsz), {'colspan':"2", 'class':"{} reportmodel{}".format(shades[shade],m_number)})
				else:
					f.td('n/a', {'colspan':"2", 'class':"{} reportmodel{}".format(shades[shade],m_number)})

				
		# RATIOS
		if ratios is not None and len(ratios)>0:
			with f.tr_:
				#f.td('Calculated Factors', {'colspan':str(len(models)*2+1), 'class':"table_category"})
				f.start("td",{'colspan':str(len(models)*2+1), 'class':'table_category'})
				f.anchor_auto_toc("Calculated Factors", "2")
				f.data("Calculated Factors")
				f.end("td")

			for pm in ratios:
				write_factor_row(pm)
	f.end('table')
#	xsign = XML_Builder("div", {'class':'larch_signature'})
#	from .version import version
#	from .util.img import favicon
#	import time
#	xsign.start('p')
#	xsign.start('img', {'width':"14", 'height':"14", 'src':"data:image/png;base64,{}".format(favicon), 'style':'position:relative;top:2px;' })
#	xsign.end('img')
#	xsign.data(" Larch {}".format(version))
#	xsign.simple('br')
#	xsign.data("multireport generated on ")
#	xsign.simple('br', attrib={'class':'noprint'})
#	xsign.data(time.strftime("%A %d %B %Y "))
#	xsign.simple('br', attrib={'class':'noprint'})
#	xsign.data(time.strftime("%I:%M:%S %p"))
#	xsign.end('p')
#	xsign.close()
#	f._f.write(xsign.dumps())

	xhtm << f
	if filename is None:
		from .util.temporaryfile import TemporaryHtml
		TemporaryHtml(content=xhtm.dump())
	else:
		xhtm.dump()
	return xhtm


