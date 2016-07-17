from . import docx, latex, pdf, txt, xhtml, alogit
import math
from ..util.xhtml import XHTML, XML_Builder, Elem
from ..util.pmath import category, pmath, rename
from ..core import LarchError
import os
import pandas
import numpy
import itertools




class AbstractReportTable():
	def __init__(self, columns=(), col_classes=()):
		self.df = pandas.DataFrame(columns=columns, index=pandas.RangeIndex())
		self.col_classes = col_classes
		self.n_thead_rows = 1
		self.silent_first_col_break = False
		self._col_width = None
	def add_blank_row(self):
		self.df.loc[len(self.df)] = None
		self._col_width = None
	def encode_cell_value(self, value, attrib=None, tag='td'):
		if pandas.isnull(value):
			return None
		if attrib is None:
			attrib = {}
		if isinstance(value, Elem):
			value.tag = tag
			for k,v in attrib.items():
				if k=='class':
					v1 = value.get(k,None)
					if v1 is not None and v not in v1:
						v = "{} {}".format(v,v1)
				value.set(k,v)
			return value
		return Elem(tag=tag, text=str(value), attrib=attrib)
	def encode_head_value(self, value, attrib=None):
		return self.encode_cell_value(value, attrib=attrib, tag='th')
	def addrow_seq_of_strings(self, str_content, attrib={}):
		if len(str_content) > len(self.df.columns):
			raise TypeError("Insufficient columns for new row")
		self.add_blank_row()
		for n,s in enumerate(str_content):
			self.df.iloc[-1,n] = self.encode_cell_value(s)
	def addrow_map_of_strings(self, str_content, attrib={}):
		self.add_blank_row()
		rowix = self.df.index[-1]
		for key,val in str_content.items():
			self.df.loc[rowix, key] = self.encode_cell_value(val)

	def set_lastrow_loc(self, colname, val, attrib=None):
		rowix = self.df.index[-1]
		self.df.loc[rowix, colname] = self.encode_cell_value(val, attrib)
		self._col_width = None
	def set_lastrow_iloc(self, colnum, val, attrib=None):
		self.df.iloc[-1, colnum] = self.encode_cell_value(val, attrib)
		self._col_width = None
	def set_lastrow_iloc_nondupe(self, colnum, val, attrib=None):
		try:
			val_text = val.text
		except AttributeError:
			val_text = str(val)
		prev = -1
		prev_text = None
		try:
			while prev_text is None:
				prev -= 1
				prev_text = self.get_text_iloc(prev,colnum, missing=None)
			if prev_text!=val_text:
				raise NameError
		except (NameError, IndexError):
			self.df.iloc[-1, colnum] = self.encode_cell_value(val, attrib)
			self._col_width = None

	def __repr__(self):
		return self.__str__()

	def _dividing_line(self, leftend="+", rightend="+", splitpoint="+", linechar="─"):
		lines = [linechar*w for w in self.min_col_widths_()]
		return leftend+splitpoint.join(lines)+rightend

	def __str__(self):
		s = self._dividing_line(leftend='┌', rightend='┐', splitpoint='┬')+"\n"
		w = self.min_col_widths()
		for r,rvalue in enumerate(self.df.index):
			if (~pandas.isnull(self.df.iloc[r,:])).sum()==1:
				catflag = True
			else:
				catflag = False
			if r==self.n_thead_rows:
				s += self._dividing_line(leftend='├', rightend='┤', splitpoint='┼')+"\n"
#			elif catflag:
#				s += self._dividing_line(leftend='├', rightend='┤', splitpoint='┴')+"\n"
			startline = True
			s += "│"
			for c,cvalue in enumerate(self.df.columns):
				cellspan = self.cellspan_iloc(r,c)
				if cellspan != (0,0):
					cw = numpy.sum(w[c:c+cellspan[1]])+cellspan[1]-1
					if catflag:
						s = s[:-1]+ "╞ {1:═<{0}s}╡".format(cw-1,self.get_text_iloc(r,c)+" ")
					else:
						s += "{1:{0}s}│".format(cw,self.get_text_iloc(r,c))
					startline = False
				elif cellspan == (0,0) and startline:
					cw = w[c]
					s += "{1:{0}s}│".format(cw,"")
				else:
					startline = False
			s += "\n"
		s += self._dividing_line(leftend='└', rightend='┘', splitpoint='┴')
		return s
	
	def xml(self, table_attrib=None):
		table = Elem(tag='table', attrib=table_attrib or {})
		thead = table.put('thead')
		tbody = table.put('tbody')
		tfoot = table.put('tfoot')
		r = 0
		while r<len(self.df.index):
			if r<self.n_thead_rows:
				tr = thead.put('tr')
				celltag = 'th'
			else:
				tr = tbody.put('tr')
				celltag = 'td'
			span = 1
			for c in range(len(self.df.columns)):
				attrib = {}
				try:
					attrib['class'] = self.col_classes[c]
				except:
					pass
				td = self.encode_cell_value(  self.df.iloc[r,c] , attrib=attrib, tag=celltag )
				if td is None:
					try:
						tr[-1].get('colspan','1')
					except IndexError:
						tr << self.encode_cell_value(  "" , attrib=attrib, tag=celltag )
					else:
						span += 1
						tr[-1].set('colspan',str(span))
				else:
					tr << td
					span = 1
			r += 1
		return table
	def cellspan_iloc(self,r,c):
		try:
			if pandas.isnull(self.df.iloc[r,c]):
				return (0,0)
		except:
			print("r:",r,type(r))
			print("c:",c,type(c))
			raise
		x,y = 1,1
		while r+y < len(self.df.index) and numpy.all(pandas.isnull(self.df.iloc[r+y,:c+1])):
			y += 1
		while c+x < len(self.df.columns) and pandas.isnull(self.df.iloc[r,c+x]):
			x += 1
		return (y,x)

	def get_text_iloc(self,r,c,missing=""):
		if pandas.isnull(self.df.iloc[r,c]):
			return missing
		return self.df.iloc[r,c].text

	def min_col_widths(self):
		if self._col_width is not None:
			return self._col_width
		w = numpy.zeros(len(self.df.columns), dtype=int)
		for r,c in itertools.product(range(len(self.df.index)),range(len(self.df.columns))):
			if self.cellspan_iloc(r,c)[1]==1:
				w[c] = max(w[c], len(self.get_text_iloc(r,c)))
		for span in range(2,len(self.df.columns)):
			for r,c in itertools.product(range(len(self.df.index)),range(len(self.df.columns))):
				if self.cellspan_iloc(r,c)[1]==span:
					shortage = len(self.get_text_iloc(r,c))  -   (numpy.sum(w[c:c+span])+span-1)
					if shortage>0:
						w[c] += shortage
		self._col_width = w
		return w

	def min_col_widths_(self):
		w = self.min_col_widths()
		if self.silent_first_col_break:
			ww = w[1:].copy()
			ww[0] += 1+w[0]
			return ww
		else:
			return w


art = AbstractReportTable





class ModelReporter(docx.DocxModelReporter,
					latex.LatexModelReporter,
					xhtml.XhtmlModelReporter,
					pdf.PdfModelReporter,
					txt.TxtModelReporter,
					alogit.AlogitModelReporter,
					):

	def report(self, style, *args, filename=None, tempfile=False, **kwargs):
		"""
		Generate a model report.
		
		Larch is capable of generating reports in five basic formats:
		text, xhtml, docx, and latex.  This function serves as a
		pass-through, to call the report generator of the given style,
		and optionally to save the results to a file.
		
		Parameters
		----------
		style : ['txt', 'docx', 'latex', 'html', 'xml']
			The style of output.  Both 'html' and 'xml' will call the 
			xhtml generator, with the only difference being the output
			type, with html generating a finished [x]html document and
			xml generating an etree for further processing by Python 
			directly.
			
		Other Parameters
		----------------
		filename : str or None
			If given, then a new stack file is created by :func:`util.filemanager.open_stack`,
			the output is generated in this file, and the file-like object is 
			returned.
		tempfile : bool
			If True then a new :class:`util.TemporaryFile` is created,
			the output is generated in this file, and the file-like object is 
			returned.
		throw_exceptions : bool
			If True, exceptions are thrown if raised while generating the report. If 
			False (the default) tracebacks are printed directly into the report for 
			each section where an exception is raised.  Setting this to True can be
			useful for testing.


		Raises
		------
		LarchError
			If both filename and tempfile evaluate as True.
		
		"""
		if filename and tempfile:
			raise LarchError("only one of filename and tempfile can be given.")
		
		from ..util.filemanager import fileopen, filenext
		
		if style.lower()=='txt':
			rpt = self.txt_report(*args, **kwargs)
			if filename is None and tempfile==False:
				return rpt
			else:
				f = fileopen(None if tempfile else filename, mode='w')
				f.write(rpt)
				f.seek(0)
				return f
		if style.lower()=='xml':
			rpt = self.xhtml_report(*args, raw_xml=True, **kwargs)
			if filename is None and tempfile==False:
				return rpt
			else:
				f = XHTML("temp" if tempfile else filenext(filename), quickhead=self, **kwargs)
				f << rpt
				f.dump()
				return f
		if style.lower()=='html':
			rpt = self.xhtml_report(*args, raw_xml=False, **kwargs)
			if filename is None and tempfile==False:
				return rpt
			else:
				use_filename = None if tempfile else filename
				if use_filename is not None:
					base, ext = os.path.splitext(use_filename)
					if ext.casefold() not in ('.html','.xhtml','.htm'):
						use_filename = use_filename + '.html'
				f = fileopen(use_filename, mode='wb')
				f.write(rpt)
				f.flush()
				f.seek(0)
				try:
					f.view()
				except:
					pass
				return f



		if style.lower()=='docx':
			raise NotImplementedError
		if style.lower()=='latex':
			raise NotImplementedError

		# otherwise, the format style is not known
		raise LarchError("Format style '{}' is not known, must be one of ['txt', 'docx', 'latex', 'html', 'xml']".format(style))

	
	def report_d(self):
		"""Shortcut to a diagnostic html report."""
		y = self.report(style='html', tempfile=True, cats='D')
		y.view()
		return y

	def report_html(self, *args, cats='*', **kwargs):
		self.report(style='html', cats=cats, **kwargs)



	def art_params(self, groups=None, display_inital=False, **format):
		"""
		Generate a div element containing the model parameters in a table.
		
		Parameters
		----------
		groups : None or list
			An ordered list of parameters names and/or categories. If given,
			this list will be used to order the resulting table.
		display_inital : bool
			Should the initial values of the parameters (the starting point 
			for estimation) be included in the report. Defaults to False.
		
		Returns
		-------
		art
			An art containing the model parameters.
		
		"""
		# keys fix
		existing_format_keys = list(format.keys())
		for key in existing_format_keys:
			if key.upper()!=key: format[key.upper()] = format[key]
		if 'PARAM' not in format: format['PARAM'] = '< 10.4g'
		if 'TSTAT' not in format: format['TSTAT'] = ' 0.2f'
		# build table

		columns = ["Parameter", None, "Estimated Value", "Std Error", "t-Stat", "Null Value"]
		col_classes = ['param_label','param_label', 'estimated_value', 'std_err', 'tstat', 'null_value']
		if display_inital:
			columns.insert(1,"Initial Value")
			col_classes.insert(1,'initial_value')

		x = art(columns=columns, col_classes=col_classes)
		x.silent_first_col_break = True

		if groups is None and hasattr(self, 'parameter_groups'):
			groups = self.parameter_groups
		if groups is None:
			groups = ()
			
		## USING GROUPS
		listed_parameters = set([p for p in groups if not isinstance(p,category)])
		for p in groups:
			if isinstance(p,category):
				listed_parameters.update( p.complete_members() )
		unlisted_parameters_set = (set(self.parameter_names()) | set(self.alias_names())) - listed_parameters
		unlisted_parameters = []
		for pname in self.parameter_names():
			if pname in unlisted_parameters_set:
				unlisted_parameters.append(pname)
		for pname in self.alias_names():
			if pname in unlisted_parameters_set:
				unlisted_parameters.append(pname)
		n_cols_params = 6 if display_inital else 5
		
		def write_param_row(p, *, force=False):
			if p is None: return
			if force or (p in self) or (p in self.alias_names()):
				if isinstance(p,category):
					x.add_blank_row()
					x.set_lastrow_iloc(0, p.name, {'class':"parameter_category"})
					for subp in p.members:
						write_param_row(subp)
				else:
					if isinstance(p,(rename, )):
						p_name = p.name
					else:
						p_name = p
					x.add_blank_row()
					if "#" in p_name:
						p_name1, p_name2 = p_name.split("#",1)
						x.set_lastrow_iloc_nondupe(0, p_name1, )
						x.set_lastrow_iloc(1, p_name2, {'name':"param"+p_name2.replace("#","_hash_")})
					elif ":" in p_name:
						p_name1, p_name2 = p_name.split(":",1)
						x.set_lastrow_iloc_nondupe(0, p_name1, )
						x.set_lastrow_iloc(1, p_name2, {'name':"param"+p_name2.replace("#","_hash_")})
					else:
						x.set_lastrow_loc('Parameter', p_name, {'name':"param"+p_name.replace("#","_hash_")})
					self.art_single_parameter_resultpart(x,p, with_inital=display_inital, **format)
		
		x.addrow_seq_of_strings(columns)
		for p in groups:
			write_param_row(p)
		if len(groups)>0 and len(unlisted_parameters)>0:
			write_param_row(category("Other Parameters"),force=True)
		if len(unlisted_parameters)>0:
			for p in unlisted_parameters:
				write_param_row(p)
		return x


















	def art_single_parameter_resultpart(self, ART, p, *, with_inital=False,
										  with_stderr=True, with_tstat=True,
										  with_nullvalue=True, tstat_parens=False, **format):
		if p is None: return
		with_stderr = bool(with_stderr)
		with_tstat = bool(with_tstat)
		with_nullvalue = bool(with_nullvalue)
		#x = XML_Builder("div", {'class':"parameter_estimate"})
		x= ART
		if isinstance(p,(rename,str)):
			try:
				model_p = self[p]
			except KeyError:
				use_shadow_p = True
			else:
				use_shadow_p = False
			if use_shadow_p:
				# Parameter not found, try shadow_parameter
				try:
					str_p = str(p.find_in(self))
				except AttributeError:
					str_p = p
				shadow_p = self.shadow_parameter[str_p]
				if with_inital:
					x.set_lastrow_loc('Initial Value', "")
				shadow_p_value = shadow_p.value
				x.set_lastrow_loc('Estimated Value', "{:{PARAM}}".format(shadow_p.value, **format))
				#x.td("{:{PARAM}}".format(shadow_p.value, **format), {'class':'estimated_value'})
				x.set_lastrow_loc('Std Error', "{}".format(shadow_p.t_stat))
				#x.td("{}".format(shadow_p.t_stat), {'colspan':str(with_stderr+with_tstat+with_nullvalue), 'class':'tstat'})
			else:
				# Parameter found, use model_p
				if with_inital:
					x.set_lastrow_loc('Initial Value', "{:{PARAM}}".format(model_p.initial_value, **format))
					#x.td("{:{PARAM}}".format(model_p.initial_value, **format), {'class':'initial_value'})
				x.set_lastrow_loc('Estimated Value', "{:{PARAM}}".format(model_p.value, **format))
				#x.td("{:{PARAM}}".format(model_p.value, **format), {'class':'estimated_value'})
				if model_p.holdfast:
					x.set_lastrow_loc('Std Error', "fixed value")
					#x.td("fixed value", {'colspan':str(with_stderr+with_tstat), 'class':'notation'})
					x.set_lastrow_loc('Null Value', "{:{PARAM}}".format(model_p.null_value, **format))
					#x.td("{:{PARAM}}".format(model_p.null_value, **format), {'class':'null_value'})
				else:
					tstat_p = model_p.t_stat
					if isinstance(tstat_p,str):
						x.set_lastrow_loc('Std Error', "{}".format(tstat_p))
						#x.td("{}".format(tstat_p), {'colspan':str(with_stderr+with_tstat+with_nullvalue), 'class':'tstat'})
					elif tstat_p is None:
						x.set_lastrow_loc('Std Error', "{:{PARAM}}".format(model_p.std_err, **format))
						#x.td("{:{PARAM}}".format(model_p.std_err, **format), {'class':'std_err'})
						x.set_lastrow_loc('t-Stat', "None")
						#x.td("None", {'class':'tstat'})
						x.set_lastrow_loc('Null Value', "{:{PARAM}}".format(model_p.null_value, **format))
						#x.td("{:{PARAM}}".format(model_p.null_value, **format), {'class':'null_value'})
					else:
						x.set_lastrow_loc('Std Error', "{:{PARAM}}".format(model_p.std_err, **format))
						#x.td("{:{PARAM}}".format(model_p.std_err, **format), {'class':'std_err'})
						x.set_lastrow_loc('t-Stat', "{:{TSTAT}}".format(tstat_p, **format))
						#x.td("{:{TSTAT}}".format(tstat_p, **format), {'class':'tstat'})
						x.set_lastrow_loc('Null Value', "{:{PARAM}}".format(model_p.null_value, **format))
						#x.td("{:{PARAM}}".format(model_p.null_value, **format), {'class':'null_value'})
		#return x.close()
