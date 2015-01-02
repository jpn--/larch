
from .core import Model2, LarchError, _core
from .array import SymmetricArray
from .utilities import category, pmath, rename
import numpy
import os
from .xhtml import XHTML, XML
import math

class Model(Model2):

	def dir(self):
		for f in dir(self):
			print(" ",f)

	def param_sum(self,*arg):
		value = 0
		found_any = False
		for p in arg:
			if isinstance(p,str) and p in self:
				value += self[p].value
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
				value *= self[p].value
				found_any = True
			elif isinstance(p,(int,float)):
				value *= p
		if not found_any:
			raise LarchError("no parameters with any of these names: {}".format(str(arg)))
		return value

	def param_ratio(self, numerator, denominator):
		if isinstance(numerator,str):
			if numerator in self:
				value = self[numerator].value
			else:
				raise LarchError("numerator {} not found".format(numerator))
		elif isinstance(numerator,(int,float)):
			value = numerator
		if isinstance(denominator,str):
			if denominator in self:
				value /= self[denominator].value
			else:
				raise LarchError("denominator {} not found".format(denominator))
		elif isinstance(denominator,(int,float)):
			value /= denominator
		return value

	def _set_nest(self, *args):
		_core.Model2_nest_set(self, *args)
		self.freshen()
	nest = property(_core.Model2_nest_get, _set_nest)
	
	def _set_link(self, *args):
		_core.Model2_link_set(self, *args)
		self.freshen()
	link = property(_core.Model2_link_get, _set_link)


	def get_data_pointer(self):
		return self._ref_to_db

	db = property(get_data_pointer, Model2.change_data_pointer, Model2.delete_data_pointer)

	def load(self, filename="@@@"):
		if filename=="@@@" and isinstance(self,str):
			filename = self
			self = Model()
		inf = numpy.inf
		nan = numpy.nan
		with open(filename) as f:
			code = compile(f.read(), filename, 'exec')
			exec(code)
		self.loaded_from = filename
		return self

	def loads(self, content="@@@"):
		if content=="@@@" and isinstance(self,(str,bytes)):
			content = self
			self = Model()
		inf = numpy.inf
		nan = numpy.nan
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

	def note(self, comment):
		if not hasattr(self,"notes"): self.notes = []
		self.notes += ["{}".format(comment).replace("\n"," -- ")]

	def xhtml_title(self, **format):
		x = XML("div", {'class':"page_header"})
		if self.title != 'Untitled Model':
			x.h1(self.title)
		else:
			x.h1("A Model")
		return x.close()

	def xhtml_computed_factors(self, groups, ignore_na=False, **format):
		# keys fix
		existing_format_keys = list(format.keys())
		for key in existing_format_keys:
			if key.upper()!=key: format[key.upper()] = format[key]
		# build table
		x = XML("div", {'class':"computed_factors"})
		x.h2("Computed Factors")
		def write_factor_row(p):
				if not isinstance(p,category) and not (p in self) and not ignore_na:
					raise LarchError("factor contains bad components")
				if p in self:
					if isinstance(p,category):
						with x.block("tr"):
							x.td(p.name, {'colspan':str(2), 'class':"parameter_category"})
						for subp in p.members:
							write_factor_row(subp)
					else:
						with x.block("tr"):
							x.td('{}'.format(p.getname()))
							if p in self:
								x.td(p.str(self))
							else:
								x.td("---")
		with x.block("table"):
			for p in groups:
				write_factor_row(p)
		return x.close()

	def xhtml_params(self, groups=None, **format):
		# keys fix
		existing_format_keys = list(format.keys())
		for key in existing_format_keys:
			if key.upper()!=key: format[key.upper()] = format[key]
		if 'PARAM' not in format: format['PARAM'] = '< 12.4g'
		if 'TSTAT' not in format: format['TSTAT'] = '0.2f'
		# build table
		x = XML("div", {'class':"parameter_estimates"})
		x.h2("Model Parameter Estimates")
		if groups is None:
			
			footer = set()
			es = self._get_estimation_statistics()
			x.table()
			# Write headers
			x.thead
			x.th("Parameter")
			x.th("Initial Value", {'class':'initial_value'})
			x.th("Estimated Value", {'class':'estimated_value'})
			x.th("Std Error", {'class':'std_err'})
			x.th("t-Stat", {'class':'tstat'})
			x.th("Null Value", {'class':'null_value'})
			x.th("", {'class':'footnote_mark'}) # footnote markers
			x.end_thead
			
			x.tbody
			
			for p in self._get_parameter():
				x.tr
				try:
					tstat = (p['value'] - p['null_value']) / p['std_err']
				except ZeroDivisionError:
					tstat = float('nan')
				x.td(str(p['name']))
				x.td("{:{PARAM}}".format(p['initial_value'],**format), {'class':'initial_value'})
				x.td("{:{PARAM}}".format(p['value'],**format), {'class':'estimated_value'})
				x.td("{:{PARAM}}".format(p['std_err'],**format), {'class':'std_err'})
				x.td("{:{TSTAT}}".format(tstat,**format), {'class':'tstat'})
				x.td("{:{PARAM}}".format(p['null_value'],**format), {'class':'null_value'})
				if p['holdfast']:
					x.td("H", {'class':'footnote_mark'})
					footer.add("H")
				else:
					x.td("", {'class':'footnote_mark'})
				x.end_tr
			x.end_tbody
			
			if len(footer):
				x.tfoot
				x.tr
				if 'H' in footer:
					x.td("H: Parameters held fixed at their initial values (not estimated)", colspan=str(7))
				x.end_tr
				x.end_tfoot
			x.end_table()
		else:
			## USING GROUPS
			listed_parameters = set([p for p in groups if not isinstance(p,category)])
			for p in groups:
				if isinstance(p,category):
					listed_parameters.update( p.complete_members() )
			unlisted_parameters = set(self.parameter_names()) - listed_parameters
			n_cols_params = 3
			def write_param_row(p, *, force=False):
				if p is None: return
				if force or (p in self):
					if isinstance(p,category):
						with x.block("tr"):
							x.td(p.name, {'colspan':str(n_cols_params), 'class':"parameter_category"})
						for subp in p.members:
							write_param_row(subp)
					else:
						if isinstance(p,rename):
							with x.block("tr"):
								x.td('{}'.format(p.name))
								x.td("{:{PARAM}}".format(self[p].value, **format), {'class':'estimated_value'})
								x.td("{:{PARAM}}".format(self[p].std_err, **format), {'class':'std_err'})
								x.td("{:{TSTAT}}".format(self[p].t_stat(), **format), {'class':'tstat'})
						else:
							with x.block("tr"):
								x.td('{}'.format(p))
								x.td("{:{PARAM}}".format(self[p].value, **format), {'class':'estimated_value'})
								x.td("{:{PARAM}}".format(self[p].std_err, **format), {'class':'std_err'})
								x.td("{:{TSTAT}}".format(self[p].t_stat(), **format), {'class':'tstat'})
			with x.block("table"):
				with x.block("thead"):
					# PARAMETER ESTIMATES
					with x.block("tr"):
						x.th('Parameter')
						x.th('Estimate')
						x.th('t-Statistic', {'class':'tstat'})
				with x.block("tbody"):
					for p in groups:
						write_param_row(p)
					if len(groups)>0 and len(unlisted_parameters)>0:
						write_param_row(category("Other Parameters"),force=True)
					if len(unlisted_parameters)>0:
						for p in unlisted_parameters:
							write_param_row(p)
		return x.close()
	xhtml_param = xhtml_parameters = xhtml_params

	# Model Estimation Statistics
	def xhtml_ll(self,**format):
		existing_format_keys = list(format.keys())
		for key in existing_format_keys:
			if key.upper()!=key: format[key.upper()] = format[key]
		if 'LL' not in format: format['LL'] = '0.2f'
		if 'RHOSQ' not in format: format['RHOSQ'] = '0.3f'
	
		es = self._get_estimation_statistics()
		x = XML("div", {'class':"statistics"})
		x.h2("Model Estimation Statistics")

		x.table
		ll = es[0]['log_like']
		if not math.isnan(ll):
			x.tr
			x.td("Log Likelihood at Convergence")
			x.td("{0:{LL}}".format(ll,**format))
			x.end_tr
		llc = es[0]['log_like_constants']
		if not math.isnan(llc):
			x.tr
			x.td("Log Likelihood at Constants")
			x.td("{0:{LL}}".format(llc,**format))
			x.end_tr
		llz = es[0]['log_like_null']
		if not math.isnan(llz):
			x.tr
			x.td("Log Likelihood at Null Parameters")
			x.td("{0:{LL}}".format(llz,**format))
			x.end_tr
		ll0 = es[0]['log_like_nil']
		if not math.isnan(ll0):
			x.tr
			x.td("Log Likelihood with No Model")
			x.td("{0:{LL}}".format(ll0,**format))
			x.end_tr
		if (not math.isnan(llz) or not math.isnan(llc) or not math.isnan(ll0)) and not math.isnan(ll):
			x.tr({'class':"top_rho_sq"})
			if not math.isnan(llc):
				rsc = 1.0-(ll/llc)
				x.td("Rho Squared w.r.t. Constants")
				x.td("{0:{RHOSQ}}".format(rsc,**format))
				x.end_tr
				if not math.isnan(llz) or not math.isnan(ll0): x.tr
			if not math.isnan(llz):
				rsz = 1.0-(ll/llz)
				x.td("Rho Squared w.r.t. Null Parameters")
				x.td("{0:{RHOSQ}}".format(rsz,**format))
				x.end_tr
				if not math.isnan(ll0): x.tr
			if not math.isnan(ll0):
				rs0 = 1.0-(ll/ll0)
				x.td("Rho Squared w.r.t. No Model")
				x.td("{0:{RHOSQ}}".format(rs0,**format))
				x.end_tr
		x.end_table
		return x.close()

	def xhtml_latest(self,**format):
		from .utilities import format_seconds
		existing_format_keys = list(format.keys())
		for key in existing_format_keys:
			if key.upper()!=key: format[key.upper()] = format[key]
		if 'LL' not in format: format['LL'] = '0.2f'
		if 'RHOSQ' not in format: format['RHOSQ'] = '0.3f'
	
		es = self._get_estimation_statistics()
		x = XML("div", {'class':"run_statistics"})
		x.h2("Latest Estimation Run Statistics")

		with x.table_:
			ers = self._get_estimation_run_statistics()
			i = ers[0]['timestamp']
			if i is not '':
				with x.tr_:
					x.td("Estimation Date")
					x.td("{0}".format(i,**format))
			i = ers[0]['iteration']
			if not math.isnan(i):
				with x.tr_:
					x.td("Number of Iterations")
					x.td("{0}".format(i,**format))
			q = ers[0]
			seconds = q['endTimeSec']+q['endTimeUSec']/1000000.0-q['startTimeSec']-q['startTimeUSec']/1000000.0
			tformat = "{}\t{}".format(*format_seconds(seconds))
			with x.tr_:
				x.td("Running Time")
				x.td("{0}".format(tformat,**format))
			i = ers[0]['notes']
			if i is not '':
				with x.tr_:
					x.td("Notes")
					x.td("{0}".format(i,**format))
			i = ers[0]['results']
			if i is not '':
				with x.tr_:
					x.td("Results")
					x.td("{0}".format(i,**format))
		return x.close()

	def xhtml_data(self,**format):
		existing_format_keys = list(format.keys())
		for key in existing_format_keys:
			if key.upper()!=key: format[key.upper()] = format[key]
		if 'LL' not in format: format['LL'] = '0.2f'
		if 'RHOSQ' not in format: format['RHOSQ'] = '0.3f'
	
		x = XML("div", {'class':"data_statistics"})
		if self.Data_Choice is None: return x.close
		x.h2("Choice and Availability Data Statistics")

		# get weights
		if self.Data_Weight:
			w = self.Data_Weight.getArray()
			w = w.reshape(w.shape+(1,))
		else:
			w = numpy.ones([self.nCases()])
		tot_w = numpy.sum(w)
		# calc avails
		if self.Data_Avail:
			av = self.Data_Avail.getArray()
			avails = numpy.sum(av,0)
			avails_weighted = numpy.sum(av*w,0)
		else:
			avails = numpy.ones([self.nAlts()]) * self.nCases()
			avails_weighted =numpy.ones([self.nAlts()]) * tot_w
		ch = self.Data_Choice.getArray()
		choices_unweighted = numpy.sum(ch,0)
		alts = self.alternative_names()
		altns = self.alternative_codes()
		choices_weighted = numpy.sum(ch*w,0)
		use_weights = bool(self.Data_Weight)
		
		with x.block("table"):
			with x.block("thead"):
				with x.block("tr"):
					x.th("Alternative")
					x.th("Condition")
					if use_weights:
						x.th("# Wgt Avail")
						x.th("# Wgt Chosen")
						x.th("# Raw Avail")
						x.th("# Raw Chosen")
					else:
						x.th("# Avail")
						x.th("# Chosen")
			with x.block("tbody"):
				for alt,altn,availw,availu,choicew,choiceu in zip(alts,altns,avails_weighted,avails,choices_weighted,choices_unweighted):
					with x.block("tr"):
						x.td("{:<19}".format(alt))
						try:
							alt_condition = self.db.queries.avail[altn]
						except:
							alt_condition = "n/a"
						x.td("{}".format(alt_condition))
						if use_weights:
							x.td("{:<15.7g}".format(availw[0]))
							x.td("{:<15.7g}".format(choicew[0]))
						x.td("{:<15.7g}".format(availu[0]))
						x.td("{:<15.7g}".format(choiceu[0]))
		return x.close()


	# Utility Data Summary
	def xhtml_utilitydata(self,**format):
		existing_format_keys = list(format.keys())
		for key in existing_format_keys:
			if key.upper()!=key: format[key.upper()] = format[key]
		if 'LL' not in format: format['LL'] = '0.2f'
		if 'RHOSQ' not in format: format['RHOSQ'] = '0.3f'
	
		x = XML("div", {'class':"utilitydata_statistics"})
		if self.Data_Choice is None: return x.close
		x.h2("Utility Data Statistics")

		if self.Data_UtilityCO:
			if self.Data_Weight:
				x.h3("idCO Utility (weighted)")
			else:
				x.h3("idCO Utility")

			means,stdevs,mins,maxs,nonzers,posis,negs,zers = self.stats_utility_co()
			names = self.Data_UtilityCO.get_variables()
			
			head_fmt = "{0:<19}\t{1:<11}\t{2:<11}\t{3:<11}\t{4:<11}\t{5:<11}"
			body_fmt = "{0:<19}\t{1:<11.7g}\t{2:<11.7g}\t{3:<11.7g}\t{4:<11.7g}\t{5:<11.7g}"
			ncols = 6
			
			stack = [names,means,stdevs,mins,maxs,zers]
			titles = ["Data","Mean","Std.Dev.","Minimum","Maximum","Zeros"]
			
			use_p = (numpy.sum(posis)>0)
			use_n = (numpy.sum(negs)>0)
			
			if numpy.sum(posis)>0:
				stack += [posis,]
				head_fmt += "\t{{{0}:<11}}".format(ncols)
				body_fmt += "\t{{{0}:<11.7g}}".format(ncols)
				titles += ["Positives",]
				ncols += 1
			if numpy.sum(negs)>0:
				stack += [negs,]
				head_fmt += "\t{{{0}:<11}}".format(ncols)
				body_fmt += "\t{{{0}:<11.7g}}".format(ncols)
				titles += ["Negatives",]
				ncols += 1
			
			x.table
			x.thead
			x.tr
			for ti in titles:
				x.th(ti)
			x.end_tr
			x.end_thead
			x.tbody
			for s in zip(*stack):
				x.tr
				for thing in s:
					if isinstance(thing,str):
						x.td("{:s}".format(thing))
					else:
						x.td("{:<11.7g}".format(thing))
				x.end_tr
			x.end_tbody
			x.end_table
		return x.close()

	def xhtml_notes(self,**format):
		x = XML("div", {'class':"notes"})
		if not hasattr(self,"notes"): return x.close()
		x.h2("Notes")
		for note in self.notes:
			x.start("p", {'class':'note'})
			x.data(note)
			x.end("p")
		return x.close()



	def report(self, cats=['title','params','LL','latest'], **format):
		import math
		from .utilities import format_seconds
		
		if cats=='*':
			cats=['title','params','LL','latest','DATA','UTILITYDATA','NOTES']
		
		# make all formatting keys uppercase
		existing_format_keys = list(format.keys())
		for key in existing_format_keys:
			if key.upper()!=key: format[key.upper()] = format[key]

		if 'STYLE' not in format: format['STYLE'] = 'txt'

		if format['STYLE'] == 'html':
		
			xhead = XML("head")
			if self.title != 'Untitled Model':
				xhead.title(self.title)
			xhead.style()
			xhead.data("""
			.error_report {color:red; font-family:monospace;}
			table, th, td { border: 1px solid black; border-collapse: collapse; padding: 3px; }
			""".replace('\n',' ').replace('\t',' '))
			xhead.end_style()
		
			with XHTML() as x:
				x.append(xhead)
				#x.append(self.xhtml_parameters(**format))
				for c in cats:
					try:
						func = getattr(type(self),"xhtml_"+c.lower())
						#func = locals()["report_"+c.upper()]
						#func = section[c.upper()]
					except (KeyError, AttributeError):
						xerr = XML("div", {'class':'error_report'})
						xerr.simple("hr")
						xerr.start("pre")
						xerr.data("Key Error: No known report section named {}\n".format(c))
						xerr.end("pre")
						xerr.simple("hr")
						x.append(xerr)
						continue
					try:
						x.append(func(self,**format))
					except:
						import traceback, sys
						xerr = XML()
						xerr.simple("hr")
						xerr.start("pre", {'class':'error_report'})
						xerr.data("Error in {}".format(c))
						xerr.simple("br")
						y = traceback.format_exception(*sys.exc_info())
						for yy in y:
							for eachline in yy.split("\n"):
								xerr.data(eachline)
								xerr.simple("br")
						xerr.end("pre")
						xerr.simple("hr")
						x.append(xerr)
				return x.dump()



		elif format['STYLE'] == 'txt':
			x = []
			# Add style options if not given
			if 'LL' not in format: format['LL'] = '0.2f'
			if 'RHOSQ' not in format: format['RHOSQ'] = '0.3f'
			if 'TABSIZE' not in format: format['TABSIZE'] = 8
			if 'PARAM' not in format: format['PARAM'] = '< 12g'
			if 'PARAM_W' not in format: format['PARAM_W'] = '12'
			if 'LINEPREFIX' not in format: format['LINEPREFIX'] = ''

			def report_TITLE():
				if self.title != 'Untitled Model':
					x = ["="]
					x += [self.title]
					return x
				else:
					return []

			# Model Parameter Estimates
			def report_PARAM():
				x = ["="]
				footer = set()
				es = self._get_estimation_statistics()
				x += ["Model Parameter Estimates"]
				x += ["-"]
				# Find max length parameter name
				max_length_freedom_name = 9
				for p in self._get_parameter():
					max_length_freedom_name = max(max_length_freedom_name, len(p['name']))
				# Write headers
				y  = "{0:<{1}}".format("Parameter",max_length_freedom_name)
				y += "\t{:{PARAM_W}}".format("InitValue",**format)
				y += "\t{:{PARAM_W}}".format("FinalValue",**format)
				y += "\t{:{PARAM_W}}".format("StdError",**format)
				y += "\t{:{PARAM_W}}".format("t-Stat",**format)
				y += "\t{:{PARAM_W}}".format("NullValue",**format)
				x.append(y)
				for p in self._get_parameter():
					try:
						tstat = (p['value'] - p['null_value']) / p['std_err']
					except ZeroDivisionError:
						tstat = float('nan')
					y  = "{0:<{1}}".format(p['name'],max_length_freedom_name)
					y += "\t{:{PARAM}}".format(p['initial_value'],**format)
					y += "\t{:{PARAM}}".format(p['value'],**format)
					y += "\t{:{PARAM}}".format(p['std_err'],**format)
					y += "\t{:{PARAM}}".format(tstat,**format)
					y += "\t{:{PARAM}}".format(p['null_value'],**format)
					if p['holdfast']:
						y += "\tH"
						footer.add("H")
					x.append(y)
				if len(footer):
					x += ["-"]
					if 'H' in footer:
						x += ["H\tParameters held fixed at their initial values (not estimated)"]
				return x

			# Model Estimation Statistics
			def report_LL():
				x = ["="]
				es = self._get_estimation_statistics()
				x += ["Model Estimation Statistics"]
				x += ["-"]
				ll = es[0]['log_like']
				if not math.isnan(ll):
					x += ["Log Likelihood at Convergence     \t{0:{LL}}".format(ll,**format)]
				llc = es[0]['log_like_constants']
				if not math.isnan(llc):
					x += ["Log Likelihood at Constants       \t{0:{LL}}".format(llc,**format)]
				llz = es[0]['log_like_null']
				if not math.isnan(llz):
					x += ["Log Likelihood at Null Parameters \t{0:{LL}}".format(llz,**format)]
				ll0 = es[0]['log_like_nil']
				if not math.isnan(ll0):
					x += ["Log Likelihood with No Model      \t{0:{LL}}".format(ll0,**format)]
				if (not math.isnan(llz) or not math.isnan(llc) or not math.isnan(ll0)) and not math.isnan(ll):
					x += ["-"]
					if not math.isnan(llc):
						rsc = 1.0-(ll/llc)
						x += ["Rho Squared w.r.t. Constants      \t{0:{RHOSQ}}".format(rsc,**format)]
					if not math.isnan(llz):
						rsz = 1.0-(ll/llz)
						x += ["Rho Squared w.r.t. Null Parameters\t{0:{RHOSQ}}".format(rsz,**format)]
					if not math.isnan(ll0):
						rs0 = 1.0-(ll/ll0)
						x += ["Rho Squared w.r.t. No Model       \t{0:{RHOSQ}}".format(rs0,**format)]

				return x

			# Model Latest Run Statistics
			def report_LATEST():
				x = ["="]
				ers = self._get_estimation_run_statistics()
				x += ["Latest Estimation Run Statistics"]
				x += ["-"]
				i = ers[0]['iteration']
				if not math.isnan(i):
					x += ["Number of Iterations \t{0}".format(i,**format)]
				q = ers[0]
				seconds = q['endTimeSec']+q['endTimeUSec']/1000000.0-q['startTimeSec']-q['startTimeUSec']/1000000.0
				tformat = "{}\t{}".format(*format_seconds(seconds))
				x += ["Running Time         \t{0}".format(tformat,**format)]
				i = ers[0]['notes']
				if i is not '':
					x += ["Notes                \t{0}".format(i,**format)]
				i = ers[0]['results']
				if i is not '':
					x += ["Results              \t{0}".format(i,**format)]
				return x

			# Data Summary
			def report_DATA():
				if self.Data_Choice is None:
					return []
				x = ["="]
				x += ["Choice and Availability Data Statistics"]
				x += ["-"]
				# get weights
				if self.Data_Weight:
					w = self.Data_Weight.getArray()
					w = w.reshape(w.shape+(1,))
				else:
					w = numpy.ones([self.nCases()])
				tot_w = numpy.sum(w)
				# calc avails
				if self.Data_Avail:
					av = self.Data_Avail.getArray()
					avails = numpy.sum(av,0)
					avails_weighted = numpy.sum(av*w,0)
				else:
					avails = numpy.ones([self.nAlts()]) * self.nCases()
					avails_weighted =numpy.ones([self.nAlts()]) * tot_w
				ch = self.Data_Choice.getArray()
				choices_unweighted = numpy.sum(ch,0)
				alts = self.alternative_names()
				choices_weighted = numpy.sum(ch*w,0)
				if self.Data_Weight:
					x += ["{0:<19}\t{1:<15}\t{2:<15}\t{3:<15}\t{4:<15}".format("Alternative","# Wgt Avail","# Wgt Chosen","# Raw Avail","# Raw Chosen")]
					for alt,availw,availu,choicew,choiceu in zip(alts,avails_weighted,avails,choices_weighted,choices_unweighted):
						x += ["{0:<19}\t{1:<15.7g}\t{2:<15.7g}\t{3:<15.7g}\t{4:<15.7g}".format(alt,availw[0],choicew[0],availu[0],choiceu[0])]
				else:
					x += ["{0:<19}\t{1:<15}\t{2:<15}".format("Alternative","# Avail","# Chosen",)]
					for alt,availw,availu,choicew,choiceu in zip(alts,avails_weighted,avails,choices_weighted,choices_unweighted):
						x += ["{0:<19}\t{1:<15.7g}\t{2:<15.7g}".format(alt,availu[0],choiceu[0])]
				return x
				
			
			# Utility Data Summary
			def report_UTILITYDATA():
				x = ["="]
				x += ["Utility Data Statistics"]
				if self.Data_UtilityCO:
					x += ["-"]
					if self.Data_Weight:
						x += ["idCO Utility (weighted)"]
					else:
						x += ["idCO Utility"]
					x += ["-"]
					means,stdevs,mins,maxs,nonzers,posis,negs,zers = self.stats_utility_co()
					names = self.Data_UtilityCO.get_variables()
					
					head_fmt = "{0:<19}\t{1:<11}\t{2:<11}\t{3:<11}\t{4:<11}\t{5:<11}"
					body_fmt = "{0:<19}\t{1:<11.7g}\t{2:<11.7g}\t{3:<11.7g}\t{4:<11.7g}\t{5:<11.7g}"
					ncols = 6
					
					stack = [names,means,stdevs,mins,maxs,zers]
					titles = ["Data","Mean","Std.Dev.","Minimum","Maximum","Zeros"]
					
					if numpy.sum(posis)>0:
						stack += [posis,]
						head_fmt += "\t{{{0}:<11}}".format(ncols)
						body_fmt += "\t{{{0}:<11.7g}}".format(ncols)
						titles += ["Positives",]
						ncols += 1
					if numpy.sum(negs)>0:
						stack += [negs,]
						head_fmt += "\t{{{0}:<11}}".format(ncols)
						body_fmt += "\t{{{0}:<11.7g}}".format(ncols)
						titles += ["Negatives",]
						ncols += 1
					
					x += [head_fmt.format(*titles)]
					for s in zip(*stack):
						if len(s[0]) > 19:
							x += [s[0]]
							x += [body_fmt.format("",*(s[1:]))]
						else:
							try:
								x += [body_fmt.format(*s)]
							except TypeError:
								print("TypeErr:",s)
								raise
				return x

			def report_NOTES():
				if not hasattr(self,"notes"): return []
				x = ["="]
				x += ["Notes"]
				x += ["-"]
				x += self.notes
				return x

			report_PARAMS = report_PARAM
			section = {
				'LL': report_LL,
				'LATEST': report_LATEST,
				'PARAM': report_PARAM,
				'PARAMS': report_PARAM,
				'DATA': report_DATA,
				'TITLE': report_TITLE,
				'UTILITYDATA': report_UTILITYDATA,
				'NOTES': report_NOTES,
			}
			for c in cats:
				try:
					func = locals()["report_"+c.upper()]
					#func = section[c.upper()]
				except KeyError:
					x += ["="]
					x += ["Key Error: No known report section named {}".format(c)]
					continue
				try:
					x += func()
				except:
					import traceback, sys
					x += ["="]
					x += ["Error in {}".format(c)]
					x += ["-"]
					y = traceback.format_exception(*sys.exc_info())
					for yy in y:
						x += yy.split("\n")
			# Bottom liner
			x += ["="]

			# What is the length longest line?  Make dividers that long
			longest = 0
			for i in range(len(x)):
				longest = max(longest,len(x[i].expandtabs(format['TABSIZE'])))
			for i in range(len(x)):
				if x[i]=='-': x[i]='-'*longest
				if x[i]=='=': x[i]='='*longest

			s = "\n".join(x)
			return format['LINEPREFIX'] + s.replace("\n", "\n"+format['LINEPREFIX'])

		# otherwise, the format style is not known
		raise LarchError("Format style '{}' is not known".format(format['STYLE']))

	def __str__(self):
		return self.report()



	def stats_utility_co_sqlite(self, where=None):
		"""
		Generate a dataframe of descriptive statistics on the model idco data read from SQLite.
		If the where argument is given, it is used as a filter on the elm_idco table.
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
		positives,negatives,zeros) on the model's idco data as loaded. Uses weights
		if available.
		"""
		x = self.Data_UtilityCO.getArray()
		if self.Data_Weight:
			w = self.Data_Weight.getArray().flatten()
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
		return (mean,stdev,mins,maxs,nonzer,pos,neg,zer)
		

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

	def estimate_constants_only(self):
		db = self._ref_to_db
		alts = db.alternatives()
		m = Model(db)
		for a in alts[1:]:
			m.utility.co('1',a[0],a[1])
		m.estimate()
		self._set_estimation_statistics(log_like_constants=m.LL())

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

	def loglike_c(self):
		return self._get_estimation_statistics()[0]['log_like_constants']

	def estimate_scipy(self, method='BFGS'):
		import scipy.optimize
		return scipy.optimize.minimize(
			self.negative_loglike,   # objective function
			self.parameter_values(), # initial values
			args=(),
			method=method,
			jac=False, #? self.d_loglike,
			hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=print,
			options=dict(disp=True))

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

	def constants_only_model(self, est=True):
		m = self.spawn("constants_only")
		m.utility_full_constants()
		m.option.calc_std_errors = False
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

