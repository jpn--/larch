

from ..util.pmath import category, pmath, rename
from ..core import LarchError, ParameterAlias, IntStringDict
from io import StringIO
from ..util.xhtml import XHTML, XML_Builder
import math
import numpy



XhtmlModelReporter_default_format = {
	'LL'         :  '0.2f',
	'RHOSQ'      :  '0.3f',
	'TABSIZE'    :  8,
	'PARAM'      :  '< 12g',
	'PARAM_W'    :  '12',
	'LINEPREFIX' :  '',
}


class XhtmlModelReporter():


	def xhtml_report(self, cats=['title','params','LL','latest'], raw_xml=False, **format):
		"""
		Generate a model report in xhtml format.
		
		Parameters
		----------
		cats : list of str, or '*'
			A list of the report components to include. Use '*' to include every
			possible component for the selected output format.
		raw_xml : bool
			If True, the resulting output is returned as a div :class:`Elem` containing a 
			subtree of the entire report. Otherwise, the results are compiled into a
			single bytes object representing a complete html document.
			
		Returns
		-------
		bytes or larch.util.xhtml.Elem
			The report content. You need to save it to a file on your own,
			if desired.
		
		"""

		if 'cats' in format:
			cats = format['cats']

		if cats=='*' and len(self.node)>0:
			cats=['title','params','LL','nesting_tree','latest','UTILITYSPEC','PROBABILITYSPEC','DATA','UTILITYDATA','NOTES','options']
		elif cats=='*':
			cats=['title','params','LL',               'latest','UTILITYSPEC',                  'DATA','UTILITYDATA','NOTES','options']

		if cats=='-' and len(self.node)>0:
			cats=['title','params','LL','nesting_tree','latest','NOTES','options']
		elif cats=='-':
			cats=['title','params','LL',               'latest','NOTES','options']

		if cats=='D' and len(self.node)>0:
			cats=['title','params','LL','nesting_tree_textonly','latest','NOTES','options','queryinfo','UTILITYSPEC',]
		elif cats=='D':
			cats=['title','params','LL',                        'latest','NOTES','options','queryinfo','UTILITYSPEC',]

		# make all formatting keys uppercase
		existing_format_keys = list(format.keys())
		for key in existing_format_keys:
			if key.upper()!=key: format[key.upper()] = format[key]
		# Add style options if not given
		for each_key, each_value in XhtmlModelReporter_default_format.items():
			each_key = each_key.upper()
			if each_key not in format:
				format[each_key] = each_value

		if raw_xml:
			from ..util.xhtml import Elem
			x = Elem('div', {'class':'model_report'})
		else:
			import base64
			x = XHTML(quickhead=self)
			
			
			
		for c in cats:
			try:
				func = getattr(type(self),"xhtml_"+c.lower())
			except (KeyError, AttributeError):
				xerr = XML_Builder("div", {'class':'error_report'})
				xerr.simple("hr")
				xerr.start("pre")
				xerr.data("Key Error: No known report section named {}\n".format(c))
				xerr.end("pre")
				xerr.simple("hr")
				x.append(xerr.close())
				continue
			try:
				to_append = func(self,**format)
				if to_append is not None:
					x.append(to_append)
			except ImportError as err:
				import traceback, sys
				xerr = XML_Builder()
				xerr.simple("hr")
				xerr.start("pre", {'class':'error_report'})
				xerr.data("Unable to provide {}: {}".format(c,str(err)))
				xerr.end("pre")
				xerr.simple("hr")
				x.append(xerr.close())
			except:
				import traceback, sys
				xerr = XML_Builder()
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
				x.append(xerr.close())
		if raw_xml:
			return x
		else:
			return x.dump()




	def xhtml_title(self, **format):
		"""
		Generate a div element containing the model title in a H1 tag.
		
		The title used is taken from the :attr:`title` of the model. There are
		no `format` keywords that are relevant for this method.
		
		Returns
		-------
		larch.util.xhtml.Elem
			A div containing the model title.
		
		"""
		x = XML_Builder("div", {'class':"page_header"})
		x.h1(self.title)
		return x.close()

	def xhtml_computed_factors(self, groups, ignore_na=False, **format):
		# keys fix
		existing_format_keys = list(format.keys())
		for key in existing_format_keys:
			if key.upper()!=key: format[key.upper()] = format[key]
		# build table
		x = XML_Builder("div", {'class':"computed_factors"})
		x.h2("Computed Factors", anchor=1)
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

	def xhtml_params(self, groups=None, display_inital=False, **format):
		"""
		Generate a div element containing the model parameters in a H1 tag.
		
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
		larch.util.xhtml.Elem
			A div containing the model parameters.
		
		Example
		-------
		>>> from larch.util.pmath import category, rename
		>>> from larch.util.xhtml import XHTML
		>>> m = larch.Model.Example(1, pre=True)
		>>> param_groups = [
		... 	category('Level of Service',
		... 			 rename('Total Time', 'tottime'),
		... 			 rename('Total Cost', 'totcost')  ),
		... 	category('Alternative Specific Constants',
		...              'ASC_SR2',
		...              'ASC_SR3P',
		...              'ASC_TRAN',
		...              'ASC_BIKE',
		...              'ASC_WALK'  ),
		... 	category('Income',
		...              'hhinc#2',
		...              'hhinc#3',
		...              'hhinc#4',
		...              'hhinc#5',
		...              'hhinc#6'   ),
		... ]
		>>> with XHTML(quickhead=m) as f:
		... 	f.append( m.xhtml_title()  )
		... 	f.append( m.xhtml_params(param_groups) )
		... 	print(f.dump())
		...
		b'<!DOCTYPE html ...>'
		"""
		# keys fix
		existing_format_keys = list(format.keys())
		for key in existing_format_keys:
			if key.upper()!=key: format[key.upper()] = format[key]
		if 'PARAM' not in format: format['PARAM'] = '< 12.4g'
		if 'TSTAT' not in format: format['TSTAT'] = '0.2f'
		# build table
		x = XML_Builder("div", {'class':"parameter_estimates"})
		x.h2("Model Parameter Estimates", anchor=1)
		
		if groups is None and hasattr(self, 'parameter_groups'):
			groups = self.parameter_groups
		
		
		if groups is None:
			
			footer = set()
			es = self._get_estimation_statistics()
			x.table()
			# Write headers
			x.thead
			x.th("Parameter")
			if display_inital:
				x.th("Initial Value", {'class':'initial_value'})
			x.th("Estimated Value", {'class':'estimated_value'})
			x.th("Std Error", {'class':'std_err'})
			x.th("t-Stat", {'class':'tstat'})
			x.th("Null Value", {'class':'null_value'})
#			x.th("", {'class':'footnote_mark'}) # footnote markers
			x.end_thead
			
			x.tbody
			
			for p in self.parameter_names():
				px = self[p]
				x.tr
				try:
					tstat = (px.value - px.null_value) / px.std_err
				except ZeroDivisionError:
					tstat = float('nan')
				x.start('td')
				x.simple_anchor("param"+p.replace("#","_hash_"))
				x.data('{}'.format(p))
				x.end('td')
				if display_inital:
					x.td("{:{PARAM}}".format(px.initial_value,**format), {'class':'initial_value'})
				x.td("{:{PARAM}}".format(px.value,**format), {'class':'estimated_value'})
				if px.holdfast:
					x.td("fixed value", {'colspan':'2','class':'notation'})
					x.td("{:{PARAM}}".format(px.null_value,**format), {'class':'null_value'})
				else:
					x.td("{:{PARAM}}".format(px.std_err,**format), {'class':'std_err'})
					x.td("{:{TSTAT}}".format(tstat,**format), {'class':'tstat'})
					x.td("{:{PARAM}}".format(px.null_value,**format), {'class':'null_value'})
#					if p['holdfast']:
#						x.td("H", {'class':'footnote_mark'})
#						footer.add("H")
#					else:
#						x.td("", {'class':'footnote_mark'})
				x.end_tr
			for p in self.alias_names():
				x.tr
				x.start('td')
				x.simple_anchor("param"+str(p).replace("#","_hash_"))
				x.data('{}'.format(str(p)))
				x.end('td')
				if display_inital:
					x.td("{:{PARAM}}".format(self.metaparameter(p).initial_value,**format), {'class':'initial_value'})
				x.td("{:{PARAM}}".format(self.metaparameter(p).value,**format), {'class':'estimated_value'})
				x.td("= {} * {}".format(self.alias(p).refers_to, self.alias(p).multiplier), {'colspan':'3'})
				x.end_tr
			x.end_tbody
			
			if len(footer):
				x.tfoot
				x.tr
				if 'H' in footer:
					x.td("H: Parameters held fixed at their initial values (not estimated)", colspan=str(6 if display_inital else 5))
				x.end_tr
				x.end_tfoot
			x.end_table()
		else:
			## USING GROUPS
			listed_parameters = set([p for p in groups if not isinstance(p,category)])
			for p in groups:
				if isinstance(p,category):
					listed_parameters.update( p.complete_members() )
			unlisted_parameters = (set(self.parameter_names()) | set(self.alias_names())) - listed_parameters
			n_cols_params = 6 if display_inital else 5
			def write_param_row(p, *, force=False):
				if p is None: return
				if force or (p in self) or (p in self.alias_names()):
					if isinstance(p,category):
						with x.block("tr"):
							x.td(p.name, {'colspan':str(n_cols_params), 'class':"parameter_category"})
						for subp in p.members:
							write_param_row(subp)
					else:
						if isinstance(p,rename):
							with x.block("tr"):
								x.start('td')
								x.simple_anchor("param"+p.name.replace("#","_hash_"))
								x.data('{}'.format(p.name))
								x.end('td')
#								x.td('{}'.format(p.name))
								if display_inital:
									x.td("{:{PARAM}}".format(self[p].initial_value, **format), {'class':'initial_value'})
								x.td("{:{PARAM}}".format(self[p].value, **format), {'class':'estimated_value'})
								if self[p].holdfast:
									x.td("fixed value", {'colspan':'2', 'class':'notation'})
									x.td("{:{PARAM}}".format(self[p].null_value, **format), {'class':'null_value'})
								else:
									x.td("{:{PARAM}}".format(self[p].std_err, **format), {'class':'std_err'})
									x.td("{:{TSTAT}}".format(self[p].t_stat, **format), {'class':'tstat'})
									x.td("{:{PARAM}}".format(self[p].null_value, **format), {'class':'null_value'})
						else:
							pwide = self.parameter_wide(p)
							if isinstance(pwide,ParameterAlias):
								with x.block("tr"):
									x.td('{}'.format(pwide.name))
									if display_inital:
										x.td("{:{PARAM}}".format(self.metaparameter(pwide.name).initial_value, **format), {'class':'initial_value'})
									x.td("{:{PARAM}}".format(self.metaparameter(pwide.name).value, **format), {'class':'estimated_value'})
									x.td("= {} * {}".format(pwide.refers_to,pwide.multiplier), {'class':'alias notation', 'colspan':'3'})
							else:
								with x.block("tr"):
									x.td('{}'.format(p))
									if display_inital:
										x.td("{:{PARAM}}".format(pwide.initial_value, **format), {'class':'initial_value'})
									x.td("{:{PARAM}}".format(pwide.value, **format), {'class':'estimated_value'})
									if pwide.holdfast:
										x.td("fixed value", {'colspan':'2', 'class':'notation'})
										x.td("{:{PARAM}}".format(pwide.null_value, **format), {'class':'null_value'})
									else:
										x.td("{:{PARAM}}".format(pwide.std_err, **format), {'class':'std_err'})
										x.td("{:{TSTAT}}".format(pwide.t_stat, **format), {'class':'tstat'})
										x.td("{:{PARAM}}".format(pwide.null_value, **format), {'class':'null_value'})
			with x.block("table"):
				with x.block("thead"):
					# PARAMETER ESTIMATES
					with x.block("tr"):
						x.th("Parameter")
						if display_inital:
							x.th("Initial Value", {'class':'initial_value'})
						x.th("Estimated Value", {'class':'estimated_value'})
						x.th("Std Error", {'class':'std_err'})
						x.th("t-Stat", {'class':'tstat'})
						x.th("Null Value", {'class':'null_value'})
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

		try:
			total_weight = float(self.Data("Weight").sum())
		except:
			total_weight = None
		if total_weight is not None:
			if round(total_weight) == self.nCases():
				total_weight = None
	
		es = self._get_estimation_statistics()
		x = XML_Builder("div", {'class':"statistics"})
		x.h2("Model Estimation Statistics", anchor=1)

		x.table
		x.tr
		x.th("Statistic")
		x.th("Aggregate")
		x.th("Per Case")
		use_colspan = '2'
		if total_weight is not None:
			x.th("Per Unit Weight")
			use_colspan = '3'
		x.end_tr
		x.tr
		x.td("Number of Cases")
		x.td("{0}".format(self.nCases()), {'colspan':use_colspan, 'class':'statistics_bridge'})
		x.end_tr
		
		if total_weight is not None:
			x.tr
			x.td("Total Weight")
			x.td("{0}".format(total_weight), {'colspan':use_colspan, 'class':'statistics_bridge'})
			x.end_tr
		
		ll = es[0]['log_like']
		if not math.isnan(ll):
			x.tr
			x.td("Log Likelihood at Convergence")
			x.td("{0:{LL}}".format(ll,**format))
			x.td("{0:{LL}}".format(ll/self.nCases(),**format))
			if total_weight is not None:
				x.td("{0:{LL}}".format(ll/total_weight,**format))
			x.end_tr
		llc = es[0]['log_like_constants']
		if not math.isnan(llc):
			x.tr
			x.td("Log Likelihood at Constants")
			x.td("{0:{LL}}".format(llc,**format))
			x.td("{0:{LL}}".format(llc/self.nCases(),**format))
			if total_weight is not None:
				x.td("{0:{LL}}".format(llc/total_weight,**format))
			x.end_tr
		llz = es[0]['log_like_null']
		if not math.isnan(llz):
			x.tr
			x.td("Log Likelihood at Null Parameters")
			x.td("{0:{LL}}".format(llz,**format))
			x.td("{0:{LL}}".format(llz/self.nCases(),**format))
			if total_weight is not None:
				x.td("{0:{LL}}".format(llz/total_weight,**format))
			x.end_tr
		ll0 = es[0]['log_like_nil']
		if not math.isnan(ll0):
			x.tr
			x.td("Log Likelihood with No Model")
			x.td("{0:{LL}}".format(ll0,**format))
			x.td("{0:{LL}}".format(ll0/self.nCases(),**format))
			if total_weight is not None:
				x.td("{0:{LL}}".format(ll0/total_weight,**format))
			x.end_tr
		if (not math.isnan(llz) or not math.isnan(llc) or not math.isnan(ll0)) and not math.isnan(ll):
			x.tr({'class':"top_rho_sq"})
			if not math.isnan(llc):
				try:
					rsc = 1.0-(ll/llc)
				except ZeroDivisionError:
					x.td("Rho Squared w.r.t. Constants")
					x.td("ZeroDivisionError", {'colspan':use_colspan, 'class':'statistics_bridge'})
				else:
					x.td("Rho Squared w.r.t. Constants")
					x.td("{0:{RHOSQ}}".format(rsc,**format), {'colspan':use_colspan, 'class':'statistics_bridge'})
				x.end_tr
				if not math.isnan(llz) or not math.isnan(ll0): x.tr
			if not math.isnan(llz):
				try:
					rsz = 1.0-(ll/llz)
				except ZeroDivisionError:
					x.td("Rho Squared w.r.t. Null Parameters")
					x.td("ZeroDivisionError", {'colspan':use_colspan, 'class':'statistics_bridge'})
				else:
					x.td("Rho Squared w.r.t. Null Parameters")
					x.td("{0:{RHOSQ}}".format(rsz,**format), {'colspan':use_colspan, 'class':'statistics_bridge'})
				x.end_tr
				if not math.isnan(ll0): x.tr
			if not math.isnan(ll0):
				try:
					rs0 = 1.0-(ll/ll0)
				except ZeroDivisionError:
					x.td("Rho Squared w.r.t. No Model")
					x.td("ZeroDivisionError", {'colspan':use_colspan, 'class':'statistics_bridge'})
				else:
					x.td("Rho Squared w.r.t. No Model")
					x.td("{0:{RHOSQ}}".format(rs0,**format), {'colspan':use_colspan, 'class':'statistics_bridge'})
				x.end_tr
		x.end_table
		return x.close()

	def xhtml_latest(self,**format):
		from ..utilities import format_seconds
		existing_format_keys = list(format.keys())
		for key in existing_format_keys:
			if key.upper()!=key: format[key.upper()] = format[key]
		if 'LL' not in format: format['LL'] = '0.2f'
		if 'RHOSQ' not in format: format['RHOSQ'] = '0.3f'
	
		es = self._get_estimation_statistics()
		x = XML_Builder("div", {'class':"run_statistics"})
		x.h2("Latest Estimation Run Statistics", anchor=1)

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
			#seconds = q['endTimeSec']+q['endTimeUSec']/1000000.0-q['startTimeSec']-q['startTimeUSec']/1000000.0
			seconds = ers[0]['total_duration_seconds']
			tformat = "{}\t{}".format(*format_seconds(seconds))
			with x.tr_:
				x.td("Running Time")
				x.td("{0}".format(tformat,**format))
			for label, dur in zip(ers[0]['process_label'],ers[0]['process_durations']):
				with x.tr_:
					x.td("- "+label)
					x.td("{0}".format(dur,**format))
			i = ers[0]['notes']
			if i is not '':
				if isinstance(i,list) and len(i)>1:
					with x.tr_:
						x.td("Notes")
						with x.td_:
							x.data("{0}".format(i[0],**format))
							for ii in i[1:]:
								x.simple("br")
								x.data("{0}".format(ii,**format))
				elif isinstance(i,list) and len(i)==1:
					with x.tr_:
						x.td("Notes")
						x.td("{0}".format(i[0],**format))
				else:
					with x.tr_:
						x.td("Notes")
						x.td("{0}".format(i,**format))
			i = ers[0]['results']
			if i is not '':
				with x.tr_:
					x.td("Results")
					x.td("{0}".format(i,**format))
			i = ers[0]['processor']
			try:
				from ..util.sysinfo import get_processor_name
				i2 = get_processor_name()
				if isinstance(i2,bytes):
					i2 = i2.decode('utf8')
			except:
				i2 = None
			if i is not '':
				with x.tr_:
					x.td("Processor")
					if i2 is None:
						x.td("{0}".format(i,**format))
					else:
						with x.td_:
							x.data("{0}".format(i,**format))
							x.simple("br")
							x.data("{0}".format(i2,**format))
			i = ers[0]['number_cpu_cores']
			if i is not '':
				with x.tr_:
					x.td("Number of CPU Cores")
					x.td("{0}".format(i,**format))
			i = ers[0]['number_threads']
			if i is not '':
				with x.tr_:
					x.td("Number of Threads Used")
					x.td("{0}".format(i,**format))
			# installed memory
			try:
				import psutil
			except ImportError:
				pass
			else:
				mem = psutil.virtual_memory().total
				if mem >= 2.0*2**30:
					mem_size = str(mem/2**30) + " GiB"
				else:
					mem_size = str(mem/2**20) + " MiB"
				with x.tr_:
					x.td("Installed Memory")
					x.td("{0}".format(mem_size,**format))
			# peak memory usage
			from ..util.sysinfo import get_peak_memory_usage
			peak = get_peak_memory_usage()
			with x.tr_:
				x.td("Peak Memory Usage")
				x.td("{0}".format(peak,**format))
		return x.close()

	def xhtml_data(self,**format):
		existing_format_keys = list(format.keys())
		for key in existing_format_keys:
			if key.upper()!=key: format[key.upper()] = format[key]
		if 'LL' not in format: format['LL'] = '0.2f'
		if 'RHOSQ' not in format: format['RHOSQ'] = '0.3f'
	
		x = XML_Builder("div", {'class':"data_statistics"})
		if self.Data("Choice") is None: return x.close
		x.h2("Choice and Availability Data Statistics", anchor=1)

		# get weights
		if bool((self.Data("Weight")!=1).any()):
			w = self.Data("Weight")
		else:
			w = numpy.ones([self.nCases()])
		tot_w = numpy.sum(w)
		# calc avails
		if self.Data("Avail") is not None:
			av = self.Data("Avail")
			avails = numpy.sum(av,0)
			avails_weighted = numpy.sum(av*w[:,numpy.newaxis,numpy.newaxis],0)
		else:
			avails = numpy.ones([self.nAlts()]) * self.nCases()
			avails_weighted =numpy.ones([self.nAlts()]) * tot_w
		ch = self.Data("Choice")
		choices_unweighted = numpy.sum(ch,0)
		alts = self.alternative_names()
		altns = self.alternative_codes()
		choices_weighted = numpy.sum(ch*w[:,numpy.newaxis,numpy.newaxis],0)
		use_weights = bool((self.Data("Weight")!=1).any())
		show_avail = not isinstance(self.db.queries.avail, str)
		show_descrip = 'alternatives' in self.descriptions
		
		with x.block("table"):
			with x.block("thead"):
				with x.block("tr"):
					x.th("Code")
					x.th("Alternative")
					if use_weights:
						x.th("# Wgt Avail")
						x.th("# Wgt Chosen")
						x.th("# Raw Avail")
						x.th("# Raw Chosen")
					else:
						x.th("# Avail")
						x.th("# Chosen")
					if show_descrip:
						x.th("Description")
					if show_avail:
						x.th("Availability Condition")
			with x.block("tbody"):
				for alt,altn,availw,availu,choicew,choiceu in zip(alts,altns,avails_weighted,avails,choices_weighted,choices_unweighted):
					with x.block("tr"):
						x.td("{:d}".format(altn))
						x.td("{:<19}".format(alt))
						if use_weights:
							x.td("{:<15.7g}".format(availw[0]))
							x.td("{:<15.7g}".format(choicew[0]))
						x.td("{:<15.7g}".format(availu[0]))
						x.td("{:<15.7g}".format(choiceu[0]))
						if show_descrip:
							try:
								alt_descrip = self.descriptions.alternatives[altn]
							except:
								alt_descrip = "n/a"
							x.td("{}".format(alt_descrip))
						if show_avail:
							try:
								alt_condition = self.db.queries.avail[altn]
							except:
								alt_condition = "n/a"
							x.td("{}".format(alt_condition))
		return x.close()


	# Utility Data Summary
	def xhtml_utilitydata(self,**format):
		existing_format_keys = list(format.keys())
		for key in existing_format_keys:
			if key.upper()!=key: format[key.upper()] = format[key]
		if 'LL' not in format: format['LL'] = '0.2f'
		if 'RHOSQ' not in format: format['RHOSQ'] = '0.3f'
	
		x = XML_Builder("div", {'class':"utilitydata_statistics"})
		if self.Data("Choice") is None: return x.close
		x.h2("Utility Data Statistics", anchor=1)


		if self.Data("UtilityCO") is not None:
			show_descrip = 'data_co' in self.descriptions
			if bool((self.Data("Weight")!=1).any()):
				x.h3("idCO Utility (weighted)")
			else:
				x.h3("idCO Utility")

			means,stdevs,mins,maxs,nonzers,posis,negs,zers,mean_nonzer = self.stats_utility_co()
			names = self.needs()["UtilityCO"].get_variables()
			
			
			ncols = 6
			
			stack = [names,means,stdevs,mins,maxs,zers,mean_nonzer]
			titles = ["Data","Mean","Std.Dev.","Minimum","Maximum","Zeros","Mean(NonZero)"]
			
			use_p = (numpy.sum(posis)>0)
			use_n = (numpy.sum(negs)>0)
			
			if numpy.sum(posis)>0:
				stack += [posis,]
				titles += ["Positives",]
				ncols += 1
			if numpy.sum(negs)>0:
				stack += [negs,]
				titles += ["Negatives",]
				ncols += 1
			if show_descrip:
				descriptions = [self.descriptions.data_co[i] if i in self.descriptions.data_co else 'n/a' for i in names]
				stack += [descriptions,]
				titles += ["Description",]
				ncols += 1
			
			x.table
			x.thead
			x.tr
			for ti in titles:
				x.th(ti)
			x.end_tr
			x.end_thead
			with x.tbody_:
				for s in zip(*stack):
					with x.tr_:
						for thing,ti in zip(s,titles):
							if ti=="Description":
								x.td("{:s}".format(thing), {'class':'strut2'})
							elif isinstance(thing,str):
								x.td("{:s}".format(thing))
							else:
								x.td("{:<11.7g}".format(thing))
			x.end_table


		if self.Data("UtilityCA") is not None:
			show_descrip = 'data_ca' in self.descriptions
			
			heads = ["idCA Utility Chosen Alternatives", "idCA Utility Unchosen Alternatives"]
			
			for summary_attrib, heading in zip( self.stats_utility_ca_chosen_unchosen(), heads ):
			
				x.h3(heading)
				
				means,stdevs,mins,maxs,nonzers,posis,negs,zers,mean_nonzer = summary_attrib
				
				
				names = self.needs()["UtilityCA"].get_variables()
				
				ncols = 6
				
				stack = [names,means,stdevs,mins,maxs,zers,mean_nonzer]
				titles = ["Data","Mean","Std.Dev.","Minimum","Maximum","Zeros","Mean(NonZero)"]
				
				use_p = (numpy.sum(posis)>0)
				use_n = (numpy.sum(negs)>0)
				
				if numpy.sum(posis)>0:
					stack += [posis,]
					titles += ["Positives",]
					ncols += 1
				if numpy.sum(negs)>0:
					stack += [negs,]
					titles += ["Negatives",]
					ncols += 1
				if show_descrip:
					descriptions = [self.descriptions.data_co[i] if i in self.descriptions.data_co else 'n/a' for i in names]
					stack += [descriptions,]
					titles += ["Description",]
					ncols += 1
				
				x.table
				x.thead
				x.tr
				for ti in titles:
					x.th(ti)
				x.end_tr
				x.end_thead
				with x.tbody_:
					for s in zip(*stack):
						with x.tr_:
							for thing,ti in zip(s,titles):
								if ti=="Description":
									x.td("{:s}".format(thing), {'class':'strut2'})
								elif isinstance(thing,str):
									x.td("{:s}".format(thing))
								else:
									x.td("{:<11.7g}".format(thing))
				x.end_table

		return x.close()

	# Utility Specification Summary for models with idca utility only
	def xhtml_utilityspec_ca_only(self,**format):
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
		
		x = XML_Builder("div", {'class':"utilityspec"})
		x.h2("Utility Specification", anchor=1)
		
		for resolved in (True, False):
			if resolved:
				headline = "Resolved Utility"
			else:
				headline = "Formulaic Utility"
			x.h3(headline, anchor=1)
			
			with x.table_:
				with x.thead_:
					with x.tr_:
						x.th('Code')
						x.th('Alternative')
						x.th(headline)
				with x.tbody_:
					altcode,altname = next(iter(self.alternatives().items()))
					altcode_ = "*"
					altname_ = "all elemental alternatives"
					with x.tr_:
						x.td(str(altcode_))
						x.td(str(altname_))
						x.start("td")
						
						first_thing = True
						
						def add_util_component(beta, resolved, x, first_thing):
							if resolved:
								beta_val = "{:{PARAM}}".format(self.metaparameter(beta.param).value, **format)
								if not first_thing:
									x.data(" + {}".format(beta_val).replace("+ -","- "))
								else: # is first thing
									x.data(beta_val)
								first_thing = False
							else:
								if not first_thing:
									x.data(" + ")
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
						
						
						for beta in self.utility.ca:
							x, first_thing = add_util_component(beta, resolved, x, first_thing)
						if altcode in self.utility.co:
							for beta in self.utility.co[altcode]:
								x, first_thing = add_util_component(beta, resolved, x, first_thing)
						

						x.end("td")
			
					G = self.networkx_digraph()
					if len(G.node)>len(self.alternative_codes())+1:
						with x.tr_:
							x.th('Code')
							x.th('Nest')
							x.th(headline)
						for altcode in self.nodes_ascending_order(exclude_elementals=True):
							if altcode==self.root_id:
								altname = 'ROOT'
								mu_name = '1'
							else:
								altname = self.nest[altcode]._altname
								mu_name = self.nest[altcode].param
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
									if not skip_mu:
										x.data("/")
										if resolved:
											beta_val = "{:{PARAM}}".format(self.metaparameter(mu_name).value, **format)
											x.data(beta_val)
										else:
											x.start('a', {'class':'parameter_reference', 'href':'#param{}'.format(mu_name.replace("#","_hash_"))})
											x.data(mu_name)
											x.end('a')
									x.data(") ")
								
								x.data(")")
								x.end("td")
		return x.close()

	# Utility Specification Summary
	def xhtml_utilityspec(self,**format):
		if len(self.utility.co)==0: return self.xhtml_utilityspec_ca_only(**format)
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
		
		x = XML_Builder("div", {'class':"utilityspec"})
		x.h2("Utility Specification", anchor=1)
		
		for resolved in (True, False):
			if resolved:
				headline = "Resolved Utility"
			else:
				headline = "Formulaic Utility"
			x.h3(headline, anchor=1)
			
			with x.table_:
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
							
							first_thing = True
							
							def add_util_component(beta, resolved, x, first_thing):
								if resolved:
									beta_val = "{:{PARAM}}".format(self.metaparameter(beta.param).value, **format)
									if not first_thing:
										x.data(" + {}".format(beta_val).replace("+ -","- "))
									else: # is first thing
										x.data(beta_val)
									first_thing = False
								else:
									if not first_thing:
										x.data(" + ")
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
							
							
							for beta in self.utility.ca:
								x, first_thing = add_util_component(beta, resolved, x, first_thing)
							if altcode in self.utility.co:
								for beta in self.utility.co[altcode]:
									x, first_thing = add_util_component(beta, resolved, x, first_thing)
							

							x.end("td")
				
					G = self.networkx_digraph()
					if len(G.node)>len(self.alternative_codes())+1:
						with x.tr_:
							x.th('Code')
							x.th('Nest')
							x.th(headline)
						for altcode in self.nodes_ascending_order(exclude_elementals=True):
							if altcode==self.root_id:
								altname = 'ROOT'
								mu_name = '1'
							else:
								altname = self.nest[altcode]._altname
								mu_name = self.nest[altcode].param
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
									if not skip_mu:
										x.data("/")
										if resolved:
											beta_val = "{:{PARAM}}".format(self.metaparameter(mu_name).value, **format)
											x.data(beta_val)
										else:
											x.start('a', {'class':'parameter_reference', 'href':'#param{}'.format(mu_name.replace("#","_hash_"))})
											x.data(mu_name)
											x.end('a')
									x.data(") ")
								
								x.data(")")
								x.end("td")
		return x.close()



	# Probability Specification Summary
	def xhtml_probabilityspec(self,**format):
		existing_format_keys = list(format.keys())
		for key in existing_format_keys:
			if key.upper()!=key: format[key.upper()] = format[key]
		if 'PARAM' not in format: format['PARAM'] = '0.4g'
		
		x = XML_Builder("div", {'class':"probabilityspec"})
		x.h2("Probability Specification", anchor=1)
		G = self.networkx_digraph()

		for resolved in (True, False):
			if resolved:
				headline = "Resolved Probablity"
			else:
				headline = "Formulaic Probablity"
			x.h3(headline, anchor=1)
		
			with x.table_:
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
								if not skip_mu:
									x.data("/")
									if resolved:
										beta_val = "{:{PARAM}}".format(self.metaparameter(mu_name).value, **format)
										x.data(beta_val)
									else:
										x.start('a', {'class':'parameter_reference', 'href':'#param{}'.format(mu_name.replace("#","_hash_"))})
										x.data(mu_name)
										x.end('a')
								x.data(")/")
								x.data("exp(Utility[{}]".format(pred_name))
								if not skip_mu:
									x.data("/")
									if resolved:
										beta_val = "{:{PARAM}}".format(self.metaparameter(mu_name).value, **format)
										x.data(beta_val)
									else:
										x.start('a', {'class':'parameter_reference', 'href':'#param{}'.format(mu_name.replace("#","_hash_"))})
										x.data(mu_name)
										x.end('a')
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



	def xhtml_notes(self,**format):
		x = XML_Builder("div", {'class':"notes"})
		if not hasattr(self,"notes"): return x.close()
		x.h2("Notes", anchor=1)
		for note in self.notes:
			x.start("p", {'class':'note'})
			x.data(note)
			x.end("p")
		for note in self.read_runstats_notes().split("\n"):
			if note:
				x.start("p", {'class':'note'})
				x.data(note)
				x.end("p")
		return x.close()

	def xhtml_options(self,**format):
		x = XML_Builder("div", {'class':"options"})
		x.h2("Options", anchor=1)
		with x.block("table"):
			for opt in sorted(dir(self.option)):
				if opt[0]=="_" or opt in ('this','thisown','copy'):
					continue
				with x.block("tr"):
					x.td(opt)
					x.td(str(self.option[opt]))
		return x.close()

	def xhtml_queryinfo(self,**format):
		x = XML_Builder("div", {'class':"query_info"})
		x.h2("Query Info", anchor=1)
		with x.block("table"):
			try:
				q = self.db.queries.idco_query
			except AttributeError:
				pass
			else:
				with x.block("tr"):
					x.td("idco query")
					x.td(str(q))

			try:
				q = self.db.queries.idca_query
			except AttributeError:
				pass
			else:
				with x.block("tr"):
					x.td("idca query")
					x.td(str(q))

			try:
				q = self.db.queries.choice
			except AttributeError:
				pass
			else:
				with x.block("tr"):
					x.td("choice")
					x.td(str(q))

			try:
				q = self.db.queries.weight
			except AttributeError:
				pass
			else:
				with x.block("tr"):
					x.td("weight")
					x.td(str(q))

			try:
				q = self.db.queries.avail
				if isinstance(q,IntStringDict):
					q = dict(q)
			except AttributeError:
				pass
			else:
				with x.block("tr"):
					x.td("avail")
					x.td(str(q))

		return x.close()


	def xhtml_nesting_tree(self,**format):
		try:
			import pygraphviz as viz
		except ImportError:
			import warnings
			warnings.warn("pygraphviz module not installed, unable to draw nesting tree in report")
			return self.xhtml_nesting_tree_textonly(**format)
		existing_format_keys = list(format.keys())
		for key in existing_format_keys:
			if key.upper()!=key: format[key.upper()] = format[key]
		if 'SUPPRESSGRAPHSIZE' not in format:
			if 'GRAPHWIDTH' not in format: format['GRAPHWIDTH'] = 6.5
			if 'GRAPHHEIGHT' not in format: format['GRAPHHEIGHT'] = 4
		if 'UNAVAILABLE' not in format: format['UNAVAILABLE'] = True
		x = XML_Builder("div", {'class':"nesting_graph"})
		x.h2("Nesting Structure", anchor=1)
		from io import BytesIO
		import xml.etree.ElementTree as ET
		ET.register_namespace("","http://www.w3.org/2000/svg")
		ET.register_namespace("xlink","http://www.w3.org/1999/xlink")
		if 'SUPPRESSGRAPHSIZE' not in format:
			G=viz.AGraph(name='Tree',directed=True,size="{GRAPHWIDTH},{GRAPHHEIGHT}".format(**format))
		else:
			G=viz.AGraph(name='Tree',directed=True)
		for n,name in self.alternatives().items():
			G.add_node(n, label='<{1} <FONT COLOR="#999999">({0})</FONT>>'.format(n,name))
		eG = G.add_subgraph(name='cluster_elemental', nbunch=self.alternative_codes(), color='#cccccc', bgcolor='#eeeeee',
					   label='Elemental Alternatives', labelloc='b', style='rounded,solid')
		unavailable_nodes = set()
		if format['UNAVAILABLE']:
			if self.is_provisioned():
				try:
					for n, ncode in enumerate(self.alternative_codes()):
#						print("AVCHEK1",ncode,'-->',numpy.sum(self.Data('Avail'),axis=0)[n,0])
						if numpy.sum(self.Data('Avail'),axis=0)[n,0]==0: unavailable_nodes.add(ncode)
				except: raise
			legible_avail = not isinstance(self.db.queries.avail, str)
			if legible_avail:
				for ncode,navail in self.db.queries.avail.items():
					try:
#						print("AVCHEK2",ncode,'-->',navail)
						if navail=='0': unavailable_nodes.add(ncode)
					except: raise
			eG.add_subgraph(name='cluster_elemental_unavailable', nbunch=unavailable_nodes, color='#bbbbbb', bgcolor='#dddddd',
						   label='Unavailable Alternatives', labelloc='b', style='rounded,solid')
		G.add_node(self.root_id, label="Root")
		for n in self.node.nodes():
			if self.node[n]._altname==self.node[n].param:
				G.add_node(n, label='<{1} <FONT COLOR="#999999">({0})</FONT>>'.format(n,self.node[n]._altname,self.node[n].param))
			else:
				G.add_node(n, label='<{1} <FONT COLOR="#999999">({0})</FONT><BR/>µ<SUB>{2}</SUB>>'.format(n,self.node[n]._altname,self.node[n].param))
		up_nodes = set()
		down_nodes = set()
		for i,j in self.link.links():
			G.add_edge(i,j)
			down_nodes.add(j)
			up_nodes.add(i)
		all_nodes = set(self.alternative_codes()) | up_nodes | down_nodes
		for j in all_nodes-down_nodes-unavailable_nodes:
			G.add_edge(self.root_id,j)
		pyg_imgdata = BytesIO()
		G.draw(pyg_imgdata, format='svg', prog='dot')       # write postscript in k5.ps with neato layout
		xx = x.close()
		xx << ET.fromstring(pyg_imgdata.getvalue().decode())
		return xx

	def xhtml_nesting_tree_textonly(self,**format):
		x = XML_Builder("div", {'class':"nesting_text"})
		x.h2("Nesting Structure", anchor=1)
		with x.block("table"):
			with x.block("tr"):
				x.th("Root")
			with x.block("tr"):
				x.td("[{}] Root Node".format(self.root_id))
			with x.block("tr"):
				x.th("Nests")
			with x.block("tr"):
				x.start("td", {'class':'nesting_text_nodes'})
				for eachline in str(self.nest).split("\n"):
					x.data(eachline)
					x.simple("br")
				x.end("td")
			with x.block("tr"):
				x.th("Links")
			with x.block("tr"):
				x.start("td", {'class':'nesting_text_links'})
				for eachline in str(self.link).split("\n"):
					x.data(eachline)
					x.simple("br")
				x.end("td")
		return x.close()

