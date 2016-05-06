
from ..util.pmath import category, pmath, rename
from ..core import LarchError, ParameterAlias
from io import StringIO
import math

def _tex_text(x):
	return x.replace('#','\#').replace('$','\$').replace('&','\&').replace('_','\_').replace('^','\^')


_generic_larch_preamble = r"""
\providecommand{\tablefont}{\sffamily}
\providecommand{\tableheadfont}{\tablefont\textbf}
\providecommand{\tablebodyfont}{\tablefont}
\providecommand{\thead}[1]{\tableheadfont{#1}}
\providecommand{\tbody}[1]{{#1}}
\providecommand{\theadc}[1]{\multicolumn{1}{c}{\thead{{#1}}}}
\providecommand{\theadl}[1]{\multicolumn{1}{l}{\thead{{#1}}}}
\providecommand{\theadr}[1]{\multicolumn{1}{r}{\thead{{#1}}}}
"""


def _slice_long_multicell_content(x, number_of_columns):
	table = StringIO()
	if len(x) > 80:
		cutpoint = x.find(' ',80)
		table.write(r'\multicolumn{{ {} }}{{|l|}}{{\thead{{{}}}}}'.format(number_of_columns, _tex_text(x[:cutpoint])))
		table.write(r'\\')
		table.write('\n')
		table.write(_slice_long_multicell_content(x[cutpoint:],number_of_columns))
	else:
		table.write(r'\multicolumn{{ {} }}{{|l|}}{{\thead{{{}}}}}'.format(number_of_columns, _tex_text(x)))
		table.write(r'\\')
		table.write('\n')
	return table.getvalue()


class LatexModelReporter():

	def latex_params(self, groups=None, display_inital=False, **format):

		# keys fix
		existing_format_keys = list(format.keys())
		for key in existing_format_keys:
			if key.upper()!=key: format[key.upper()] = format[key]
		if 'PARAM' not in format: format['PARAM'] = '< 12.4g'
		if 'TSTAT' not in format: format['TSTAT'] = ' 0.2f'
		if 'CENTER' not in format: format['CENTER'] = True
		if 'PREAMBLE' not in format: format['PREAMBLE'] = ""

		number_of_columns = 5
		if display_inital:
			number_of_columns += 1


		if groups is None and hasattr(self, 'parameter_groups'):
			groups = self.parameter_groups

		table = StringIO()
		
		table.write(format['PREAMBLE'])
		table.write(_generic_larch_preamble)


		def append_simple_row(name, initial_value, value, std_err, tstat, nullvalue, holdfast):
			table.write(_tex_text(name))
			table.write(r' & ')
			if display_inital:
				table.write("{:{PARAM}}".format(initial_value, **format     ))
				table.write(r' & ')
			table.write("{:{PARAM}}".format(value , **format))
			table.write(r' & ')
			if holdfast:
				table.write(r'\multicolumn{2}{l}{fixed value}')
				table.write(r' & ')
			else:
				table.write("{:.3g}".format(std_err   , **format))
				table.write(r' & ')
				table.write("{:{TSTAT}}".format(tstat , **format  ))
				table.write(r' & ')
			table.write("{:.1f}".format(nullvalue , **format))
			table.write('\n')
			table.write(r'\\')
			table.write('\n')

		def append_derivative_row(name, initial_value, value, refers_to, multiplier):
			table.write(_tex_text(name))
			table.write(r' & ')
			if display_inital:
				table.write("{:{PARAM}}".format(initial_value, **format     ))
				table.write(r' & ')
			table.write("{:{PARAM}}".format(value , **format))
			table.write(r' & ')
			table.write(r"\multicolumn{{3}}{{l}}{{= {} * {} }}".format(refers_to,multiplier))
			table.write('\n')
			table.write(r'\\')
			table.write('\n')

		#table.write(r'{\tablebodyfont\begin{tabular*}{\textwidth}{ |@{\extracolsep{\fill} }l|')
		table.write(r'{\tablebodyfont')
		if format['CENTER']:
			table.write(r'\begin{center}')
		table.write(r'\begin{tabular}{ |l|')
		if display_inital: table.write(r'c|')
		table.write(r'c|c|c|c| }')
		table.write('\n')

		table.write(r'\theadc{Parameter} & ')
		if display_inital:
			table.write(r'\theadc{Initial Value} & ')
		table.write(r'\theadc{Estimated Value} & ')
		table.write(r'\theadc{Std Error} & ')
		table.write(r'\theadc{t-Stat} & ')
		table.write(r'\theadc{Null Value} ')
		table.write('\n')
		table.write(r'\\')
		table.write('\n\hline\n')

		if groups is None:
			for par in self.parameter_names():
				append_simple_row(
					par,
					self.parameter(par).initial_value,
					self.parameter(par).value,
					self.parameter(par).std_err,
					self.parameter(par).t_stat,
					self.parameter(par).null_value,
					self.parameter(par).holdfast
				)

		else:
			
			## USING GROUPS
			listed_parameters = set([p for p in groups if not isinstance(p,category)])
			for p in groups:
				if isinstance(p,category):
					listed_parameters.update( p.complete_members() )
			unlisted_parameters = (set(self.parameter_names()) | set(self.alias_names())) - listed_parameters


			def write_param_row(p, *, force=False):
				if p is None: return
				if force or (p in self) or (p in self.alias_names()):
					if isinstance(p,category):
						table.write('\hline\n')
						table.write(_slice_long_multicell_content(p.name, number_of_columns))
						#table.write(r'\multicolumn{{ {} }}{{|l|}}{{ \thead{{ {} }} }}'.format(number_of_columns, _tex_text(p.name)))
						for subp in p.members:
							write_param_row(subp)
					else:
						if isinstance(p,rename):
							append_simple_row(p.name,
								self[p].initial_value,
								self[p].value,
								self[p].std_err,
								self[p].t_stat,
								self[p].null_value,
								self[p].holdfast
							)
						else:
							pwide = self.parameter_wide(p)
							if isinstance(pwide,ParameterAlias):
								append_derivative_row(pwide.name,
									self.metaparameter(pwide.name).initial_value,
									self.metaparameter(pwide.name).value,
									pwide.refers_to,
									pwide.multiplier
								)
							else:
								append_simple_row(pwide.name,
									pwide.initial_value,
									pwide.value,
									pwide.std_err,
									pwide.t_stat,
									pwide.null_value,
									pwide.holdfast
								)


			# end def
			for p in groups:
				write_param_row(p)
			if len(groups)>0 and len(unlisted_parameters)>0:
				write_param_row(category("Other Parameters"),force=True)
			if len(unlisted_parameters)>0:
				for p in unlisted_parameters:
					write_param_row(p)
		table.write('\n')
		table.write(r'\hline\end{tabular}')
		if format['CENTER']:
			table.write(r'\end{center}')
		table.write('}\n')
		return table.getvalue().replace('\hline\n\hline','\hline')
	latex_param = latex_parameters = latex_params



	# Model Estimation Statistics
	def latex_ll(self,**format):
		existing_format_keys = list(format.keys())
		for key in existing_format_keys:
			if key.upper()!=key: format[key.upper()] = format[key]
		if 'LL' not in format: format['LL'] = '0.2f'
		if 'RHOSQ' not in format: format['RHOSQ'] = '0.3f'
		if 'CENTER' not in format: format['CENTER'] = True
		if 'PREAMBLE' not in format: format['PREAMBLE'] = ""
	
		es = self._get_estimation_statistics()
		table = StringIO()
		
		table.write(format['PREAMBLE'])
		table.write(_generic_larch_preamble)

		def append_simple_row(name, agg, percase):
			table.write(_tex_text(name))
			table.write(r' & ')
			if percase:
				table.write("{:{LL}}".format(agg, **format     ))
				table.write(r' & ')
				table.write("{:{LL}}".format(agg/self.nCases(), **format     ))
			else:
				table.write(r'\multicolumn{{2}}{{|c|}}{{{:{LL}}}}'.format(agg, **format))
			table.write('\n\\\\\n')

		def append_rhosq_row(name, agg):
			table.write(_tex_text(name))
			table.write(r' & ')
			table.write(r'\multicolumn{{2}}{{|c|}}{{{:{RHOSQ}}}}'.format(agg, **format))
			table.write('\n\\\\\n')

		def append_general_row(name, agg):
			table.write(_tex_text(name))
			table.write(r' & ')
			table.write(r'\multicolumn{{2}}{{|c|}}{{{:}}}'.format(agg, **format))
			table.write('\n\\\\\n')
		
		table.write(r'{\tablebodyfont')
		if format['CENTER']:
			table.write(r'\begin{center}')
		table.write(r'\begin{tabular}{ |l|c|c| }')
		table.write('\n')

		table.write(r'\theadc{Statistic} & ')
		table.write(r'\theadc{Aggregate} & ')
		table.write(r'\theadc{Per Case} ')
		table.write('\n')
		table.write(r'\\')
		table.write('\n\hline\n')
		
		
		append_general_row("Number of Cases", self.nCases())

		
		ll = es[0]['log_like']
		
		if not math.isfinite(ll) and math.isfinite(es[0]['log_like_best']):
			ll = es[0]['log_like_best']
		
		
		if not math.isnan(ll):
			append_simple_row("Log Likelihood at Convergence", ll, True)
		llc = es[0]['log_like_constants']
		if not math.isnan(llc):
			append_simple_row("Log Likelihood at Constants", llc, True)
		llz = es[0]['log_like_null']
		if not math.isnan(llz):
			append_simple_row("Log Likelihood at Null Parameters", llz, True)
		ll0 = es[0]['log_like_nil']
		if not math.isnan(ll0):
			append_simple_row("Log Likelihood with No Model", ll0, True)

		if (not math.isnan(llz) or not math.isnan(llc) or not math.isnan(ll0)) and not math.isnan(ll):
			if not math.isnan(llc):
				rsc = 1.0-(ll/llc)
				append_rhosq_row("Rho Squared w.r.t. Constants", rsc)
			if not math.isnan(llz):
				rsz = 1.0-(ll/llz)
				append_rhosq_row("Rho Squared w.r.t. Null Parameters", rsz)
			if not math.isnan(ll0):
				rs0 = 1.0-(ll/ll0)
				append_rhosq_row("Rho Squared w.r.t. No Model", rs0)

		table.write('\n')
		table.write(r'\hline\end{tabular}')
		if format['CENTER']:
			table.write(r'\end{center}')
		table.write('}\n')
		return table.getvalue().replace('\hline\n\hline','\hline')










def latex_simple_joint_report(models, modeltitles, groups, **format):
	existing_format_keys = list(format.keys())
	for key in existing_format_keys:
		if key.upper()!=key: format[key.upper()] = format[key]
	if 'LL' not in format: format['LL'] = '0.2f'
	if 'RHOSQ' not in format: format['RHOSQ'] = '0.3f'
	if 'CENTER' not in format: format['CENTER'] = True
	if 'PREAMBLE' not in format: format['PREAMBLE'] = ""
	if 'PARAM' not in format: format['PARAM'] = '< 12.4g'
	if 'TSTAT' not in format: format['TSTAT'] = ' 0.2f'

	table = StringIO()
	table.write(format['PREAMBLE'])
	table.write(_generic_larch_preamble)

	number_of_columns = 3*len(models) + 1

	table.write(r'{\tablebodyfont')
	if format['CENTER']:
		table.write(r'\begin{center}')
	table.write(r'\begin{tabular}{ |l|')
	table.write('ccc|'*3)

	table.write('}\n')

	table.write(r'\theadc{}')
	for m in modeltitles:
		table.write(r' & ')
		table.write(r'\multicolumn{3}{|c|}{\thead{')
		table.write( _tex_text(m) )
		table.write(r'}}')
	table.write(r'\\')

	table.write(r'\theadc{Parameter}')
	for m in models:
		table.write(r' & \multicolumn{1}{|c}{\thead{{Estimate}}}')
		table.write(r' & \multicolumn{1}{c}{\thead{{Std Error}}}')
		table.write(r' & \multicolumn{1}{c|}{\thead{{t-Stat}}}')
	table.write('\n')
	table.write(r'\\')
	table.write('\n\hline\n')


	## USING GROUPS
	listed_parameters = set([p for p in groups if not isinstance(p,category)])
	for p in groups:
		if isinstance(p,category):
			listed_parameters.update( p.complete_members() )
	unlisted_parameters = set()
	for m in models:
		unlisted_parameters |= set(m.parameter_names())
		unlisted_parameters |= set(m.alias_names())
	unlisted_parameters -= listed_parameters

	def write_partial_row(value, std_err, tstat, holdfast):
		table.write(r' & ')
		table.write("{:{PARAM}}".format(value , **format))
		if holdfast:
			table.write(r' & ')
			table.write(r'\multicolumn{2}{c|}{\textit{fixed value}}')
		else:
			table.write(r' & ')
			table.write("{:.3g}".format(std_err   , **format))
			table.write(r' & ')
			table.write("{:{TSTAT}}".format(tstat , **format  ))

	def write_partial_row_alias(value, refers_to, multiplier):
		table.write(r' & ')
		table.write("{:{PARAM}}".format(value , **format))
		table.write(r' & ')
		table.write(r"\multicolumn{{2}}{{l}}{{= {} * {} }}".format(refers_to,multiplier))

	def write_param_row(p, *, force=False):
		if p is None: return
		actually = False
		if force:
			actually = True
		else:
			for m in models:
				if (p in m) or (p in m.alias_names()):
					actually = True
					break
		if actually:
			if isinstance(p,category):
				table.write('\\hline\n')
				table.write(_slice_long_multicell_content(p.name, number_of_columns))
				for subp in p.members:
					write_param_row(subp)
			elif isinstance(p,rename):
				table.write(_tex_text(p.name))
				for m in models: write_partial_row(m[p].value, m[p].std_err, m[p].t_stat, m[p].holdfast)
				table.write('\\\\\n')
			else:
				table.write(_tex_text(p))
				for m in models:
					pwide = m.parameter_wide(p)
					if isinstance(pwide,ParameterAlias):
						write_partial_row_alias(pwide.value, pwide.refers_to, pwide.multiplier)
					else:
						write_partial_row(pwide.value, pwide.std_err, pwide.t_stat, pwide.holdfast)
				table.write('\\\\\n')
			# end def

	for p in groups:
		write_param_row(p)
	if len(groups)>0 and len(unlisted_parameters)>0:
		write_param_row(category("Other Parameters"),force=True)
	if len(unlisted_parameters)>0:
		for p in unlisted_parameters:
			write_param_row(p)
	table.write('\n')


	es = [m._get_estimation_statistics() for m in models]
	for each_es in es:
		if not math.isfinite(each_es[0]['log_like']) and math.isfinite(each_es[0]['log_like_best']):
			each_es[0]['log_like'] = each_es[0]['log_like_best']



	def write_summary_row_partial(value, fmt):
		if math.isfinite(value):
			table.write(r" & \multicolumn{{3}}{{|c|}}{{ {0:{1}} }}".format(value, fmt))
		else:
			table.write(r" & \multicolumn{{3}}{{|c|}}{{ n/a }}".format(value, fmt))

	use_null = False
	use_nil = False
	use_c = False

	for each_es in es:
		if math.isfinite(each_es[0]['log_like_constants']): use_c = True
		if math.isfinite(each_es[0]['log_like_null']):      use_null = True
		if math.isfinite(each_es[0]['log_like_nil']):       use_nil = True

	table.write("\n\hline\hline\n")
	table.write(_slice_long_multicell_content('Estimation Statistics', number_of_columns))
	table.write("\hline\n")

	table.write("Log Likelihood at Convergence")
	for each_es in es: write_summary_row_partial(each_es[0]['log_like'], format['LL'])
	table.write("\\\\\n")

	if use_c:
		table.write("Log Likelihood at Constants")
		for each_es in es: write_summary_row_partial(each_es[0]['log_like_constants'], format['LL'])
		table.write("\\\\\n")

	if use_null:
		table.write("Log Likelihood at Null Parameters")
		for each_es in es: write_summary_row_partial(each_es[0]['log_like_null'], format['LL'])
		table.write("\\\\\n")

	if use_nil:
		table.write("Log Likelihood with No Model")
		for each_es in es: write_summary_row_partial(each_es[0]['log_like_nil'], format['LL'])
		table.write("\\\\\n")

	table.write("\\hline\n")

	if use_c:
		table.write("Rho Squared w.r.t. Constants")
		for each_es in es:
			try:
				rs = 1.0-(each_es[0]['log_like']/each_es[0]['log_like_constants'])
			except ZeroDivisionError:
				rs = numpy.nan
			write_summary_row_partial(rs, format['RHOSQ'])
		table.write("\\\\\n")

	if use_null:
		table.write("Rho Squared w.r.t. Null Parameters")
		for each_es in es:
			try:
				rs = 1.0-(each_es[0]['log_like']/each_es[0]['log_like_null'])
			except ZeroDivisionError:
				rs = numpy.nan
			write_summary_row_partial(rs, format['RHOSQ'])
		table.write("\\\\\n")

	if use_nil:
		table.write("Rho Squared w.r.t. No Model")
		for each_es in es:
			try:
				rs = 1.0-(each_es[0]['log_like']/each_es[0]['log_like_nil'])
			except ZeroDivisionError:
				rs = numpy.nan
			write_summary_row_partial(rs, format['RHOSQ'])
		table.write("\\\\\n")


	table.write(r'\hline\end{tabular}')
	if format['CENTER']:
		table.write(r'\end{center}')
	table.write('}\n')
	return table.getvalue().replace('\hline\n\hline','\hline')

