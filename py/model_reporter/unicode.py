


from ..util.pmath import category, pmath, rename
from ..core import LarchError, ParameterAlias
from io import StringIO
from ..util.xhtml import XHTML, XML_Builder
import math
import numpy
from ..utilities import format_seconds


UnicodeModelReporter_default_format = {
	'LL'         :  '0.2f',
	'RHOSQ'      :  '0.3f',
	'TABSIZE'    :  8,
	'PARAM'      :  '< 12g',
	'PARAM_W'    :  '12',
	'LINEPREFIX' :  '',
}



class UnicodeReport(str):
	def __repr__(self):
		return self


class UnicodeModelReporter():


	def unicode_report(self, cats=['title','params','LL','latest'], throw_exceptions=False, **format):
		"""
		Generate a model report in unicode format.
		
		Parameters
		----------
		cats : list of str, or '*'
			A list of the report components to include. Use '*' to include every
			possible component for the selected output format.
		throw_exceptions : bool
			If True, exceptions are thrown if raised while generating the report. If 
			False (the default) tracebacks are printed directly into the report for 
			each section where an exception is raised.  Setting this to True can be
			useful for testing.
			
		Returns
		-------
		str
			The report content. You need to save it to a file on your own,
			if desired.
		
		"""
		if cats=='*' and len(self.node)>0:
			cats=['title','params','LL','nesting_tree','latest','UTILITYSPEC','PROBABILITYSPEC','DATA','UTILITYDATA','NOTES']
		elif cats=='*':
			cats=['title','params','LL',               'latest','UTILITYSPEC',                  'DATA','UTILITYDATA','NOTES']
	
		# make all formatting keys uppercase
		existing_format_keys = list(format.keys())
		for key in existing_format_keys:
			if key.upper()!=key: format[key.upper()] = format[key]
		# Add style options if not given
		for each_key, each_value in UnicodeModelReporter_default_format.items():
			each_key = each_key.upper()
			if each_key not in format:
				format[each_key] = each_value
		if "SIGFIGS" in format:
			format['PARAM'] = "< {}.{}".format(format['PARAM_W'], format['SIGFIGS'])

		x = []
		
		for c in cats:
			try:
				func = getattr(type(self),"unicode_"+c.lower())
			except (KeyError, AttributeError):
				if throw_exceptions: raise
				x.append("Error: No known report section named {}".format(c))
				continue
			try:
				x.append(func(self,**format))
			except:
				if throw_exceptions: raise
				import traceback, sys
				xerr = "Error in {}\n".format(c)
				xerr += traceback.format_exception(*sys.exc_info())
				x.append(xerr)

		s = "\n\n".join(x)
		return UnicodeReport(format['LINEPREFIX'] + s.replace("\n", "\n"+format['LINEPREFIX']))




	def unicode_title(self, **format):
		if self.title != 'Untitled Model':
			return UnicodeReport("\n"+self.title)
		else:
			return UnicodeReport("")


	def unicode_params(self, groups=None, display_inital=False, display_id=False, **format):
		"""
		Generate a unicode box table containing the model parameters.
		
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
		str
			A unicode string containing the model parameters in a rectangular table.
		
		Example
		-------
		>>> from larch.util.pmath import category, rename
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
		>>> m.unicode_params(param_groups)
		
		 Model Parameter Estimates
		┌──────────┬───────────────┬──────────┬──────┬──────────┐
		│Parameter │Estimated Value│Std Error │t-Stat│Null Value│
		├──────────┼───────────────┼──────────┼──────┼──────────┤
		╞ Level of Service ═════════════════════════════════════╡
		│Total Time│-0.05134       │ 0.003099 │-16.56│ 0        │
		│Total Cost│-0.00492       │ 0.0002389│-20.60│ 0        │
		╞ Alternative Specific Constants ═══════════════════════╡
		│ASC_SR2   │-2.178         │ 0.1046   │-20.82│ 0        │
		│ASC_SR3P  │-3.725         │ 0.1777   │-20.96│ 0        │
		│ASC_TRAN  │-0.671         │ 0.1326   │-5.06 │ 0        │
		│ASC_BIKE  │-2.376         │ 0.3045   │-7.80 │ 0        │
		│ASC_WALK  │-0.2068        │ 0.1941   │-1.07 │ 0        │
		╞ Income ═══════════════════════════════════════════════╡
		│hhinc   │2│-0.00217       │ 0.001553 │-1.40 │ 0        │
		│        │3│ 0.0003577     │ 0.002538 │ 0.14 │ 0        │
		│        │4│-0.005286      │ 0.001829 │-2.89 │ 0        │
		│        │5│-0.01281       │ 0.005324 │-2.41 │ 0        │
		│        │6│-0.009686      │ 0.003033 │-3.19 │ 0        │
		└──────────┴───────────────┴──────────┴──────┴──────────┘
		"""
		art = self.art_params(groups=groups, display_inital=display_inital, display_id=display_id, **format)
		return UnicodeReport("\n"+str(art))

	unicode_param = unicode_parameters = unicode_params


	def unicode_latest(self,**format):
		art = self.art_latest(**format)
		return UnicodeReport("\n"+str(art))

	def unicode_ll(self,**format):
		"""
		Generate a unicode box table containing the model estimation statistics.
		
		Returns
		-------
		str
			A unicode string containing the model estimation statistics.
		
		Example
		-------
		>>> m = larch.Model.Example(1, pre=True)
		>>> m.unicode_ll()

		 Model Estimation Statistics
		┌──────────────────────────────────┬─────────┬────────┐
		│Statistic                         │Aggregate│Per Case│
		├──────────────────────────────────┼─────────┼────────┤
		│Number of Cases                   │       5029       │
		│Log Likelihood at Convergence     │-3626.19 │-0.72   │
		│Log Likelihood at Null Parameters │-7309.60 │-1.45   │
		│Rho Squared w.r.t. Null Parameters│      0.504       │
		└──────────────────────────────────┴─────────┴────────┘
		"""
		art = self.art_ll(**format)
		return UnicodeReport("\n"+str(art))







