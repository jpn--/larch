


from ..util.pmath import category, pmath, rename
from ..core import LarchError, ParameterAlias
from io import StringIO
from ..util.xhtml import XHTML, XML_Builder
import math
import numpy
from ..utilities import format_seconds


TxtModelReporter_default_format = {
	'LL'         :  '0.2f',
	'RHOSQ'      :  '0.3f',
	'TABSIZE'    :  8,
	'PARAM'      :  '< 12g',
	'PARAM_W'    :  '12',
	'LINEPREFIX' :  '',
}



class TxtModelReporter():


	def txt_report(self, cats=['title','params','LL','latest'], **format):
		"""
		Generate a model report in text format.
		
		Parameters
		----------
		cats : list of str, or '*'
			A list of the report components to include. Use '*' to include every
			possible component for the selected output format.
			
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
		for each_key, each_value in TxtModelReporter_default_format.items():
			each_key = each_key.upper()
			if each_key not in format:
				format[each_key] = each_value
		if "SIGFIGS" in format:
			format['PARAM'] = "< {}.{}".format(format['PARAM_W'], format['SIGFIGS'])

		x = []
		
		for c in cats:
			try:
				func = getattr(type(self),"txt_"+c.lower())
			except KeyError:
				x += ["="]
				x += ["Key Error: No known report section named {}".format(c)]
				continue
			try:
				x += func(self,**format)
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




	def txt_title(self, **format): # report_TITLE():
		if self.title != 'Untitled Model':
			x = ["="]
			x += [self.title]
			return x
		else:
			return []

	# Model Parameter Estimates
	def txt_param(self, **format): #report_PARAM():
		x = ["="]
		footer = set()
		x += ["Model Parameter Estimates"]
		x += ["-"]
		# Find max length parameter name
		max_length_freedom_name = 9
		for p in self.parameter_names():
			max_length_freedom_name = max(max_length_freedom_name, len(p))
		# Write headers
		y  = "{0:<{1}}".format("Parameter",max_length_freedom_name)
		y += "\t{:{PARAM_W}}".format("InitValue",**format)
		y += "\t{:{PARAM_W}}".format("FinalValue",**format)
		y += "\t{:{PARAM_W}}".format("StdError",**format)
		y += "\t{:{PARAM_W}}".format("t-Stat",**format)
		y += "\t{:{PARAM_W}}".format("NullValue",**format)
		x.append(y)
		for p in self.parameter_names():
			px = self[p]
			try:
				tstat = (px.value - px.null_value) / px.std_err
			except ZeroDivisionError:
				tstat = float('nan')
			y  = "{0:<{1}}".format(p,max_length_freedom_name)
			y += "\t{:{PARAM}}".format(px.initial_value,**format)
			y += "\t{:{PARAM}}".format(px.value,**format)
			y += "\t{:{PARAM}}".format(px.std_err,**format)
			y += "\t{:{PARAM}}".format(tstat,**format)
			y += "\t{:{PARAM}}".format(px.null_value,**format)
			if px.holdfast:
				y += "\tH"
				footer.add("H")
			x.append(y)
		if len(footer):
			x += ["-"]
			if 'H' in footer:
				x += ["H\tParameters held fixed at their initial values (not estimated)"]
		return x

	# Model Estimation Statistics
	def txt_ll(self, **format): #def report_LL():
		x = ["="]
		x += ["Model Estimation Statistics"]
		x += ["-"]
		ll = self._LL_best
		if not math.isnan(ll):
			x += ["Log Likelihood at Convergence     \t{0:{LL}}".format(ll,**format)]
		llc = self._LL_constants
		if not math.isnan(llc):
			x += ["Log Likelihood at Constants       \t{0:{LL}}".format(llc,**format)]
		llz = self._LL_null
		if not math.isnan(llz):
			x += ["Log Likelihood at Null Parameters \t{0:{LL}}".format(llz,**format)]
		ll0 = self._LL_nil
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
	def txt_latest(self, **format): #def report_LATEST():
		x = ["="]
		ers = self._get_estimation_run_statistics()
		x += ["Latest Estimation Run Statistics"]
		x += ["-"]
		i = ers[0]['iteration']
		if not math.isnan(i):
			x += ["Number of Iterations    \t{0}".format(i,**format)]
		q = ers[0]
		try:
			#seconds = q['endTimeSec']+q['endTimeUSec']/1000000.0-q['startTimeSec']-q['startTimeUSec']/1000000.0
			seconds = ers[0]['total_duration_seconds']
			tformat = "{}\t{}".format(*format_seconds(seconds))
			x += ["Running Time            \t{0}".format(tformat,**format)]
		except KeyError:
			x += ["Running Time:".format(**format)]
		for label, dur in zip(ers[0]['process_label'],ers[0]['process_durations']):
			x += ["- {0:22s}\t{1}".format(label,dur,**format)]
		i = ers[0]['notes']
		if i is not '' and i is not []:
			if isinstance(i, list):
				if len(i)>=1:
					x += ["Notes                   \t{0}".format(i[0],**format)]
				for j in range(1,len(i)):
					x += ["                        \t{0}".format(i[j],**format)]
			else:
				x += ["Notes                   \t{0}".format(i,**format)]
		i = ers[0]['results']
		if i is not '':
			x += ["Results                 \t{0}".format(i,**format)]
		return x

	# Data Summary
	def txt_data(self, **format): #def report_DATA():
		if self.Data("Choice") is None:
			x = ["="]
			x += ["Choice and availability data not provisioned"]
			return x
		x = ["="]
		x += ["Choice and Availability Data Statistics"]
		x += ["-"]
		# get weights
		if bool((self.Data("Weight")!=1).any()):
			w = self.Data("Weight")
			w = w.reshape(w.shape+(1,))
		else:
			w = numpy.ones([self.nCases()])
		tot_w = numpy.sum(w)
		# calc avails
		import time
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
		choices_weighted = numpy.sum(ch*w[:,numpy.newaxis,numpy.newaxis],0)
		if bool((self.Data("Weight")!=1).any()):
			x += ["{0:<19}\t{1:<15}\t{2:<15}\t{3:<15}\t{4:<15}".format("Alternative","# Wgt Avail","# Wgt Chosen","# Raw Avail","# Raw Chosen")]
			for alt,availw,availu,choicew,choiceu in zip(alts,avails_weighted,avails,choices_weighted,choices_unweighted):
				x += ["{0:<19}\t{1:<15.7g}\t{2:<15.7g}\t{3:<15.7g}\t{4:<15.7g}".format(alt,availw[0],choicew[0],availu[0],choiceu[0])]
		else:
			x += ["{0:<19}\t{1:<15}\t{2:<15}".format("Alternative","# Avail","# Chosen",)]
			for alt,availw,availu,choicew,choiceu in zip(alts,avails_weighted,avails,choices_weighted,choices_unweighted):
				x += ["{0:<19}\t{1:<15.7g}\t{2:<15.7g}".format(alt,availu[0],choiceu[0])]
		return x
		

	# Utility Data Summary
	def txt_utilitydata(self, **format): #report_UTILITYDATA():
		if self.Data("UtilityCO") is None:
			x = ["="]
			x += ["Utility CO data not provisioned"]
			return x
		if self.Data("UtilityCO") is not None:
			x = ["="]
			x += ["Utility Data Statistics"]
			x += ["-"]
			if bool((self.Data("Weight")!=1).any()):
				x += ["idCO Utility (weighted)"]
			else:
				x += ["idCO Utility"]
			x += ["-"]
			means,stdevs,mins,maxs,nonzers,posis,negs,zers,mean_nonzer = self.stats_utility_co()
			names = self.needs()["UtilityCO"].get_variables()
			
			head_fmt = "{0:<19}\t{1:<11}\t{2:<11}\t{3:<11}\t{4:<11}\t{5:<11}\t{5:<11}"
			body_fmt = "{0:<19}\t{1:<11.7g}\t{2:<11.7g}\t{3:<11.7g}\t{4:<11.7g}\t{5:<11.7g}\t{5:<11.7g}"
			ncols = 6
			
			stack = [names,means,stdevs,mins,maxs,zers,mean_nonzer]
			titles = ["Data","Mean","Std.Dev.","Minimum","Maximum","Zeros","Mean(NonZeros)"]
			
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

	def txt_notes(self,**format): #report_NOTES():
		if not hasattr(self,"notes") and self.read_runstats_notes()=="":
			return []
		x = ["="]
		x += ["Notes"]
		x += ["-"]
		x += self.notes
		for n in self.read_runstats_notes().split("\n"):
			x += n
		return x

	def txt_utilityspec(self,**format): #report_UTILITYSPEC():
		return [] # not implemented in txt reports
		
	def txt_probabilityspec(self,**format): #report_PROBABILITYSPEC():
		return [] # not implemented in txt reports
		
	def txt_nesting_tree(self,**format): #report_NESTING_TREE():
		if len(self.nest)>0:
			x = ["="]
			x += ["Nesting Structure"]
			x += ["-"]
			x += ["Nodes"]
			x += ["-"]
			nestrep = str(self.nest)
			x += nestrep.split("\n")
			x += ["-"]
			x += ["Links"]
			x += ["-"]
			linkrep = str(self.link)
			x += linkrep.split("\n")
		else:
			return [] # nothing to report

	txt_params = txt_param










