
from .roles import X
from .model_reporter.art import AbstractReportTable
import pandas, numpy

class AnalyzeVariable:
	def __init__(self, name, splits=None, formatting=None, *, include_total=False,
					all_discrete_values=False, discrete_values=None,
					omit_zeros=False):
		self.name = name
		self.splits = splits
		self.formatting = formatting
		self.descrip = X(name).descrip or name
		self.include_total = include_total
		self.all_discrete_values = all_discrete_values
		self.discrete_values = discrete_values
		self.omit_zeros = omit_zeros
		if splits is None and not all_discrete_values and discrete_values is None:
			raise TypeError("must give splits or discrete_values or set all_discrete_values to True")


class AnalyticsObj():
	pass


def _nearly_integer(y):
	if numpy.isclose(y, numpy.round(y)):
		return numpy.round(y)
	return y

class observations_vs_predictions(AnalyticsObj):

	def __init__(self, vars, pct_str=True, title=None, short_title=None):
		"""
		Generate an analysis of choice observation vs model predictions.
		
		Parameters
		----------
		vars : AnalyzeVariable or sequence of AnalyzeVariable
			The variables to analyze
		pct_str : bool
			If true (the default) the percentage values in the result are transformed to
			preformatted percentage strings.
			
		Returns
		-------
		pandas.DataFrame
		"""
		self.vars = vars
		self.pct_str = pct_str
		self.title = title
		self.short_title = short_title



	def __call__(self, m, *, return_art=True):
		"""
		Generate an analysis of choice observation vs model predictions.
		
		Parameters
		----------
		m : Model
			The choice model generating the predictions.
			
		Returns
		-------
		pandas.DataFrame
		"""
		
		# dt: DT, varnames, varsplits, varformats
		
		labeler = lambda x: X(x).descrip or x
		ch = m.df.array_choice().squeeze()
		pr = m.probability()[:,:m.nAlts()].squeeze()
		df = pandas.DataFrame(index=pandas.MultiIndex.from_product([(), (), ()], names=['Variable','Value[s]','For']),
							  columns=list(m.df.alternative_names())+['Count'])
		if isinstance(self.vars, AnalyzeVariable):
			vars = [self.vars, ]
		else:
			vars = self.vars
		varnames = [v.name for v in vars]
		co = m.df.array_idco(*varnames)
		for i in range(co.shape[1]):
			zero_actually_here = False
			choice_in_range_total = numpy.zeros(m.df.nAlts())
			probab_in_range_total = numpy.zeros(m.df.nAlts())
			varlabel = vars[i].descrip
			varformat = vars[i].formatting
			prev_lowbound = -numpy.inf
			lowbound = -numpy.inf
			highbound = -numpy.inf
			discrete_values = None
			if vars[i].all_discrete_values:
				discrete_values = numpy.unique(co[:,i])
			elif vars[i].discrete_values is not None:
				discrete_values = vars[i].discrete_values
			if discrete_values is not None:
				for eachvalue in discrete_values:
					if vars[i].omit_zeros and all_discrete_values and eachvalue==0:
						zero_actually_here = True
						continue
					choice_in_range = ch[(co[:, i] == eachvalue),:].sum(0)
					probab_in_range = pr[(co[:, i] == eachvalue),:].sum(0)
					try:
						range_label = "= {0:{1}}".format(eachvalue,varformat)
					except ValueError:
						range_label = "= {0}".format(eachvalue)
					cnt = choice_in_range.sum()
					if cnt:
						choice_in_range_total += choice_in_range.squeeze()
						probab_in_range_total += probab_in_range.squeeze()
						df.loc[(varlabel, range_label, 'Observed'), 'Count'] = _nearly_integer(cnt)
						df.loc[(varlabel, range_label, 'Predicted'), 'Count'] = _nearly_integer(probab_in_range.sum())
						choice_in_range /= choice_in_range.sum() / 100
						probab_in_range /= probab_in_range.sum() / 100
						df.loc[(varlabel, range_label, 'Observed'), :].values[:-1] = choice_in_range.squeeze()
						df.loc[(varlabel, range_label, 'Predicted'), :].values[:-1] = probab_in_range.squeeze()
			else:
				for split_n in range(len(vars[i].splits)+1):
					prev_lowbound = lowbound
					lowbound = highbound
					if split_n==len(vars[i].splits):
						highbound = numpy.inf
					else:
						highbound = vars[i].splits[split_n]
					if vars[i].omit_zeros and highbound==0 and lowbound==0:
						zero_actually_here = True
						continue
					if highbound==lowbound:
						choice_in_range = ch[(co[:, i] == lowbound),:].sum(0)
						probab_in_range = pr[(co[:, i] == lowbound),:].sum(0)
						try:
							range_label = "X = {0:{1}}".format(lowbound,varformat)
						except ValueError:
							range_label = "X = {0}".format(lowbound)
					elif numpy.isinf(highbound) and prev_lowbound == lowbound:
						choice_in_range = ch[(co[:, i] > lowbound),:].sum(0)
						probab_in_range = pr[(co[:, i] > lowbound),:].sum(0)
						try:
							range_label = "X > {0:{1}}".format(lowbound,varformat)
						except ValueError:
							range_label = "X > {0}".format(lowbound)
					elif numpy.isinf(highbound) and prev_lowbound != lowbound:
						choice_in_range = ch[(co[:, i] >= lowbound),:].sum(0)
						probab_in_range = pr[(co[:, i] >= lowbound),:].sum(0)
						try:
							range_label = "X ≥ {0:{1}}".format(lowbound,varformat)
						except ValueError:
							range_label = "X ≥ {0}".format(lowbound)
					elif numpy.isinf(lowbound):
						choice_in_range = ch[(co[:, i] < highbound),:].sum(0)
						probab_in_range = pr[(co[:, i] < highbound),:].sum(0)
						try:
							range_label = "X < {0:{1}}".format(highbound,varformat)
						except ValueError:
							range_label = "X < {0}".format(highbound)
					elif prev_lowbound == lowbound:
						choice_in_range = ch[(co[:, i] < highbound) & (co[:, i] > lowbound),:].sum(0)
						probab_in_range = pr[(co[:, i] < highbound) & (co[:, i] > lowbound),:].sum(0)
						try:
							range_label = "{0:{2}} < X < {1:{2}}".format(lowbound, highbound,varformat)
						except ValueError:
							range_label = "{0} < X < {1}".format(lowbound, highbound)
					else:
						choice_in_range = ch[(co[:, i] < highbound) & (co[:, i] >= lowbound),:].sum(0)
						probab_in_range = pr[(co[:, i] < highbound) & (co[:, i] >= lowbound),:].sum(0)
						try:
							range_label = "{0:{2}} ≤ X < {1:{2}}".format(lowbound, highbound,varformat)
						except ValueError:
							range_label = "{0} ≤ X < {1}".format(lowbound, highbound)
					cnt = choice_in_range.sum()
					if cnt:
						choice_in_range_total += choice_in_range.squeeze()
						probab_in_range_total += probab_in_range.squeeze()
						df.loc[(varlabel, range_label, 'Observed'), 'Count'] = _nearly_integer(cnt)
						df.loc[(varlabel, range_label, 'Predicted'), 'Count'] = _nearly_integer(probab_in_range.sum())
						choice_in_range /= choice_in_range.sum() / 100
						probab_in_range /= probab_in_range.sum() / 100
						df.loc[(varlabel, range_label, 'Observed'), :].values[:-1] = choice_in_range.squeeze()
						df.loc[(varlabel, range_label, 'Predicted'), :].values[:-1] = probab_in_range.squeeze()
			if vars[i].include_total:
				if zero_actually_here:
					tot_label = 'TOTAL (X≠0)'
				else:
					tot_label = 'TOTAL'
				df.loc[(varlabel, tot_label, 'Observed'), 'Count'] = _nearly_integer(choice_in_range_total.sum())
				df.loc[(varlabel, tot_label, 'Predicted'), 'Count'] = _nearly_integer(probab_in_range_total.sum())
				choice_in_range_total /= choice_in_range_total.sum() / 100
				probab_in_range_total /= probab_in_range_total.sum() / 100
				df.loc[(varlabel, tot_label, 'Observed'), :].values[:-1] = choice_in_range_total.squeeze()
				df.loc[(varlabel, tot_label, 'Predicted'), :].values[:-1] = probab_in_range_total.squeeze()
		if self.pct_str:
			for a in m.df.alternative_names():
				df[a] = df[a].apply(lambda x:"{:.2f}%".format(x))
		if return_art:
			art= AbstractReportTable(from_dataframe=df, title=self.title, short_title=self.short_title)
			from .util.xhtml import Elem
			blacker = lambda y: Elem(tag='td', text=y, attrib={'style':'border-bottom:None;'})
			greener = lambda y: Elem(tag='td', text=y, attrib={'style':'color:#80bd01;border-top:None;'})
			art.df.ix[art.df['For']=='Observed', 2:] = art.df.ix[art.df['For']=='Observed', 2:].applymap(blacker)
			art.df.ix[art.df['For']=='Predicted', 2:] = art.df.ix[art.df['For']=='Predicted', 2:].applymap(greener)
			return art
		return df

