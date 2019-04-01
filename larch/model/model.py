import pandas
import numpy
import copy

from .controller import Model5c as _Model5c
from ..dataframes import DataFrames, get_dataframe_format
from .linear import LinearFunction_C, DictOfLinearFunction_C
from ..general_precision import l4_float_dtype
from typing import Sequence

import logging
from ..log import logger_name
logger = logging.getLogger(logger_name+'.model')

class Model(_Model5c):
	"""A discrete choice model.

	Parameters
	----------
	parameters : Sequence, optional
		The names of parameters used in this model.  It is generally not
		necessary to define parameter names at initialization, as the names
		can (and will) be collected from the utility function and nesting
		components later.
	utility_ca : LinearFunction_C, optional
		The utility_ca function, which represents the qualitative portion of
		utility for attributes that vary by alternative.
	utility_co : DictOfLinearFunction, optional
		The utility_co function, which represents the qualitative portion of
		utility for attributes that vary by decision maker but not by alternative.
	quantity_ca : LinearFunction_C, optional
		The quantity_ca function, which represents the quantitative portion of
		utility for attributes that vary by alternative.
	quantity_scale : str, optional
		The name of the parameter used to scale the quantitative portion of
		utility.
	graph : NestingTree, optional
		The nesting tree for this choice model.
	dataservice : DataService, optional
		An object that can act as a DataService to generate the data needed for
		this model.

	"""

	utility_co = DictOfLinearFunction_C()
	utility_ca = LinearFunction_C()
	quantity_ca = LinearFunction_C()

	@classmethod
	def Example(cls, n=1):
		from ..examples import example
		return example(n)

	def __init__(self,
				 utility_ca=None,
				 utility_co=None,
				 quantity_ca=None,
				 **kwargs):
		import sys
		self._sklearn_data_format = 'idce'
		self.utility_co = utility_co
		self.utility_ca = utility_ca
		self.quantity_ca = quantity_ca
		super().__init__(**kwargs)
		self._scan_all_ensure_names()
		self.mangle()

	def get_params(self, deep=True):
		p = dict()
		if deep:
			p['frame'] = self.pf.copy()
			p['utility_ca'] = LinearFunction_C(self.utility_ca.copy())
			p['utility_co'] = self.utility_co.copy_without_touch_callback()
			p['quantity_ca'] = LinearFunction_C(self.quantity_ca.copy())
			p['quantity_scale'] = self.quantity_scale.copy() if self.quantity_scale is not None else None
			p['graph'] = copy.deepcopy(self.graph)
			p['is_clone'] = True
		else:
			p['frame'] = self.pf
			p['utility_ca'] = self.utility_ca
			p['utility_co'] = self.utility_co
			p['quantity_ca'] = self.quantity_ca
			p['quantity_scale'] = self.quantity_scale
			p['graph'] = self.graph
			p['is_clone'] = True
		return p

	def set_params(self, **kwargs):
		if 'frame' in kwargs and kwargs['frame'] is not None:
			self.frame = kwargs['frame']

		self.utility_ca = kwargs.get('utility_ca', None)
		self.utility_co = kwargs.get('utility_co', None)
		self.quantity_ca = kwargs.get('quantity_ca', None)
		self.quantity_scale = kwargs.get('quantity_scale', None)
		self.graph = kwargs.get('graph', None)


	def fit(self, X, y, sample_weight=None, **kwargs):
		"""Estimate the parameters of this model from the training set (X, y).

		Parameters
		----------
		X : pandas.DataFrame
			This DataFrame can be in idca, idce, or idco formats.
			If given in idce format, this is a DataFrame with *n_casealts* rows, and
			a two-level MultiIndex.
		y : array-like or str
			The target choice values.  If given as a ``str``, use that named column of `X`.
		sample_weight : array-like, shape = [n_cases] or [n_casealts], or None
			Sample weights. If None, then samples are equally weighted. If shape is *n_casealts*,
			the array is collapsed to *n_cases* by taking only the first weight in each case.

		Returns
		-------
		self : Model
		"""

		if not isinstance(X, pandas.DataFrame):
			raise TypeError(f'must fit on an {self._sklearn_data_format} dataframe')

		if sample_weight is not None:
			raise NotImplementedError('sample_weight not implemented')

		self._sklearn_data_format = get_dataframe_format(X)

		if self._sklearn_data_format == 'idce':

			if sample_weight is not None:
				if isinstance(sample_weight, str):
					sample_weight = X[sample_weight]
				if len(sample_weight) == X.shape[0]:
					sample_weight = sample_weight.groupby(X.index.labels[0]).first()

			if isinstance(y, str):
				y = X[y].unstack().fillna(0)
			elif isinstance(y, (pandas.DataFrame, pandas.Series)):
				y = y.unstack().fillna(0)
			else:
				y = pandas.Series(y, index=X.index).unstack().fillna(0)

			from ..dataframes import _check_dataframe_of_dtype
			try:
				if _check_dataframe_of_dtype(X, l4_float_dtype):
					# when the dataframe is an array of the correct type,
					# it is efficient to use it directly
					self.dataframes = DataFrames(
						ce = X,
						ch = y,
						wt = sample_weight,
					)
				else:
					# when the dataframe is not an array of the correct type,
					# it is efficient to only manipulate needed columns
					self.dataframes = DataFrames(
						ce = X[self.required_data().ca],
						ch = y,
						wt = sample_weight,
					)
			except KeyError:
				# not all keys were available in natural form, try computing them
				dfs1 = DataFrames( ce = X, )
				dfs = dfs1.make_dataframes(self.required_data())
				dfs.data_ch = y
				dfs.data_wt = sample_weight
				self.dataframes = dfs
		else:
			raise NotImplementedError(self._sklearn_data_format)

		self.maximize_loglike(**kwargs)

		return self

	def predict(self, X):
		"""Predict choices for X.

		This method returns the index of the maximum probability choice, not the probability.
		To recover the probability, which is probably what you want (pun intended), see
		:meth:`predict_proba`.

		Parameters
		----------
		X : pandas.DataFrame

		Returns
		-------
		y : array of shape = [n_cases]
			The predicted choices.
		"""
		if not isinstance(X, pandas.DataFrame):
			raise TypeError("X must be a pandas.DataFrame")

		if self._sklearn_data_format in ('idce', 'idca'):
			pr = self.predict_proba(X)
			pr = pr.unstack()
		elif self._sklearn_data_format in ('idco',):
			pr = self.predict_proba(X)
		else:
			raise NotImplementedError(self._sklearn_data_format)

		result = numpy.nanargmax(pr.values, axis=1)

		if self._sklearn_data_format in ('idce', 'idca'):
			pr.values[~numpy.isnan(pr.values)] = 0
			pr.values[numpy.arange(pr.shape[0]), result] = 1
			result = pr.stack()

		return result

	def predict_proba(self, X):
		"""Predict probability for X.

		Parameters
		----------
		X : pandas.DataFrame

		Returns
		-------
		y : array of shape = [n_cases, n_alts]
			The predicted probabilities.
		"""

		if not isinstance(X, pandas.DataFrame):
			raise TypeError(f'predict_proba requires an {self._sklearn_data_format} dataframe')

		if self._sklearn_data_format == 'idce':
			try:
				self.dataframes = DataFrames(
					ce = X[self.required_data().ca],
				)
			except KeyError:
				# not all keys were available in natural form, try computing them
				dfs1 = DataFrames( ce = X, )
				dfs = dfs1.make_dataframes({'ca':self.required_data().ca})
				self.dataframes = dfs

		elif self._sklearn_data_format == 'idca':
			try:
				self.dataframes = DataFrames(
					ca = X[self.required_data().ca],
				)
			except KeyError:
				# not all keys were available in natural form, try computing them
				dfs1 = DataFrames( ca = X, )
				dfs = dfs1.make_dataframes({'ca':self.required_data().ca})
				self.dataframes = dfs
		elif self._sklearn_data_format == 'idco':
			try:
				self.dataframes = DataFrames(
					co = X[self.required_data().co],
				)
			except KeyError:
				# not all keys were available in natural form, try computing them
				dfs1 = DataFrames(co=X, )
				dfs = dfs1.make_dataframes({'co': self.required_data().co})
				self.dataframes = dfs
		else:
			raise NotImplementedError(self._sklearn_data_format)

		result = self.probability(return_dataframe=self._sklearn_data_format)

		return result

	def score(self, X, y, sample_weight=None):
		"""
		Returns the mean negative log loss on the given test data and labels.

		Note that the log loss is defined as the negative of the log likelihood,
		and thus the mean negative log loss is also just mean log likelihood.

		Parameters
		----------
		X : pandas.DataFrame
			If given in idce format, a dataFrame with *n_casealts* rows.
		y : array-like or str
			The target choice values.  If given as a ``str``, use that named column of `X`.
		sample_weight : array-like, shape = [n_casealts], or None
			Sample weights. If None, then samples are equally weighted.

		Returns
		-------
		score : float
			Mean negative log loss of self.predict_proba(X) wrt. y.
		"""
		if isinstance(y, str):
			y = X[y].values.reshape(-1)
		elif isinstance(y, (pandas.DataFrame, pandas.Series)):
			y = y.values.reshape(-1)
		else:
			y = y.reshape(-1)

		pr = self.predict_proba(X)

		weight_adjust = numpy.sum(y) / self.dataframes.n_cases

		if sample_weight is not None:
			sample_weight = sample_weight[y>0] / weight_adjust

		pr = pr[y>0]
		y = y[y>0]

		if sample_weight is None:
			return numpy.sum(numpy.log(pr) * y / weight_adjust) / self.dataframes.n_cases
		else:
			return numpy.sum(numpy.log(pr) * y * sample_weight) / numpy.sum(sample_weight)


	def __parameter_table_section(self, pname):

		from xmle import Elem

		pname_str = str(pname)
		pf = self.pf
		# if pname in self.rename_parameters:
		# 	colspan = 0
		# 	if 'std err' in pf.columns:
		# 		colspan += 1
		# 	if 't stat' in pf.columns:
		# 		colspan += 1
		# 	if 'nullvalue' in pf.columns:
		# 		colspan += 1
		# 	return [
		# 		Elem('td', text="{:.4g}".format(pf.loc[self.rename_parameters[pname],'value'])),
		# 		Elem('td', text="= "+self.rename_parameters[pname], colspan=str(colspan)),
		# 	]
		if pf.loc[pname_str,'holdfast']:
			colspan = 0
			if 'std err' in pf.columns:
				colspan += 1
			if 't stat' in pf.columns:
				colspan += 1
			if 'nullvalue' in pf.columns:
				colspan += 1
			# if pf.loc[pname_str,'holdfast'] == holdfast_reasons.asc_but_never_chosen:
			# 	return [
			# 		Elem('td', text="{:.4g}".format(pf.loc[pname_str,'value'])),
			# 		Elem('td', text="fixed value, never chosen", colspan=str(colspan)),
			# 	]
			# elif pf.loc[pname_str, 'holdfast'] == holdfast_reasons.snap_to_constant_eq:
			# 	return [
			# 		Elem('td', text="{:.4g}".format(pf.loc[pname_str, 'value'])),
			# 		Elem('td', text="fixed value", colspan=str(colspan)),
			# 	]
			# elif pf.loc[pname_str, 'holdfast'] == holdfast_reasons.snap_to_constant_le:
			# 	return [
			# 		Elem('td', text="{:.4g}".format(pf.loc[pname_str, 'value'])),
			# 		Elem('td', text="≤ {:.4g}".format(pf.loc[pname_str, 'value']), colspan=str(colspan)),
			# 	]
			# elif pf.loc[pname_str, 'holdfast'] == holdfast_reasons.snap_to_constant_ge:
			# 	return [
			# 		Elem('td', text="{:.4g}".format(pf.loc[pname_str, 'value'])),
			# 		Elem('td', text="≥ {:.4g}".format(pf.loc[pname_str, 'value']), colspan=str(colspan)),
			# 	]
			# elif pf.loc[pname_str, 'note'] != "" and not pandas.isnull(pf.loc[pname_str, 'note']):
			# 	return [
			# 		Elem('td', text="{:.4g}".format(pf.loc[pname_str,'value'])),
			# 		Elem('td', text=pf.loc[pname_str, 'note'], colspan=str(colspan)),
			# 	]
			# else:
			return [
				Elem('td', text="{:.4g}".format(pf.loc[pname_str,'value'])),
				Elem('td', text="fixed value", colspan=str(colspan), style="text-align: left;"),
			]
		else:
			result = [ Elem('td', text="{:.4g}".format(pf.loc[pname_str,'value'])) ]
			if 'std err' in pf.columns:
				result += [ Elem('td', text="{:#.3g}".format(pf.loc[pname_str, 'std err'])), ]
			if 't stat' in pf.columns:
				result += [ Elem('td', text="{:#.2f}".format(pf.loc[pname_str, 't stat'])), ]
			if 'nullvalue' in pf.columns:
				result += [ Elem('td', text="{:#.2g}".format(pf.loc[pname_str, 'nullvalue'])), ]
			return result


	# def pfo(self):
	# 	if self.ordering is None:
	# 		return self.pf
	# 	paramset = set(self.pf.index)
	# 	out = []
	# 	import re
	# 	if self.ordering:
	# 		for category in self.ordering:
	# 			category_name = category[0]
	# 			category_params = []
	# 			for category_pattern in category[1:]:
	# 				category_params.extend(sorted(i for i in paramset if re.match(category_pattern, i) is not None))
	# 				paramset -= set(category_params)
	# 			out.append( [category_name, category_params] )
	# 	if len(paramset):
	# 		out.append( ['Other', sorted(paramset)] )
	#
	# 	tuples = []
	# 	for c,pp in out:
	# 		for p in pp:
	# 			tuples.append((c,p))
	#
	# 	ix = pandas.MultiIndex.from_tuples(tuples, names=['Category','Parameter'])
	#
	# 	f = self.pf
	# 	f = f.reindex(ix.get_level_values(1))
	# 	f.index = ix
	# 	return f

	def parameter_summary(self):
		pfo = self.pfo()

		ordered_p = list(pfo.index)

		any_colons = False
		for rownum in range(len(ordered_p)):
			if ":" in ordered_p[rownum][1]:
				any_colons = True
				break

		from xmle import ElemTable

		div = ElemTable('div')
		table = div.put('table')

		thead = table.put('thead')
		tr = thead.put('tr')
		tr.put('th', text='Category', style="text-align: left;")
		if any_colons:
			tr.put('th', text='Parameter', colspan='2', style="text-align: left;")
		else:
			tr.put('th', text='Parameter', style="text-align: left;")

		tr.put('th', text='Value')
		if 'std err' in pfo.columns:
			tr.put('th', text='Std Err')
		if 't stat' in pfo.columns:
			tr.put('th', text='t Stat')
		if 'nullvalue' in pfo.columns:
			tr.put('th', text='Null Value')

		tbody = table.put('tbody')

		swallow_categories = 0
		swallow_subcategories = 0

		for rownum in range(len(ordered_p)):
			tr = tbody.put('tr')
			if swallow_categories > 0:
				swallow_categories -= 1
			else:
				nextrow = rownum + 1
				while nextrow<len(ordered_p) and ordered_p[nextrow][0] == ordered_p[rownum][0]:
					nextrow += 1
				swallow_categories = nextrow - rownum - 1
				if swallow_categories:
					tr.put('th', text=ordered_p[rownum][0], rowspan=str(swallow_categories+1), style="vertical-align: top; text-align: left;")
				else:
					tr.put('th', text=ordered_p[rownum][0], style="vertical-align: top; text-align: left;")
			parameter_name = ordered_p[rownum][1]
			if ":" in parameter_name:
				parameter_name = parameter_name.split(":",1)
				if swallow_subcategories > 0:
					swallow_subcategories -= 1
				else:
					nextrow = rownum + 1
					while nextrow < len(ordered_p) and ":" in ordered_p[nextrow][1] and ordered_p[nextrow][1].split(":",1)[0] == parameter_name[0]:
						nextrow += 1
					swallow_subcategories = nextrow - rownum - 1
					if swallow_subcategories:
						tr.put('th', text=parameter_name[0], rowspan=str(swallow_subcategories+1), style="vertical-align: top; text-align: left;")
					else:
						tr.put('th', text=parameter_name[0], style="vertical-align: top; text-align: left;")
				tr.put('th', text=parameter_name[1], style="vertical-align: top; text-align: left;")
			else:
				if any_colons:
					tr.put('th', text=parameter_name, colspan="2", style="vertical-align: top; text-align: left;")
				else:
					tr.put('th', text=parameter_name, style="vertical-align: top; text-align: left;")
			tr << self.__parameter_table_section(ordered_p[rownum][1])

		return div


	def __repr__(self):
		s = "<larch.Model"
		if self.is_mnl():
			s += " (MNL)"
		else:
			s += " (GEV)"
		if self.title != "Untitled":
			s += f' "{self.title}"'
		s += ">"
		return s


	def utility_functions(self, subset=None, resolve_parameters=True):
		from xmle import Elem
		x = Elem('div')
		t = x.elem('table', style="margin-top:1px;", attrib={'class':'floatinghead'})
		if len(self.utility_co):
			# t.elem('caption', text=f"Utility Functions",
			# 	   style="caption-side:top;text-align:left;font-family:Roboto;font-weight:700;"
			# 			 "font-style:normal;font-size:100%;padding:0px;color:black;")
			t_head = t.elem('thead')
			tr = t_head.elem('tr')
			tr.elem('th', text="alt")
			tr.elem('th', text='formula')
			t_body = t.elem('tbody')
			for j in self.utility_co.keys():
				if subset is None or j in subset:
					tr = t_body.elem('tr')
					tr.elem('td', text=str(j))
					utilitycell = tr.elem('td')
					utilitycell.elem('div')
					anything = False
					if len(self.utility_ca):
						utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + "
						utilitycell << list(self.utility_ca.__xml__(linebreaks=True, resolve_parameters=self if resolve_parameters else None))
						anything = True
					if j in self.utility_co:
						if anything:
							utilitycell << Elem('br')
						v = self.utility_co[j]
						utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + "
						utilitycell << list(v.__xml__(linebreaks=True, resolve_parameters=self if resolve_parameters else None))
						anything = True
					if len(self.quantity_ca):
						if anything:
							utilitycell << Elem('br')
						if self.quantity_scale:
							utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + "
							from .roles import ParameterRef
							utilitycell << list(ParameterRef(self.quantity_scale).__xml__(resolve_parameters=self if resolve_parameters else None))
							utilitycell[-1].tail = (utilitycell[-1].tail or "") + " * log("
						else:
							utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + log("
						content = self.quantity_ca.__xml__(linebreaks=True, lineprefix="  ",
														   exponentiate_parameters=True, resolve_parameters=self if resolve_parameters else None)
						utilitycell << list(content)
						utilitycell.elem('br', tail=")")
		else:
			# there is no differentiation by alternatives, just give one formula
			# t.elem('caption', text=f"Utility Function",
			# 	   style="caption-side:top;text-align:left;font-family:Roboto;font-weight:700;"
			# 			 "font-style:normal;font-size:100%;padding:0px;color:black;")
			tr = t.elem('tr')
			utilitycell = tr.elem('td')
			utilitycell.elem('div')
			anything = False
			if len(self.utility_ca):
				utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + "
				utilitycell << list(self.utility_ca.__xml__(linebreaks=True, resolve_parameters=self if resolve_parameters else None))
				anything = True
			if len(self.quantity_ca):
				if anything:
					utilitycell << Elem('br')
				if self.quantity_scale:
					utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + "
					from ..roles import ParameterRef
					utilitycell << list(ParameterRef(self.quantity_scale).__xml__(resolve_parameters=self if resolve_parameters else None))
					utilitycell[-1].tail = (utilitycell[-1].tail or "") + " * log("
				else:
					utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + log("
				content = self.quantity_ca.__xml__(linebreaks=True, lineprefix="  ", exponentiate_parameters=True, resolve_parameters=self if resolve_parameters else None)
				utilitycell << list(content)
				utilitycell.elem('br', tail=")")
		return x


	def required_data(self):
		"""
		What data is required in DataFrames for this model to be used.

		Returns
		-------
		Dict
		"""
		try:
			from ..util import dictx
			req_data = dictx()

			if self.utility_ca is not None and len(self.utility_ca):
				if 'ca' not in req_data:
					req_data.ca = set()
				for i in self.utility_ca:
					req_data.ca.add(str(i.data))

			if self.quantity_ca is not None and len(self.quantity_ca):
				if 'ca' not in req_data:
					req_data.ca = set()
				for i in self.quantity_ca:
					req_data.ca.add(str(i.data))

			if self.utility_co is not None and len(self.utility_co):
				if 'co' not in req_data:
					req_data.co = set()
				for alt, func in self.utility_co.items():
					for i in func:
						if str(i.data)!= '1':
							req_data.co.add(str(i.data))

			if 'ca' in req_data:
				req_data.ca = list(sorted(req_data.ca))
			if 'co' in req_data:
				req_data.co = list(sorted(req_data.co))

			if self.choice_ca_var:
				req_data.choice_ca = self.choice_ca_var
			elif self.choice_co_vars:
				req_data.choice_co = self.choice_co_vars
			elif self.choice_co_code:
				req_data.choice_co_code = self.choice_co_code

			if self.weight_co_var:
				req_data.weight_co = self.weight_co_var

			if self.availability_var:
				req_data.avail_ca = self.availability_var
			elif self.availability_co_vars:
				req_data.avail_co = self.availability_co_vars

			return req_data
		except:
			logger.exception("error in required_data")


	def __contains__(self, item):
		return (item in self.pf.index) or (item in self.rename_parameters)

