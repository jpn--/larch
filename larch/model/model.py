import pandas
import numpy
import copy

from .controller import Model5c as _Model5c
from ..dataframes import DataFrames, get_dataframe_format
from .linear import LinearFunction_C, DictOfLinearFunction_C
from ..general_precision import l4_float_dtype
from .constraints import ParametricConstraintList
from typing import Sequence, Mapping
from numbers import Number
import numpy as np



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
	"""DictOfLinearFunction_C : The portion of the utility function computed from idco data.
	
	The keys of this mapping are alternative codes for the applicable elemental
	alteratives, and the values are linear functions to compute for the indicated
	alternative.  Each alternative that has any idco utility components must have
	a unique linear function given.
	"""

	utility_ca = LinearFunction_C()
	"""LinearFunction_C : The portion of the utility function computed from idca data.
	
	Examples
	--------
		
	>>> from larch import Model, P, X
	>>> m = Model()
	>>> m.utility_ca = P.Param1 * X.Data1 + P.Param2 * X.Data2
	>>> print(m.utility_ca)
	P.Param1 * X.Data1 + P.Param2 * X.Data2
	
	"""

	quantity_ca = LinearFunction_C()
	"""LinearFunction_C : The portion of the quantity function computed from idca data.
	
	Note that for the quantity function, the actual computed linear function
	uses the exponential of the parameter value(s), not the raw values. Thus, 
	if the quantity function is given as `P.Param1 * X.Data1 + P.Param2 * X.Data2`,
	the computed values will actually be `exp(P.Param1) * X.Data1 + exp(P.Param2) * X.Data2`.
	This transformation ensures that the outcome from the quantity function is 
	always positive, so long as at all of the data terms in the function are
	positive.  The `LinearFunction_C` class itself is not intrinsically aware
	of this implementation detail, but the `Model.utility_functions()` method is, 
	and will render the complete utility function in a mathematically correct form.
	
	Examples
	--------
		
	>>> from larch import Model, P, X
	>>> m = Model()
	>>> m.quantity_ca = P.Param1 * X.Data1 + P.Param2 * X.Data2
	>>> print(m.quantity_ca)
	P.Param1 * X.Data1 + P.Param2 * X.Data2

	"""

	constraints = ParametricConstraintList()

	@classmethod
	def Example(cls, n=1):
		from ..examples import example
		return example(n)

	@classmethod
	def load(cls, filename):
		import os
		if not os.path.exists(filename):
			raise FileNotFoundError(filename)
		_, ext = os.path.splitext(filename)
		if ext == '.xlsx':
			from ..util.excel import load_metadata
			return load_metadata(filename, key='_self_')
		import pickle
		return pickle.load(filename)

	def __init__(self,
				 utility_ca=None,
				 utility_co=None,
				 quantity_ca=None,
				 constraints=None,
				 **kwargs):
		import sys
		self._sklearn_data_format = 'idce'
		self.utility_co = utility_co
		self.utility_ca = utility_ca
		self.quantity_ca = quantity_ca
		self.constraints = constraints
		super().__init__(**kwargs)
		self._scan_all_ensure_names()
		self.mangle()

	def dumps(self):
		"""
		Use pickle to dump the contents of this Model to a bytestring.

		Any associated data (dataframes and dataservice) are not included.

		Returns
		-------
		bytes
		"""
		import pickle
		return pickle.dumps(self)



	def get_params(self, deep=True):
		p = dict()
		if deep:
			p['frame'] = self.pf.copy()
			p['utility_ca'] = LinearFunction_C(self.utility_ca.copy())
			p['utility_co'] = self.utility_co.copy_without_touch_callback()
			p['quantity_ca'] = LinearFunction_C(self.quantity_ca.copy())
			p['quantity_scale'] = self.quantity_scale.copy() if self.quantity_scale is not None else None
			p['graph'] = copy.deepcopy(self.graph)
			p['constraints'] = copy.deepcopy(self.constraints)
			p['is_clone'] = True
		else:
			p['frame'] = self.pf
			p['utility_ca'] = self.utility_ca
			p['utility_co'] = self.utility_co
			p['quantity_ca'] = self.quantity_ca
			p['quantity_scale'] = self.quantity_scale
			p['graph'] = self.graph
			p['constraints'] = self.constraints
			p['is_clone'] = True
		return p

	def set_params(self, **kwargs):
		if 'frame' in kwargs and kwargs['frame'] is not None:
			self._frame = kwargs['frame']

		self.utility_ca = kwargs.get('utility_ca', None)
		self.utility_co = kwargs.get('utility_co', None)
		self.quantity_ca = kwargs.get('quantity_ca', None)
		self.quantity_scale = kwargs.get('quantity_scale', None)
		self.graph = kwargs.get('graph', None)
		self.constraints = kwargs.get('constraints', [])


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
					sample_weight = sample_weight.groupby(X.index.codes[0]).first()

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


	def utility_functions(self, subset=None, resolve_parameters=False):
		"""
		Generate an XHTML output of the utility function(s).

		Parameters
		----------
		subset : Collection, optional
			A collection of alternative codes to include. This only has effect if
			there are separate utility_co functions set by alternative. It is
			recommended to use this parameter if there are a very large number of
			alternatives, and the utility functions of most (or all) of them
			can be effectively communicated by showing only a few.
		resolve_parameters : bool, default False
			Whether to resolve the parameters to the current (estimated) value
			in the output.

		Returns
		-------
		xmle.Elem
		"""
		self.unmangle()
		from xmle import Elem
		x = Elem('div')
		t = x.elem('table', style="margin-top:1px;", attrib={'class':'floatinghead'})
		if len(self.utility_co):
			# t.elem('caption', text=f"Utility Functions",
			# 	   style="caption-side:top;text-align:left;font-family:Roboto;font-weight:700;"
			# 			 "font-style:normal;font-size:100%;padding:0px;color:black;")

			# iterate over all alternatives if a dataframes is attached and lists the alternatives
			try:
				if self.dataservice is not None:
					alts = self.dataservice.alternative_codes()
				elif self.dataframes is not None:
					alts = self.dataframes.alternative_codes()
				else:
					alts = self.utility_co.keys()
			except:
				alts = self.utility_co.keys()
			t_head = t.elem('thead')
			tr = t_head.elem('tr')
			tr.elem('th', text="alt")
			tr.elem('th', text='formula', attrib={'style':'text-align:left;'})
			t_body = t.elem('tbody')
			for j in alts:
				if subset is None or j in subset:
					tr = t_body.elem('tr')
					tr.elem('td', text=str(j))
					utilitycell = tr.elem('td', attrib={'style':'text-align:left;'})
					utilitycell.elem('div')
					anything = False
					if len(self.utility_ca):
						utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + "
						utilitycell << list(self.utility_ca.__xml__(linebreaks=True, resolve_parameters=self, value_in_tooltips=not resolve_parameters))
						anything = True
					if j in self.utility_co:
						v = self.utility_co[j]
						if len(v):
							if anything:
								utilitycell << Elem('br')
							utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + "
							utilitycell << list(v.__xml__(linebreaks=True, resolve_parameters=self, value_in_tooltips=not resolve_parameters))
							anything = True
					if len(self.quantity_ca):
						if anything:
							utilitycell << Elem('br')
						if self.quantity_scale:
							utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + "
							from .linear import ParameterRef_C
							utilitycell << list(ParameterRef_C(self.quantity_scale).__xml__(resolve_parameters=self, value_in_tooltips=not resolve_parameters))
							utilitycell[-1].tail = (utilitycell[-1].tail or "") + " * log("
						else:
							utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + log("
						content = self.quantity_ca.__xml__(linebreaks=True, lineprefix="  ",
														   exponentiate_parameters=True, resolve_parameters=self, value_in_tooltips=not resolve_parameters)
						utilitycell << list(content)
						utilitycell.elem('br', tail=")")
		else:
			# there is no differentiation by alternatives, just give one formula
			# t.elem('caption', text=f"Utility Function",
			# 	   style="caption-side:top;text-align:left;font-family:Roboto;font-weight:700;"
			# 			 "font-style:normal;font-size:100%;padding:0px;color:black;")
			tr = t.elem('tr')
			utilitycell = tr.elem('td', attrib={'style':'text-align:left;'})
			utilitycell.elem('div')
			anything = False
			if len(self.utility_ca):
				utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + "
				utilitycell << list(self.utility_ca.__xml__(linebreaks=True, resolve_parameters=self, value_in_tooltips=not resolve_parameters))
				anything = True
			if len(self.quantity_ca):
				if anything:
					utilitycell << Elem('br')
				if self.quantity_scale:
					utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + "
					from .linear import ParameterRef_C
					utilitycell << list(ParameterRef_C(self.quantity_scale).__xml__(resolve_parameters=self, value_in_tooltips=not resolve_parameters))
					utilitycell[-1].tail = (utilitycell[-1].tail or "") + " * log("
				else:
					utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + log("
				content = self.quantity_ca.__xml__(linebreaks=True, lineprefix="  ", exponentiate_parameters=True, resolve_parameters=self, value_in_tooltips=not resolve_parameters)
				utilitycell << list(content)
				utilitycell.elem('br', tail=")")
		return x

	def _utility_functions_as_frame(self, subset=None, resolve_parameters=False):
		"""
		Generate a tabular output of the utility function(s).

		Parameters
		----------
		subset : Collection, optional
			A collection of alternative codes to include. This only has effect if
			there are separate utility_co functions set by alternative. It is
			recommended to use this parameter if there are a very large number of
			alternatives, and the utility functions of most (or all) of them
			can be effectively communicated by showing only a few.
		resolve_parameters : bool, default False
			Whether to resolve the parameters to the current (estimated) value
			in the output.  Not implemented.

		Returns
		-------
		xmle.Elem
		"""
		self.unmangle()

		tf = pandas.DataFrame(
			index=pandas.MultiIndex.from_tuples([], names=['Alt', 'Line']),
			columns=['Formula'],
		)

		if len(self.utility_co):
			# iterate over all alternatives if a dataframes is attached and lists the alternatives
			try:
				if self.dataservice is not None:
					alts = self.dataservice.alternative_codes()
				elif self.dataframes is not None:
					alts = self.dataframes.alternative_codes()
				else:
					alts = self.utility_co.keys()
			except:
				alts = self.utility_co.keys()

			for a in alts:
				if not (subset is None or a in subset):
					continue
				line = 1
				op = ' '
				for i in self.utility_ca:
					tf.loc[(a, line), :] = f"{op} {str(i)}"
					op = '+'
					line += 1
				if a in self.utility_co:
					for i in self.utility_co[a]:
						tf.loc[(a, line), :] = f"{op} {str(i)}"
						op = '+'
						line += 1
				if len(self.quantity_ca):
					if self.quantity_scale:
						from .linear import ParameterRef_C
						q = ParameterRef_C(self.quantity_scale)
						tf.loc[(a, line), :] = f"{op} {str(q)} * log("
					else:
						tf.loc[(a, line), :] = f"{op} log("
					op = ' '
					line += 1
					for i in self.quantity_ca:
						tf.loc[(a, line), :] = f"    {op} {i._str_exponentiate()}"
						op = '+'
						line += 1
					tf.loc[(a, line), :] = f")"
					op = '+'
					line += 1

		else:
			# there is no differentiation by alternatives, just give one formula
			a = '*'
			line = 1
			op = ' '
			for i in self.utility_ca:
				tf.loc[(a, line), :] = f"{op} {str(i)}"
				op = '+'
				line += 1
			if len(self.quantity_ca):
				if self.quantity_scale:
					from .linear import ParameterRef_C
					q = ParameterRef_C(self.quantity_scale)
					tf.loc[(a, line), :] = f"{op} {str(q)} * log("
				else:
					tf.loc[(a, line), :] = f"{op} log("
				op = ' '
				line += 1
				for i in self.quantity_ca:
					tf.loc[(a, line), :] = f"    {op} {i._str_exponentiate()}"
					op = '+'
					line += 1
				tf.loc[(a, line), :] = f")"
				op = '+'
				line += 1

		return tf

	def required_data(self):
		"""
		What data is required in DataFrames for this model to be used.

		Returns
		-------
		dictx
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
			elif self.choice_any:
				req_data.choice_any = True

			if self.weight_co_var:
				req_data.weight_co = self.weight_co_var

			if self.availability_var:
				req_data.avail_ca = self.availability_var
			elif self.availability_co_vars:
				req_data.avail_co = self.availability_co_vars
			elif self.availability_any:
				req_data.avail_any = True

			return req_data
		except:
			logger.exception("error in required_data")

	def __contains__(self, item):
		return (item in self.pf.index) #or (item in self.rename_parameters)

	def doctor(
			self,
			repair_ch_av=None,
			repair_ch_zq=None,
			repair_asc=None,
			repair_noch_nowt=None,
			repair_nan_wt=None,
			repair_nan_data_co=None,
			verbose=3,
	):
		self.unmangle(True)
		from ..troubleshooting import doctor
		return doctor(
			self,
			repair_ch_av=repair_ch_av,
			repair_ch_zq=repair_ch_zq,
			repair_asc=repair_asc,
			repair_noch_nowt=repair_noch_nowt,
			repair_nan_wt=repair_nan_wt,
			repair_nan_data_co=repair_nan_data_co,
			verbose=verbose,
		)

	def copy(self):
		"""
		Create a copy of the Model.

		The copy shares the same DataService, but does not
		share or copy any loaded DataFrames, which must be
		reloaded if desired.

		Returns
		-------
		Model
		"""
		import pickle
		result = pickle.loads(self.dumps())
		result.dataservice = self.dataservice
		result.graph.set_touch_callback(result.mangle)
		return result

	def remove_unused_parameters(self, verbose=True):
		"""
		Remove parameters that are not used in the model.
		"""
		self.unmangle(True)
		old_pf = self.pf.copy()
		self.pf.drop(self.pf.index, inplace=True)
		self.unmangle(True)
		overlay = pandas.merge(self.pf.iloc[:, 0:0], old_pf, left_index=True, right_index=True, how='left')
		for col in overlay.columns:
			f = ~overlay[col].isna()
			self.pf.loc[f, col] = overlay.loc[f, col]
		if verbose:
			dropped_params = [p for p in old_pf.index if p not in self.pf.index]
			if len(dropped_params) == 0:
				pass
			elif len(dropped_params) < 4:
				logger.warning(f"dropped {len(dropped_params)} parameters: {', '.join(dropped_params)}")
			else:
				logger.warning(f"dropped {len(dropped_params)} parameters including: {', '.join(dropped_params[:3])}")

	def data_statistics(self):
		"""
		Generate a summary of data statistics.

		This summary includes several sections, including
		statistics on idco and idca data as appropriate, as well as
		choice and availability statistics.

		Returns
		-------
		xmle.Elem
		"""
		if self.dataframes is not None:
			return self.dataframes.statistics(graph=self.graph)

	def _get_constraints(self, method):
		if method.lower() in ('slsqp', 'cobyla'):
			constraint_dicts = []
			for c in self.constraints:
				constraint_dicts.extend(c.as_constraint_dicts())
			return constraint_dicts
		if method.lower() in ('trust-constr'):
			constraints = []
			for c in self.constraints:
				constraints.extend(c.as_linear_constraints())
			return constraints
		return ()

	def _add_constraint(self, c):
		"""
		Add a parametric constraint to the model.

		Parameters
		----------
		c : ParametricConstraint
			The constraint to potentially add.  Constraints are
			not actually added if they duplicate an existing
			constraint.

		Returns
		-------
		bool
			Whether this constraint was actually added to the model.
		"""
		from .constraints import ParametricConstraint
		if not isinstance(c, ParametricConstraint):
			raise TypeError(f"constraint must be ParametricConstraint not {type(c)}")
		duplicate_constraint = False
		for i in self.constraints:
			if i==c:
				duplicate_constraint = True
				break
		if not duplicate_constraint:
			self.constraints.append(c)

	def _get_bounds_constraints(self, binding_tol=1e-4):
		"""
		Convert bounds to parametric constraints on the model.

		Parameters
		----------
		binding_tol : Number or Mapping
			The binding tolerance to use, which determines
			whether a constraint is considered active or not.

		Returns
		-------
		list
			A list of constraints.
		"""
		from .constraints import FixedBound
		if isinstance(binding_tol, Number):
			default_binding_tol = binding_tol
		else:
			default_binding_tol = 1e-4

		if not isinstance(binding_tol, Mapping):
			binding_tol = {}

		get_bind_tol = lambda x: binding_tol.get(x, default_binding_tol)

		bounds = []
		self.unmangle()
		for pname, pf_row in self.pf.iterrows():
			if pf_row['holdfast']:
				# don't create bounds constraints on holdfast parameters
				continue
			b = (pf_row['minimum'], pf_row['maximum'])
			if b[0] != -numpy.inf or b[1] != numpy.inf:
				bounds.append(FixedBound(pname, *b, model=self, binding_tol=get_bind_tol(pname)))

		return bounds

	def get_all_contraints(self):
		"""
		All constraints, including bounds.

		Returns
		-------
		list
		"""
		constraints = list(self.constraints)
		try:
			constraints.extend(self._get_bounds_constraints())
		except:
			pass
		return constraints


	def get_all_binding_contraints(self):
		"""
		All binding constraints, including min-max bounds.

		Returns
		-------
		list
		"""
		binding_constraints = list()

		for c in self.get_all_contraints():
			if np.absolute(c.fun(self.pf.value)) < c.binding_tol:
				binding_constraints.append(c)

		return binding_constraints

