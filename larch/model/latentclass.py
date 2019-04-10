
import numpy
import pandas
from typing import MutableMapping

from ..util import Dict
from ..dataframes import DataFrames
from ..model.controller import ParameterNotInModelWarning
from .linear import ParameterRef_C
from ..general_precision import l4_float_dtype


def sync_frames(*models):
	"""
	Synchronize model parameter frames.

	Parameters
	----------
	*models : Sequence[Model]
	"""
	# check if all frames are already in sync
	in_sync = True
	pf1 = models[0].pf
	for m in models[1:]:
		if m.pf is not pf1:
			in_sync = False
	if not in_sync:
		joined = pandas.concat([m.pf for m in models])
		joined = joined[~joined.index.duplicated(keep='first')]
		for m in models:
			m.set_frame(joined)

class LatentClassModel:

	def __init__(self, k_membership, k_models:MutableMapping, dataservice=None):
		self._k_membership = k_membership
		if not isinstance(k_models, MutableMapping):
			raise ValueError(f'k_models must be a MutableMapping, not {type(k_models)}')
		self._k_models = k_models
		self._dataservice = dataservice
		self._dataframes = None
		self._mangled = True

	def _k_model_names(self):
		return list(sorted(self._k_models.keys()))

	@property
	def pf(self):
		self.unmangle()
		return self._k_membership.pf

	def required_data(self):
		# combine all required_data from all class-level submodels
		req = Dict()
		for k_name, k_model in self._k_models.items():
			k_req = k_model.required_data()
			for i in ['ca','co']:
				if i in k_req or i in req:
					req[i] = list(set(req.get(i,[])) | set(k_req.get(i,[])))
			for i in ['weight_co', 'avail_ca', 'avail_co', 'choice_ca', 'choice_co_code', 'choice_co_vars', ]:
				if i in req:
					if i in k_req and req[i] != k_req[i]:
						raise ValueError(f'mismatched {i}')
					else:
						pass
				else:
					if i in k_req:
						req[i] = k_req[i]
					else:
						pass
		top_req = self._k_membership.required_data()
		if 'co' in top_req:
			req['co'] = list(sorted(set(req.get('co', [])) | set(top_req.get('co', []))))
		return req

	def __prep_for_compute(self, x=None):
		self.unmangle()
		if x is not None:
			self._k_membership.set_values(x)

	def class_membership_probability(self, x=None, start_case=0, stop_case=-1, step_case=1):
		self.__prep_for_compute(x)
		return self._k_membership.probability(
			x=None,
			return_dataframe='idco',
			start_case=start_case,
			stop_case=stop_case,
			step_case=step_case,
		)

	def class_membership_d_probability(self, x=None, start_case=0, stop_case=-1, step_case=1):
		self.__prep_for_compute(x)
		return self._k_membership.d_probability(
			x=None,
			start_case=start_case,
			stop_case=stop_case,
			step_case=step_case,
		)

	def probability(self, x=None, start_case=0, stop_case=-1, step_case=1, return_dataframe=False,):
		self.__prep_for_compute(x)

		if start_case >= self.dataframes.n_cases:
			raise IndexError("start_case >= n_cases")

		if stop_case == -1:
			stop_case = self.dataframes.n_cases

		if start_case > stop_case:
			raise IndexError("start_case > stop_case")

		if step_case <= 0:
			raise IndexError("non-positive step_case")

		n_rows = ((stop_case - start_case) // step_case) + (1 if (stop_case - start_case) % step_case else 0)

		p = numpy.zeros([n_rows, self.dataframes.n_alts])

		import warnings
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=ParameterNotInModelWarning)
			k_membership_probability = self.class_membership_probability(
				start_case=start_case, stop_case=stop_case, step_case=step_case,
			)
			for k_name, k_model in self._k_models.items():
				k_pr = k_model.probability(start_case=start_case, stop_case=stop_case, step_case=step_case)
				p += (
						numpy.asarray( k_pr[:,:self.dataframes.n_alts] )
						* k_membership_probability.loc[:,k_name].values[:, None]
				)

		if return_dataframe:
			return pandas.DataFrame(
				p,
				index=self._dataframes.caseindex[start_case:stop_case:step_case],
				columns=self._dataframes.alternative_codes(),
			)
		return p


	def d_probability(self, x=None, start_case=0, stop_case=-1, step_case=1,):
		"""
		Compute the partial derivative of probability w.r.t. the parameters.

		Note this function is known to be incomplete.  It computes the
		derivative only within the classes, not for the class membership model.

		Parameters
		----------
		x
		start_case
		stop_case
		step_case
		return_dataframe

		Returns
		-------

		"""
		self.__prep_for_compute(x)

		if start_case >= self.dataframes.n_cases:
			raise IndexError("start_case >= n_cases")

		if stop_case == -1:
			stop_case = self.dataframes.n_cases

		if start_case > stop_case:
			raise IndexError("start_case > stop_case")

		if step_case <= 0:
			raise IndexError("non-positive step_case")

		n_rows = ((stop_case - start_case) // step_case) + (1 if (stop_case - start_case) % step_case else 0)

		p = numpy.zeros([n_rows, self.dataframes.n_alts, len(self.pf)])

		import warnings
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=ParameterNotInModelWarning)
			k_membership_probability = self.class_membership_probability(
				start_case=start_case, stop_case=stop_case, step_case=step_case,
			)
			k_membership_d_probability = self.class_membership_d_probability(
				start_case=start_case, stop_case=stop_case, step_case=step_case,
			)
			for k_name, k_model in self._k_models.items():
				k_pr = k_model.probability(start_case=start_case, stop_case=stop_case, step_case=step_case)
				k_d_pr = k_model.d_probability(start_case=start_case, stop_case=stop_case, step_case=step_case)
				p += (
						numpy.asarray( k_d_pr[:,:self.dataframes.n_alts,:] )
						* k_membership_probability.loc[:,k_name].values[:, None, None]
				)
				k_position = k_membership_probability.columns.get_loc(k_name)
				p += (
						numpy.asarray( k_pr[:,:self.dataframes.n_alts, None] )
						* k_membership_d_probability[:,k_position,:][:,None,:]
				)
		return p

	def d_loglike(
			self,
			x=None,
			start_case=0,
			stop_case=-1,
			step_case=1,
	):
		self.__prep_for_compute(x)
		pr = self.probability(x=None, start_case=start_case, stop_case=stop_case, step_case=step_case,)
		d_p = self.d_probability(x=None, start_case=start_case, stop_case=stop_case, step_case=step_case,)
		from .nl import d_loglike_from_d_probability

		if stop_case == -1:
			stop_case = self.dataframes.n_cases

		if self.dataframes.data_wt is not None:
			wt = self.dataframes.data_wt.iloc[start_case:stop_case:step_case]
		else:
			wt = None

		d_LL = d_loglike_from_d_probability(
			pr,
			d_p,
			self.dataframes.array_ch()[start_case:stop_case:step_case],
			wt,
			False
		)
		return d_LL

	def loglike(
			self,
			x=None,
			*,
			start_case=0, stop_case=-1, step_case=1,
			persist=0,
			leave_out=-1, keep_only=-1, subsample=-1,
			probability_only=False,
	):
		"""
		Compute a log likelihood value.

		Parameters
		----------
		x : {'null', 'init', 'best', array-like, dict, scalar}, optional
			Values for the parameters.  See :ref:`set_values` for details.
		start_case : int, default 0
			The first case to include in the log likelihood computation.  To include all
			cases, start from 0 (the default).
		stop_case : int, default -1
			One past the last case to include in the log likelihood computation.  This is processed as usual for
			Python slicing and iterating, and negative values count backward from the end.  To include all cases,
			end at -1 (the default).
		step_case : int, default 1
			The step size of the case iterator to use in likelihood calculation.  This is processed as usual for
			Python slicing and iterating.  To include all cases, step by 1 (the default).
		persist : int, default 0
			Whether to return a variety of internal and intermediate arrays in the result dictionary.
			If set to 0, only the final `ll` value is included.
		leave_out, keep_only, subsample : int, optional
			Settings for cross validation calculations.
			If `leave_out` and `subsample` are set, then case rows where rownumber % subsample == leave_out are dropped.
			If `keep_only` and `subsample` are set, then only case rows where rownumber % subsample == keep_only are used.
		probability_only : bool, default False
			Compute only the probability and ignore the likelihood.  If this setting is active, the
			dataframes need not include the "actual" choice data.

		Returns
		-------
		Dict or float or array
			The log likelihood or the probability.  Other arrays are also included if `persist` is set to True.

		"""
		self.__prep_for_compute(x)
		pr = self.probability(
			x=None,
			start_case=start_case,
			stop_case=stop_case,
			step_case=step_case,
			return_dataframe=False,
		)
		if probability_only:
			return pr
		from .mnl import loglike_from_probability

		if stop_case == -1:
			stop_case = self.dataframes.n_cases

		if self.dataframes.data_wt is not None:
			wt = self.dataframes.data_wt.iloc[start_case:stop_case:step_case]
		else:
			wt = None

		ll = loglike_from_probability(
			pr,
			self.dataframes.array_ch()[start_case:stop_case:step_case],
			wt
		)

		return ll



	@property
	def dataframes(self):
		return self._dataframes

	@dataframes.setter
	def dataframes(self, x):
		self._dataframes = x
		top_data = DataFrames(
			co = x.make_idco(*self._k_membership.required_data().get('co', [])),
			alt_codes=numpy.arange(1, len(self._k_models)+1),
			alt_names=self._k_model_names(),
			av=1,
		)
		self._k_membership.dataframes = top_data
		for k_name, k_model in self._k_models.items():
			k_model.dataframes = DataFrames(
				co=x.data_co,
				ca=x.data_ca,
				ce=x.data_ce,
				av=x.data_av,
				ch=x.data_ch,
				wt=x.data_wt,
				alt_names=x.alternative_names(),
				alt_codes=x.alternative_codes(),
			)

	def mangle(self, *args, **kwargs):
		#self.clear_best_loglike()
		self._mangled = True
		self._k_membership.mangle()
		for m in self._k_models.values():
			m.mangle()

	def unmangle(self, force=False):
		if self._mangled or force:
			self._k_membership.unmangle(force=force)
			for m in self._k_models.values():
				m.unmangle(force=force)
			sync_frames(self._k_membership, *self._k_models.values())
			self._k_membership.unmangle()
			for m in self._k_models.values():
				m.unmangle()
			self._mangled = False

	def load_data(self, dataservice=None, autoscale_weights=True):
		self.unmangle()
		if dataservice is not None:
			self._dataservice = dataservice
		if self._dataservice is not None:
			dfs = self._dataservice.make_dataframes(self.required_data())
			if autoscale_weights and dfs.data_wt is not None:
				dfs.autoscale_weights()
			self.dataframes = dfs
		else:
			raise ValueError('dataservice is not defined')

	def set_value(self, name, value=None, **kwargs):
		if isinstance(name, ParameterRef_C):
			name = str(name)
		if name not in self.pf.index:
			self.unmangle()
		if value is not None:
			# value = numpy.float64(value)
			# self.frame.loc[name,'value'] = value
			kwargs['value'] = value
		for k,v in kwargs.items():
			if k in self.pf.columns:
				if self.pf.dtypes[k] == 'float64':
					v = numpy.float64(v)
				elif self.pf.dtypes[k] == 'float32':
					v = numpy.float32(v)
				elif self.pf.dtypes[k] == 'int8':
					v = numpy.int8(v)
				self.pf.loc[name, k] = v

			# update init values when they are implied
			if k=='value':
				if 'initvalue' not in kwargs and pandas.isnull(self.pf.loc[name, 'initvalue']):
					self.pf.loc[name, 'initvalue'] = l4_float_dtype(v)

		# update null values when they are implied
		if 'nullvalue' not in kwargs and pandas.isnull(self.pf.loc[name, 'nullvalue']):
			self.pf.loc[name, 'nullvalue'] = 0

		# refresh everything # TODO: only refresh what changed
		if self._k_membership.dataframes is not None:
			try:
				self._k_membership.dataframes.read_in_model_parameters()
			except AttributeError:
				# the model may not have been ever unmangled yet
				pass
		for m in self._k_models.values():
			try:
				if m.dataframes is not None:
					m.dataframes.read_in_model_parameters()
			except AttributeError:
				# the model may not have been ever unmangled yet
				pass

	@property
	def pvals(self):
		self.unmangle()
		return self.pf['value'].values.copy()

	@property
	def pnames(self):
		self.unmangle()
		return self.pf.index.values.copy()

	def check_d_loglike(self, stylize=True, skip_zeros=False):
		"""
		Check that the analytic and finite-difference gradients are approximately equal.

		Primarily used for debugging.

		Parameters
		----------
		stylize : bool, default True
			See :ref:`check_gradient` for details.
		skip_zeros : bool, default False
			See :ref:`check_gradient` for details.

		Returns
		-------
		pandas.DataFrame or Stylized DataFrame
		"""
		from ..math.optimize import check_gradient
		epsilon=numpy.sqrt(numpy.finfo(float).eps)
		return check_gradient(
			self.loglike,
			self.d_loglike,
			self.pvals.copy(),
			names=self.pnames,
			stylize=stylize,
			skip_zeros=skip_zeros,
			epsilon=epsilon,
		)
