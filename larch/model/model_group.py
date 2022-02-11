import numpy as np
import pandas as pd

from .abstract_model import AbstractChoiceModel
from . import persist_flags
from ..exceptions import ParameterNotInModelWarning
from .constraints import ParametricConstraintList
from collections.abc import MutableSequence

class ModelGroup(AbstractChoiceModel, MutableSequence):

	constraints = ParametricConstraintList()

	def __init__(
			self,
			models,
			*,
			parameters=None,
			frame=None,
			title=None,
			dataservice=None,
			constraints=None,
	):
		super().__init__(
			parameters=parameters,
			frame=frame,
			title=title,
		)
		self._k_models = list()
		for model in models:
			if isinstance(model, ModelGroup):
				self._k_models.extend(model._k_models)
			else:
				self._k_models.append(model)
		self._dataservice = dataservice
		self._dataframes = None
		self._mangled = True
		self.constraints = constraints

	def __getitem__(self, x):
		return self._k_models[x]

	def __setitem__(self, i, value):
		assert isinstance(value, AbstractChoiceModel)
		self._k_models[i] = value

	def __delitem__(self, x):
		del self._k_models[x]

	def __len__(self):
		return len(self._k_models)

	def __contains__(self, x):
		return (x in self.pf.index)

	def insert(self, i, value):
		assert isinstance(value, AbstractChoiceModel)
		self._k_models.insert(i,value)

	@property
	def dataframes(self):
		return 'access dataframes of group members individually'

	@property
	def n_cases(self):
		"""int : Total number of cases in the attached data of all grouped models."""
		return sum(k.n_cases for k in self._k_models)

	def total_weight(self):
		"""
		The total weight of cases in the attached data of all grouped models.

		Returns
		-------
		float
		"""
		return sum(k.total_weight() for k in self._k_models)

	def unmangle(self, force=False):
		super().unmangle(force)
		for k in self._k_models:
			rows_to_import = [i for i in k.pf.index if i not in self._frame.index]
			if len(rows_to_import):
				new_frame = pd.concat([
					self._frame, k.pf.loc[rows_to_import]
				]).astype(self._frame.dtypes)

				if not (
					np.array_equal(new_frame['value'].values, self._frame['value'].values)
					and np.array_equal(new_frame.index, self._frame.index)
				):
					frame_values_have_changed = True
				else:
					frame_values_have_changed = False

				self._frame = new_frame
				if frame_values_have_changed:
					self._frame_values_have_changed()

			k.unmangle(force)

	def set_values(self, values=None, *, respect_holdfast=True, **kwargs):
		"""
		Set the parameter values for one or more parameters.

		Parameters
		----------
		values : {'null', 'init', 'best', array-like, dict, scalar}, optional
			New values to set for the parameters.
			If 'null' or 'init', the current values are set
			equal to the null or initial values given in
			the 'nullvalue' or 'initvalue' column of the
			parameter frame, respectively.
			If 'best', the current values are set equal to
			the values given in the 'best' column of the
			parameter frame, if that columns exists,
			otherwise a ValueError exception is raised.
			If given as array-like, the array must be a
			vector with length equal to the length of the
			parameter frame, and the given vector will replace
			the current values.  If given as a dictionary,
			the dictionary is used to update `kwargs` before
			they are processed.
		kwargs : dict
			Any keyword arguments (or if `values` is a
			dictionary) are used to update the included named
			parameters only.  A warning will be given if any key of
			the dictionary is not found among the existing named
			parameters in the parameter frame, and the value
			associated with that key is ignored.  Any parameters
			not named by key in this dictionary are not changed.

		Notes
		-----
		Setting parameters both in the `values` argument and
		through keyword assignment is not explicitly disallowed,
		although it is not recommended and may be disallowed in the future.

		Raises
		------
		ValueError
			If setting to 'best' but there is no 'best' column in
			the `pf` parameters DataFrame.
		"""
		super().set_values(values=values, respect_holdfast=respect_holdfast, **kwargs)
		self._frame_values_have_changed()

	def _frame_values_have_changed(self):
		vals = self.pf.value.copy()
		import warnings
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=ParameterNotInModelWarning)
			for k in self._k_models:
				k.set_values(**vals)
				k._frame = k._frame.copy()
		for k in self._k_models:
			k._frame_values_have_changed()

	def __prep_for_compute(self, x=None):
		self.unmangle()
		if x is not None:
			self.set_values(x)
		vals = self.pf.value.copy()
		import warnings
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=ParameterNotInModelWarning)
			for k in self._k_models:
				k.set_values(**vals)

	def load_data(self, dataservice=None, autoscale_weights=True, log_warnings=True):
		for k in self._k_models:
			k.load_data(
				dataservice=dataservice,
				autoscale_weights=autoscale_weights,
				log_warnings=log_warnings,
			)

	def loglike(
			self,
			x=None,
			*,
			start_case=0,
			stop_case=-1,
			step_case=1,
			persist=0,
			leave_out=-1,
			keep_only=-1,
			subsample=-1,
			return_series=True,
			probability_only=False,
	):
		"""
		Compute a log likelihood value.

		Parameters
		----------
		x : {'null', 'init', 'best', array-like, dict, scalar}, optional
			Values for the parameters.  See :ref:`set_values` for details.

		Returns
		-------
		dictx
			The log likelihood is given by key 'll' and the first derivative by key 'dll'.
			Other arrays are also included if `persist` is set to True.

		"""

		if start_case != 0:
			raise NotImplementedError('start_case != 0')
		if stop_case != -1:
			raise NotImplementedError('stop_case != -1')
		if step_case != 1:
			raise NotImplementedError('step_case != 1')
		if leave_out != -1:
			raise NotImplementedError('leave_out != -1')
		if keep_only != -1:
			raise NotImplementedError('keep_only != -1')
		if subsample != -1:
			raise NotImplementedError('subsample != -1')
		if probability_only:
			raise NotImplementedError('probability_only != False')

		from ..util import dictx
		self.__prep_for_compute(x)
		ll2_parts = [m.loglike(persist=persist) for m in self._k_models]
		if not persist:
			result = sum(ll2_parts)
			self._check_if_best(result)
			return result
		ll2 = dictx(
			ll=sum(y.ll for y in ll2_parts),
		)
		if persist & persist_flags.PERSIST_PARTS:
			ll2.ll_parts = [y.ll for y in ll2_parts]
		for key in ll2_parts[0].keys():
			if key != 'll':
				ll2[key] = list(y[key] for y in ll2_parts)
		self._check_if_best(ll2.ll)
		return ll2


	def loglike2(
			self,
			x=None,
			*,
			start_case=0,
			stop_case=-1,
			step_case=1,
			persist=0,
			leave_out=-1,
			keep_only=-1,
			subsample=-1,
			return_series=True,
			probability_only=False,

	):
		"""
		Compute a log likelihood value and its first derivative.

		Parameters
		----------
		x : {'null', 'init', 'best', array-like, dict, scalar}, optional
			Values for the parameters.  See :ref:`set_values` for details.

		Returns
		-------
		dictx
			The log likelihood is given by key 'll' and the first derivative by key 'dll'.
			Other arrays are also included if `persist` is set to True.

		"""

		if start_case != 0:
			raise NotImplementedError('start_case != 0')
		if stop_case != -1:
			raise NotImplementedError('stop_case != -1')
		if step_case != 1:
			raise NotImplementedError('step_case != 1')
		if leave_out != -1:
			raise NotImplementedError('leave_out != -1')
		if keep_only != -1:
			raise NotImplementedError('keep_only != -1')
		if subsample != -1:
			raise NotImplementedError('subsample != -1')
		if probability_only:
			raise NotImplementedError('probability_only != False')

		from ..util import dictx
		self.__prep_for_compute(x)
		ll2_parts = [m.loglike2(persist=persist) for m in self._k_models]
		dll = ll2_parts[0].dll
		for y in ll2_parts[1:]:
			dll = dll.add(y.dll, fill_value=0)
		dll = dll[self.pf.index]
		ll2 = dictx(
			ll=sum(y.ll for y in ll2_parts),
			dll=dll,
		)
		for key in ll2_parts[0].keys():
			if key not in {'ll','dll'}:
				ll2[key] = list(y[key] for y in ll2_parts)
		self._check_if_best(ll2.ll)
		return ll2

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
		problems = []
		for k in self._k_models:
			k.unmangle(True)
			problems.append(k.doctor(
				repair_ch_av=repair_ch_av,
				repair_ch_zq=repair_ch_zq,
				repair_asc=repair_asc,
				repair_noch_nowt=repair_noch_nowt,
				repair_nan_wt=repair_nan_wt,
				repair_nan_data_co=repair_nan_data_co,
				verbose=verbose,
			))
		return problems

	def to_xlsx(self, filename, save_now=True, **kwargs):
		from ..util.excel import _make_excel_writer
		_make_excel_writer(self, filename, save_now=save_now, **kwargs)
