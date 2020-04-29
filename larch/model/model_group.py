

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
		self._k_models = list(models)
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

	def insert(self, i, value):
		assert isinstance(value, AbstractChoiceModel)
		self._k_models.insert(i,value)

	@property
	def dataframes(self):
		return 'access dataframes of group members individually'

	@property
	def n_cases(self):
		"""int : Total number of cases in the attached dataframes of all grouped models."""
		return sum(k.n_cases for k in self._k_models)

	def total_weight(self):
		"""
		The total weight of cases in the attached dataframes of all grouped models.

		Returns
		-------
		float
		"""
		return sum(k.total_weight() for k in self._k_models)

	def unmangle(self):
		super().unmangle()
		for k in self._k_models:
			for i in k.pf.index:
				if i not in self._frame.index:
					self._frame.loc[i] = k.pf.loc[i]

	def _frame_values_have_changed(self):
		for k in self._k_models:
			k._frame_values_have_changed()

	def __prep_for_compute(self, x=None):
		self.unmangle()
		if x is not None:
			self.set_values(x)
		vals = self.pf.value
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
		Compute a log likelihood value and it first derivative.

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
		Compute a log likelihood value and it first derivative.

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
