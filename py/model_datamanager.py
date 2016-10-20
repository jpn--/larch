from .util.attribute_dict import dictal
from .util.text_manip import case_insensitive_close_matches

class DataManager:
	"""Manages data for a :class:`Model`."""

	def __init__(self, model, readonly=True):
		self._model = model
		self._readonly = readonly

	def _access(self, name):
		if self._readonly:
			return self._model.Data(name)
		else:
			return self._model.DataEdit(name)
	
	@property
	def UtilityCO(self):
		return self._access("UtilityCO")

	@property
	def UtilityCA(self):
		return self._access("UtilityCA")

	@property
	def QuantityCA(self):
		return self._access("QuantityCA")

	@property
	def Choice(self):
		return self._access("Choice")

	@property
	def Avail(self):
		return self._access("Avail")

	@property
	def Weight(self):
		return self._access("Weight")

	utilityco = UtilityCO
	utilityca = UtilityCA
	choice = Choice
	avail = Avail
	quantity = QuantityCA
	weight = Weight

	def needs(self):
		return dictal(self._model.needs())

	@property
	def utilityco_vars(self):
		return self._model.needs()['UtilityCO'].get_variables()

	@property
	def utilityca_vars(self):
		return self._model.needs()['UtilityCA'].get_variables()

	@property
	def quantity_vars(self):
		return self._model.needs()['QuantityCA'].get_variables()

	def utilityco_varindex(self, var):
		vars = self._model.needs()['UtilityCO'].get_variables()
		try:
			return vars.index(var)
		except ValueError:
			case_insensitive_close_matches(var, vars, excpt=KeyError)

	def utilityca_varindex(self, var):
		vars = self._model.needs()['UtilityCA'].get_variables()
		try:
			return vars.index(var)
		except ValueError:
			case_insensitive_close_matches(var, vars, excpt=KeyError)




class WorkspaceManager:
	"""Manages computational arrays for a :class:`Model`."""

	def __init__(self, model):
		self._model = model

	@property
	def probability(self):
		return self._model.Probability()

	@property
	def utility(self):
		if self._model.Utility().shape == (0,):
			raise TypeError("this model did not allocate a seperate utility computational array")
		return self._model.Utility()


