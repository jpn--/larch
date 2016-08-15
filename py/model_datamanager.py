from .util.attribute_dict import dictal


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
	def Choice(self):
		return self._access("Choice")

	@property
	def Avail(self):
		return self._access("Avail")

	utilityco = UtilityCO
	utilityca = UtilityCA
	choice = Choice
	avail = Avail


	def needs(self):
		return dictal(self._model.needs())



class WorkspaceManager:
	"""Manages computational arrays for a :class:`Model`."""

	def __init__(self, model):
		self._model = model

	@property
	def probability(self):
		return self._model.Probability()

	@property
	def utility(self):
		if m.Utility().shape == (0,):
			raise TypeError("this model did not allocate a seperate utility computational array")
		return self._model.Utility()


