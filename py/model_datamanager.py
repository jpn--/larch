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