
import numpy, pandas
from .parameter_collection import ParameterCollection
from .data_collection import DataCollection
from .workspace_collection import WorkspaceCollection

class Model(ParameterCollection):

	def __init__(self, *,
				 parameters = (),
				 cases = (),
				 alts = (),
				 graph = None,
				 datasource = None,
				 **kwarg):

		if datasource is not None:
			if isinstance(cases, (numpy.ndarray, pandas.Series, pandas.DataFrame, list, tuple, set)) and len(cases)>0:
				pass # use the override cases
			else:
				cases = datasource.caseids()
			if isinstance(alts, (numpy.ndarray, pandas.Series, pandas.DataFrame, list, tuple, set)) and len(alts)>0:
				pass # use the override alts
			else:
				alts = datasource.alternative_codes()

		super().__init__(names=parameters, altindex=alts, **kwarg)

		self._graph = graph

		self.data = DataCollection(caseindex=cases, altindex=alts)
		self.work = WorkspaceCollection(data_coll=self.data, parameter_coll=self, graph=self._graph)

