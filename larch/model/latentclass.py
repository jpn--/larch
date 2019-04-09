
import numpy
import pandas
from typing import MutableMapping

from ..util import Dict
from ..dataframes import DataFrames
from ..model.controller import ParameterNotInModelWarning


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

	def class_membership_probability(self, x=None):
		return self._k_membership.probability(x, return_dataframe='idco')

	def probability(self, x=None):
		self.__prep_for_compute(x)
		p = numpy.zeros([self.dataframes.n_cases, self.dataframes.n_alts])
		import warnings
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=ParameterNotInModelWarning)
			k_membership_probability = self.class_membership_probability()
			for k_name, k_model in self._k_models.items():
				p += (
						numpy.asarray( k_model.probability()[:,:self.dataframes.n_alts] )
						* k_membership_probability.loc[:,k_name].values[:, None]
				)
		return pandas.DataFrame(
			p,
			index=self._dataframes.caseindex,
			columns=self._dataframes.alternative_codes(),
		)

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

