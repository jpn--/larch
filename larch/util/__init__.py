from .signal_dict import SignalDict

import addict_yaml
import pprint

class Dict(addict_yaml.Dict):

	def __xml__(self):
		from xmle import Elem
		x = Elem('div')
		t = x.elem('table', style="margin-top:1px;")
		if len(self):
			tr = t.elem('tr')
			tr.elem('th', text="key")
			tr.elem('th', text='value', style='text-align:left;')
			for k,v in self.items():
				tr = t.elem('tr')
				tr.elem('td', text=str(k))
				try:
					v_ = v.__xml__()
				except AttributeError:
					tr.elem('td', text=str(v), style='text-align:left;')
				else:
					tr.elem('td') << v_
		else:
			tr = t.elem('tr')
			tr.elem('td', text="<empty>")
		return x

	def _repr_html_(self):
		return self.__xml__().tostring()

	def _ipython_display_(self):
		raise AttributeError('_ipython_display_')

	def __getattr__(self, item):
		if item[:5]=="_repr" and item[-1]=="_":
			raise AttributeError(item)
		return self.__getitem__(item)


class dictx(dict):
	"""Python dict with attribute access and xml output."""

	def __getattr__(self, item):
		if item[0]=="_" and item[-1]=="_":
			raise AttributeError(item)
		return self.__getitem__(item)

	def __setattr__(self, name, value):
		if hasattr(dictx, name):
			raise AttributeError("'dictx' object attribute "
								 "'{0}' is read-only".format(name))
		else:
			self[name] = value

	def __xml__(self):
		from xmle import Elem, Show
		x = Elem('div')
		t = x.elem('table', style="margin-top:1px;")
		if len(self):
			tr = t.elem('tr')
			tr.elem('th', text="key")
			tr.elem('th', text='value', style='text-align:left;')
			for k,v in self.items():
				tr = t.elem('tr')
				tr.elem('td', text=str(k))
				try:
					v_ = Show(v)
				except AttributeError:
					tr.elem('td', text=pprint.pformat(v), style='text-align:left;')
				else:
					tr.elem('td') << v_
		else:
			tr = t.elem('tr')
			tr.elem('td', text="<empty>")
		return x

	def _repr_html_(self):
		return self.__xml__().tostring()


dicta = Dict


# Add statistics to pandas.DataFrame and pandas.Series
import pandas
from .statistics import statistics_for_dataframe, statistics_for_array5
pandas.DataFrame.statistics = statistics_for_dataframe
pandas.Series.statistics = statistics_for_array5

from .dataframe import compute
pandas.DataFrame.compute = compute
