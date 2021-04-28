from .signal_dict import SignalDict

import pprint
import datetime
from .addict_yaml import Dict
from .data_expansion import piece, hard_sigmoid


def _prettyprint_fallback(x):
	if isinstance(x, dictx):
		return repr(x)
	else:
		return pprint.pformat(x)



class dictx(dict):
	"""Python dict with attribute access and xml output."""

	def __repr__(self):
		if self.keys():
			m = max(map(len, list(str(_) for _ in self.keys()))) + 1
			return '\n'.join(['┣'+str(k).rjust(m) + ': ' + _prettyprint_fallback(v).replace('\n','\n┃'+' '*(m+2)) for k, v in self.items()])
		else:
			return self.__class__.__name__ + "()"

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
					if isinstance(v, str) and '\n' not in v and v[:1] != "#":
						raise ValueError()
					v_ = Show(v)
				except (AttributeError, ValueError):
					plaintext = pprint.pformat(v)
					if isinstance(v, datetime.timedelta):
						plaintext = str(v)
					if "\n" in plaintext:
						tr.elem('td', style='text-align:left;').elem('pre', text=plaintext)
					else:
						tr.elem('td', style='text-align:left;', text=plaintext)
				else:
					tr.elem('td', style='text-align:left;') << v_
		else:
			tr = t.elem('tr')
			tr.elem('td', text="<empty>")
		return x

	def _repr_html_(self):
		return self.__xml__().tostring()

	def copy(self):
		return dictx(super().copy())

dicta = Dict


# Add statistics to pandas.DataFrame and pandas.Series
import pandas
from .statistics import statistics_for_dataframe, statistics_for_array5, uniques, invmap
pandas.DataFrame.statistics = statistics_for_dataframe
pandas.Series.statistics = statistics_for_array5
pandas.Series.uniques = uniques
pandas.Series.invmap = invmap

from .dataframe import compute
pandas.DataFrame.compute = compute
