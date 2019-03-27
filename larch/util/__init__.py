from .signal_dict import SignalDict

import addict_yaml

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

dicta = Dict
