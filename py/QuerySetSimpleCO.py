
def info(self):
	s = ""
	from .core import IntStringDict
	d = lambda x: x if not isinstance(x,IntStringDict) else dict(x)
	p = lambda x: str(d(x)).replace("\n","\n           \t")
	s += "idco query:\t{}\n".format(p(self.get_idco_query()))
	s += "alts query:\t{}\n".format(p(self.get_alts_query()))
	s += "choice:    \t{}\n".format(p(self.choice))
	s += "weight:    \t{}\n".format(p(self.weight))
	s += "avail:     \t{}".format(p(self.avail))
	return s


weight = property(lambda self: self.get_weight_column(), lambda self,w: self.set_weight_column(str(w)), None, "The weight indicator")


def set_choice(self, x):
	if isinstance(x,str):
		self.set_choice_column(x)
	else:
		self.set_choice_column_map(x)

def get_choice(self):
	x = self.get_choice_column()
	if x!="":
		return x
	return self.get_choice_column_map()


choice = property(lambda self: self.get_choice(), lambda self,w: self.set_choice(w), None, "The choice indicator")


def set_avail(self, x):
	if isinstance(x,int) and x==1:
		self.set_avail_all()
	elif isinstance(x,bool) and x is True:
		self.set_avail_all()
	else:
		self.set_avail_column_map(x)

def get_avail(self):
	if self.all_alts_always_available():
		return True
	return self.get_avail_column_map()

avail = property(lambda self: self.get_avail(), lambda self,w: self.set_avail(w), None, "The avail indicator")




alts_query = property(lambda self: self.get_alts_query(), lambda self,w: self.set_alts_query(w), None, "The alts query")
idco_query = property(lambda self: self.get_idco_query(), lambda self,w: self.set_idco_query(w), None, "The idco query")








## Load these methods into core.QuerySetSimpleCO

import_these = dict(locals())
from .core import QuerySetSimpleCO
for k,f in import_these.items():
	if len(k)>0 and k[0]!='_':
		setattr(QuerySetSimpleCO,k,f)
