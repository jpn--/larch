



def info(self):
	print("QuerySetTwoTable.info")

def info(self):
	s = ""
	from .core import IntStringDict
	d = lambda x: x if not isinstance(x,IntStringDict) else dict(x)
	p = lambda x: str(d(x)).replace("\n","\n           \t")
	s += "idco query:\t{}\n".format(p(self.get_idco_query()))
	s += "idca query:\t{}\n".format(p(self.get_idca_query()))
	s += "alts query:\t{}\n".format(p(self.get_alts_query()))
	try:
		ch = self.choice
	except AttributeError:
		ch = "<error/not implemented>"
	try:
		wg = self.weight
	except AttributeError:
		wg = "<error/not implemented>"
	try:
		av = self.avail
	except AttributeError:
		av = "<error>"
	s += "choice:    \t{}\n".format(p(ch))
	s += "weight:    \t{}\n".format(p(wg))
	s += "avail:     \t{}".format(p(av))
	return s




alts_query = property(lambda self: self.get_alts_query(), lambda self,w: self.set_alts_query(w), None, "The alts query")
idco_query = property(lambda self: self.get_idco_query(), lambda self,w: self.set_idco_query(w), None, "The idco query")
idco_query = property(lambda self: self.get_idca_query(), lambda self,w: self.set_idca_query(w), None, "The idca query")



def set_avail(self, x):
	if isinstance(x,int) and x==1:
		self.set_avail_all()
	elif isinstance(x,bool) and x is True:
		self.set_avail_all()
	elif isinstance(x,str):
		self.set_avail_ca_column(x)
	else:
		self.set_avail_co_column_map(x)

def get_avail(self):
	if self.all_alts_always_available():
		return True
	if self.get_avail_ca_column()!="" and len(self.get_avail_co_column_map())==0:
		return self.get_avail_ca_column()
	if self.get_avail_ca_column()=="" and len(self.get_avail_co_column_map())>0:
		return self.get_avail_co_column_map()
	raise LarchError("bad avail specification")

avail = property(lambda self: self.get_avail(), lambda self,w: self.set_avail(w), None, "The avail indicator")





## Load these methods into core.QuerySetTwoTable

import_these = dict(locals())
from .core import QuerySetTwoTable
for k,f in import_these.items():
	if len(k)>0 and k[0]!='_':
		setattr(QuerySetTwoTable,k,f)


