



def info(self):
	print("QuerySetTwoTable.info")







## Load these methods into core.QuerySetTwoTable

import_these = dict(locals())
from .core import QuerySetTwoTable
for k,f in import_these.items():
	if len(k)>0 and k[0]!='_':
		setattr(QuerySetTwoTable,k,f)
