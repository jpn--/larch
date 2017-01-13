from .logging import flogger, easy



import resource
def curmem():
	return "{}MB".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024)



flog = flogger(level=30, label="flog")
flog('QUICKLOG INITIALIZED')

def flogm(msg, *arg, **kwarg):
	flog(msg.format(*arg, **kwarg) +'   ~'+ curmem())
