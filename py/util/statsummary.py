
import numpy
from .plotting import spark_histogram

class statistical_summary():

	def __init__(self, **kwargs):
		self.mean = None
		self.stdev = None
		# Range
		self.minimum = None
		self.maximum = None
		# Counts
		self.n_positives = None
		self.n_negatives = None
		self.n_zeros = None
		self.n_nonzeros = None
		# Adj Stats
		self.mean_nonzero = None
		# Figures
		self.histogram = None
		for kw,val in kwargs.items():
			setattr(self,kw,val)
	
	@staticmethod
	def compute(xxx):
		ss = statistical_summary()
		ss.mean = numpy.mean(xxx,0)
		ss.stdev = numpy.std(xxx,0)
		ss.minimum = numpy.amin(xxx,0)
		ss.maximum = numpy.amax(xxx,0)
		ss.n_nonzeros = tuple(numpy.count_nonzero(xxx[:,i]) for i in range(xxx.shape[1]))
		ss.n_positives = tuple(int(numpy.sum(xxx[:,i]>0)) for i in range(xxx.shape[1]))
		ss.n_negatives = tuple(int(numpy.sum(xxx[:,i]<0)) for i in range(xxx.shape[1]))
		ss.n_zeros = tuple(xxx[:,i].size-numpy.count_nonzero(xxx[:,i]) for i in range(xxx.shape[1]))
		sumx_ = numpy.sum(xxx,0)
		ss.mean_nonzero = sumx_ / numpy.asarray(ss.n_nonzeros)
		ss.histogram = numpy.apply_along_axis(lambda x:[spark_histogram(x)], 0, xxx).squeeze()
		return ss
