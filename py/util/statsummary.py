
import numpy
import pandas
from .plotting import spark_histogram

class statistical_summary():

	def __init__(self, xxx=None, *arg, **kwargs):
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
		self.unique_values = None
		# Adj Stats
		self.mean_nonzero = None
		# Figures
		self.histogram = None
		for kw,val in kwargs.items():
			setattr(self,kw,val)

	def __repr__(self):
		s = "<statistical_summary>"
		s += "\n          mean: {}".format(self.mean)
		s += "\n         stdev: {}".format(self.stdev)
		s += "\n       minimum: {}".format(self.minimum)
		s += "\n       maximum: {}".format(self.maximum)
		s += "\n   n_positives: {}".format(self.n_positives)
		s += "\n   n_negatives: {}".format(self.n_negatives)
		s += "\n       n_zeros: {}".format(self.n_zeros)
		s += "\n    n_nonzeros: {}".format(self.n_nonzeros)
		s += "\n  mean_nonzero: {}".format(self.mean_nonzero)
		if self.unique_values:
			s += "\n unique values: {}".format(repr(self.unique_values).replace("\n","\n                "))
		return s

	@staticmethod
	def compute(xxx, histogram_bins='auto', count_uniques=False):
		ss = statistical_summary()
		ss.mean = numpy.mean(xxx,0)
		ss.stdev = numpy.std(xxx,0)
		ss.minimum = numpy.amin(xxx,0)
		ss.maximum = numpy.amax(xxx,0)
		try:
			xxx_shape_1 = xxx.shape[1]
		except IndexError:
			ss.n_nonzeros = numpy.count_nonzero(xxx)
			ss.n_positives = int(numpy.sum(xxx>0))
			ss.n_negatives = int(numpy.sum(xxx<0))
			ss.n_zeros = xxx.size-ss.n_nonzeros
		else:
			ss.n_nonzeros = tuple(numpy.count_nonzero(xxx[:,i]) for i in range(xxx_shape_1))
			ss.n_positives = tuple(int(numpy.sum(xxx[:,i]>0)) for i in range(xxx_shape_1))
			ss.n_negatives = tuple(int(numpy.sum(xxx[:,i]<0)) for i in range(xxx_shape_1))
			ss.n_zeros = tuple(xxx[:,i].size-numpy.count_nonzero(xxx[:,i]) for i in range(xxx_shape_1))
		sumx_ = numpy.sum(xxx,0)
		ss.mean_nonzero = sumx_ / numpy.asarray(ss.n_nonzeros)
		ss.notes = set()
		if len(xxx.shape) == 1:
			ss.histogram = [spark_histogram(xxx, bins=histogram_bins, notetaker=ss.notes),]
		else:
			ss.histogram = numpy.apply_along_axis(lambda x:[spark_histogram(x, bins=histogram_bins, notetaker=ss.notes)], 0, xxx).squeeze()
		# Make sure that the histogram field is iterable
		if isinstance(ss.histogram, numpy.ndarray):
			ss.histogram = numpy.atleast_1d(ss.histogram)
#		try:
#			iter(ss.histogram)
#		except:
#			ss.histogram = [ss.histogram, ]
		if count_uniques:
			q1,q2 = numpy.unique(xxx, return_counts=True)
			ss.unique_values = pandas.Series(q2,q1)
		return ss
