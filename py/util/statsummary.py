
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
		self.notes = set()
		for kw,val in kwargs.items():
			setattr(self,kw,val)


	def __iter__(self):
		self._iter_n = 0
		return self
	
	def __next__(self):
		if self.mean is None or self._iter_n > len(self.mean):
			raise StopIteration
		s = statistical_summary()
		s.mean = self.mean[self._iter_n]
		s.stdev = self.stdev[self._iter_n]
		s.minimum = self.minimum[self._iter_n]
		s.maximum = self.maximum[self._iter_n]
		s.n_positives = self.n_positives[self._iter_n]
		s.n_negatives = self.n_negatives[self._iter_n]
		s.n_zeros = self.n_zeros[self._iter_n]
		s.n_nonzeros = self.n_nonzeros[self._iter_n]
		s.mean_nonzero = self.mean_nonzero[self._iter_n]
		s.histogram = self.histogram[self._iter_n]
		self._iter_n += 1
		return s
		

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

	def empty(self):
		if self.mean is None and \
			self.stdev is None and \
			self.minimum is None and \
			self.maximum is None and \
			self.n_positives is None and \
			self.n_negatives is None and \
			self.n_zeros is None and \
			self.n_nonzeros is None and \
			self.unique_values is None and \
			self.mean_nonzero is None and \
			self.histogram is None and \
			len(self.notes)==0: return True
		return False

	@staticmethod
	def compute(xxx, histogram_bins='auto', count_uniques=False, dimzer=lambda x: x, full_xxx=None, weights=None, spark_kwargs=None):
		if spark_kwargs is None:
			spark_kwargs = {}
		if len(xxx)==0:
			return statistical_summary()
		else:
			ss = statistical_summary()
			if weights is None or not bool((weights!=1).any()):
				ss.mean = dimzer( numpy.mean(xxx,0) )
				ss.stdev = dimzer( numpy.std(xxx,0) )
				ss.minimum = dimzer( numpy.amin(xxx,0) )
				ss.maximum = dimzer( numpy.amax(xxx,0) )
				try:
					xxx_shape_1 = xxx.shape[1]
				except IndexError:
					ss.n_nonzeros = dimzer( numpy.count_nonzero(xxx) )
					ss.n_positives = dimzer( int(numpy.sum(xxx>0)) )
					ss.n_negatives = dimzer( int(numpy.sum(xxx<0)) )
					ss.n_zeros = dimzer( xxx.size-ss.n_nonzeros )
					ss.histogram = (spark_histogram(xxx, bins=histogram_bins, notetaker=ss.notes, data_for_bins=full_xxx, **spark_kwargs),)
				else:
					ss.n_nonzeros = tuple(numpy.count_nonzero(xxx[:,i]) for i in range(xxx_shape_1))
					ss.n_positives = tuple(int(numpy.sum(xxx[:,i]>0)) for i in range(xxx_shape_1))
					ss.n_negatives = tuple(int(numpy.sum(xxx[:,i]<0)) for i in range(xxx_shape_1))
					ss.n_zeros = tuple(xxx[:,i].size-numpy.count_nonzero(xxx[:,i]) for i in range(xxx_shape_1))
					if full_xxx is None:
						ss.histogram = tuple(spark_histogram(xxx[:,i], bins=histogram_bins, notetaker=ss.notes, data_for_bins=None, **spark_kwargs) for i in range(xxx_shape_1))
					else:
						ss.histogram = tuple(spark_histogram(xxx[:,i], bins=histogram_bins, notetaker=ss.notes, data_for_bins=full_xxx[:,i], **spark_kwargs) for i in range(xxx_shape_1))
				sumx_ = dimzer( numpy.sum(xxx,0) )
				try:
					ss.mean_nonzero = sumx_ / numpy.asarray(ss.n_nonzeros)
				except ValueError:
					ss.mean_nonzero = sumx_ / numpy.apply_along_axis(numpy.count_nonzero, 0, xxx)
				
				# Make sure that the histogram field is iterable
				if isinstance(ss.histogram, numpy.ndarray):
					ss.histogram = numpy.atleast_1d(ss.histogram)
				if count_uniques:
					q1,q2 = numpy.unique(xxx, return_counts=True)
					ss.unique_values = pandas.Series(q2,q1)
				return ss
			else:

				w = weights.flatten()
				ss.mean = dimzer( numpy.average(xxx, axis=0, weights=w) )
				variance = numpy.average((xxx-ss.mean)**2, axis=0, weights=w)
				ss.stdev = dimzer( numpy.average((xxx-ss.mean)**2, axis=0, weights=w) )
				ss.minimum = dimzer( numpy.amin(xxx[w>0],0) )
				ss.maximum = dimzer( numpy.amax(xxx[w>0],0) )
				try:
					xxx_shape_1 = xxx.shape[1]
				except IndexError:
					ss.n_nonzeros = dimzer( (w[xxx!=0]) )
					ss.n_positives = dimzer( (w[xxx>0]) )
					ss.n_negatives = dimzer( (w[xxx<0]) )
					ss.n_zeros = dimzer( (w[xxx==0]) )
					ss.histogram = (spark_histogram(xxx, bins=histogram_bins, notetaker=ss.notes, data_for_bins=full_xxx, **spark_kwargs),)
				else:
					ss.n_nonzeros = tuple( numpy.sum(w[xxx[:,i]!=0]) for i in range(xxx_shape_1))
					ss.n_positives = tuple( numpy.sum(w[xxx[:,i]>0]) for i in range(xxx_shape_1))
					ss.n_negatives = tuple( numpy.sum(w[xxx[:,i]<0]) for i in range(xxx_shape_1))
					ss.n_zeros = tuple( numpy.sum(w[xxx[:,i]==0]) for i in range(xxx_shape_1))
					if full_xxx is None:
						ss.histogram = tuple(spark_histogram(xxx[:,i], bins=histogram_bins, notetaker=ss.notes, data_for_bins=None, **spark_kwargs) for i in range(xxx_shape_1))
					else:
						ss.histogram = tuple(spark_histogram(xxx[:,i], bins=histogram_bins, notetaker=ss.notes, data_for_bins=full_xxx[:,i], **spark_kwargs) for i in range(xxx_shape_1))
					ss.notes.add('Graphs are unweighted representations.')

				w_nonzero = w.copy().reshape(w.shape[0],1) * numpy.ones([1,xxx.shape[1]])
				w_nonzero[xxx==0] = 0
				if sum(w_nonzero)==0:
					ss.mean_nonzero = numpy.zeros_like(ss.mean)
				else:
					ss.mean_nonzero = numpy.average(xxx, axis=0, weights=w_nonzero)

				# Make sure that the histogram field is iterable
				if isinstance(ss.histogram, numpy.ndarray):
					ss.histogram = numpy.atleast_1d(ss.histogram)
				if count_uniques:
					q1,q2 = numpy.unique(xxx, return_counts=True)
					ss.unique_values = pandas.Series(q2,q1)
				return ss




	@staticmethod
	def nan_compute(xxx, histogram_bins='auto', count_uniques=False, full_xxx=None, weights=None, spark_kwargs=None, bar_not_pie=True, dictionary=None):
		"""
		Compute statistics, ignoring NaNs.
		
		Parameters
		----------
		xxx : array
			Input data, 1-d or 2-d
		"""
		if spark_kwargs is None:
			spark_kwargs = {}
		if len(xxx)==0:
			return statistical_summary()
		else:
			ss = statistical_summary()
			if weights is None or not bool((weights!=1).any()):
				if numpy.all(numpy.isnan(xxx)):
					try:
						xxx_shape_1 = xxx.shape[1]
					except IndexError:
						# input data is 1-d, return only scalars
						ss.mean = numpy.nan
						ss.stdev = numpy.nan
						ss.minimum = numpy.nan
						ss.maximum = numpy.nan
						ss.n_nonzeros = 0
						ss.n_positives = 0
						ss.n_negatives = 0
						ss.n_zeros = 0
						ss.n_nans = int(numpy.sum(numpy.isnan(xxx)))
						ss.n_values = xxx.shape[0]
						ss.histogram = None
						ss.mean_nonzero = numpy.nan
					else:
						# input data is 1-d, return tuples
						ss.mean = numpy.asarray([numpy.nan for i in range(xxx_shape_1)])
						ss.stdev = numpy.asarray([numpy.nan for i in range(xxx_shape_1)])
						ss.minimum = numpy.asarray([numpy.nan for i in range(xxx_shape_1)])
						ss.maximum = numpy.asarray([numpy.nan for i in range(xxx_shape_1)])
						ss.n_nonzeros = numpy.asarray([0 for i in range(xxx_shape_1)])
						ss.n_positives = numpy.asarray([0 for i in range(xxx_shape_1)])
						ss.n_negatives = numpy.asarray([0 for i in range(xxx_shape_1)])
						ss.n_zeros = numpy.asarray([0 for i in range(xxx_shape_1)])
						ss.n_nans = numpy.asarray([int(numpy.sum(numpy.isnan(xxx[:,i]))) for i in range(xxx_shape_1)])
						ss.n_values = xxx.shape[0]
						ss.histogram = tuple(None for i in range(xxx_shape_1))
						ss.mean_nonzero = numpy.asarray([numpy.nan for i in range(xxx_shape_1)])
					
					# Make sure that the histogram field is iterable
					if isinstance(ss.histogram, numpy.ndarray):
						ss.histogram = numpy.atleast_1d(ss.histogram)
					if count_uniques:
						q1,q2 = numpy.unique(xxx, return_counts=True)
						ss.unique_values = pandas.Series([None],[None])
					return ss
				else:
					ss.mean = numpy.nanmean(xxx,0)
					ss.stdev = numpy.nanstd(xxx,0)
					ss.minimum = numpy.nanmin(xxx,0)
					ss.maximum = numpy.nanmax(xxx,0)
					try:
						xxx_shape_1 = xxx.shape[1]
					except IndexError:
						# input data is 1-d, return only scalars
						ss.n_nonzeros = numpy.count_nonzero(xxx)
						ss.n_positives = int(numpy.sum(xxx>0))
						ss.n_negatives = int(numpy.sum(xxx<0))
						ss.n_zeros = xxx.size-ss.n_nonzeros
						ss.n_nans = int(numpy.sum(numpy.isnan(xxx)))
						ss.n_values = xxx.shape[0]
						ss.histogram = spark_histogram(xxx, bins=histogram_bins, notetaker=ss.notes, data_for_bins=full_xxx,
														bar_not_pie=bar_not_pie, dictionary=dictionary, **spark_kwargs)
					else:
						# input data is 1-d, return tuples
						ss.n_nonzeros = numpy.asarray([numpy.count_nonzero(xxx[:,i]) for i in range(xxx_shape_1)])
						ss.n_positives = numpy.asarray([int(numpy.sum(xxx[:,i]>0)) for i in range(xxx_shape_1)])
						ss.n_negatives = numpy.asarray([int(numpy.sum(xxx[:,i]<0)) for i in range(xxx_shape_1)])
						ss.n_zeros = numpy.asarray([xxx[:,i].size-numpy.count_nonzero(xxx[:,i]) for i in range(xxx_shape_1)])
						ss.n_nans = numpy.asarray([int(numpy.sum(numpy.isnan(xxx[:,i]))) for i in range(xxx_shape_1)])
						ss.n_values = xxx.shape[0]
						if full_xxx is None:
							ss.histogram = tuple(spark_histogram(xxx[:,i], bins=histogram_bins, notetaker=ss.notes, data_for_bins=None,
																	bar_not_pie=bar_not_pie, dictionary=dictionary, **spark_kwargs) for i in range(xxx_shape_1))
						else:
							ss.histogram = tuple(spark_histogram(xxx[:,i], bins=histogram_bins, notetaker=ss.notes, data_for_bins=full_xxx[:,i],
																	bar_not_pie=bar_not_pie, dictionary=dictionary, **spark_kwargs) for i in range(xxx_shape_1))
					sumx_ = numpy.nansum(xxx,0)
					try:
						ss.mean_nonzero = sumx_ / numpy.asarray(ss.n_nonzeros)
					except ValueError:
						ss.mean_nonzero = sumx_ / numpy.apply_along_axis(numpy.count_nonzero, 0, xxx)
					
					# Make sure that the histogram field is iterable
					if isinstance(ss.histogram, numpy.ndarray):
						ss.histogram = numpy.atleast_1d(ss.histogram)
					if count_uniques:
						q1,q2 = numpy.unique(xxx, return_counts=True)
						ss.unique_values = pandas.Series(q2,q1)
					return ss
			else:
				raise NotImplementedError('weights are not implemented for nan_compute')

				w = weights.flatten()
				ss.mean = dimzer( numpy.average(xxx, axis=0, weights=w) )
				variance = numpy.average((xxx-ss.mean)**2, axis=0, weights=w)
				ss.stdev = dimzer( numpy.average((xxx-ss.mean)**2, axis=0, weights=w) )
				ss.minimum = dimzer( numpy.amin(xxx[w>0],0) )
				ss.maximum = dimzer( numpy.amax(xxx[w>0],0) )
				try:
					xxx_shape_1 = xxx.shape[1]
				except IndexError:
					ss.n_nonzeros = dimzer( (w[xxx!=0]) )
					ss.n_positives = dimzer( (w[xxx>0]) )
					ss.n_negatives = dimzer( (w[xxx<0]) )
					ss.n_zeros = dimzer( (w[xxx==0]) )
					ss.n_nans = 0 #todo
					ss.n_values = 0 #todo
					ss.histogram = (spark_histogram(xxx, bins=histogram_bins, notetaker=ss.notes, data_for_bins=full_xxx,
													bar_not_pie=bar_not_pie, dictionary=dictionary, **spark_kwargs),)
				else:
					ss.n_nonzeros = tuple( numpy.sum(w[xxx[:,i]!=0]) for i in range(xxx_shape_1))
					ss.n_positives = tuple( numpy.sum(w[xxx[:,i]>0]) for i in range(xxx_shape_1))
					ss.n_negatives = tuple( numpy.sum(w[xxx[:,i]<0]) for i in range(xxx_shape_1))
					ss.n_zeros = tuple( numpy.sum(w[xxx[:,i]==0]) for i in range(xxx_shape_1))
					ss.n_nans = 0 #todo
					ss.n_values = 0 #todo
					if full_xxx is None:
						ss.histogram = tuple(spark_histogram(xxx[:,i], bins=histogram_bins, notetaker=ss.notes, data_for_bins=None,
																bar_not_pie=bar_not_pie, dictionary=dictionary, **spark_kwargs) for i in range(xxx_shape_1))
					else:
						ss.histogram = tuple(spark_histogram(xxx[:,i], bins=histogram_bins, notetaker=ss.notes, data_for_bins=full_xxx[:,i],
																bar_not_pie=bar_not_pie, dictionary=dictionary, **spark_kwargs) for i in range(xxx_shape_1))
					ss.notes.add('Graphs are unweighted representations.')

				w_nonzero = w.copy().reshape(w.shape[0],1) * numpy.ones([1,xxx.shape[1]])
				w_nonzero[xxx==0] = 0
				if sum(w_nonzero)==0:
					ss.mean_nonzero = numpy.zeros_like(ss.mean)
				else:
					ss.mean_nonzero = numpy.average(xxx, axis=0, weights=w_nonzero)

				# Make sure that the histogram field is iterable
				if isinstance(ss.histogram, numpy.ndarray):
					ss.histogram = numpy.atleast_1d(ss.histogram)
				if count_uniques:
					q1,q2 = numpy.unique(xxx, return_counts=True)
					ss.unique_values = pandas.Series(q2,q1)
				return ss





	@staticmethod
	def compute_v3(xxx, histogram_bins='auto', count_uniques=False, dimzer=lambda x: x, xxt=None):
		if len(xxx)==0:
			return statistical_summary()
		else:
			ss = statistical_summary()
			ss.mean = dimzer( numpy.mean(xxx,0) )
			ss.stdev = dimzer( numpy.std(xxx,0) )
			ss.minimum = dimzer( numpy.amin(xxx,0) )
			ss.maximum = dimzer( numpy.amax(xxx,0) )
			try:
				xxx_shape_1 = xxx.shape[1]
			except IndexError:
				ss.n_nonzeros = dimzer( numpy.count_nonzero(xxx) )
				ss.n_positives = dimzer( int(numpy.sum(xxx>0)) )
				ss.n_negatives = dimzer( int(numpy.sum(xxx<0)) )
				ss.n_zeros = dimzer( xxx.size-ss.n_nonzeros )
				ss.histogram = (spark_histogram(xxx, bins=histogram_bins, notetaker=ss.notes),)
			else:
				ss.n_nonzeros = tuple(numpy.count_nonzero(xxx[:,i]) for i in range(xxx_shape_1))
				ss.n_positives = tuple(int(numpy.sum(xxx[:,i]>0)) for i in range(xxx_shape_1))
				ss.n_negatives = tuple(int(numpy.sum(xxx[:,i]<0)) for i in range(xxx_shape_1))
				ss.n_zeros = tuple(xxx[:,i].size-numpy.count_nonzero(xxx[:,i]) for i in range(xxx_shape_1))
				ss.histogram = tuple(spark_histogram(xxx[:,i], bins=histogram_bins, notetaker=ss.notes) for i in range(xxx_shape_1))
			sumx_ = dimzer( numpy.sum(xxx,0) )
			ss.mean_nonzero = sumx_ / numpy.asarray(ss.n_nonzeros)
#			if len(xxx.shape) == 1:
#				ss.histogram = [spark_histogram(xxx, bins=histogram_bins, notetaker=ss.notes),]
#			else:
#				ss.histogram = numpy.apply_along_axis(lambda x:[spark_histogram(x, bins=histogram_bins, notetaker=ss.notes)], 0, xxx).squeeze()
			
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




	@classmethod
	def compute_v2(cls, xxx, histogram_bins='auto', count_uniques=False, duo_filter=None):
		axis=0
		if len(xxx)==0:
			return cls()
		else:
			if duo_filter is None:
				ss = cls()
				ss.mean = numpy.atleast_1d( numpy.mean(xxx,axis) )
				ss.stdev = numpy.atleast_1d( numpy.std(xxx,axis) )
				ss.minimum = numpy.atleast_1d( numpy.amin(xxx,axis) )
				ss.maximum = numpy.atleast_1d( numpy.amax(xxx,axis) )
				try:
					xxx_shape_1 = xxx.shape[1]
				except IndexError:
					# xxx is one dimensional
					ss.n_nonzeros = numpy.atleast_1d( numpy.count_nonzero(xxx) )
					ss.n_positives = numpy.atleast_1d( int(numpy.sum(xxx>0)) )
					ss.n_negatives = numpy.atleast_1d( int(numpy.sum(xxx<0)) )
					ss.n_zeros = numpy.atleast_1d( xxx.size-ss.n_nonzeros )
					ss.histogram = (spark_histogram(xxx, bins=histogram_bins, notetaker=ss.notes, duo_filter=duo_filter),)
				else:
					ss.n_nonzeros = numpy.asarray([numpy.count_nonzero(xxx[:,i]) for i in range(xxx_shape_1)])
					ss.n_positives = numpy.asarray(tuple(int(numpy.sum(xxx[:,i]>0)) for i in range(xxx_shape_1)))
					ss.n_negatives = numpy.asarray(tuple(int(numpy.sum(xxx[:,i]<0)) for i in range(xxx_shape_1)))
					ss.n_zeros = numpy.asarray(tuple(xxx[:,i].size-numpy.count_nonzero(xxx[:,i]) for i in range(xxx_shape_1)))
					ss.histogram = (tuple(
						spark_histogram(xxx[:,i], bins=histogram_bins, notetaker=ss.notes, duo_filter=duo_filter)
						for i in range(xxx_shape_1))
					)
				sumx_ = numpy.atleast_1d( numpy.sum(xxx,axis) )
				ss.mean_nonzero = sumx_ / numpy.asarray(ss.n_nonzeros)
				if count_uniques:
					q1,q2 = numpy.unique(xxx, return_counts=True)
					ss.unique_values = pandas.Series(q2,q1)
				return ss
			else:
				ss = cls()
				ss.mean = numpy.atleast_1d( numpy.mean(xxx[duo_filter],axis) ), numpy.atleast_1d( numpy.mean(xxx[~duo_filter],axis) )
				ss.stdev = numpy.atleast_1d( numpy.std(xxx[duo_filter],axis) ), numpy.atleast_1d( numpy.std(xxx[~duo_filter],axis) )
				ss.minimum = numpy.atleast_1d( numpy.amin(xxx[duo_filter],axis) ), numpy.atleast_1d( numpy.amin(xxx[~duo_filter],axis) )
				ss.maximum = numpy.atleast_1d( numpy.amax(xxx[duo_filter],axis) ), numpy.atleast_1d( numpy.amax(xxx[~duo_filter],axis) )
				try:
					xxx_shape_1 = xxx.shape[1]
				except IndexError:
					# xxx is one dimensional
					ss.n_nonzeros = numpy.atleast_1d( numpy.count_nonzero(xxx[duo_filter]) ), numpy.atleast_1d( numpy.count_nonzero(xxx[~duo_filter]) )
					ss.n_positives = numpy.atleast_1d( int(numpy.sum(xxx[duo_filter]>0)) ), numpy.atleast_1d( int(numpy.sum(xxx[~duo_filter]>0)) )
					ss.n_negatives = numpy.atleast_1d( int(numpy.sum(xxx[duo_filter]<0)) ), numpy.atleast_1d( int(numpy.sum(xxx[~duo_filter]<0)) )
					ss.n_zeros = numpy.atleast_1d( xxx[duo_filter].size-ss.n_nonzeros[0] ), numpy.atleast_1d( xxx[~duo_filter].size-ss.n_nonzeros[1] )
					htemp = spark_histogram(xxx, bins=histogram_bins, notetaker=ss.notes, duo_filter=duo_filter)
					ss.histogram = (htemp[0],), (htemp[1],)
				else:
					ss.n_nonzeros  = numpy.asarray([numpy.count_nonzero(xxx[duo_filter,i]) for i in range(xxx_shape_1)]), numpy.asarray([numpy.count_nonzero(xxx[~duo_filter,i]) for i in range(xxx_shape_1)])
					ss.n_positives = numpy.asarray(tuple(int(numpy.sum(xxx[duo_filter,i]>0)) for i in range(xxx_shape_1))), numpy.asarray(tuple(int(numpy.sum(xxx[~duo_filter,i]>0)) for i in range(xxx_shape_1)))
					ss.n_negatives = numpy.asarray(tuple(int(numpy.sum(xxx[duo_filter,i]<0)) for i in range(xxx_shape_1))), numpy.asarray(tuple(int(numpy.sum(xxx[~duo_filter,i]<0)) for i in range(xxx_shape_1)))
					ss.n_zeros = numpy.asarray(tuple(xxx[duo_filter,i].size-numpy.count_nonzero(xxx[duo_filter,i]) for i in range(xxx_shape_1))), numpy.asarray(tuple(xxx[~duo_filter,i].size-numpy.count_nonzero(xxx[~duo_filter,i]) for i in range(xxx_shape_1)))
					htemps = tuple(
						spark_histogram(xxx[:,i], bins=histogram_bins, notetaker=ss.notes, duo_filter=duo_filter)
						for i in range(xxx_shape_1))
					ss.histogram = [h[0] for h in htemps], [h[1] for h in htemps]
				sumx_ = numpy.atleast_1d( numpy.sum(xxx[duo_filter],axis) ), numpy.atleast_1d( numpy.sum(xxx[~duo_filter],axis) )
				ss.mean_nonzero = sumx_[0] / numpy.asarray(ss.n_nonzeros[0]), sumx_[1] / numpy.asarray(ss.n_nonzeros[1])
				if count_uniques:
					q1,q2 = numpy.unique(xxx[duo_filter], return_counts=True)
					q3,q4 = numpy.unique(xxx[~duo_filter], return_counts=True)
					ss.unique_values = pandas.Series(q2,q1), pandas.Series(q4,q3)
				return ss

