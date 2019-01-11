


import logging
import numpy
import pandas
import os
from appdirs import user_cache_dir
from sklearn.externals import joblib

from .general_precision import l4_float_dtype


def user_cache_file(filename, appname=None, appauthor=None, version=None, opinion=True):
	d = user_cache_dir(appname=appname, appauthor=appauthor, version=version, opinion=opinion)
	os.makedirs(d, exist_ok=True)
	return os.path.join(d, filename)


class Prelearner():

	def __init__(
			self,
			training_X,
			training_Y,
			training_W=None,
			classifier=None,
			fit=True,
			cache_file=None,
			output_name='prelearned_utility',
			appname='larch',
			**kwargs,
	):
		"""

		Parameters
		----------
		training_X : pandas.DataFrame
		training_Y
		training_Y
		classifier
		fit
		cache_file : str, optional
			A cache file name to store the trained prelearner.  If just a filename is given,
			it will be stored in `appdirs.user_cache_file()`. If instead an absolute path or
			a relative path beginning with '.' is given, that location will be used.
			If the file exists, it will be loaded instead of re-training.
		output_name
		kwargs
		"""

		if classifier is None:
			raise ValueError('must give a classifier')

		logger = logging.getLogger('L5')

		self.input_columns = training_X.columns
		self.output_column = output_name

		if cache_file is not None:
			if os.path.isabs(cache_file) or cache_file[:2] in ('./', '..', '.\\'):
				cache_clf_file = cache_file
			else:
				cache_clf_file = user_cache_file(cache_file, appname=appname)
		else:
			cache_clf_file = None

		if cache_clf_file is not None and os.path.exists(cache_clf_file):
			logger.info(f'LOADING {cache_clf_file}...')
			clf = joblib.load(cache_clf_file)
			logger.info(f'COMPLETED LOADING {cache_clf_file}')
		else:
			clf = classifier(**kwargs)
			if fit:
				logger.info(f'FITTING {classifier}...')
				if training_W is not None:
					clf.fit(training_X, training_Y, sample_weight=training_W)
				else:
					clf.fit(training_X, training_Y)
				logger.info(f'FITTED {classifier}')
			if cache_clf_file is not None:
				joblib.dump(clf, cache_clf_file)
				logger.info(f'SAVED {cache_clf_file}')

		self.clf = clf
		self._predict_type = 'predict_proba col 1'


	def apply(
			self,
			X,
			dtype=None,
	):
		if dtype is None:
			dtype = l4_float_dtype
		if self._predict_type == 'predict_proba col 1':
			X[self.output_column] = numpy.log(self.clf.predict_proba(X[self.input_columns])[:, 1]).astype(dtype)
		elif self._predict_type == 'predict':
			X[self.output_column] = numpy.log(self.clf.predict(X[self.input_columns])).astype(dtype)
		else:
			raise TypeError(self._predict_type)
		return X



class RandomForestPrelearner(Prelearner):
	def __init__(
			self,
			training_X,
			training_Y,
			training_W=None,
			cache_file=None,
			fit=True,
			output_name='prelearned_utility',
			**kwargs,
	):
		from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

		default_kwargs = dict(
			n_estimators=200,
			warm_start=False,
			max_features=None,
			oob_score=True,
			n_jobs=-1,
			random_state=123,
		)

		default_kwargs.update(kwargs)

		super().__init__(
			training_X,
			training_Y,
			training_W=training_W,
			classifier=RandomForestClassifier,
			fit=fit,
			cache_file=cache_file,
			output_name=output_name,
			**default_kwargs,
		)


class XGBoostHardPrelearner(Prelearner):

	def __init__(
			self,
			training_X,
			training_Y,
			training_W=None,
			cache_file=None,
			fit=True,
			output_name='prelearned_utility',
			**kwargs,
	):
		from xgboost import XGBClassifier

		default_kwargs = dict(
			max_depth=11,
			learning_rate=0.01,
			n_estimators=500,
			silent=True,
			objective='binary:logistic',
			booster='gbtree',
			n_jobs=-1,
			max_delta_step=0,
			subsample=1,
			colsample_bytree=1,
			colsample_bylevel=1,
			reg_alpha=0,
			reg_lambda=1,
			scale_pos_weight=1,
			base_score=0.5,
			random_state=123,
		)

		default_kwargs.update(kwargs)

		super().__init__(
			training_X,
			training_Y,
			training_W=training_W,
			classifier=XGBClassifier,
			fit=fit,
			cache_file=cache_file,
			output_name=output_name,
			**default_kwargs,
		)


class XGBoostSoftPrelearner(Prelearner):

	def __init__(
			self,
			training_X,
			training_Y,
			training_W=None,
			cache_file=None,
			fit=True,
			output_name='prelearned_utility',
			**kwargs,
	):
		from xgboost import XGBRegressor

		default_kwargs = dict(
			max_depth=11,
			learning_rate=0.01,
			n_estimators=500,
			silent=True,
			objective='reg:logistic',
			booster='gbtree',
			n_jobs=-1,
			max_delta_step=0,
			subsample=1,
			colsample_bytree=1,
			colsample_bylevel=1,
			reg_alpha=0,
			reg_lambda=1,
			scale_pos_weight=1,
			base_score=0.5,
			random_state=123,
		)

		default_kwargs.update(kwargs)

		super().__init__(
			training_X,
			training_Y,
			training_W=training_W,
			classifier=XGBRegressor,
			fit=fit,
			cache_file=cache_file,
			output_name=output_name,
			**default_kwargs,
		)

		self._predict_type = 'predict'




class XGBoostPrelearner(Prelearner):

	def __init__(
			self,
			training_X,
			training_Y,
			training_W=None,
			cache_file=None,
			fit=True,
			output_name='prelearned_utility',
			**kwargs,
	):
		from xgboost import XGBRegressor, XGBClassifier

		use_soft = numpy.any((training_Y != 0) & (training_Y != 1.0))

		default_kwargs = dict(
			max_depth=11,
			learning_rate=0.01,
			n_estimators=500,
			silent=True,
			objective='reg:logistic' if use_soft else 'binary:logistic',
			booster='gbtree',
			n_jobs=-1,
			max_delta_step=0,
			subsample=1,
			colsample_bytree=1,
			colsample_bylevel=1,
			reg_alpha=0,
			reg_lambda=1,
			scale_pos_weight=1,
			base_score=0.5,
			random_state=123,
		)

		default_kwargs.update(kwargs)

		super().__init__(
			training_X,
			training_Y,
			training_W=training_W,
			classifier=XGBRegressor if use_soft else XGBClassifier,
			fit=fit,
			cache_file=cache_file,
			output_name=output_name,
			**default_kwargs,
		)

		self._predict_type = 'predict' if use_soft else 'predict_proba col 1'


