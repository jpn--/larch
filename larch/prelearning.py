


import logging
import numpy
import pandas
import os
from appdirs import user_cache_dir
import joblib
from typing import MutableMapping

from .general_precision import l4_float_dtype
from .log import logger_name
from .dataframes import DataFrames

def user_cache_file(filename, appname=None, appauthor=None, version=None, opinion=True):
	d = user_cache_dir(appname=appname, appauthor=appauthor, version=version, opinion=opinion)
	os.makedirs(d, exist_ok=True)
	return os.path.join(d, filename)


class Prelearner():
	"""
	A prelearner for use with Larch.

	A prelearner uses a machine learning classifier to make an initial
	prediction of the result.  This initial prediction is then added
	as an input data column for Larch, effectively creating a chained
	classifier.

	Parameters
	----------
	training_X : pandas.DataFrame
		The exogenous variables.
	training_Y : pandas.DataFrame
		The observed choices in the training data.
	training_W : pandas.DataFrame, optional
		The weights.
	classifier : sklearn Classifier or Regressor
		This is the class object for the selected classifier, not
		an existing instance.  This classifier or Regressor will be
		instantiated and trained using the data above to generate
		the prediction.
	fit : dict or False, optional
		A dictionary of arguments to pass to the `fit` method of the
		classifier, or set to False to not fit the classifier
		during the initialization of this object.
	cache_file : str, optional
		A cache file name to store the trained prelearner.  If just a filename is given,
		it will be stored in `appdirs.user_cache_file()`. If instead an absolute path or
		a relative path beginning with '.' is given, that location will be used.
		If the file exists, it will be loaded instead of re-training.
	output_name : str, default 'prelearned_utility'
		The name of the output column from this prelearner.
	grid_cv_params : dict or List[dict], optional
		If given, this is used as the `param_grid` argument
		to initialize a :class:`sklearn.model_selection.GridSearchCV`
		wrapped around the classifier, instead of using the
		classifier directly.
	grid_cv_kwds : dict, optional
		If `grid_cv_params` is given, this dict gives other keyword
		arguments given to :class:`sklearn.model_selection.GridSearchCV`.
	**kwargs
		Any other keyword arguments are passed through to the classifier's
		constructor.
	"""

	def __init__(
			self,
			dataframes,
			ca_columns=None,
			co_columns=None,
			classifier=None,
			fit=True,
			cache_file=None,
			output_name='prelearned_utility',
			appname='larch',
			grid_cv_params=None,
			grid_cv_kwds=None,
			validation_dataframes=None,
			**kwargs,
	):

		if classifier is None:
			raise ValueError('must give a classifier')

		if fit is True:
			fit = {}

		logger = logging.getLogger(logger_name)

		self.input_ca_columns = ca_columns if ca_columns is not None else []
		self.input_co_columns = co_columns

		self.eval_set_names = fit.pop('eval_set_names', [])

		if isinstance(fit, MutableMapping):
			if 'validation_percent' in fit and validation_dataframes is None:
				vpct = fit.pop('validation_percent')
				dataframes, validation_dataframes = dataframes.split([100-vpct, vpct])
			else:
				vpct = 'preset'
			if validation_dataframes is not None:
				validation_X = self.filter_and_join_columns(
					validation_dataframes.data_ca_as_ce(),
					validation_dataframes.data_co,
				)
				validation_Y = validation_dataframes.array_ch_as_ce()
				validation_W = validation_dataframes.array_wt_as_ce()

				fit['eval_set'] = fit.get('eval_set', []) + [(validation_X, validation_Y)]
				if validation_W is not None:
					fit['sample_weight_eval_set'] = fit.get('sample_weight_eval_set', []) + [validation_W]
				self.eval_set_names += [f'validation_{vpct}']

		training_X = self.filter_and_join_columns(
			dataframes.data_ca_as_ce(),
			dataframes.data_co,
		)
		training_Y = dataframes.array_ch_as_ce()
		training_W = dataframes.array_wt_as_ce()
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
			if grid_cv_params is not None:
				from sklearn.model_selection import GridSearchCV
				clf = GridSearchCV(
					classifier(**kwargs),
					grid_cv_params,
					**grid_cv_kwds,
				)
			else:
				clf = classifier(**kwargs)
			if fit is not False:

				if 'train_as_eval' in fit:
					fit.pop('train_as_eval')
					fit['eval_set'] = [(training_X, training_Y),] + fit.get('eval_set',[])
					if training_W is not None:
						fit['sample_weight_eval_set'] = [training_W,]+fit.get('sample_weight_eval_set',[])
					self.eval_set_names = ['training'] + self.eval_set_names

				logger.info(f'FITTING {classifier}...')
				if training_W is not None:
					clf.fit(training_X, training_Y, sample_weight=training_W, **fit)
				else:
					clf.fit(training_X, training_Y, **fit)
				logger.info(f'FITTED {classifier}')
			if cache_clf_file is not None:
				joblib.dump(clf, cache_clf_file)
				logger.info(f'SAVED {cache_clf_file}')

		self.clf = clf
		self._predict_type = 'predict_proba col 1'


	def filter_ca_columns(self, X):
		# filter the columns of the input into the correct form for the prelearner.
		try:
			X1 = X[self.input_ca_columns]
		except KeyError:
			X1 = pandas.DataFrame(
				X.eval(self.input_ca_columns).T.astype(float),
				index=X.index,
				columns=self.input_ca_columns,
			)
		return X1

	def filter_and_join_columns(self, X_ca, X_co):
		training_X = self.filter_ca_columns(X_ca)
		if self.input_co_columns:
			try:
				X_co = X_co[self.input_co_columns]
			except KeyError:
				X_co = pandas.DataFrame(
					X_co.eval(self.input_co_columns).T.astype(float),
					index=X_co.index,
					columns=self.input_co_columns,
				)
			training_X = training_X.join(X_co, on=training_X.index.levels[0].name, how='left').fillna(0)
		return training_X

	def apply(
			self,
			X,
			dtype=None,
			output_name=None,
			**kwargs,
	):
		"""
		Apply the prelearner to compute pseudo-utility.

		Parameters
		----------
		X : pandas.DataFrame
		dtype : dtype, default float
			The dtype to use for the output column.
		output_name : str, optional
			The name of the output column from this
			application of the prelearner.
		**kwargs
			Other keyword arguments are forwarded to the
			`predict` or `predict_proba` method of the
			`clf` member.

		Returns
		-------
		pandas.DataFrame
		"""
		if dtype is None:
			dtype = l4_float_dtype

		if isinstance(X, DataFrames):
			X_ca = X._data_ca_or_ce
			X_co = X.data_co
		else:
			X_ca = X
			X_co = None

		X_in = self.filter_and_join_columns(
			X_ca,
			X_co,
		)

		if output_name is None:
			output_name = self.output_column

		if self._predict_type == 'predict_proba col 1':
			X_ca.loc[:,output_name] = numpy.log(self.clf.predict_proba(X_in, **kwargs)[:, 1]).astype(dtype)
		elif self._predict_type == 'predict':
			X_ca.loc[:,output_name] = numpy.log(self.clf.predict(X_in, **kwargs)).astype(dtype)
		else:
			raise TypeError(self._predict_type)
		return X



class RandomForestPrelearner(Prelearner):
	def __init__(
			self,
			dataframes,
			ca_columns=None,
			co_columns=None,
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
			dataframes=dataframes,
			ca_columns=ca_columns,
			co_columns=co_columns,
			classifier=RandomForestClassifier,
			fit=fit,
			cache_file=cache_file,
			output_name=output_name,
			**default_kwargs,
		)


class XGBoostHardPrelearner(Prelearner):

	def __init__(
			self,
			dataframes,
			ca_columns=None,
			co_columns=None,
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
			# silent=True,
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
			dataframes=dataframes,
			ca_columns=ca_columns,
			co_columns=co_columns,
			classifier=XGBClassifier,
			fit=fit,
			cache_file=cache_file,
			output_name=output_name,
			**default_kwargs,
		)


class XGBoostSoftPrelearner(Prelearner):

	def __init__(
			self,
			dataframes,
			ca_columns=None,
			co_columns=None,
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
			# silent=True,
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
			dataframes=dataframes,
			ca_columns=ca_columns,
			co_columns=co_columns,
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
			dataframes,
			ca_columns=None,
			co_columns=None,
			cache_file=None,
			fit=True,
			output_name='prelearned_utility',
			**kwargs,
	):
		from xgboost import XGBRegressor, XGBClassifier

		training_Y = dataframes.array_ch_as_ce()
		use_soft = numpy.any((training_Y != 0) & (training_Y != 1.0))

		default_kwargs = dict(
			max_depth=11,
			learning_rate=0.01,
			n_estimators=500,
			# silent=True,
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
			dataframes=dataframes,
			ca_columns=ca_columns,
			co_columns=co_columns,
			classifier=XGBRegressor if use_soft else XGBClassifier,
			fit=fit,
			cache_file=cache_file,
			output_name=output_name,
			**default_kwargs,
		)

		self._predict_type = 'predict' if use_soft else 'predict_proba col 1'


	def evals_result(self):
		j = [
			pandas.DataFrame({mk:numpy.asarray(mv) for mk, mv in ev.items()})
			for ek, ev in self.clf.evals_result_.items()
		]
		k = [
			ek
			for ek, ev in self.clf.evals_result_.items()
		]
		for i in range(len(self.eval_set_names)):
			if len(k)>i:
				k[i] = self.eval_set_names[i]
		return pandas.concat(j, axis=1, keys=k, sort=False)
