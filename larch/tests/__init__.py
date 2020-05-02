from .test_qmnl import test_saturate_cpu

def stable_df(df, filename):
	import os, pandas
	f = os.path.join(os.path.dirname(__file__), 'stored_dataframes', f'{filename}.pkl.gz')
	if os.path.exists(f):
		comparison = pandas.read_pickle(f)
		pandas.testing.assert_frame_equal(df, comparison)
	else:
		try:
			df.to_pickle(f)
		except FileNotFoundError:
			raise FileNotFoundError(f)
