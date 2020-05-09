

def stable_df(df, filename):
	import os, pandas
	from pandas.io.formats.style import Styler
	if isinstance(df, Styler):
		df = df.data
	f = os.path.join(os.path.dirname(__file__), f'{filename}.pkl.gz')
	if os.path.exists(f):
		comparison = pandas.read_pickle(f)
		pandas.testing.assert_frame_equal(df, comparison)
	else:
		try:
			df.to_pickle(f)
		except FileNotFoundError:
			raise FileNotFoundError(f)

