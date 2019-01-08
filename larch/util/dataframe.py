import pandas, numpy
from xmle import Elem

class DataFrameViewer(pandas.DataFrame):

	def __xml__(self):
		z = Elem('div')
		table = z.put('table', {'class':'dataframe larch_dataframe'})
		thead = table.put('thead')
		for i in range(self.index.nlevels):
			thead.put('th', text=self.index.names[i])
		for c in self.columns:
			thead.put('th', text=str(c))
		tbody = table.put('tbody')
		for i in self.index:
			row = tbody.put('tr')
			if self.index.nlevels==1:
				row.put('th', text=str(i))
			else:
				for ii in i:
					row.put('th', text=str(ii))
			for c in self.columns:
				cell = self.loc[i, c]
				try:
					cell_content = cell.__xml__()
				except AttributeError:
					if isinstance(cell, float):
						row.put('td', text="{0:.6g}".format(cell))
					else:
						row.put('td', text=str(cell))
				else:
					row.put('td') << cell_content
		return z

	def _repr_html_(self):
		return self.__xml__().tostring()

	def __getitem__(self, item):
		result = super().__getitem__(item)
		if type(result) == pandas.DataFrame:
			result.__class__ = DataFrameViewer
		return result

	def drop(self, *args, **kwargs):
		result = super().drop(*args, **kwargs)
		if type(result) == pandas.DataFrame:
			result.__class__ = DataFrameViewer
		return result


#####################
# DataFrame Styling
#####################

from matplotlib import colors
import seaborn

def global_background_gradient(s, m, M, cmap=None, low=0, high=0):
	if cmap is None:
		cmap = seaborn.light_palette("seagreen", as_cmap=True)
	rng = M - m
	norm = colors.Normalize(m - (rng * low),
							M + (rng * high))
	normed = norm(s.values)
	c = [colors.rgb2hex(x) for x in cmap(normed)]
	return ['background-color: %s' % color for color in c]



def apply_global_background_gradient(df, override_min=None, override_max=None, cmap=None, subset=None):
	if cmap is None:
		seagreen = seaborn.light_palette("seagreen", as_cmap=True)
		cmap = seagreen
	df = df.apply(
		global_background_gradient,
		cmap=cmap,
		m=override_min if override_min is not None else df.data.min().min(),
		M=override_max if override_max is not None else df.data.max().max(),
		subset=subset,
	)
	return df

###################

def columnize(df, name, inplace=True):
	"""Add a computed column to a DataFrame."""

	from ..roles import LinearFunction
	if isinstance(name, LinearFunction):
		datanames = [str(_.data) for _ in name]
		df1 = pandas.concat([
			columnize(df, _, False)
			for _ in datanames
		], axis=1)
		if inplace:
			df[datanames] = df1
			return
		else:
			return df1

	from tokenize import tokenize, untokenize, NAME, OP, STRING, NUMBER
	from .aster import asterize
	DOT = (OP, '.')
	COLON = (OP, ':')
	COMMA = (OP, ',')
	OBRAC = (OP, '[')
	CBRAC = (OP, ']')
	OPAR = (OP, '(')
	CPAR = (OP, ')')
	EQUAL = (OP, '=')
	from io import BytesIO
	if inplace:
		recommand = [(NAME, 'df'), OBRAC, (STRING, f"'{name}'"), CBRAC, EQUAL]
	else:
		recommand = []
	try:
		name_encode = name.encode('utf-8')
	except AttributeError:
		name_encode = str(name).encode('utf-8')
	reader = BytesIO(name_encode)
	g = tokenize(reader.readline)
	for toknum, tokval, _, _, _ in g:
		if toknum == NAME:
			pass
			if tokval in df.columns:
				# replace NAME tokens
				partial = [
					(NAME, 'df'),
					OBRAC, (STRING, f"'{tokval}'"), CBRAC,
				]
				recommand.extend(partial)
			# break
			else:
				# no dat contains this natural name
				# put the name back in raw, maybe it works cause it's a global, more likely
				# the exception manager below will catch it.
				recommand.append((toknum, tokval))
		else:
			recommand.append((toknum, tokval))
	try:
		ret = untokenize(recommand).decode('utf-8')
	except:
		print("<recommand>")
		print(recommand)
		print("</recommand>")
		raise
	g.close()
	reader.close()
	# print("<ret>")
	# print(ret)
	# print("</ret>")
	j = asterize(ret, mode="exec" if inplace else "eval")
	from .aster import inXd
	from numpy import log, exp, log1p, absolute, fabs, sqrt, isnan, isfinite, logaddexp, fmin, fmax, nan_to_num, sin, cos, pi
	from ..util.common_functions import piece, normalize
	try:
		if inplace:
			_result = exec(j)
		else:
			_result = eval(j)
			_result.name = name
	except Exception as exc:
		args = exc.args
		if not args:
			arg0 = ''
		else:
			arg0 = args[0]
		arg0 = arg0 + '\nwithin parsed command: "{!s}"'.format(name)
		if "max" in name:
			arg0 = arg0 + '\n(note to get the maximum of arrays use "fmax" not "max")'.format(name)
		if "min" in name:
			arg0 = arg0 + '\n(note to get the maximum of arrays use "fmin" not "min")'.format(name)
		if isinstance(exc, NameError):
			badname = str(exc).split("'")[1]
			goodnames = {
				'log', 'exp', 'log1p', 'absolute', 'fabs', 'sqrt', 'isnan',
				'isfinite', 'logaddexp', 'fmin', 'fmax', 'nan_to_num', 'piece', 'normalize',
			}
			goodnames |= set(df.columns)
			from ..util.text_manip import case_insensitive_close_matches
			did_you_mean_list = case_insensitive_close_matches(badname, goodnames, n=3, cutoff=0.1, excpt=None)
			if len(did_you_mean_list) > 0:
				arg0 = arg0 + '\n' + "did you mean {}?".format(
					" or ".join("'{}'".format(s) for s in did_you_mean_list))
		exc.args = (arg0,) + args[1:]
		raise
	return _result

###################
### Typical Calibration Table

def counts_and_shares(
		tours_df,
		rows='tour_mode',
		cross='distH2Work',
		bins=None,
		stack=True,
		title=None,
		blurb=None,
		rowtotals=True,
):

	if isinstance(rows, str):
		if not isinstance(tours_df[rows].dtype, pandas.core.dtypes.dtypes.CategoricalDtype):
			raise TypeError("column for rows must be categorical")
		cats = tours_df[rows].copy()
	else:
		if not isinstance(rows.dtype, pandas.core.dtypes.dtypes.CategoricalDtype):
			raise TypeError("column for rows must be categorical")
		cats = rows.copy()

	firstcat, lastcat = cats.cat.categories[0], cats.cat.categories[-1]

	cats = cats.cat.add_categories(["TOTAL", "% of TOT"])

	if isinstance(cross, str):
		if bins=='cat':
			pivot_cols = pandas.Series(pandas.Categorical(tours_df[cross]), index=tours_df.index)
		elif bins is not None:
			pivot_cols = pandas.cut(
				tours_df[cross],
				bins=bins,
			)
		else:
			pivot_cols = tours_df[cross]
	elif isinstance(cross, pandas.Series):
		if isinstance(cross.dtype, pandas.core.dtypes.dtypes.CategoricalDtype):
			pivot_cols = cross
		else:
			raise TypeError('if a Series, cross must be a Categorical Series')
	else:
		raise TypeError(f'bad cross of type {type(cross)}')

	result1 = pandas.pivot_table(
		tours_df,
		values='personId',
		index=cats,
		columns=pivot_cols,
		aggfunc=len,
	).fillna(0).astype(int)

	result1.columns = result1.columns.astype(str)

	if rowtotals:
		result1.loc[:, 'TOTAL'] = result1.sum(1)

	result2 = (result1 / result1.sum(0))
	result2.loc['TOTAL', :] = numpy.nan
	result2.loc['% of TOT', :] = numpy.nan

	if rowtotals:
		result1.loc['TOTAL', :] = result1.sum(0)
		result1.loc['% of TOT', :] = -result1.loc['TOTAL', :] * 2 / result1.loc['TOTAL', :].sum()
	else:
		result1.loc['TOTAL', :] = result1.sum(0)
		result1.loc['% of TOT', :] = -result1.loc['TOTAL', :] / result1.loc['TOTAL', :].sum()

	def pct_or_thousands(x):
		if x < 0.0:
			return "{:.1%}".format(-x)
		else:
			return "{:,.0f}".format(x)

	def hide_nan_else_pct(x):
		if pandas.isnull(x):
			return ""
		else:
			return "{:.2%}".format(x)

	if stack:
		result = pandas.concat(
			[result1, result2],
			axis=1,
			keys=['Counts', 'Shares'],
		)
		result = result.style.format({
			**{i: pct_or_thousands for i in result.columns if i[0] == 'Counts'},
			**{i: hide_nan_else_pct for i in result.columns if i[0] == 'Shares'}
		})

	else:
		result = (
			result1.style.format("{:,.0f}"),
			result2.style.format("{:.2%}"),
		)

	# try:
	# 	result = apply_global_background_gradient(
	# 		result,
	# 		override_max=0.75,
	# 		override_min=0.01,
	# 		# subset=[i for i in result.columns if i[0]=='Shares'],
	# 		subset=pandas.IndexSlice[firstcat:lastcat, [i for i in result.columns if i[0] == 'Shares']]
	# 	)
	# except ValueError:
	# 	pass

	result.applymap(
		lambda y: 'text-align:right;',
		# subset=pandas.IndexSlice[firstcat:lastcat, [i for i in result.columns if i[0]=='Counts']]
	)

	if title is None:
		return result
	else:
		from IPython.display import display, display_html, HTML
		display_html(HTML(f"<h2>{title}</h2>"))
		if blurb is not None:
			display_html(HTML(f"<p>{blurb}</p>"))
		display(result)
