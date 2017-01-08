

from ..util.xhtml import Elem, XML_Builder
from ..util.statsummary import statistical_summary
from ..util.plotting import spark_histogram, spark_pie_maker
import tables
import warnings
from .. import jupyter

_stat_rows = ('mean', 'stdev',    'minimum', 'maximum', 'n_positives', 'n_negatives', 'n_zeros', 'n_nonzeros', 'n_nans', 'mean_nonzero')
_stat_labs = ('Mean', 'Std.Dev.', 'Minimum', 'Maximum', '# Positives', '# Negatives', '# Zeros', '# Nonzeros', '# NaNs', 'Mean (Nonzeros)')


def clear_look_cache(self):
	self.wipe_vault("looker_cache_.*")

def look_idco(self, *names, headlevel=3, spark_kwargs=None, cache=True, regex=None, title="Variable Analysis (idco)", titlelevel=2):
	top = XML_Builder()
	
	if title:
		top.hn(titlelevel, title, anchor=title)
	
	spark_kw = dict(
		figwidth=4,
		figheight=3, 
		show_labels=True, 
		subplots_adjuster=0.1,
		frame=True,
		xticks=True,
		pie_chart_cutoff=10,
		pie_chart_type='bar',
	)
	if spark_kwargs is not None:
		spark_kw.update(spark_kwargs)
	
	if len(names)==1 and names[0] == '*':
		names = sorted(self.idco._v_children_keys_including_extern)

	if len(names)==0 and regex is not None:
		import re
		names = [i for i in sorted(self.idco._v_children_keys_including_extern) if re.search(regex, i)]



	if 'IPython' in jupyter.ipython_status():
		try:
			use_jupyter = True
		except:
			use_jupyter = False
	else:
		use_jupyter = False



	if use_jupyter:
		try:
			from IPython.display import display, HTML
		except ImportError:
			step = lambda x: top.append(x)
		else:
			step = lambda x: display(HTML(x.dump().decode()))
	else:
		step = lambda x: top.append(x)


	for name in names:
		
		if self.in_vault("looker_cache_"+name):
			top.append(self.from_vault("looker_cache_"+name))
		else:
			
			x = XML_Builder()
			try:
				z = self.idco[name][:]
			except (TypeError,tables.NoSuchNodeError):
				try:
					z = self.array_idco(name, screen="None").squeeze()
				except:
					x.hn(headlevel, name)
					x.data("Not available")
					continue

			try:
				z_dict = self.idco[name]._v_attrs.DICTIONARY
			except AttributeError:
				z_dict = None

			if z.dtype.kind == 'S':
				x.hn(headlevel, name, anchor=name)

				if name in self.idco:
					descrip = self.idco[name]._v_attrs.TITLE
					if self.idco[name]._v_attrs.TITLE:
						x.data(descrip)
						x.simple('br')

				x.data('Text field, summary function not implemented')

			else:
				ss = statistical_summary.nan_compute(z, dictionary=z_dict, spark_kwargs=spark_kw)

				x.hn(headlevel, name, anchor=name)

				if name in self.idco:
					descrip = self.idco[name]._v_attrs.TITLE
					if self.idco[name]._v_attrs.TITLE:
						x.data(descrip)
						x.simple('br')

				x.start('table', {'style':'font-size: 11pt; border:hidden;'})
				x.start('td', {'style':'border:hidden;'})

				# Table of stats
				if 1:
					x.start('table')
					for i,j in zip(_stat_rows, _stat_labs):
						x.start('tr')
						x.td(j, {'style':'text-align:right; font-family: "Roboto Slab", serif;'})
						x.td(str(getattr(ss,i)), {'style':'font-family: "Roboto Mono", monospace; '})
						x.end('tr')
					x.end('table')
					
				x.end('td')
				x.start('td')
				
				# Graph of distribution

				#histo = spark_histogram(z, pie_chart_cutoff=4, notetaker=None, prerange=None, duo_filter=None, data_for_bins=None)
				histo = ss.histogram
				if isinstance(histo, tuple):
					histo = histo[0]
				x.append(histo)

				x.end('td')


				# Pie of Fundamental Data
				pie_funda, pie_labels, pie_explode = [],[],[]
				if ss.n_nans:
					pie_funda += [ss.n_nans,]
					pie_labels += ['NaN',]
					pie_explode += [0,]

				if ss.n_positives:
					pie_funda += [ss.n_positives,]
					pie_labels += ['+',]
					pie_explode += [0,]

				if ss.n_zeros:
					pie_funda += [ss.n_zeros,]
					pie_labels += ['0',]
					pie_explode += [0,]

				if ss.n_negatives:
					pie_funda += [ss.n_negatives,]
					pie_labels += ['-',]
					pie_explode += [0,]

				if len(pie_funda)>1:
					x.start('td', {'style':'border:hidden;'})
					pie = spark_pie_maker(pie_funda, notetaker=None, figheight=2.0, figwidth=2.0, labels=pie_labels,
											show_labels=True, shadow=False, subplots_adjuster=0.1,
											explode=pie_explode, wedge_linewidth=1, tight=True,
											solid_joinstyle='round')
					x.append(pie)
					x.end('td')




				if ss.notes:
					x.simple('caption', content=ss.notes, attrib={'style':'font-size:75%;'})
				x.end('table')

				x_div = x.close()

				with warnings.catch_warnings():
					warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
					self.to_vault("looker_cache_"+name, x_div)

				step(x)
				
	result = top.close()
	return result


