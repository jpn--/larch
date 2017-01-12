

from ..util.xhtml import Elem, XML_Builder, XHTML
from ..util.statsummary import statistical_summary
from ..util.plotting import spark_histogram, spark_pie_maker
from ..util.filemanager import filename_safe
import tables
import warnings
from .. import jupyter
import os
import re

_stat_rows = ('n_values', 'mean', 'stdev',    'minimum', 'maximum', 'n_positives', 'n_negatives', 'n_zeros', 'n_nonzeros', 'n_nans', 'mean_nonzero')
_stat_labs = ('# Values', 'Mean', 'Std.Dev.', 'Minimum', 'Maximum', '# Positives', '# Negatives', '# Zeros', '# Nonzeros', '# NaNs', 'Mean (Nonzeros)')


def clear_look_cache(self):
	self.wipe_vault("looker_cache_.*")






def look_site(self, directory=None, screen="None"):
	
	if directory is None:
		raise NotImplementedError('give a directory')

	os.makedirs(os.path.join(directory, 'idco'), exist_ok=True)
	
	with XHTML(os.path.join(directory, "index.html"), overwrite=True, view_on_exit=False) as f_top:
		
		names = sorted(self.idco._v_children_keys_including_extern)
		if len(names):
			os.makedirs(os.path.join(directory, 'idco'), exist_ok=True)
			f_top.hn(1, 'idco Variables', anchor='idco Variables')

			file_names = [filename_safe(directory, 'idco', re.sub('[^\w\s-]', '_', name)+".html") for name in names]
			relfile_names = [os.path.join('.', re.sub('[^\w\s-]', '_', name)+".html") for name in names]

			f_top_table = f_top.body.put(tag="table")
			f_top_table_headrow = f_top_table.put(tag="thead").put(tag="tr")
			f_top_table_headrow.put(tag="th", text='Variable')
			f_top_table_headrow.put(tag="th", text='dtype')
			
			for slot in range(len(names)):
				
				name = names[slot]
				fname = file_names[slot]
				
				trow = f_top_table.put(tag="tr")
				trow.put(tag="td").put(tag="a", attrib={'href':'./idco/'+name+".html"}, text=name)

				try:
					z_dtype = self.idco[name].dtype
				except (TypeError,tables.NoSuchNodeError):
					try:
						z_dtype = self.idco[name]._values_.dtype
					except:
						z_dtype = "?"

				trow.put(tag="td", text=str(z_dtype))
				
				with XHTML(fname, overwrite=True, view_on_exit=False) as f:
					navbar = Elem(tag='span', attrib={'style':'font:Roboto, monospace; font-size:80%; font-weight:900;'}, text='', tail=' ')
					if slot==0:
						navbar << Elem(tag='span', attrib={'style':'color:#bbbbbb;'}, text='<< PREV', tail=' ')
					else:
						navbar << Elem(tag='a', attrib={'href':relfile_names[slot-1]}, text='<< PREV', tail=' ')
					navbar << Elem(tag='a', attrib={'href':'../index.html'}, text=' ^TOP^ ', tail=' ')
					if slot+1==len(names):
						navbar << Elem(tag='span', attrib={'style':'color:#bbbbbb;'}, text='NEXT >>')
					else:
						navbar << Elem(tag='a', attrib={'href':relfile_names[slot+1]}, text='NEXT >>')
					
					f << navbar
					f << self.look_idco(name, tall=True, title=None, headlevel=1, screen=screen)

		names = sorted(self.idca._v_children_keys_including_extern)
		if len(names):
			os.makedirs(os.path.join(directory, 'idca'), exist_ok=True)
			f_top.hn(1, 'idca Variables', anchor='idca Variables')
			
			file_names = [filename_safe(directory, 'idca', re.sub('[^\w\s-]', '_', name)+".html") for name in names]
			relfile_names = [os.path.join('.', re.sub('[^\w\s-]', '_', name)+".html") for name in names]

			f_top_table = f_top.body.put(tag="table")
			f_top_table_headrow = f_top_table.put(tag="thead").put(tag="tr")
			f_top_table_headrow.put(tag="th", text='Variable')
			f_top_table_headrow.put(tag="th", text='dtype')

			for slot in range(len(names)):
				
				name = names[slot]
				fname = file_names[slot]
				
				trow = f_top_table.put(tag="tr")
				trow.put(tag="td").put(tag="a", attrib={'href':'./idca/'+name+".html"}, text=name)
				try:
					z_dtype = self.idca[name].dtype
				except (TypeError,tables.NoSuchNodeError):
					try:
						z_dtype = self.idca[name]._values_.dtype
					except:
						z_dtype = "?"
				trow.put(tag="td", text=str(z_dtype))

				with XHTML(fname, overwrite=True, view_on_exit=False) as f:
					navbar = Elem(tag='span', attrib={'style':'font:Roboto, monospace; font-size:80%; font-weight:900;'}, text='', tail=' ')
					if slot==0:
						navbar << Elem(tag='span', attrib={'style':'color:#bbbbbb;'}, text='<< PREV', tail=' ')
					else:
						navbar << Elem(tag='a', attrib={'href':relfile_names[slot-1]}, text='<< PREV', tail=' ')
					navbar << Elem(tag='a', attrib={'href':'../index.html'}, text=' ^TOP^ ', tail=' ')
					if slot+1==len(names):
						navbar << Elem(tag='span', attrib={'style':'color:#bbbbbb;'}, text='NEXT >>')
					else:
						navbar << Elem(tag='a', attrib={'href':relfile_names[slot+1]}, text='NEXT >>')
					
					f << navbar
					f << self.look_idca(name, tall=True, title=None, headlevel=1, screen=screen)




def look_idco(self, *names, headlevel=3, spark_kwargs=None, cache=True, regex=None, title="Variable Analysis (idco)", titlelevel=2, tall=False, screen="None"):
	top = XML_Builder()

	if title:
		top.hn(titlelevel, title, anchor=title)
	
	spark_kw = dict(
		figwidth=6 if tall else 4,
		figheight=4 if tall else 3,
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

	use_jupyter = False # trouble here, when in ipython but not in jupyter, so just say no

	if use_jupyter:
		try:
			from IPython.display import display, HTML
		except ImportError:
			step = lambda x: top.append(x)
		else:
			step = lambda x: display(HTML(x.dump().decode()))
	else:
		step = lambda x: top.append(x)

	# scr is a slice-type obj
	if screen is None:
		try:
			scr = self.h5top.screen[:]
		except tables.NoSuchNodeError:
			scr = slice(None)
	elif screen is "None":
		scr = slice(None)
	else:
		scr = screen

	for name in names:
		
		if self.in_vault("looker_cache_"+name):
			top.append(self.from_vault("looker_cache_"+name))
		else:
			
			x = XML_Builder()
			try:
				z = self.idco[name][scr]
			except (TypeError,tables.NoSuchNodeError):
				try:
					z = self.array_idco(name, screen=screen).squeeze()
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
					
					try:
						orig_source = self.idco[name]._v_attrs.ORIGINAL_SOURCE
					except AttributeError:
						pass
					else:
						if orig_source:
							x.start('span', {'style':'font-style:italic;'})
							x.data("Original Source: ")
							x.data(orig_source)
							x.end('span')
							x.simple('br')

				if tall:
					x.hn(headlevel+1, "Summary Statistics", anchor='Statistics')
				else:
					x.start('table', {'style':'font-size: 11pt; border:hidden;'})
					x.start('tr')
					x.start('td', {'style':'border:hidden;'})

				# Table of stats
				if 1:
					x.start('table')
					for i,j in zip(_stat_rows, _stat_labs):
						x.start('tr')
						x.td(j, {'style':'text-align:right; font-family: "Roboto Slab", serif;'})
						x.td(str(getattr(ss,i)), {'style':'font-family: "Roboto Mono", monospace; '})
						x.end('tr')

					x.start('tr')
					x.td('dtype', {'style':'text-align:right; font-family: "Roboto Slab", serif;'})
					x.td(str(z.dtype), {'style':'font-family: "Roboto Mono", monospace; '})
					x.end('tr')
					
					x.end('table')
					
				if tall:
					x.hn(headlevel+1, "Distribution of Values", anchor='Distribution')
				else:
					x.end('td')
					x.start('td')

				# Graph of distribution

				#histo = spark_histogram(z, pie_chart_cutoff=4, notetaker=None, prerange=None, duo_filter=None, data_for_bins=None)
				histo = ss.histogram
				if isinstance(histo, tuple):
					histo = histo[0]
				x.append(histo)

				if tall:
					if ss.notes:
						x.simple('span', content=ss.notes, attrib={'style':'font-size:75%;'})
				else:
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

					if tall:
						x.hn(headlevel+1, "Fundamentals", anchor='Fundamentals')
					else:
						x.start('td', {'style':'border:hidden;'})

					pie = spark_pie_maker(pie_funda, notetaker=None, figheight=2.0, figwidth=2.0, labels=pie_labels,
											show_labels=True, shadow=False, subplots_adjuster=0.1,
											explode=pie_explode, wedge_linewidth=1,
											solid_joinstyle='round')
					x.append(pie)
					if tall:
						pass
					else:
						x.end('td')


				if z_dict is not None:
					if tall:
						x.hn(headlevel+1, "Dictionary", anchor='Dictionary')
					else:
						x.start('td', {'style':'border:hidden;'})
						x.simple(tag='span', content="Dictionary", attrib={'style':'font-weight:bold;'})
					x.start('table')
					x.start('tr', {'style':'font-family: "Roboto Slab", serif; '})
					x.th('Value')
					x.th('Meaning')
					x.end('tr')
					for i,j in z_dict.items():
						x.start('tr')
						x.td(str(i), {'style':'font-family: "Roboto Mono", monospace; '})
						x.td(str(j), {'style':'font-family: "Roboto Mono", monospace; '})
						x.end('tr')
					x.end('table')
					
					if tall:
						pass
					else:
						x.end('td')

				if ss.notes and not tall:
					x.simple('caption', content=ss.notes, attrib={'style':'font-size:75%;'})

				if not tall:
					x.end('tr')
					x.end('table')

				x_div = x.close()

				with warnings.catch_warnings():
					warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
					self.to_vault("looker_cache_"+name, x_div)

				step(x)
				
	result = top.close()
	return result







def look_idca(self, *names, headlevel=3, spark_kwargs=None, cache=True, regex=None,
				title="Variable Analysis (idca)", titlelevel=2, tall=False, screen="None"):
	top = XML_Builder()

	if title:
		top.hn(titlelevel, title, anchor=title)
	
	spark_kw = dict(
		figwidth=6 if tall else 4,
		figheight=4 if tall else 3,
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
		names = sorted(self.idca._v_children_keys_including_extern)

	if len(names)==0 and regex is not None:
		import re
		names = [i for i in sorted(self.idca._v_children_keys_including_extern) if re.search(regex, i)]



	if 'IPython' in jupyter.ipython_status():
		try:
			use_jupyter = True
		except:
			use_jupyter = False
	else:
		use_jupyter = False

	use_jupyter = False # trouble here, when in ipython but not in jupyter, so just say no

	if use_jupyter:
		try:
			from IPython.display import display, HTML
		except ImportError:
			step = lambda x: top.append(x)
		else:
			step = lambda x: display(HTML(x.dump().decode()))
	else:
		step = lambda x: top.append(x)


	if screen is None:
		try:
			scr = self.h5top.screen[:]
		except tables.NoSuchNodeError:
			scr = slice(None)
	elif screen is "None":
		scr = slice(None)
	else:
		scr = screen

	av = self.array_avail(screen=screen).ravel()

	for name in names:
		
		if self.in_vault("looker_cache_"+name):
			top.append(self.from_vault("looker_cache_"+name))
		else:
			
			x = XML_Builder()
			try:
				z = self.idca[name][scr]
			except (TypeError,tables.NoSuchNodeError):
				try:
					z = self.array_idca(name, screen=screen).squeeze()
				except:
					x.hn(headlevel, name)
					x.data("Not available")
					continue

			z = z.ravel()[av]

			try:
				z_dict = self.idca[name]._v_attrs.DICTIONARY
			except AttributeError:
				z_dict = None

			if z.dtype.kind == 'S':
				x.hn(headlevel, name, anchor=name)

				if name in self.idca:
					descrip = self.idca[name]._v_attrs.TITLE
					if self.idca[name]._v_attrs.TITLE:
						x.data(descrip)
						x.simple('br')

				x.data('Text field, summary function not implemented')

			else:
				ss = statistical_summary.nan_compute(z, dictionary=z_dict, spark_kwargs=spark_kw)

				x.hn(headlevel, name, anchor=name)

				if name in self.idca:
					descrip = self.idca[name]._v_attrs.TITLE
					if self.idca[name]._v_attrs.TITLE:
						x.data(descrip)
						x.simple('br')
					
					try:
						orig_source = self.idca[name]._v_attrs.ORIGINAL_SOURCE
					except AttributeError:
						pass
					else:
						if orig_source:
							x.start('span', {'style':'font-style:italic;'})
							x.data("Original Source: ")
							x.data(orig_source)
							x.end('span')
							x.simple('br')

				if tall:
					x.hn(headlevel+1, "Summary Statistics", anchor='Statistics')
				else:
					x.start('table', {'style':'font-size: 11pt; border:hidden;'})
					x.start('tr')
					x.start('td', {'style':'border:hidden;'})

				# Table of stats
				if 1:
					x.start('table')
					for i,j in zip(_stat_rows, _stat_labs):
						x.start('tr')
						x.td(j, {'style':'text-align:right; font-family: "Roboto Slab", serif;'})
						x.td(str(getattr(ss,i)), {'style':'font-family: "Roboto Mono", monospace; '})
						x.end('tr')

					x.start('tr')
					x.td('dtype', {'style':'text-align:right; font-family: "Roboto Slab", serif;'})
					x.td(str(z.dtype), {'style':'font-family: "Roboto Mono", monospace; '})
					x.end('tr')
					
					x.end('table')
					
				if tall:
					x.hn(headlevel+1, "Distribution of Values", anchor='Distribution')
				else:
					x.end('td')
					x.start('td')

				# Graph of distribution

				#histo = spark_histogram(z, pie_chart_cutoff=4, notetaker=None, prerange=None, duo_filter=None, data_for_bins=None)
				histo = ss.histogram
				if isinstance(histo, tuple):
					histo = histo[0]
				x.append(histo)

				if tall:
					if ss.notes:
						x.simple('span', content=ss.notes, attrib={'style':'font-size:75%;'})
				else:
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

					if tall:
						x.hn(headlevel+1, "Fundamentals", anchor='Fundamentals')
					else:
						x.start('td', {'style':'border:hidden;'})

					pie = spark_pie_maker(pie_funda, notetaker=None, figheight=2.0, figwidth=2.0, labels=pie_labels,
											show_labels=True, shadow=False, subplots_adjuster=0.1,
											explode=pie_explode, wedge_linewidth=1,
											solid_joinstyle='round')
					x.append(pie)
					if tall:
						pass
					else:
						x.end('td')


				if z_dict is not None:
					if tall:
						x.hn(headlevel+1, "Dictionary", anchor='Dictionary')
					else:
						x.start('td', {'style':'border:hidden;'})
						x.simple(tag='span', content="Dictionary", attrib={'style':'font-weight:bold;'})
					x.start('table')
					x.start('tr', {'style':'font-family: "Roboto Slab", serif; '})
					x.th('Value')
					x.th('Meaning')
					x.end('tr')
					for i,j in z_dict.items():
						x.start('tr')
						x.td(str(i), {'style':'font-family: "Roboto Mono", monospace; '})
						x.td(str(j), {'style':'font-family: "Roboto Mono", monospace; '})
						x.end('tr')
					x.end('table')
					
					if tall:
						pass
					else:
						x.end('td')

				if ss.notes and not tall:
					x.simple('caption', content=ss.notes, attrib={'style':'font-size:75%;'})

				if not tall:
					x.end('tr')
					x.end('table')

				x_div = x.close()

				with warnings.catch_warnings():
					warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
					self.to_vault("looker_cache_"+name, x_div)

				step(x)
				
	result = top.close()
	return result


