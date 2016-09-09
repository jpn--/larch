
import numpy, pandas
from itertools import count
from ..model_reporter.art import AbstractReportTable
from .statsummary import statistical_summary
from ..roles import X


def basic_idco_variable_analysis(arr, names, weights=None, description_catalog=None, title="Analytics", short_title=None):
	"""
	Parameters
	----------
	arr : ndarray
		The data to analyze.  Should have shape (cases, vars)
	names : sequence of str
		The names of the variables.  Should have length matching last dim of `arr`
	"""

	if description_catalog is None:
		description_catalog = {}
		from ..roles import _data_description_catalog
		description_catalog.update(_data_description_catalog)
	
	description_catalog_keys = list(description_catalog.keys())
	description_catalog_keys.sort(key=len, reverse=True)
	
	descriptions = numpy.asarray(names)
	
	for dnum, descr in enumerate(descriptions):
		if descr in description_catalog:
			descriptions[dnum] = description_catalog[descr]
		else:
			for key in description_catalog_keys:
				if key in descr:
					descr = descr.replace(key,description_catalog[key])
			descriptions[dnum] = descr

	show_descrip = (numpy.asarray(descriptions)!=numpy.asarray(names)).any()

	use_weighted = weights is not None and bool((weights!=1).any())

	if use_weighted:
		title += " (weighted)"

	ss = statistical_summary.compute(arr, dimzer=numpy.atleast_1d)
	if use_weighted:
		w = weights.flatten()
		ss.mean = numpy.average(arr, axis=0, weights=w)
		variance = numpy.average((arr-ss.mean)**2, axis=0, weights=w)
		ss.stdev = numpy.sqrt(variance)
		w_nonzero = w.copy().reshape(w.shape[0],1) * numpy.ones(arr.shape[1])
		w_nonzero[arr==0] = 0
		ss.mean_nonzero = numpy.average(arr, axis=0, weights=w_nonzero)

	ncols = 0
	stack = []
	titles = []

	if show_descrip:
		stack += [descriptions,]
		titles += ["Description",]
		ncols += 1
	else:
		stack += [names,]
		titles += ["Data",]
		ncols += 1

	ncols += 5
	stack += [ss.mean,ss.stdev,ss.minimum,ss.maximum,ss.n_zeros,ss.mean_nonzero]
	titles += ["Mean","Std.Dev.","Minimum","Maximum","Zeros","Mean(NonZero)"]
	
	use_p = (numpy.sum(ss.n_positives)>0)
	use_n = (numpy.sum(ss.n_negatives)>0)
	
	if use_p:
		stack += [ss.n_positives,]
		titles += ["Positives",]
		ncols += 1
	if use_n:
		stack += [ss.n_negatives,]
		titles += ["Negatives",]
		ncols += 1

	# Histograms
	stack += [ss.histogram,]
	titles += ["Distribution",]
	ncols += 1

	if show_descrip:
		stack += [names,]
		titles += ["Data",]
		ncols += 1

	x = AbstractReportTable(title=title, short_title=short_title, columns=titles)
	x.addrow_seq_of_strings(titles)
	for s in zip(*stack):
		s_formatted = []
		for si in s:
			try:
				is_int = (int(si)==float(si))
			except:
				s_formatted += [si,]
			else:
				use_fmt = "{:.0f}" if is_int else "{:.5g}"
				s_formatted += [use_fmt.format(float(si)),]
		x.addrow_seq_of_strings(s_formatted)
	
	for footnote in sorted(ss.notes):
		x.footnotes.append(footnote)
	
	return x





def basic_idca_variable_analysis(arr, names, av, weights=None, description_catalog=None, title="Analytics", short_title=None):
	"""
	Parameters
	----------
	arr : ndarray
		The data to analyze.  Should have shape (cases, vars)
	names : sequence of str
		The names of the variables.  Should have length matching last dim of `arr`
	"""

	arr_original_shape = arr.shape
	av = av.reshape(arr.shape[0]*arr.shape[1])
	arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])[av]

	if description_catalog is None:
		description_catalog = {}
		from ..roles import _data_description_catalog
		description_catalog.update(_data_description_catalog)
	
	description_catalog_keys = list(description_catalog.keys())
	description_catalog_keys.sort(key=len, reverse=True)
	
	descriptions = numpy.asarray(names)
	
	for dnum, descr in enumerate(descriptions):
		if descr in description_catalog:
			descriptions[dnum] = description_catalog[descr]
		else:
			for key in description_catalog_keys:
				if key in descr:
					descr = descr.replace(key,description_catalog[key])
			descriptions[dnum] = descr

	show_descrip = (numpy.asarray(descriptions)!=numpy.asarray(names)).any()

	use_weighted = weights is not None and bool((weights!=1).any())

	if use_weighted:
		title += " (weighted)"

	ss = statistical_summary.compute(arr, dimzer=numpy.atleast_1d)
	if use_weighted:
		w = numpy.broadcast_to(weights, arr_original_shape).flatten()[av]
		ss.mean = numpy.average(arr, axis=0, weights=w)
		variance = numpy.average((arr-ss.mean)**2, axis=0, weights=w)
		ss.stdev = numpy.sqrt(variance)
		w_nonzero = w.copy().reshape(w.shape[0],1) * numpy.ones(arr.shape[1])
		w_nonzero[arr==0] = 0
		ss.mean_nonzero = numpy.average(arr, axis=0, weights=w_nonzero)

	ncols = 0
	stack = []
	titles = []

	if show_descrip:
		stack += [descriptions,]
		titles += ["Description",]
		ncols += 1
	else:
		stack += [names,]
		titles += ["Data",]
		ncols += 1

	ncols += 5
	stack += [ss.mean,ss.stdev,ss.minimum,ss.maximum,ss.n_zeros,ss.mean_nonzero]
	titles += ["Mean","Std.Dev.","Minimum","Maximum","Zeros","Mean(NonZero)"]
	
	use_p = (numpy.sum(ss.n_positives)>0)
	use_n = (numpy.sum(ss.n_negatives)>0)
	
	if use_p:
		stack += [ss.n_positives,]
		titles += ["Positives",]
		ncols += 1
	if use_n:
		stack += [ss.n_negatives,]
		titles += ["Negatives",]
		ncols += 1

	# Histograms
	stack += [ss.histogram,]
	titles += ["Distribution",]
	ncols += 1

	if show_descrip:
		stack += [names,]
		titles += ["Data",]
		ncols += 1

	x = AbstractReportTable(title=title, short_title=short_title, columns=titles)
	x.addrow_seq_of_strings(titles)
	for s in zip(*stack):
		s_formatted = []
		for si in s:
			try:
				is_int = (int(si)==float(si))
			except:
				s_formatted += [si,]
			else:
				use_fmt = "{:.0f}" if is_int else "{:.5g}"
				s_formatted += [use_fmt.format(float(si)),]
		x.addrow_seq_of_strings(s_formatted)
	
	for footnote in sorted(ss.notes):
		x.footnotes.append(footnote)
	
	return x





def _make_buckets(arr, ch, av, for_altcode, altcodes):
	"""
	
	Parameters
	----------
	arr : ndarray
		The data to analyze.  Should have shape (cases, alts, vars)
	ch : ndarray
		The data to analyze.  Should have shape (cases, alts, 1) and dtype float
	av : ndarray
		The data to analyze.  Should have shape (cases, alts, 1) and dtype bool
	altcodes : sequence of int
		The code numbers for elemental alternatives
	"""
	
	altslots = numpy.zeros([len(altcodes)], dtype=bool)
	# CONVERT altcode to SLOT
	for anum,acode in enumerate(altcodes):
		if acode == for_altcode:
			altslots[anum] = True
			break

	ch_asbool = ch.astype(bool).reshape(ch.shape[0],ch.shape[1])
	ch_asbool[:,~altslots] = False
	
	x_chosen = arr[ch_asbool]
	
	ch_asbool[:,altslots] = ~ch_asbool[:,altslots] & av[:,altslots].squeeze(axis=2)
	x_unchosen = arr[ch_asbool]
	x_total = arr[av[:,altslots].squeeze(),altslots].squeeze()
	
	ss_chosen = statistical_summary.compute(x_chosen, dimzer=numpy.atleast_1d, full_xxx=x_total)
	ss_unchosen = statistical_summary.compute(x_unchosen, dimzer=numpy.atleast_1d, full_xxx=x_total)
	
	return ss_chosen, ss_unchosen,


def _make_buckets_flattened(arr, ch, av):
	"""
	
	Parameters
	----------
	arr : ndarray
		The data to analyze.  Should have shape (cases, alts, vars)
	ch : ndarray
		The data to analyze.  Should have shape (cases, alts, 1) and dtype float
	av : ndarray
		The data to analyze.  Should have shape (cases, alts, 1) and dtype bool
	altcodes : sequence of int
		The code numbers for elemental alternatives
	"""
	
	ch_asbool = ch.astype(bool).reshape(ch.shape[0],ch.shape[1])
	
	x_chosen = arr[ch_asbool]
	
	ch_asbool[:,:] = ~ch_asbool[:,:] & av[:,:].squeeze(axis=2)
	x_unchosen = arr[ch_asbool]
	x_total = arr[av[:,:].squeeze(axis=2)]
	
	ss_chosen = statistical_summary.compute(x_chosen, dimzer=numpy.atleast_1d, full_xxx=x_total)
	ss_unchosen = statistical_summary.compute(x_unchosen, dimzer=numpy.atleast_1d, full_xxx=x_total)
	
	return ss_chosen, ss_unchosen,



def basic_variable_analysis_by_alt(arr, ch, av, names, altcodes, altnames, title="Analytics", short_title=None, picks=None):
	"""
	Selected statistics about idca data.
	
	This method requires the data to be currently provisioned in the
	:class:`Model`.
	
	Parameters
	----------
	arr : ndarray
		The data to analyze.  Should have shape (cases, alts, vars) or (cases, vars)
	ch : ndarray
		The choice data to analyze.  Should have shape (cases, alts, 1) and dtype float
	av : ndarray
		The availability data to analyze.  Should have shape (cases, alts, 1) and dtype bool
	names : sequence of str
		The names of the variables.  Should have length matching last dim of `arr`
	altcodes : sequence of int
		The code numbers for elemental alternatives
	altnames : sequence of str
		The names of the elemental alternatives, same len and order as codes
	
	Returns
	-------
	:class:`pandas.DataFrame`
		A DataFrame containing selected statistics.
	
	"""
	if len(arr.shape)==2:
		arr = arr[:, None, :][:, numpy.zeros(len(altcodes), dtype=int), :]
	footnotes = set()
	table_cache = pandas.DataFrame(columns=['namecounter',"altcode","altname","data","filter","mean","stdev","min","max","zeros","mean_nonzero","positives","negatives","descrip","histogram",])
	if picks is None:
		pickcodes = None
	else:
		pickcodes = set(i[0] for i in picks)

	for acode,aname in zip(altcodes, altnames):
		if pickcodes is None or acode in pickcodes:
			bucket = _make_buckets( arr, ch, av, acode, altcodes )
			bucket_types = ["Chosen", "Unchosen"]
			for summary_attrib, bucket_type in zip( bucket, bucket_types ):
				if summary_attrib.empty():
					continue
				for name_,n_,sa_ in zip(names,count(),summary_attrib):
					if picks is None or (acode,name_) in picks:
						newrow = {'altcode':acode, 'altname':aname, 'filter':bucket_type,
									'data':name_, 'mean':sa_.mean, 'stdev':sa_.stdev,
									'min':sa_.minimum,'max':sa_.maximum,'zeros':sa_.n_zeros,'mean_nonzero':sa_.mean_nonzero,
									'positives':sa_.n_positives,'negatives':sa_.n_negatives,'namecounter':n_,
									'histogram':sa_.histogram,
						}
						table_cache = table_cache.append(newrow, ignore_index=True)
						footnotes |= summary_attrib.notes
#				means = summary_attrib.mean
#				stdevs = summary_attrib.stdev
#				mins = summary_attrib.minimum
#				maxs = summary_attrib.maximum
#				nonzers = summary_attrib.n_nonzeros
#				posis = summary_attrib.n_positives
#				negs = summary_attrib.n_negatives
#				zers = summary_attrib.n_zeros
#				mean_nonzer = summary_attrib.mean_nonzero
#				histos = summary_attrib.histogram
#				footnotes |= summary_attrib.notes
#				try:
#					for s in zip(names,means,stdevs,mins,maxs,zers,mean_nonzer,posis,negs,count(),histos,):
#						if picks is None or (acode,s[0]) in picks:
#							newrow = {'altcode':acode, 'altname':aname, 'filter':bucket_type,
#										'data':s[0], 'mean':s[1], 'stdev':s[2],
#										'min':s[3],'max':s[4],'zeros':s[5],'mean_nonzero':s[6],
#										'positives':s[7],'negatives':s[8],'namecounter':s[9],
#										'histogram':s[10],
#							}
#							table_cache = table_cache.append(newrow, ignore_index=True)
#				except:
#					print('names',type(names),names)
#					raise
	table_cache.sort_values(['altcode','namecounter','filter'], inplace=True)
	table_cache.index = range(len(table_cache))



	display_cols = [
		('Filter','filter', "{}"),
		('Mean',"mean", "{:.5g}"),
		('Std.Dev.',"stdev", "{:.5g}"),
		('Minimum',"min", "{}"),
		('Maximum',"max", "{}"),
		('Mean (Nonzeros)',"mean_nonzero", "{:.5g}"),
		('# Zeros',"zeros", "{:.0f}"),
	]
	
	if table_cache['positives'].sum()>0:
		display_cols += [('# Positives',"positives", "{:.0f}"),]
	if table_cache['negatives'].sum()>0:
		display_cols += [('# Negatives',"negatives", "{:.0f}"),]



	x = AbstractReportTable(columns=['alt','data']+[t[0] for t in display_cols]+['dist'], title=title, short_title=short_title)
	x.add_blank_row()
	x.set_lastrow_iloc(0,'Alternative')
	x.set_lastrow_iloc(1,'Data')
	cn = 2
	for coltitle,colvalue,_ in display_cols:
		x.set_lastrow_iloc(cn,coltitle)
		cn += 1
	x.set_lastrow_iloc(cn,'Distribution')

	for acode,aname in zip(altcodes, altnames):
		block = table_cache[table_cache['altcode']==acode]
		block1 = True
		for rownum in block.index:
			x.add_blank_row()
			
			try:
				x.set_lastrow_iloc_nondupe(0, aname)
				x.set_lastrow_iloc_nondupe_wide(1, block.loc[rownum,'data'])
				cn = 2
				for coltitle,colvalue,colfmt in display_cols:
					x.set_lastrow_iloc(cn,colfmt.format( block.loc[rownum,colvalue] ) )
					cn += 1
				x.set_lastrow_iloc(cn, block.loc[rownum,'histogram'])
			except:
				print("Exception in Code")
				print(block)
				raise

	x.footnotes = sorted(footnotes)

	return x




def basic_variable_analysis_all(arr, ch, av, names, title="Analytics", short_title=None, picks=None):
	"""
	Selected statistics about idca data.
	
	This method requires the data to be currently provisioned in the
	:class:`Model`.
	
	Parameters
	----------
	arr : ndarray
		The data to analyze.  Should have shape (cases, alts, vars) or (cases, vars)
	ch : ndarray
		The choice data to analyze.  Should have shape (cases, alts, 1) and dtype float
	av : ndarray
		The availability data to analyze.  Should have shape (cases, alts, 1) and dtype bool
	names : sequence of str
		The names of the variables.  Should have length matching last dim of `arr`
	
	Returns
	-------
	:class:`pandas.DataFrame`
		A DataFrame containing selected statistics.
	
	"""
	if len(arr.shape)==2:
		raise TypeError('only for idca')
	footnotes = set()
	table_cache = pandas.DataFrame(columns=['namecounter',"data","filter","mean","stdev","min","max","zeros","mean_nonzero","positives","negatives","descrip","histogram",])

	bucket = _make_buckets_flattened( arr, ch, av )
	bucket_types = ["Chosen", "Unchosen"]
	for summary_attrib, bucket_type in zip( bucket, bucket_types ):
		if summary_attrib.empty():
			continue
		for name_,n_,sa_ in zip(names,count(),summary_attrib):
			if picks is None or name_ in picks:
				newrow = {'filter':bucket_type,
							'data':name_, 'mean':sa_.mean, 'stdev':sa_.stdev,
							'min':sa_.minimum,'max':sa_.maximum,'zeros':sa_.n_zeros,'mean_nonzero':sa_.mean_nonzero,
							'positives':sa_.n_positives,'negatives':sa_.n_negatives,'namecounter':n_,
							'histogram':sa_.histogram,
				}
				table_cache = table_cache.append(newrow, ignore_index=True)
				footnotes |= summary_attrib.notes
	table_cache.sort_values(['namecounter','filter'], inplace=True)
	table_cache.index = range(len(table_cache))

	display_cols = [
		('Filter','filter', "{}"),
		('Mean',"mean", "{:.5g}"),
		('Std.Dev.',"stdev", "{:.5g}"),
		('Minimum',"min", "{}"),
		('Maximum',"max", "{}"),
		('Mean (Nonzeros)',"mean_nonzero", "{:.5g}"),
		('# Zeros',"zeros", "{:.0f}"),
	]
	
	if table_cache['positives'].sum()>0:
		display_cols += [('# Positives',"positives", "{:.0f}"),]
	if table_cache['negatives'].sum()>0:
		display_cols += [('# Negatives',"negatives", "{:.0f}"),]



	x = AbstractReportTable(columns=['data']+[t[0] for t in display_cols]+['dist'], title=title, short_title=short_title)
	x.add_blank_row()
	x.set_lastrow_iloc(0,'Data')
	cn = 1
	for coltitle,colvalue,_ in display_cols:
		x.set_lastrow_iloc(cn,coltitle)
		cn += 1
	x.set_lastrow_iloc(cn,'Distribution')

	block = table_cache
	block1 = True
	for rownum in block.index:
		x.add_blank_row()
		
		try:
			x.set_lastrow_iloc_nondupe_wide(0, block.loc[rownum,'data'])
			cn = 1
			for coltitle,colvalue,colfmt in display_cols:
				x.set_lastrow_iloc(cn,colfmt.format( block.loc[rownum,colvalue] ) )
				cn += 1
			x.set_lastrow_iloc(cn, block.loc[rownum,'histogram'])
		except:
			print("Exception in Code")
			print(block)
			raise

	x.footnotes = sorted(footnotes)

	return x






#
#
#
#def basic_idco_variable_analysis_by_alt(arr, ch, av, names, altcodes, altnames, title="Analytics", short_title=None):
#	"""
#	Selected statistics about idca data.
#	
#	This method requires the data to be currently provisioned in the
#	:class:`Model`.
#	
#	Parameters
#	----------
#	arr : ndarray
#		The data to analyze.  Should have shape (cases, alts, vars)
#	ch : ndarray
#		The choice data to analyze.  Should have shape (cases, alts, 1) and dtype float
#	av : ndarray
#		The availability data to analyze.  Should have shape (cases, alts, 1) and dtype bool
#	names : sequence of str
#		The names of the variables.  Should have length matching last dim of `arr`
#	altcodes : sequence of int
#		The code numbers for elemental alternatives
#	altnames : sequence of str
#		The names of the elemental alternatives, same len and order as codes
#	
#	Returns
#	-------
#	:class:`pandas.DataFrame`
#		A DataFrame containing selected statistics.
#	
#	"""
#	footnotes = set()
#	table_cache = pandas.DataFrame(columns=["altcode","altname","data","filter","mean","stdev","min","max","zeros","mean_nonzero","positives","negatives","descrip","histogram",])
#	for acode,aname in zip(altcodes, altnames):
#		bucket = _make_buckets( arr, ch, av, acode, altcodes )
#		bucket_types = ["All Avail", "Chosen", "Unchosen"]
#		for summary_attrib, bucket_type in zip( bucket, bucket_types ):
#			if summary_attrib.empty():
#				continue
#			means = summary_attrib.mean
#			stdevs = summary_attrib.stdev
#			mins = summary_attrib.minimum
#			maxs = summary_attrib.maximum
#			nonzers = summary_attrib.n_nonzeros
#			posis = summary_attrib.n_positives
#			negs = summary_attrib.n_negatives
#			zers = summary_attrib.n_zeros
#			mean_nonzer = summary_attrib.mean_nonzero
#			histos = summary_attrib.histogram
#			footnotes |= summary_attrib.notes
#			try:
#				for s in zip(names,means,stdevs,mins,maxs,zers,mean_nonzer,posis,negs,count(),histos,):
#					newrow = {'altcode':acode, 'altname':aname, 'filter':bucket_type,
#								'data':s[0], 'mean':s[1], 'stdev':s[2],
#								'min':s[3],'max':s[4],'zeros':s[5],'mean_nonzero':s[6],
#								'positives':s[7],'negatives':s[8],'namecounter':s[9],
#								'histogram':s[10],
#					}
#					table_cache = table_cache.append(newrow, ignore_index=True)
#			except:
#				print('names',type(names),names)
#				print('means',type(means),means)
#				raise
#	table_cache.sort_values(['altcode','namecounter','filter'], inplace=True)
#	table_cache.index = range(len(table_cache))
#
#
#
#	display_cols = [
#		('Filter','filter', "{}"),
#		('Mean',"mean", "{:.5g}"),
#		('Std.Dev.',"stdev", "{:.5g}"),
#		('Minimum',"min", "{}"),
#		('Maximum',"max", "{}"),
#		('Mean (Nonzeros)',"mean_nonzero", "{:.5g}"),
#		('# Zeros',"zeros", "{:.0f}"),
#	]
#	
#	if table_cache['positives'].sum()>0:
#		display_cols += [('# Positives',"positives", "{:.0f}"),]
#	if table_cache['negatives'].sum()>0:
#		display_cols += [('# Negatives',"negatives", "{:.0f}"),]
#
#
#
#	x = AbstractReportTable(columns=['alt','data']+[t[0] for t in display_cols]+['dist'], title=title, short_title=short_title)
#	x.add_blank_row()
#	x.set_lastrow_iloc(0,'Alternative')
#	x.set_lastrow_iloc(1,'Data')
#	cn = 2
#	for coltitle,colvalue,_ in display_cols:
#		x.set_lastrow_iloc(cn,coltitle)
#		cn += 1
#	x.set_lastrow_iloc(cn,'Distribution')
#
#	for acode,aname in zip(altcodes, altnames):
#		block = table_cache[table_cache['altcode']==acode]
#		block1 = True
#		for rownum in block.index:
#			x.add_blank_row()
#			
#			try:
#				x.set_lastrow_iloc_nondupe(0, aname)
#				x.set_lastrow_iloc_nondupe(1, block.loc[rownum,'data'])
#				cn = 2
#				for coltitle,colvalue,colfmt in display_cols:
#					x.set_lastrow_iloc(cn,colfmt.format( block.loc[rownum,colvalue] ) )
#					cn += 1
#				x.set_lastrow_iloc(cn, block.loc[rownum,'histogram'])
#			except:
#				print("Exception in Code")
#				print(block)
#				raise
#
#	x.footnotes = sorted(footnotes)
#
#	return x
#
#
#
#
#
#






def art_choice_distributions(m, pct_str=True):

	if not m.is_provisioned():
		m.provision()

	co_label = lambda x: X(x).descrip or x

	ch = m.Data("Choice")
	df = pandas.DataFrame(index=pandas.MultiIndex.from_product([(), ()], names=['Variable','Value[s]']),
	                      columns=list(m.alternative_names())+['Count'])

	co_sources = "UtilityCO", "SamplingCO"

	for co_source in co_sources:

		co = m.Data(co_source)
		if co_source not in m.needs():
			continue
		co_names = m.needs()[co_source].get_variables()

		for i in range(co.shape[1]):
			varlabel = co_label(co_names[i])
			uniq = numpy.unique(co[:100,i])
			if len(uniq)<=4:
				uniq = numpy.unique(co[:,i])

			if len(uniq) == 1:
				continue

			elif len(uniq) <= 4:
				if len(uniq)==2 and uniq[0]==0 and uniq[1]==1:
					# Yes
					choice_when_uniq = ch[co[:, i] == 1].sum(0)
					df.loc[(varlabel, 'Yes'), 'Count'] = choice_when_uniq.sum()
					choice_when_uniq /= choice_when_uniq.sum() / 100
					df.loc[(varlabel, 'Yes'), :].values[:-1] = choice_when_uniq.squeeze()
					# No
					choice_when_uniq = ch[co[:, i] == 0].sum(0)
					df.loc[(varlabel, 'No'), 'Count'] = choice_when_uniq.sum()
					choice_when_uniq /= choice_when_uniq.sum() / 100
					df.loc[(varlabel, 'No'), :].values[:-1] = choice_when_uniq.squeeze()
				else:
					for uniqslot in range(len(uniq)):
						choice_when_uniq = ch[co[:,i]==uniq[uniqslot]].sum(0)
						df.loc[(varlabel, uniq[uniqslot]),'Count'] = choice_when_uniq.sum()
						choice_when_uniq /= choice_when_uniq.sum()/100
						df.loc[(varlabel, uniq[uniqslot]),:].values[:-1] = choice_when_uniq.squeeze()

			else: # more than 4 unique values, report by thirds plus zeros
				use_zeros = (co[:, i]==0)


				highmark = numpy.nanpercentile(co[~use_zeros, i], 66.67)
				lowmark = numpy.nanpercentile(co[~use_zeros, i], 33.33)
				choice_when_high = ch[(co[:, i] > highmark)&(~use_zeros)].sum(0)
				choice_when_mid = ch[(co[:, i] <= highmark) & (co[:, i] >= lowmark)&(~use_zeros)].sum(0)
				choice_when_low = ch[(co[:, i] < lowmark)&(~use_zeros)].sum(0)
				choice_when_zero = ch[use_zeros].sum(0)
				low_label = "< {:.2g}".format(lowmark)
				mid_label = "{:.2g} to {:.2g}".format(lowmark,highmark)
				if lowmark==highmark:
					mid_label = "= {:.2g}".format(lowmark)
				highlabel = "> {:.2g}".format(highmark)
				zerolabel = "= 0"

				cnt = choice_when_high.sum()
				if cnt:
					df.loc[(varlabel, highlabel), 'Count'] = cnt
					choice_when_high /= choice_when_high.sum() / 100
					df.loc[(varlabel, highlabel), :].values[:-1] = choice_when_high.squeeze()

				cnt = choice_when_mid.sum()
				if cnt:
					df.loc[(varlabel, mid_label), 'Count'] = cnt
					choice_when_mid /= choice_when_mid.sum() / 100
					df.loc[(varlabel, mid_label), :].values[:-1] = choice_when_mid.squeeze()

				cnt = choice_when_low.sum()
				if cnt:
					df.loc[(varlabel, low_label), 'Count'] = cnt
					choice_when_low /= choice_when_low.sum() / 100
					df.loc[(varlabel, low_label), :].values[:-1] = choice_when_low.squeeze()

				cnt = choice_when_zero.sum()
				if cnt:
					df.loc[(varlabel, zerolabel), 'Count'] = cnt
					choice_when_zero /= choice_when_zero.sum() / 100
					df.loc[(varlabel, zerolabel), :].values[:-1] = choice_when_zero.squeeze()

#	ca_sources = "UtilityCA", "QuantityCA"
#
#	cha = ch.flatten()[m.Data("Avail").flatten()]
#
#	for ca_source in ca_sources:
#
#		ca = m.Data(ca_source)
#		if ca_source not in m.needs():
#			continue
#		ca_names = m.needs()[ca_source].get_variables()
#
#		co = ca.reshape(ca.shape[0]*ca.shape[1], ca.shape[2])
#		co = co[m.Data("Avail").flatten()]
#
#		for i in range(co.shape[1]):
#			varlabel = co_label(ca_names[i])
#			uniq = numpy.unique(co[:100,i])
#			if len(uniq)<=4:
#				uniq = numpy.unique(co[:,i])
#
#			if len(uniq) == 1:
#				continue
#
#			elif len(uniq) <= 4:
#				if len(uniq)==2 and uniq[0]==0 and uniq[1]==1:
#					# Yes
#					choice_when_uniq = ch[co[:, i] == 1].sum(0)
#					df.loc[(varlabel, 'Yes'), 'Count'] = choice_when_uniq.sum()
#					choice_when_uniq /= choice_when_uniq.sum() / 100
#					df.loc[(varlabel, 'Yes'), :].values[:-1] = choice_when_uniq.squeeze()
#					# No
#					choice_when_uniq = ch[co[:, i] == 0].sum(0)
#					df.loc[(varlabel, 'No'), 'Count'] = choice_when_uniq.sum()
#					choice_when_uniq /= choice_when_uniq.sum() / 100
#					df.loc[(varlabel, 'No'), :].values[:-1] = choice_when_uniq.squeeze()
#				else:
#					for uniqslot in range(len(uniq)):
#						choice_when_uniq = ch[co[:,i]==uniq[uniqslot]].sum(0)
#						df.loc[(varlabel, uniq[uniqslot]),'Count'] = choice_when_uniq.sum()
#						choice_when_uniq /= choice_when_uniq.sum()/100
#						df.loc[(varlabel, uniq[uniqslot]),:].values[:-1] = choice_when_uniq.squeeze()
#
#			else: # more than 4 unique values, report by thirds plus zeros
#				use_zeros = (co[:, i]==0)
#
#
#				highmark = numpy.nanpercentile(co[~use_zeros, i], 66.67)
#				lowmark = numpy.nanpercentile(co[~use_zeros, i], 33.33)
#				choice_when_high = ch[(co[:, i] > highmark)&(~use_zeros)].sum(0)
#				choice_when_mid = ch[(co[:, i] <= highmark) & (co[:, i] >= lowmark)&(~use_zeros)].sum(0)
#				choice_when_low = ch[(co[:, i] < lowmark)&(~use_zeros)].sum(0)
#				choice_when_zero = ch[use_zeros].sum(0)
#				low_label = "< {:.2g}".format(lowmark)
#				mid_label = "{:.2g} to {:.2g}".format(lowmark,highmark)
#				if lowmark==highmark:
#					mid_label = "= {:.2g}".format(lowmark)
#				highlabel = "> {:.2g}".format(highmark)
#				zerolabel = "= 0"
#
#				cnt = choice_when_high.sum()
#				if cnt:
#					df.loc[(varlabel, highlabel), 'Count'] = cnt
#					choice_when_high /= choice_when_high.sum() / 100
#					df.loc[(varlabel, highlabel), :].values[:-1] = choice_when_high.squeeze()
#
#				cnt = choice_when_mid.sum()
#				if cnt:
#					df.loc[(varlabel, mid_label), 'Count'] = cnt
#					choice_when_mid /= choice_when_mid.sum() / 100
#					df.loc[(varlabel, mid_label), :].values[:-1] = choice_when_mid.squeeze()
#
#				cnt = choice_when_low.sum()
#				if cnt:
#					df.loc[(varlabel, low_label), 'Count'] = cnt
#					choice_when_low /= choice_when_low.sum() / 100
#					df.loc[(varlabel, low_label), :].values[:-1] = choice_when_low.squeeze()
#
#				cnt = choice_when_zero.sum()
#				if cnt:
#					df.loc[(varlabel, zerolabel), 'Count'] = cnt
#					choice_when_zero /= choice_when_zero.sum() / 100
#					df.loc[(varlabel, zerolabel), :].values[:-1] = choice_when_zero.squeeze()

	if len(df)==0:
		from ..model_reporter.art import SkipReportTable
		return SkipReportTable()

	if pct_str:
		for a in m.alternative_names():
			df[a] = df[a].apply(lambda x:"{:.2f}%".format(x))

	return AbstractReportTable.FromDataFrame(df, title="Choice Distributions by Data Attributes", short_title="Choice Distributions")






