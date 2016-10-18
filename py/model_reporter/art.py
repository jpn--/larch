from . import docx, latex, pdf, txt, xhtml, alogit
import math
from ..util.xhtml import XHTML, XML_Builder, Elem
from ..util.pmath import category, pmath, rename
from ..util.categorize import CategorizerLabel, Categorizer, Renamer
from ..core import LarchError
import os
import pandas
import numpy
import itertools
from ..utilities import uid as _uid




class colorize:
	_PURPLE = '\033[95m'
	_CYAN = '\033[96m'
	_DARKCYAN = '\033[36m'  # colorama ok
	_BLUE = '\033[94m'
	_GREEN = '\033[32m'
	_LIGHTGREEN = '\033[92m'
	_BOLDGREEN = '\033[92;1m'
	_YELLOW = '\033[93m'
	_REDBRIGHT = '\033[91m'
	_RED = '\033[31m'
	_BOLD = '\033[1m'
	_UNDERLINE = '\033[4m'
	_END = '\033[0m'
	@classmethod
	def bold(cls, x):
		return cls._BOLD + str(x) + cls._END
	@classmethod
	def red1(cls, x):
		return cls._REDBRIGHT + str(x) + cls._END
	@classmethod
	def red(cls, x):
		return cls._RED + str(x) + cls._END
	@classmethod
	def boldgreen(cls, x):
		return cls._GREEN + cls._BOLD+ str(x) + cls._END
	@classmethod
	def green(cls, x):
		return cls._GREEN +  str(x) + cls._END
	@classmethod
	def darkcyan(cls, x):
		return cls._DARKCYAN + str(x) + cls._END



class _MergeFrom():
	def __init__(self, row_offset=0, col_offset=0):
		self.row_offset = row_offset
		self.col_offset = col_offset



class AbstractReportTables():
	def __init__(self, *arg, newlist=None, title=None, short_title=None):
		if newlist is None:
			self.arts = [a for a in arg]
		else:
			self.arts = [a for a in arg] + newlist
		self.title = title
		self.short_title = short_title
	def __add__(self, other):
		if other is None:
			return AbstractReportTables(newlist=self.arts)
		if isinstance(other, AbstractReportTables):
			return AbstractReportTables(newlist=self.arts+other.arts)
		if isinstance(other, AbstractReportTable):
			return AbstractReportTables(newlist=self.arts+[other])
		raise TypeError
	def __iadd__(self, other):
		if isinstance(other, AbstractReportTables):
			self.arts+=other.arts
			return self
		if isinstance(other, AbstractReportTable):
			self.arts+=[other,]
			return self
		if other is None:
			return self
		raise TypeError

	def __xml__(self, table_attrib=None, headlevel=2):
		div = XML_Builder("div")
		if self.title is None:
			for a in self.arts:
				div << a.xml(table_attrib=table_attrib)
		else:
			div.h2(self.title, anchor=self.short_title or self.title)
			for a in self.arts:
				div << a.xml(table_attrib=table_attrib, headlevel=headlevel+1)
		return div.close()

	xml = __xml__

	def _repr_html_(self):
		return self.__xml__().tostring().decode()

	def __repr__(self):
		return "\n\n".join(repr(a) for a in self.arts)


class SkipReportTable:
	def __init__(self, *arg, **kwarg):
		pass
	def xml(self, *arg, **kwarg):
		return None


class AbstractReportTable():

	def __add__(self, other):
		if isinstance(other, AbstractReportTables):
			return AbstractReportTables(newlist=[self,]+other.arts)
		if isinstance(other, AbstractReportTable):
			return AbstractReportTables(newlist=[self, other])
		if other is None:
			return AbstractReportTables(newlist=[self, ])
		raise TypeError
	

	def __init__(self, columns=('0',), col_classes=(), n_head_rows=1, from_dataframe=None, title=None, short_title=None):
		self.df = pandas.DataFrame(columns=columns, index=pandas.RangeIndex(0))
		self.col_classes = col_classes
		self.n_thead_rows = n_head_rows
		self.use_columns_as_thead = False
		self.silent_first_col_break = False
		self._col_width = None
		self.title = title
		self.short_title = short_title
		self.footnotes = []
		if from_dataframe is not None:
			self.from_dataframe(from_dataframe)

	def from_dataframe(self, d, keep_index=True, keep_columns=True):
		temp = d.copy()
		if keep_index:
			if isinstance(temp.index, pandas.MultiIndex):
				for ii in range(len(temp.index.levels)):
					subname = temp.index.names[ii]
					temp.insert(ii, subname, temp.index.get_level_values(ii))
					dupes = (temp[subname].iloc[1:] == temp[subname].iloc[:-1])
					temp[subname][1:][dupes] = numpy.nan
			else:
				newcol = ' '
				while newcol in temp.columns:
					newcol = newcol+' '
				temp.insert(0, ' ', temp.index)
			temp.index = pandas.RangeIndex(0,len(temp.index))
		if keep_columns:
			temp.loc[-1] = temp.columns
			temp.sort_index(inplace=True)
			temp.index = pandas.RangeIndex(0,len(temp.index))
			self.n_thead_rows = 1
		self.df = temp

	@classmethod
	def FromDataFrame(cls, *arg, title=None, short_title=None, **kwarg):
		x = cls(title=title, short_title=short_title)
		x.from_dataframe(*arg, **kwarg)
		return x

	def add_blank_row(self):
		self.df.loc[len(self.df)] = None
		self._col_width = None
	def encode_cell_value(self, value, attrib=None, tag='td', anchorlabel=None, auto_toc_level=None):
		if pandas.isnull(value):
			return None
		if attrib is None:
			attrib = {}
		if isinstance(value, _MergeFrom):
			return value
		if isinstance(value, numpy.ndarray) and value.shape==():
			value = value[()]
		elif isinstance(value, numpy.ndarray):
			value = str(value)
		if isinstance(value, Elem):
			value.tag = tag
			for k,v in attrib.items():
				if k=='class':
					v1 = value.get(k,None)
					if v1 is not None and v not in v1:
						v = "{} {}".format(v,v1)
				value.set(k,v)
			return value
		if anchorlabel is None:
			e = Elem(tag=tag, text=str(value), attrib=attrib)
		else:
			e = Elem(tag=tag, attrib=attrib)
			if auto_toc_level is None:
				e.append(Elem(tag='a', attrib={'name':str(anchorlabel)}, tail=str(value)))
			else:
				e.append(Elem(tag='a', attrib={'name':_uid(), 'reftxt':str(anchorlabel), 'class':'toc', 'toclevel':str(auto_toc_level)}, tail=str(value)))
		return e
	def encode_head_value(self, value, attrib=None):
		return self.encode_cell_value(value, attrib=attrib, tag='th')
	def addrow_seq_of_strings(self, str_content, attrib={}):
		if len(str_content) > len(self.df.columns):
			raise TypeError("Insufficient columns for new row")
		self.add_blank_row()
		for n,s in enumerate(str_content):
			self.df.iloc[-1,n] = self.encode_cell_value(s)
	def addrow_map_of_strings(self, str_content, attrib={}):
		self.add_blank_row()
		rowix = self.df.index[-1]
		for key,val in str_content.items():
			self.df.loc[rowix, key] = self.encode_cell_value(val)
	def addrow_kwd_strings(self, **str_content):
		self.add_blank_row()
		rowix = self.df.index[-1]
		for key,val in str_content.items():
			self.df.loc[rowix, key] = self.encode_cell_value(val)

	def set_lastrow_loc(self, colname, val, attrib=None, anchorlabel=None):
		rowix = self.df.index[-1]
		if colname not in self.df.columns:
			self.df[colname] = None
		newval = self.encode_cell_value(val, attrib, anchorlabel=anchorlabel)
		self.df.loc[rowix, colname] = newval
		self._col_width = None
	def set_lastrow_iloc(self, colnum, val, attrib=None, anchorlabel=None):
		self.df.iloc[-1, colnum] = self.encode_cell_value(val, attrib, anchorlabel=anchorlabel)
		self._col_width = None
	def set_lastrow_iloc_nondupe(self, colnum, val, attrib=None, anchorlabel=None):
		try:
			val_text = val.text
		except AttributeError:
			val_text = str(val)
		prev = -1
		prev_text = None
		try:
			while prev_text is None:
				prev -= 1
				prev_text = self.get_text_iloc(prev,colnum, missing=None)
			if prev_text!=val_text:
				raise NameError
		except (NameError, IndexError):
			self.df.iloc[-1, colnum] = self.encode_cell_value(val, attrib, anchorlabel=anchorlabel)
			self._col_width = None

	def set_lastrow_iloc_nondupe_wide(self, colnum, val, attrib=None, anchorlabel=None):
		try:
			val_text = val.text
		except AttributeError:
			val_text = str(val)
		prev = -1
		prev_text = None
		try:
			for c in range(colnum):
				if self.get_text_iloc(prev,c, missing=None) is not None:
					raise NameError
			while prev_text is None:
				prev -= 1
				prev_text = self.get_text_iloc(prev,colnum, missing=None)
			if prev_text!=val_text:
				raise NameError
		except (NameError, IndexError):
			self.df.iloc[-1, colnum] = self.encode_cell_value(val, attrib, anchorlabel=anchorlabel)
			self._col_width = None

	def __repr__(self):
		if self.title:
			s = " {}\n".format(colorize.boldgreen(self.title))
		else:
			s = ""
		s += self.unicodebox()
		for footnote in self.footnotes:
			s += "\n {}".format(colorize.darkcyan(footnote))
		return s

	def _dividing_line(self, leftend="+", rightend="+", splitpoint="+", linechar="─"):
		lines = [linechar*w for w in self.min_col_widths_()]
		return leftend+splitpoint.join(lines)+rightend

	def __str__(self):
		if self.title:
			s = " {}\n".format(self.title)
		else:
			s = ""
		s += self.unicodebox()
		for footnote in self.footnotes:
			s += "\n {}".format(footnote)
		return s

	def _text_output(self, topleft   ='┌', topsplit   ='┬', topright   ='┐',
	                       middleleft='├', middlesplit='┼', middleright='┤',
	                       bottomleft='└', bottomsplit='┴', bottomright='┘',
						   leftvert  ='│', othervert  ='│',
						   horizbar='─',
						   catleft='╞', catright='╡', cathorizbar='═',
						   suppress_internal_newlines=True,
						   ):
		s = self._dividing_line(leftend=topleft, rightend=topright, splitpoint=topsplit, linechar=horizbar)+"\n"
		w = self.min_col_widths()
		for r,rvalue in enumerate(self.df.index):
			if (~pandas.isnull(self.df.iloc[r,1:])).sum()==0:
				catflag = True
			else:
				catflag = False
			if r==self.n_thead_rows and self.n_thead_rows>0:
				s += self._dividing_line(leftend=middleleft, rightend=middleright, splitpoint=middlesplit, linechar=horizbar)+"\n"
			startline = True
			s += leftvert
			for c,cvalue in enumerate(self.df.columns):
				cellspan = self.cellspan_iloc(r,c)
				if cellspan != (0,0):
					cw = numpy.sum(w[c:c+cellspan[1]])+cellspan[1]-1
					if suppress_internal_newlines:
						thistext = self.get_text_iloc(r,c).replace('\t'," ").replace('\n'," ")
					else:
						thistext = self.get_text_iloc(r,c).replace('\t'," ")
					if catflag:
						if len(leftvert)>0:
							s = s[:-len(leftvert)]+ catleft+" {1:{2}<{0}s}".format(cw-1,thistext+" ",cathorizbar)+catright
						else:
							s += catleft+" {1:{2}<{0}s}".format(cw-1,thistext+" ",cathorizbar)+catright
					elif self.is_centered_cell(r,c):
						s += "{1: ^{0}s}".format(cw,thistext)+othervert
					else:
						s += "{1:{0}s}".format(cw,thistext)+othervert
					startline = False
				elif cellspan == (0,0) and startline:
					cw = w[c]
					if cw>0:
						s += "{1:{0}s}".format(cw,"")+othervert
					else:
						s += othervert
				else:
					startline = False
			s += "\n"
		s += self._dividing_line(leftend=bottomleft, rightend=bottomright, splitpoint=bottomsplit, linechar=horizbar)
		return s



	def unicodebox(self):
#		s = self._dividing_line(leftend='┌', rightend='┐', splitpoint='┬')+"\n"
#		w = self.min_col_widths()
#		for r,rvalue in enumerate(self.df.index):
#			if (~pandas.isnull(self.df.iloc[r,1:])).sum()==0:
#				catflag = True
#			else:
#				catflag = False
#			if r==self.n_thead_rows and self.n_thead_rows>0:
#				s += self._dividing_line(leftend='├', rightend='┤', splitpoint='┼')+"\n"
#			startline = True
#			s += "│"
#			for c,cvalue in enumerate(self.df.columns):
#				cellspan = self.cellspan_iloc(r,c)
#				if cellspan != (0,0):
#					cw = numpy.sum(w[c:c+cellspan[1]])+cellspan[1]-1
#					if catflag:
#						s = s[:-1]+ "╞ {1:═<{0}s}╡".format(cw-1,self.get_text_iloc(r,c)+" ")
#					else:
#						s += "{1:{0}s}│".format(cw,self.get_text_iloc(r,c))
#					startline = False
#				elif cellspan == (0,0) and startline:
#					cw = w[c]
#					s += "{1:{0}s}│".format(cw,"")
#				else:
#					startline = False
#			s += "\n"
#		s += self._dividing_line(leftend='└', rightend='┘', splitpoint='┴')
#		return s
		return self._text_output()

	def tabdelim(self):
#		s = self._dividing_line(leftend='', rightend='', splitpoint='\t')+"\n"
#		w = self.min_col_widths()
#		for r,rvalue in enumerate(self.df.index):
#			if (~pandas.isnull(self.df.iloc[r,:])).sum()==1:
#				catflag = True
#			else:
#				catflag = False
#			if r==self.n_thead_rows:
#				s += self._dividing_line(leftend='', rightend='', splitpoint='\t')+"\n"
#			startline = True
#			for c,cvalue in enumerate(self.df.columns):
#				cellspan = self.cellspan_iloc(r,c)
#				if cellspan != (0,0):
#					cw = numpy.sum(w[c:c+cellspan[1]])+cellspan[1]-1
#					if catflag:
#						s = s[:-1]+ " {1: <{0}s}".format(cw-1,self.get_text_iloc(r,c)+" ")
#					else:
#						s += "{1:{0}s}\t".format(cw,self.get_text_iloc(r,c))
#					startline = False
#				elif cellspan == (0,0) and startline:
#					cw = w[c]
#					s += "{1:{0}s}\t".format(cw,"")
#				else:
#					startline = False
#			s += "\n"
#		s += self._dividing_line(leftend='', rightend='', splitpoint='\t')
#		return s
		return self._text_output(  topleft   ='', topsplit   ='\t', topright   ='',
								   middleleft='', middlesplit='\t', middleright='',
								   bottomleft='', bottomsplit='\t', bottomright='',
								   leftvert  ='', othervert  ='\t', horizbar='-',
								   catleft='', catright='', cathorizbar=' ')

	def ascii(self):
		return self._text_output(  topleft   ='+', topsplit   ='+', topright   ='+',
								   middleleft='+', middlesplit='+', middleright='+',
								   bottomleft='+', bottomsplit='+', bottomright='+',
								   leftvert  =' ', othervert  =' ', horizbar='-',
								   catleft='=', catright='=', cathorizbar='=')

	def xml(self, table_attrib=None, headlevel=2):
		table = Elem(tag='table', attrib=table_attrib or {})
		thead = table.put('thead')
		tbody = table.put('tbody')
		tfoot = table.put('tfoot')
		r = 0
		while r<len(self.df.index):
			if r<self.n_thead_rows:
				tr = thead.put('tr')
				celltag = 'th'
			else:
				tr = tbody.put('tr')
				celltag = 'td'
			span = 1
			for c in range(len(self.df.columns)):
				attrib = {}
				try:
					attrib['class'] = self.col_classes[c]
				except:
					pass
				td = self.encode_cell_value(  self.df.iloc[r,c] , attrib=attrib, tag=celltag )
				if td is None or isinstance(td, _MergeFrom):
					try:
						tr.findall('./'+celltag)[-1].get('colspan','1')
					except IndexError:
						# There are no previous cells matching celltag
						# check if first cell in row, and cell above is same value
						upper_colspan = 1
						if c==0:
							rc_text, rx, cx = self.get_implied_text_iloc(r,c)
							if rx<0:
								try:
									tbody[rx-1][0].set('rowspan',str(1-rx))
									tbody[rx-1][0].set('style','vertical-align:top;')
									tr << Elem(tag='div', attrib={'class':'dummycell'})
								except IndexError:
									tr << self.encode_cell_value(  "" , attrib=attrib, tag=celltag )
							else:
								tr << self.encode_cell_value(  "" , attrib=attrib, tag=celltag )
						else:
							tr << self.encode_cell_value(  "" , attrib=attrib, tag=celltag )
					else:
						span += 1
						tr.findall('./'+celltag)[-1].set('colspan',str(span))
						tr << Elem(tag='div', attrib={'class':'dummycell'})
				else:
					tr << td
					span = 1
			r += 1
		if len(self.footnotes):
			sorted_footnotes = sorted(self.footnotes)
			caption = table.put('caption', text=sorted_footnotes[0])
			for footnote in sorted_footnotes[1:]:
				caption.put('br', tail=footnote)

		if headlevel is not None and self.title is not None:
			try:
				headlevel = int(headlevel)
			except:
				pass
			else:
				div = XML_Builder("div")
				div.hn(headlevel, self.title, anchor=self.short_title or self.title)
				div << table
				return div.close()
		return table
	
	__xml__ = xml

	def _repr_html_(self):
		return self.__xml__().tostring().decode()

	def to_xlsx(self, workbook, worksheet_name=None, r_top=0, c_left=0,
	            freeze_panes=True, hide_gridlines=True,
				metahead=None, buffercol=True):
		
		if worksheet_name is not None and worksheet_name not in workbook.sheetnames:
			worksheet = workbook.add_worksheet(worksheet_name)
		elif worksheet_name is None:
			worksheet = workbook.add_worksheet(self.short_title)
		else:
			worksheet = workbook.get_worksheet_by_name(worksheet_name)
		
		title_format = workbook.add_format({'bold': True, 'font_size':16})
		category_format = workbook.add_format({'bold': True, 'italic':True, 'bg_color':'#f4f4f4','border':1, 'border_color':'#AAAAAA',})
		tablehead = workbook.add_format({'bold': True, 'align':'center'})
		merged = workbook.add_format({'valign':'top', 'border':1, 'border_color':'#AAAAAA'})
		merged_tablehead = workbook.add_format({'valign':'top','bold': True, 'align':'center'})
		foot_format = workbook.add_format({'italic': True, 'font_size':10})

		current_format = (tablehead, merged_tablehead)

		# Buffer Column
		if buffercol:
			worksheet.set_column(c_left, c_left, 1)
			c_left += 1
		
		# MetaHeading
		if metahead is not None:
			metatitle_format = workbook.add_format({'bold': True, 'font_size':10})
			worksheet.write(r_top, c_left, metahead, metatitle_format)
			r_top += 1
		
		# Heading
		worksheet.write(r_top, c_left, self.title, title_format)
		r_top += 1

		# Content
		for r,rvalue in enumerate(self.df.index):
			if (~pandas.isnull(self.df.iloc[r,1:])).sum()==0:
				catflag = True
			else:
				catflag = False
			if r==self.n_thead_rows:
				# Have now completed header rows
				current_format = (merged, merged)
			startline = True
			for c,cvalue in enumerate(self.df.columns):
				cellspan = self.cellspan_iloc(r,c)
				if cellspan != (0,0):
					# This is a cell with real content
					if cellspan == (1,1):
						worksheet.write(r_top+r, c_left+c, self.get_text_iloc(r,c), current_format[0])
					else:
						worksheet.merge_range(r_top+r, c_left+c, r_top+r+cellspan[0]-1, c_left+c+cellspan[1]-1, self.get_text_iloc(r,c), category_format if catflag else current_format[1])
				else:
					startline = False

		# Col widths
		wid = self.min_col_widths()
		for c,w in enumerate(wid):
			worksheet.set_column(c+c_left, c+c_left, w)
		
		if freeze_panes:
			worksheet.freeze_panes(r_top+self.n_thead_rows, 0)
	
		if hide_gridlines:
			worksheet.hide_gridlines(2)

		r_top += len(self.df.index)

		from ..util.img import favicon_raw
		from io import BytesIO
		from ..version import version
		import time
		worksheet.write(r_top+1, c_left, "      Larch {}".format(version), foot_format)
		worksheet.insert_image(r_top+1, c_left, "larch_favicon.png",
								{'image_data': BytesIO(favicon_raw),
								 'x_scale':0.5, 'y_scale':0.5,
								 'x_offset':    2, 'y_offset':    2,})
		worksheet.write(r_top+2, c_left, "Report generated on "+time.strftime("%A %d %B %Y ")+time.strftime("%I:%M:%S %p"), foot_format)



	def cellspan_iloc(self,r,c):
		try:
			if pandas.isnull(self.df.iloc[r,c]):
				return (0,0)
		except:
			print("r:",r,type(r))
			print("c:",c,type(c))
			raise
		x,y = 1,1
		while r+y < len(self.df.index) and numpy.all(pandas.isnull(self.df.iloc[r+y,:c+1])):
			y += 1
		while c+x < len(self.df.columns) and pandas.isnull(self.df.iloc[r,c+x]):
			x += 1
		return (y,x)

	def is_centered_cell(self,r,c):
		if pandas.isnull(self.df.iloc[r,c]):
			return False
		if isinstance(self.df.iloc[r,c], Elem):
			cls = self.df.iloc[r,c].get('class','')
			if 'centered_cell' in cls:
				return True
		return False

	def get_text_iloc(self,r,c,missing=""):
		if pandas.isnull(self.df.iloc[r,c]):
			return missing
		if isinstance(self.df.iloc[r,c], Elem):
			txt = str(self.df.iloc[r,c].text or "")
			for subelement in self.df.iloc[r,c]:
				txt += str(subelement.text or "") + str(subelement.tail or "")
		else:
			txt = str(self.df.iloc[r,c])
		return txt

	def get_implied_text_iloc(self,r,c,missing=""):
		"""Get the implied text value of the field after merges, and the source cell.
		
		Parameters
		----------
		r : int
			The row of the cell to get
		c : int
			The columns of the cell to get
		missing : str
			Text to return if there is no implied text value
		
		Returns
		-------
		str
			The implied text value
		int
			The offset to the row of the cell generating this value
		int
			The offset to the column of the cell generating this value
		
		"""
		cx = c
		rx = r
		if c==0 and pandas.isnull(self.df.iloc[r,c]):
			# first column, null value so look up
			while rx>0 and pandas.isnull(self.df.iloc[rx,cx]):
				rx -=1
			if pandas.isnull(self.df.iloc[rx,cx]):
				return missing, 0, 0
			else:
				if isinstance(self.df.iloc[rx,cx], Elem):
					return self.df.iloc[rx,cx].text, rx-r, cx-c
				else:
					return str(self.df.iloc[rx,cx]), rx-r, cx-c
		while cx>0 and pandas.isnull(self.df.iloc[r,cx]):
			cx -= 1
		if pandas.isnull(self.df.iloc[r,cx]):
			return missing, 0, 0
		if isinstance(self.df.iloc[rx,cx], Elem):
			return self.df.iloc[rx,cx].text, rx-r, cx-c
		else:
			return str(self.df.iloc[rx,cx]), rx-r, cx-c

	def min_col_widths(self):
		if self._col_width is not None:
			return self._col_width
		w = numpy.zeros(len(self.df.columns), dtype=int)
		for r,c in itertools.product(range(len(self.df.index)),range(len(self.df.columns))):
			if self.cellspan_iloc(r,c)[1]==1:
				w[c] = max(w[c], len(self.get_text_iloc(r,c)))
		for span in range(2,len(self.df.columns)):
			for r,c in itertools.product(range(len(self.df.index)),range(len(self.df.columns))):
				if self.cellspan_iloc(r,c)[1]==span:
					shortage = len(self.get_text_iloc(r,c))  -   (numpy.sum(w[c:c+span])+span-1)
					if shortage>0:
						w[c] += shortage
		self._col_width = w
		return w

	def min_col_widths_(self):
		w = self.min_col_widths()
		if self.silent_first_col_break:
			ww = w[1:].copy()
			ww[0] += 1+w[0]
			return ww
		else:
			return w

	def __call__(self, m):
		return self


ART = AbstractReportTable

class AbstractReportTableFactory():
	"""This class generalizes the ART for both preprocessed and postprocessing tables."""
	def __init__(self, *, art=None, func=None, args=(), kwargs=None, title=None, short_title=None):
		if art is None and func is None:
			raise TypeError("art or func must be given")
		if art is not None and func is not None:
			raise TypeError("only one of art or func must be given")
		self.art = art
		self.func = func
		self.func_args = args
		self.func_kwargs = kwargs or {}
		self.title = title
		self.short_title = short_title or title
	def __call__(self, m):
		if self.func is None:
			candidate = self.art
		else:
			candidate = self.func(m, *self.func_args, **self.func_kwargs)
		if isinstance(candidate, pandas.DataFrame):
			candidate = AbstractReportTable.FromDataFrame(candidate)
		if candidate is None:
			print('candidate is None')
			return None
		if candidate.title is None and self.title is not None:
			candidate.title = self.title
		if candidate.short_title is None and self.short_title is not None:
			candidate.short_title = self.short_title
		return candidate





class ArtModelReporter():


	def art_new(self, handle, factory, title=None, short_title=None):
		try:
			self._user_defined_arts
		except AttributeError:
			self._user_defined_arts = {}
		if isinstance(factory, AbstractReportTableFactory):
			self._user_defined_arts[handle.lower()] = factory
		elif isinstance(factory, pandas.DataFrame):
			self._user_defined_arts[handle.lower()] = AbstractReportTableFactory(art=factory, title=title, short_title=short_title)
		elif callable(factory):
			self._user_defined_arts[handle.lower()] = AbstractReportTableFactory(func=factory, title=title, short_title=short_title)
		else:
			self._user_defined_arts[handle.lower()] = factory


	def _art_params_categorize(self, groups, display_inital=False, display_id=False, display_null=True, **format):
		"""
		Generate a ART containing the model parameters.
		
		Parameters
		----------
		groups : Categorizer
			An ordered list of parameters names and/or categories. If given,
			this list will be used to order the resulting table.
		display_inital : bool
			Should the initial values of the parameters (the starting point 
			for estimation) be included in the report. Defaults to False.
		display_id : bool
			Should the actual parameter names be shown in an id column.
			Defaults to False.  This can be useful if the groups include 
			renaming.
		display_null : bool
			Should the null values of the parameters (the reference point
			for a null model, typically the default no-information value
			of the parameter) be included in the report. Defaults to True.
		
		Returns
		-------
		AbstractReportTable
			An ART containing the model parameters.
		
		"""
		# keys fix
		existing_format_keys = list(format.keys())
		for key in existing_format_keys:
			if key.upper()!=key: format[key.upper()] = format[key]
		if 'PARAM' not in format: format['PARAM'] = '< 10.4g'
		if 'TSTAT' not in format: format['TSTAT'] = ' 0.2f'
		# build table

		columns = ["Parameter", None, "Estimated Value", "Std Error", "t-Stat"]
		col_classes = ['param_label','param_label', 'estimated_value', 'std_err', 'tstat']
		if display_inital:
			columns.insert(1,"Initial Value")
			col_classes.insert(1,'initial_value')
		if display_null:
			columns.append("Null Value")
			col_classes.append('null_value')
		if display_id:
			columns.append('id')
			col_classes.append('id')

		x = AbstractReportTable(columns=columns, col_classes=col_classes)
		x.silent_first_col_break = True
		x.title = "Model Parameter Estimates"
		x.short_title="Parameter Estimates"
		
		## USING GROUPS
		namelist = self.parameter_names() + list(self.alias_names())
		present_order = groups.match(namelist)[1]
				
		n_cols_params = 6 if display_inital else 5
		if display_id:
			n_cols_params += 1
		
		def write_param_row(p, *, force=False):
			if p is None: return
			if isinstance(p, CategorizerLabel):
				x.add_blank_row()
				x.set_lastrow_iloc(0, x.encode_cell_value(p.label, auto_toc_level=3, anchorlabel=p.label), {'class':"parameter_category"})
			else:
				if isinstance(p, Renamer):
					p_name = p.label
					p_decode = p.decode(namelist)
				else:
					p_name = p
					p_decode = p
				if p_decode is None:
					return
				x.add_blank_row()
				if "#" in p_name:
					p_name1, p_name2 = p_name.split("#",1)
					x.set_lastrow_iloc_nondupe(0, p_name1, )
					x.set_lastrow_iloc(1, p_name2, anchorlabel="param"+p_name2.replace("#","_hash_"))
				elif ":" in p_name:
					p_name1, p_name2 = p_name.split(":",1)
					x.set_lastrow_iloc_nondupe(0, p_name1, )
					x.set_lastrow_iloc(1, p_name2, anchorlabel="param"+p_name2.replace("#","_hash_"))
				else:
					x.set_lastrow_loc('Parameter', p_name, anchorlabel="param"+p_name.replace("#","_hash_"))
				self.art_single_parameter_resultpart(x,p_decode, with_inital=display_inital, with_nullvalue=display_null, **format)
				if display_id:
					x.set_lastrow_loc('id', p_decode)
					
		x.addrow_seq_of_strings(columns)
		for p in present_order.unpack():
			write_param_row(p)
		return x




	def art_params(self, groups=None, display_inital=False, display_id=False, display_null=True, **format):
		"""
		Generate a ART containing the model parameters.
		
		Parameters
		----------
		groups : None or list
			An ordered list of parameters names and/or categories. If given,
			this list will be used to order the resulting table.
		display_inital : bool
			Should the initial values of the parameters (the starting point 
			for estimation) be included in the report. Defaults to False.
		display_id : bool
			Should the actual parameter names be shown in an id column.
			Defaults to False.  This can be useful if the groups include 
			renaming.
		display_null : bool
			Should the null values of the parameters (the reference point
			for a null model, typically the default no-information value
			of the parameter) be included in the report. Defaults to True.
		
		Returns
		-------
		AbstractReportTable
			An ART containing the model parameters.
		
		"""
		if groups is None and hasattr(self, 'parameter_groups'):
			groups = self.parameter_groups

		if isinstance(groups, (tuple,list)):
			groups = Categorizer(None, *groups)

		if groups is None:
			groups = ()
			
		if isinstance(groups, Categorizer):
			return self._art_params_categorize(groups, display_inital=display_inital, display_id=display_id, display_null=display_null, **format)


		# keys fix
		existing_format_keys = list(format.keys())
		for key in existing_format_keys:
			if key.upper()!=key: format[key.upper()] = format[key]
		if 'PARAM' not in format: format['PARAM'] = '< 10.4g'
		if 'TSTAT' not in format: format['TSTAT'] = ' 0.2f'
		# build table

		columns = ["Parameter", None, "Estimated Value", "Std Error", "t-Stat", ]
		col_classes = ['param_label','param_label', 'estimated_value', 'std_err', 'tstat', ]
		if display_inital:
			columns.insert(1,"Initial Value")
			col_classes.insert(1,'initial_value')
		if display_null:
			columns.append("Null Value")
			col_classes.append('null_value')
		if display_id:
			columns.append('id')
			col_classes.append('id')

		x = AbstractReportTable(columns=columns, col_classes=col_classes)
		x.silent_first_col_break = True
		x.title = "Model Parameter Estimates"
		x.short_title="Parameter Estimates"


		## USING GROUPS
		listed_parameters = set([p for p in groups if not isinstance(p,category)])
		for p in groups:
			if isinstance(p,category):
				listed_parameters.update( p.complete_members() )
		unlisted_parameters_set = (set(self.parameter_names()) | set(self.alias_names())) - listed_parameters
		unlisted_parameters = []
		for pname in self.parameter_names():
			if pname in unlisted_parameters_set:
				unlisted_parameters.append(pname)
		for pname in self.alias_names():
			if pname in unlisted_parameters_set:
				unlisted_parameters.append(pname)
		n_cols_params = 5 if display_inital else 4
		if display_null:
			n_cols_params += 1
		if display_id:
			n_cols_params += 1
		
		def write_param_row(p, *, force=False):
			if p is None: return
			if force or (p in self) or (p in self.alias_names()):
				if isinstance(p,category):
					x.add_blank_row()
					x.set_lastrow_iloc(0, x.encode_cell_value(p.name, auto_toc_level=3, anchorlabel=p.name), {'class':"parameter_category"})
					for subp in p.members:
						write_param_row(subp)
				else:
					if isinstance(p,(rename, )):
						p_name = p.name
					else:
						p_name = p
					x.add_blank_row()
					if "#" in p_name:
						p_name1, p_name2 = p_name.split("#",1)
						x.set_lastrow_iloc_nondupe(0, p_name1, )
						x.set_lastrow_iloc(1, p_name2, anchorlabel="param"+p_name2.replace("#","_hash_"))
					elif ":" in p_name:
						p_name1, p_name2 = p_name.split(":",1)
						x.set_lastrow_iloc_nondupe(0, p_name1, )
						x.set_lastrow_iloc(1, p_name2, anchorlabel="param"+p_name2.replace("#","_hash_"))
					else:
						x.set_lastrow_loc('Parameter', p_name, anchorlabel="param"+p_name.replace("#","_hash_"))
					self.art_single_parameter_resultpart(x,p, with_inital=display_inital, with_nullvalue=display_null, **format)
					if display_id:
						if isinstance(p,(rename, )):
							p_id = p.find_in(self)
						else:
							p_id = p
						x.set_lastrow_loc('id', p_id)
		
		x.addrow_seq_of_strings(columns)
		for p in groups:
			write_param_row(p)
		if len(groups)>0 and len(unlisted_parameters)>0:
			write_param_row(category("Other Parameters"),force=True)
		if len(unlisted_parameters)>0:
			for p in unlisted_parameters:
				write_param_row(p)
		return x



	def art_single_parameter_resultpart(self, ART, p, *, with_inital=False,
										  with_stderr=True, with_tstat=True,
										  with_nullvalue=True, tstat_parens=False, **format):
		if p is None: return
		with_stderr = bool(with_stderr)
		with_tstat = bool(with_tstat)
		with_nullvalue = bool(with_nullvalue)
		x= ART
		if isinstance(p,(rename,Renamer,str)):
			try:
				model_p = self[p]
			except KeyError:
				use_shadow_p = True
			else:
				use_shadow_p = False
			if use_shadow_p:
				# Parameter not found, try shadow_parameter
				try:
					str_p = str(p.find_in(self))
				except AttributeError:
					str_p = p
				shadow_p = self.shadow_parameter[str_p]
				if with_inital:
					x.set_lastrow_loc('Initial Value', "")
				shadow_p_value = shadow_p.value
				x.set_lastrow_loc('Estimated Value', "{:{PARAM}}".format(shadow_p.value, **format))
				x.set_lastrow_loc('Std Error', "{}".format(shadow_p.t_stat))
			else:
				# Parameter found, use model_p
				if with_inital:
					x.set_lastrow_loc('Initial Value', "{:{PARAM}}".format(model_p.initial_value, **format))
				x.set_lastrow_loc('Estimated Value', "{:{PARAM}}".format(model_p.value, **format))
				if model_p.holdfast:
					x.set_lastrow_loc('Std Error', "fixed value")
					if with_nullvalue:
						x.set_lastrow_loc('Null Value', "{:{PARAM}}".format(model_p.null_value, **format))
				else:
					tstat_p = model_p.t_stat
					if isinstance(tstat_p,str):
						x.set_lastrow_loc('Std Error', "{}".format(tstat_p))
					elif tstat_p is None:
						x.set_lastrow_loc('Std Error', "{:{PARAM}}".format(model_p.std_err, **format))
						x.set_lastrow_loc('t-Stat', "None")
						if with_nullvalue:
							x.set_lastrow_loc('Null Value', "{:{PARAM}}".format(model_p.null_value, **format))
					else:
						x.set_lastrow_loc('Std Error', "{:{PARAM}}".format(model_p.std_err, **format))
						x.set_lastrow_loc('t-Stat', "{:{TSTAT}}".format(tstat_p, **format))
						if with_nullvalue:
							x.set_lastrow_loc('Null Value', "{:{PARAM}}".format(model_p.null_value, **format))



	def art_latest(self,**format):
		from ..utilities import format_seconds
		existing_format_keys = list(format.keys())
		for key in existing_format_keys:
			if key.upper()!=key: format[key.upper()] = format[key]
		if 'LL' not in format: format['LL'] = '0.2f'
		if 'RHOSQ' not in format: format['RHOSQ'] = '0.3f'
	
		es = self._get_estimation_statistics()

		x = AbstractReportTable(columns=["attr", "subattr", "value", ])
		x.silent_first_col_break = True
		x.n_thead_rows = 0
		
		x.title = "Latest Estimation Run Statistics"
		x.short_title = "Latest Estimation Run"

		try:
			last = self.maximize_loglike_results
		except AttributeError:
			x.add_blank_row()
			x.set_lastrow_iloc(0, "Warning")
			x.set_lastrow_iloc(2, "Latest estimation run statistics not available")
			return x
		try:
			last_stat = last.stats
		except AttributeError:
			pass
		else:
			x.add_blank_row()
			x.set_lastrow_iloc(0, "Estimation Date")
			x.set_lastrow_iloc(2, last_stat.timestamp)

			x.add_blank_row()
			x.set_lastrow_iloc(0, "Results")
			x.set_lastrow_iloc(2, last_stat.results)

			if len(last.intermediate) > 1:
				x.add_blank_row()
				x.set_lastrow_iloc(0, "Messages")
				x.set_lastrow_iloc(1, "Final")
				x.set_lastrow_iloc(2, str(last.message))
				for intermed in last.intermediate:
					x.add_blank_row()
					x.set_lastrow_iloc_nondupe(0, "Messages")
					x.set_lastrow_iloc(1, intermed.method)
					x.set_lastrow_iloc(2, intermed.message)
			else:
				x.add_blank_row()
				x.set_lastrow_iloc(0, "Message")
				x.set_lastrow_iloc(2, str(last.message))
				x.add_blank_row()
				x.set_lastrow_iloc(0, "Optimization Method")
				x.set_lastrow_iloc(2, last.intermediate[0].method)
			

			if len(last.niter) > 1:
				x.add_blank_row()
				x.set_lastrow_iloc(0, "Number of Iterations")
				x.set_lastrow_iloc(1, "Total")
				x.set_lastrow_iloc(2, last_stat.iteration)
				for iter in last.niter:
					x.add_blank_row()
					x.set_lastrow_iloc_nondupe(0, "Number of Iterations")
					x.set_lastrow_iloc(1, iter[0])
					x.set_lastrow_iloc(2, iter[1])
			else:
				x.add_blank_row()
				x.set_lastrow_iloc(0, "Number of Iterations")
				x.set_lastrow_iloc(2, last_stat.iteration)

			if len(last.intermediate) > 1:
				x.add_blank_row()
				x.set_lastrow_iloc(0, "Log Likelihood")
				x.set_lastrow_iloc(1, "Final")
				x.set_lastrow_iloc(2, str(last.loglike))
				for intermed in last.intermediate:
					x.add_blank_row()
					x.set_lastrow_iloc_nondupe(0, "Log Likelihood")
					x.set_lastrow_iloc(1, intermed.method)
					try:
						x.set_lastrow_iloc(2, str(-intermed.fun))
					except AttributeError:
						x.set_lastrow_iloc(2, str("err: no fun"))


			seconds = last_stat.dictionary()['total_duration_seconds']
			tformat = "{}\t{}".format(*format_seconds(seconds))
			x.add_blank_row()
			x.set_lastrow_iloc_nondupe(0, "Running Time")
			x.set_lastrow_iloc(1, "Total")
			x.set_lastrow_iloc(2, "{0}".format(tformat,**format))
			for label, dur in zip(last_stat.process_label,last_stat.dictionary()['process_durations']):
				x.add_blank_row()
				x.set_lastrow_iloc_nondupe(0, "Running Time")
				x.set_lastrow_iloc(1, label)
				x.set_lastrow_iloc(2, "{0}".format(dur,**format))
				
			i = last_stat.notes()
			if i is not '':
				if isinstance(i,str):
					i = i.split("\n")
				if isinstance(i,list):
					if len(i)>1:
						for inum, ii in enumerate(i):
							x.add_blank_row()
							x.set_lastrow_iloc_nondupe(0, "Notes")
							x.set_lastrow_iloc(1, "({})".format(inum))
							x.set_lastrow_iloc(2, str(ii))
					else:
						for ii in i:
							x.add_blank_row()
							x.set_lastrow_iloc_nondupe(0, "Notes")
							x.set_lastrow_iloc(2, str(ii))
				else:
					x.add_blank_row()
					x.set_lastrow_iloc_nondupe(0, "Notes")
					x.set_lastrow_iloc(2, str(i))

			import socket
			fqdn = socket.getfqdn()
			if fqdn:
				x.add_blank_row()
				x.set_lastrow_iloc(0, "Computer")
				x.set_lastrow_iloc(2, fqdn)

			i = last_stat.processor
#			try:
#				from ..util.sysinfo import get_processor_name
#				i2 = get_processor_name()
#				if isinstance(i2,bytes):
#					i2 = i2.decode('utf8')
#			except:
#				i2 = None
#			if i is not '':
#				x.add_blank_row()
#				x.set_lastrow_iloc(0, "Processor")
#				x.set_lastrow_iloc(2, str(i))
#				if i2 is not None:
#					x.add_blank_row()
#					x.set_lastrow_iloc_nondupe(0, "Processor")
#					x.set_lastrow_iloc(1, "Detail")
#					x.set_lastrow_iloc(2, str(i2))
			x.add_blank_row()
			x.set_lastrow_iloc(0, "Processor")
			x.set_lastrow_iloc(2, i)

			x.add_blank_row()
			x.set_lastrow_iloc(0, "Number of CPU Cores")
			x.set_lastrow_iloc(2, last_stat.number_cpu_cores)

			x.add_blank_row()
			x.set_lastrow_iloc(0, "Number of Threads Used")
			x.set_lastrow_iloc(2, last_stat.number_threads)

			# installed memory
#			try:
#				import psutil
#			except ImportError:
#				pass
#			else:
#				mem = psutil.virtual_memory().total
#				if mem >= 2.0*2**30:
#					mem_size = str(mem/2**30) + " GiB"
#				else:
#					mem_size = str(mem/2**20) + " MiB"
#				x.add_blank_row()
#				x.set_lastrow_iloc(0, "Installed Memory")
#				x.set_lastrow_iloc(2, "{0}".format(mem_size,**format))
			try:
				mem_size = last.installed_memory
			except AttributeError:
				pass
			else:
				x.add_blank_row()
				x.set_lastrow_iloc(0, "Installed Memory")
				x.set_lastrow_iloc(2, "{0}".format(mem_size,**format))

			# peak memory usage
			try:
				peak = last.peak_memory_usage
			except AttributeError:
				pass
			else:
				x.add_blank_row()
				x.set_lastrow_iloc(0, "Peak Memory Usage")
				x.set_lastrow_iloc(2, "{0}".format(peak,**format))
		return x






	# Model Estimation Statistics
	def art_ll(self,**format):
		"""
		Generate an ART containing the model estimation statistics.
		
		Returns
		-------
		AbstractReportTable
			An ART containing the estimation statistics.
		"""
		existing_format_keys = list(format.keys())
		for key in existing_format_keys:
			if key.upper()!=key: format[key.upper()] = format[key]
		if 'LL' not in format: format['LL'] = '0.2f'
		if 'RHOSQ' not in format: format['RHOSQ'] = '0.3f'

		try:
			total_weight = float(self.Data("Weight").sum())
		except:
			total_weight = None
		if total_weight is not None:
			if round(total_weight) == self.nCases():
				total_weight = None
	
		es = self._get_estimation_statistics()

		if total_weight is not None:
			cols = ["statistic", "aggregate", "per_case", "per_unit_weight"]
		else:
			cols = ["statistic", "aggregate", "per_case"]

		x = AbstractReportTable(columns=[cols])
		x.n_thead_rows = 1
		x.title = "Model Estimation Statistics"
		x.short_title = "Estimation Statistics"

		x.add_blank_row()
		x.set_lastrow_iloc(0, "Statistic")
		x.set_lastrow_iloc(1, "Aggregate")
		x.set_lastrow_iloc(2, "Per Case")

		x.add_blank_row()
		x.set_lastrow_iloc(0, "Number of Cases")
		x.set_lastrow_iloc(1, self.nCases(), {'class':'statistics_bridge centered_cell'})


		if total_weight is not None:
			x.add_blank_row()
			x.set_lastrow_iloc(0, "Total Weight")
			x.set_lastrow_iloc(1, total_weight, {'class':'statistics_bridge centered_cell'})

		ll = es[0]['log_like']
		if not math.isnan(ll):
			x.add_blank_row()
			x.set_lastrow_iloc(0, "Log Likelihood at Convergence")
			x.set_lastrow_iloc(1, "{0:{LL}}".format(ll,**format))
			x.set_lastrow_iloc(2, "{0:{LL}}".format(ll/numpy.int64(self.nCases()),**format))
			if total_weight is not None:
				x.set_lastrow_iloc(3,"{0:{LL}}".format(ll/total_weight,**format))

		llc = es[0]['log_like_constants']
		if not math.isnan(llc):
			x.add_blank_row()
			x.set_lastrow_iloc(0,"Log Likelihood at Constants")
			x.set_lastrow_iloc(1,"{0:{LL}}".format(llc,**format))
			x.set_lastrow_iloc(2,"{0:{LL}}".format(llc/numpy.int64(self.nCases()),**format))
			if total_weight is not None:
				x.set_lastrow_iloc(3,"{0:{LL}}".format(llc/total_weight,**format))

		llz = es[0]['log_like_null']
		if not math.isnan(llz):
			x.add_blank_row()
			x.set_lastrow_iloc(0,"Log Likelihood at Null Parameters")
			x.set_lastrow_iloc(1,"{0:{LL}}".format(llz,**format))
			x.set_lastrow_iloc(2,"{0:{LL}}".format(llz/numpy.int64(self.nCases()),**format))
			if total_weight is not None:
				x.set_lastrow_iloc(3,"{0:{LL}}".format(llz/total_weight,**format))

		ll0 = es[0]['log_like_nil']
		if not math.isnan(ll0):
			x.add_blank_row()
			x.set_lastrow_iloc(0,"Log Likelihood with No Model")
			x.set_lastrow_iloc(1,"{0:{LL}}".format(ll0,**format))
			x.set_lastrow_iloc(2,"{0:{LL}}".format(ll0/numpy.int64(self.nCases()),**format))
			if total_weight is not None:
				x.set_lastrow_iloc(3,"{0:{LL}}".format(ll0/total_weight,**format))

		if (not math.isnan(llz) or not math.isnan(llc) or not math.isnan(ll0)) and not math.isnan(ll):
			x.add_blank_row()
			if not math.isnan(llc):
				try:
					rsc = 1.0-(ll/llc)
				except ZeroDivisionError:
					x.set_lastrow_iloc(0,"Rho Squared w.r.t. Constants")
					x.set_lastrow_iloc(1,"ZeroDivisionError", {'class':'statistics_bridge centered_cell'})
				else:
					x.set_lastrow_iloc(0,"Rho Squared w.r.t. Constants")
					x.set_lastrow_iloc(1,"{0:{RHOSQ}}".format(rsc,**format), {'class':'statistics_bridge centered_cell'})
				if not math.isnan(llz) or not math.isnan(ll0): x.add_blank_row()

			if not math.isnan(llz):
				try:
					rsz = 1.0-(ll/llz)
				except ZeroDivisionError:
					x.set_lastrow_iloc(0,"Rho Squared w.r.t. Null Parameters")
					x.set_lastrow_iloc(1,"ZeroDivisionError", {'class':'statistics_bridge centered_cell'})
				else:
					x.set_lastrow_iloc(0,"Rho Squared w.r.t. Null Parameters")
					x.set_lastrow_iloc(1,"{0:{RHOSQ}}".format(rsz,**format), {'class':'statistics_bridge centered_cell'})
				if not math.isnan(ll0): x.add_blank_row()
			if not math.isnan(ll0):
				try:
					rs0 = 1.0-(ll/ll0)
				except ZeroDivisionError:
					x.set_lastrow_iloc(0,"Rho Squared w.r.t. No Model")
					x.set_lastrow_iloc(1,"ZeroDivisionError", {'class':'statistics_bridge centered_cell'})
				else:
					x.set_lastrow_iloc(0,"Rho Squared w.r.t. No Model")
					x.set_lastrow_iloc(1,"{0:{RHOSQ}}".format(rs0,**format), {'class':'statistics_bridge centered_cell'})
		return x


	def art_excludedcases(self, **format):
		try:
			return AbstractReportTable.FromDataFrame(self.df.exclusion_summary, title='Excluded Cases')
		except AttributeError:
			return SkipReportTable()

	def art_datasummary(self, **format):
		not_too_many_alts = (self.nAlts() < 20)
		a = AbstractReportTables(title="Various Data Statistics", short_title="Data Stats")
		a += self.art_stats_utility_co()
		a += self.art_stats_utility_ca()
		a += self.art_stats_quantity_ca()
		if not_too_many_alts:
			a += self.art_stats_utility_co_by_alt()
			a += self.art_stats_utility_ca_by_alt()
			a += self.art_stats_quantity_ca_by_alt()
		else:
			a += self.art_stats_utility_ca_by_all()
			a += self.art_stats_quantity_ca_by_all()
		return a


	from ..util.analytics import art_choice_distributions
