from . import docx, latex, pdf, txt, xhtml, alogit, art, unicode
import math
from ..util.xhtml import XHTML, XML_Builder, Elem
from ..util.pmath import category, pmath, rename
from ..core import LarchError
import os
import pandas
import numpy
import itertools


class ModelReporter(docx.DocxModelReporter,
					latex.LatexModelReporter,
					xhtml.XhtmlModelReporter,
					pdf.PdfModelReporter,
					txt.TxtModelReporter,
					alogit.AlogitModelReporter,
					unicode.UnicodeModelReporter,
					art.ArtModelReporter,
					):

	def report(self, style, *args, filename=None, tempfile=False, **kwargs):
		"""
		Generate a model report.
		
		Larch is capable of generating reports in five basic formats:
		text, xhtml, docx, and latex.  This function serves as a
		pass-through, to call the report generator of the given style,
		and optionally to save the results to a file.
		
		Parameters
		----------
		style : ['txt', 'docx', 'latex', 'html', 'xml']
			The style of output.  Both 'html' and 'xml' will call the 
			xhtml generator, with the only difference being the output
			type, with html generating a finished [x]html document and
			xml generating an etree for further processing by Python 
			directly.
			
		Other Parameters
		----------------
		filename : str or None
			If given, then a new stack file is created by :func:`util.filemanager.open_stack`,
			the output is generated in this file, and the file-like object is 
			returned.
		tempfile : bool
			If True then a new :class:`util.TemporaryFile` is created,
			the output is generated in this file, and the file-like object is 
			returned.
		throw_exceptions : bool
			If True, exceptions are thrown if raised while generating the report. If 
			False (the default) tracebacks are printed directly into the report for 
			each section where an exception is raised.  Setting this to True can be
			useful for testing.


		Raises
		------
		LarchError
			If both filename and tempfile evaluate as True.
		
		"""
		if filename and tempfile:
			raise LarchError("only one of filename and tempfile can be given.")
		
		from ..util.filemanager import fileopen, filenext
		
		if style.lower()=='txt':
			rpt = self.txt_report(*args, **kwargs)
			if filename is None and tempfile==False:
				return rpt
			else:
				f = fileopen(None if tempfile else filename, mode='w')
				f.write(rpt)
				f.seek(0)
				return f
		if style.lower()=='xml':
			rpt = self.xhtml_report(*args, raw_xml=True, **kwargs)
			if filename is None and tempfile==False:
				return rpt
			else:
				f = XHTML("temp" if tempfile else filenext(filename), quickhead=self, **kwargs)
				f << rpt
				f.dump()
				return f
		if style.lower()=='html':
			rpt = self.xhtml_report(*args, raw_xml=False, **kwargs)
			if filename is None and tempfile==False:
				return rpt
			else:
				use_filename = None if tempfile else filename
				if use_filename is not None:
					base, ext = os.path.splitext(use_filename)
					if ext.casefold() not in ('.html','.xhtml','.htm'):
						use_filename = use_filename + '.html'
				f = fileopen(use_filename, mode='wb')
				f.write(rpt)
				f.flush()
				f.seek(0)
				try:
					f.view()
				except:
					pass
				return f



		if style.lower()=='docx':
			raise NotImplementedError
		if style.lower()=='latex':
			raise NotImplementedError

		# otherwise, the format style is not known
		raise LarchError("Format style '{}' is not known, must be one of ['txt', 'docx', 'latex', 'html', 'xml']".format(style))

	
	def report_d(self):
		"""Shortcut to a diagnostic html report."""
		y = self.report(style='html', tempfile=True, cats='D')
		y.view()
		return y

	def report_html(self, *args, cats='*', **kwargs):
		self.report(style='html', cats=cats, **kwargs)


