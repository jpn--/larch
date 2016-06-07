

from ..logging import logging as baselogging
from .. import logging
from .xhtml import XHTML, XML_Builder
import os.path
from .temporaryfile import TemporaryFile
import inspect

def roll(self, filename=None, loglevel=baselogging.INFO, cats='-', use_ce=False, sourcecode=True, **format):
	"""Estimate a model and generate a report.
	
	This method rolls together model estimation, reporting, and saving results
	into a single handy function.
	
	Parameters
	----------
	filename : str, optional
		The filename into which the output report will be saved.  If not given,
		a temporary file will be created.  If the given file already exists, a
		new file will be created with a number appended to the base filename.
	loglevel : int, optional
		The log level that will be used while estimating the model.  Smaller numbers
		result in a more verbose log, the contents of which appear at the end of
		the HTML report. See the standard Python :mod:`logging` module for more details.
	cats : list of str, or '-' or '+'
		A list of report sections to include in the report. The default is '-', which
		includes a minimal list of report setions. Giving '+' will dump every available
		report section, which could be a lot and might take a lot of time (and 
		computer memory) to compute.
	sourcecode : bool
		If true (the default), this method will attempt to access the source code
		of the file where this function was called, and insert the contents of
		that file into a section of the resulting report.  This is done because the
		source code may be more instructive as to how the model was created,
		and how different (but related) future models might be created.
	
	"""
	m = self
	local_log = False
	log = m.logger()
	if log is None:
		local_log = True
		log = m.logger(1)

	if filename is None:
		use_filename = 'temp'
	else:
		use_filename = os.path.splitext(filename)[0] + '.html'

	templog = TemporaryFile('log')

	fh = baselogging.StreamHandler(templog)
	fh.setLevel(loglevel)
	fh.setFormatter(logging.default_formatter())
	log.addHandler(fh)

	if log.getEffectiveLevel() > loglevel:
		log.setLevel(loglevel)
	
	if use_ce:
		m.setup_utility_ce()

	m.maximize_loglike()

	log.removeHandler(fh)
	if local_log:
		m.logger(0)

	#m.save(filename)
	fh.flush()

	# SourceFile grabber
	try:
		frame = inspect.stack()[1]
		sourcefile = inspect.getsourcefile(frame[0])
	except:
		sourcefile = None
	
	with XHTML(use_filename, quickhead=m) as f:
		f << m.report(cats=cats, style='xml', **format)

		if sourcefile is not None and os.path.isfile(sourcefile):
			xsource = XML_Builder("div", {'class':'raw_source'})
			xsource.h2("Source Code", anchor=1)
			xsource.data("From: {!s}".format(sourcefile))
			xsource.simple("hr")
			xsource.start("pre")
			with open(sourcefile, 'r') as sf:
				xsource.data(sf.read())
			xsource.end("pre")
			xsource.simple("hr")
			f.append(xsource.close())

		xlog = XML_Builder("div", {'class':'raw_log'})
		xlog.h2("Estimation Log", anchor=1)
		xlog.simple("hr")
		xlog.start("pre")
		templog.seek(0)
		xlog.data(templog.read())
		xlog.end("pre")
		xlog.simple("hr")
		f.append(xlog.close())

