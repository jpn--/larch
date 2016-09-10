

from ..logging import logging as baselogging
from .. import logging
from .xhtml import XHTML, XML_Builder, xhtml_rawtext_as_div, _default_css
import os.path
from .temporaryfile import TemporaryFile
import inspect

def roll(self, filename=None, loglevel=baselogging.INFO, cats='-', use_ce=False, sourcecode=True, maxlik_args=(), **format):
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
	cats : list of str, or '-' or '*'
		A list of report sections to include in the report. The default is '-', which
		includes a minimal list of report setions. Giving '*' will dump every available
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
	
	use_jupyter = False

	if filename is None:
		use_filename = 'temp'
	elif filename == "None":
		use_filename = None
	elif filename == "jupyter":
		use_filename = None
		use_jupyter = True
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

	m.maximize_loglike(*maxlik_args)

	log.removeHandler(fh)
	if local_log:
		m.logger(0)

	#m.save(filename)
	fh.flush()

	# SourceFile grabber
	if sourcecode:
		try:
			frame = inspect.stack()[1]
			sourcefile = inspect.getsourcefile(frame[0])
		except:
			sourcefile = None

	css = None
	if 'css' in format:
		css = format['css']
	elif use_jupyter:
		css = _default_css + """
		.strut2 {min-width:1in}
		"""

	with XHTML(use_filename, quickhead=m, css=css) as f:
		f << m.report(cats=cats, style='xml', **format)
		if sourcecode:
			f << xhtml_rawtext_as_div(filename=sourcefile, classtype='raw_source', title="Source Code")
		f << xhtml_rawtext_as_div(filehandle=templog, classtype='raw_log', title="Estimation Log")



		if use_jupyter:
			try:
				from IPython.display import display, HTML
			except ImportError:
				pass
			else:
				display(HTML(f.dump(toc=False,sign=True).decode()))
		elif use_filename is None:
			return f.dump()

	return self
