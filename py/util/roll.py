

from ..logging import logging as baselogging
from .. import logging
from .xhtml import XHTML, XML_Builder
import os.path
from .temporaryfile import TemporaryFile

def roll(m, filename=None, loglevel=baselogging.INFO, cats='-', **format):
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
	
	m.estimate()

	log.removeHandler(fh)
	if local_log:
		m.logger(0)

	#m.save(filename)
	fh.flush()

	with XHTML(use_filename, quickhead=m) as f:
		f << m.report(cats=cats, style='xml', **format)

		xlog = XML_Builder("div", {'class':'raw_log'})
		xlog.h2("Estimation Log", anchor=1)
		xlog.simple("hr")
		xlog.start("pre")
		templog.seek(0)
		xlog.data(templog.read())
		xlog.end("pre")
		xlog.simple("hr")
		f.append(xlog.close())

