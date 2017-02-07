

import numpy
import os
import zipfile
from ..util.xhtml import XHTML
from ..util.temporaryfile import TemporaryFile

class Exporter():

	def export_idco(self, file, varnames=None, screen="None", **formats):
		'''Export the :ref:`idco` data to a csv file.
		
		Only the :ref:`idco` table is exported, the :ref:`idca` table is ignored.  Future versions
		of Larch may provide a facility to export idco and idca data together in a 
		single idco output file.
		
		Parameters
		----------
		file : str or file-like
			If a string, this is the file name to give to the `open` command. Otherwise,
			this object is passed to :class:`csv.writer` directly.
		varnames : sequence of str, or None
			The variables to export.  If None, all regular variables are exported.
			
		Notes
		-----
		This method uses a :class:`pandas.DataFrame` object to write the output file, using
		:meth:`pandas.DataFrame.to_csv`. Any keyword
		arguments not listed here are passed through to the writer.
		'''
		if varnames is None:
			data = self.dataframe_idco(*self.variables_co(), screen=screen)
		else:
			data = self.dataframe_idco(*varnames, screen=screen)
		try:
			if os.path.splitext(file)[1] == '.gz':
				if 'compression' not in formats:
					formats['compression'] = 'gzip'
		except:
			pass
		data.to_csv(file, index_label='caseid', **formats)





	def export_zip_package(self, filename, idco_varnames=None, idca_varnames=None, screen="None"):
		'''Export the data to zip file.

		Only the :ref:`idco` table is exported whole, the :ref:`idca` data is
		exported into seperate columns.

		Parameters
		----------
		file : str
			This is the file name to give to the zipfile.  The .zip extension will be added if not given.
		idco_varnames, idca_varnames : sequence of str, or None
			The variables to export.  If None, all regular variables of the relevant type are exported.
		screen : str or array
			The screen to use.  Defaults to no screen.

		'''
		filename_base, filename_ext = os.path.splitext(filename)
		
		with zipfile.ZipFile(filename_base+'.zip', 'w', compression=zipfile.ZIP_DEFLATED) as myzip:

#			Coming for Python 3.6
#			with myzip.open(filename_base+'_idco.csv', 'w', force_zip64=True) as idcodata:
#				self.export_idco(idcodata, idco_varnames, screen)
#
#			with myzip.open(filename_base+'_info.html', 'w', force_zip64=True) as infohtml:
#				x = XHTML()
#				x << self.info(3)
#				infohtml.write(x.dump())

			import io
			idcodata = io.StringIO()
			self.export_idco(idcodata, idco_varnames, screen)
			myzip.writestr(os.path.basename(filename_base+'_idco.csv'), idcodata.getvalue())

			x = XHTML()
			x << self.info(3)
			myzip.writestr(os.path.basename(filename_base+'_info.html'), x.dump())
		
			if idca_varnames is None:
				idca_varnames = self.variables_ca()
	
			for idca_var in idca_varnames:
				tempfile = TemporaryFile(suffix='txt', mode='wb+')
				if screen in ("None","*"):
					numpy.savetxt(tempfile, self.idca[idca_var][:], fmt='%.8e', delimiter=',', newline='\n')
				else:
					numpy.savetxt(tempfile, self.array_idca(idca_var, screen=screen), fmt='%.8e', delimiter=',', newline='\n')
				tempfile.flush()
				myzip.write(tempfile.name, arcname=os.path.basename(filename_base+'_idca_'+idca_var+'.csv'))
				tempfile.close()
				del tempfile
		return
