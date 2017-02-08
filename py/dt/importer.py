





import os
import zipfile

try:
	import tables as _tb
	import numpy
except ImportError:
	from .mock_module import Mock
	_tb = Mock()
	numpy = Mock()
	_tb_success = False
else:
	_tb_success = True

from ..core import IntStringDict


class Importer():





	@classmethod
	def FromDB(cls, db, filename=None, temp=True):
		'''Generate a DT data file from a DB file.
		
		Larch comes with a few example data sets, which are used in documentation
		and testing. This function copies the data into a HDF5 file, which you can
		freely edit without damaging the original data.
		
		Parameters
		----------
		db : DB
			Which example dataset should be used.
		filename : str
			A filename to open the HDF5 file (even in-memory files need a name).
		temp : bool
			The example database be created in-memory; if `temp` is false,
			the file will be dumped to disk when closed.
			
		Returns
		-------
		DT
			An open connection to the HDF5 example data.
		
		'''

		h5filters = _tb.Filters(complevel=1)

		if filename is None:
			filename = '{}.h5'.format(os.path.splitext(os.path.basename(db.source_filename))[0])

		from ..util.filemanager import next_stack
		n=0
		while 1:
			try:
				tryname = next_stack(filename, plus=n, allow_natural=(n==0))
				h5f = _tb.open_file(tryname, 'w', filters=h5filters, driver="H5FD_CORE", driver_core_backing_store=0 if temp else 1)
			except ValueError:
				n += 1
				if n>1000:
					raise RuntimeError("cannot open HDF5 at {}".format(filename))
			else:
				break


		from ..db import DB
		if not isinstance(db, DB):
			raise TypeError('db must be DB')
		
		edb = db
		self = cls(filename, 'w', h5f=h5f)

		descrip_larch = {}
		descrip_alts = {
			'altid': _tb.Int64Col(pos=1, dflt=-999),
			'name': _tb.StringCol(itemsize=127, pos=2, dflt=""),
		}
		descrip_co = {}
		descrip_ca = {}
		vars_co = edb.variables_co()
		vars_ca = edb.variables_ca()
		for i in vars_co:
			if i == 'caseid':
				descrip_co[i] = _tb.Int64Col(pos=len(descrip_co), dflt=-999)
			else:
				descrip_co[i] = _tb.Float64Col(pos=len(descrip_co), dflt=numpy.nan)
		for i in vars_ca:
			if i in ('caseid','altid'):
				descrip_ca[i] = _tb.Int64Col(pos=len(descrip_ca), dflt=-999)
			else:
				descrip_ca[i] = _tb.Float64Col(pos=len(descrip_ca), dflt=numpy.nan)

		larchnode = h5f._get_or_create_path("/larch", True)
		larchidca = h5f._get_or_create_path("/larch/idca", True)
		larchidco = h5f._get_or_create_path("/larch/idco", True)
		larchalts = h5f._get_or_create_path("/larch/alts", True)

		for var_ca in vars_ca:
			if var_ca not in ('caseid', 'casenum', 'IDCASE' ):
				h5var = h5f.create_carray(larchidca, var_ca, _tb.Float64Atom(), shape=(edb.nCases(), edb.nAlts()), filters=h5filters)
				arr, caseids = edb.array_idca(var_ca)
				h5var[:,:] = arr.squeeze()

		for var_co in vars_co:
			if var_co not in ('caseid', 'casenum', 'IDCASE'):
				h5var = h5f.create_carray(larchidco, var_co, _tb.Float64Atom(), shape=(edb.nCases(),), filters=h5filters)
				arr, caseids = edb.array_idco(var_co)
				h5var[:] = arr.squeeze()

		h5caseids = h5f.create_carray(larchnode, 'caseids', _tb.Int64Atom(), shape=(edb.nCases(),), filters=h5filters)
		h5caseids[:] = caseids.squeeze()

		h5scrn = h5f.create_carray(larchnode, 'screen', _tb.BoolAtom(), shape=(edb.nCases(),), filters=h5filters)
		h5scrn[:] = True

		h5altids = h5f.create_carray(larchalts, 'altids', _tb.Int64Atom(), shape=(edb.nAlts(),), filters=h5filters, title='elemental alternative code numbers')
		h5altids[:] = edb.alternative_codes()

		h5altnames = h5f.create_vlarray(larchalts, 'names', _tb.VLUnicodeAtom(), filters=h5filters, title='elemental alternative names')
		for an in edb.alternative_names():
			h5altnames.append( str(an) )
		
		if isinstance(edb.queries.avail, (dict, IntStringDict)):
			self.avail_idco = dict(edb.queries.avail)
		else:
			h5avail = h5f.create_carray(larchidca, '_avail_', _tb.BoolAtom(), shape=(edb.nCases(), edb.nAlts()), filters=h5filters)
			arr, caseids = edb.array_avail()
			h5avail[:,:] = arr.squeeze()

		try:
			ch_ca = edb.queries.get_choice_ca()
			h5f.create_soft_link(larchidca, '_choice_', target='/larch/idca/'+ch_ca)
		except AttributeError:
			h5ch = h5f.create_carray(larchidca, '_choice_', _tb.Float64Atom(), shape=(edb.nCases(), edb.nAlts()), filters=h5filters)
			arr, caseids = edb.array_choice()
			h5ch[:,:] = arr.squeeze()

		wgt = edb.queries.weight
		if wgt:
			h5f.create_soft_link(larchidco, '_weight_', target='/larch/idco/'+wgt)

		return self
