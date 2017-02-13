





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
from ..util.naming import make_valid_identifier


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




	def import_idco(self, filepath_or_buffer, caseid_column=None, overwrite=0, *args, **kwargs):
		"""Import an existing CSV or similar file in idco format into this HDF5 file.
		
		This function relies on :func:`pandas.read_csv` to read and parse the input data.
		All arguments other than those described below are passed through to that function.
		
		Parameters
		----------
		filepath_or_buffer : str or buffer or :class:`pandas.DataFrame`
			This argument will be fed directly to the :func:`pandas.read_csv` function.
			If a string is given and the file extension is ".xlsx" then the :func:`pandas.read_excel`
			function will be used instead, ot if the file extension is ".dbf" then 
			:func:`simpledbf.Dbf5.to_dataframe` is used.  Alternatively, you can just pass a pre-loaded
			:class:`pandas.DataFrame`.
		caseid_column : None or str
			If given, this is the column of the input data file to use as caseids.  If not given,
			arbitrary sequential caseids will be created.  If it is given and
			the caseids do already exist, a `LarchError` is raised.
		overwrite : int
			If positive, existing data with same name will be overwritten.  If zero (the default)
			existing data with same name will be not overwritten and tables.exceptions.NodeError
			will be raised.  If negative, existing data will not be overwritten but no errors will be raised.
		
		Returns
		-------
		DT
			self
		
		Raises
		------
		LarchError
			If caseids exist and are also given,
			or if the caseids are not integer values.
		"""
		import pandas
		from .. import logging
		log = logging.getLogger('DT')
		log("READING %s",str(filepath_or_buffer))
		original_source = None
		if isinstance(filepath_or_buffer, str) and filepath_or_buffer.casefold()[-5:]=='.xlsx':
			df = pandas.read_excel(filepath_or_buffer, *args, **kwargs)
			original_source = filepath_or_buffer
		elif isinstance(filepath_or_buffer, str) and filepath_or_buffer.casefold()[-5:]=='.dbf':
			from simpledbf import Dbf5
			dbf = Dbf5(filepath_or_buffer, codec='utf-8')
			df = dbf.to_dataframe()
			original_source = filepath_or_buffer
		elif isinstance(filepath_or_buffer, pandas.DataFrame):
			df = filepath_or_buffer
		else:
			df = pandas.read_csv(filepath_or_buffer, *args, **kwargs)
			original_source = filepath_or_buffer
		log("READING COMPLETE")
		try:
			for col in df.columns:
				log("LOADING %s",col)
				col_array = df[col].values
				try:
					tb_atom = _tb.Atom.from_dtype(col_array.dtype)
				except ValueError:
					log.warn("  column %s is not an simple compatible datatype",col)
					try:
						maxlen = int(df[col].str.len().max())
					except ValueError:
						import datetime
						if isinstance(df[col][0],datetime.time):
							log.warn("  column %s is datetime.time, converting to Time32",col)
							tb_atom = _tb.atom.Time32Atom()
							#convert_datetime_time_to_epoch_seconds = lambda tm: tm.hour*3600+ tm.minute*60 + tm.second
							def convert_datetime_time_to_epoch_seconds(tm):
								try:
									return tm.hour*3600+ tm.minute*60 + tm.second
								except:
									if numpy.isnan(tm):
										return 0
									else:
										raise
							col_array = df[col].apply(convert_datetime_time_to_epoch_seconds).astype(numpy.int32).values
						else:
							import __main__
							__main__.err_df = df
							raise
					else:
						maxlen = max(maxlen,8)
						log.warn("  column %s, converting to S%d",col,maxlen)
						try:
							col_array = df[col].astype('S{}'.format(maxlen)).values
						except SystemError:
							log.warn("  SystemError in converting column %s to S%d, data is being discarded",col,maxlen)
							tb_atom = None
						except UnicodeEncodeError:
							log.warn("  UnicodeEncodeError in converting column %s to S%d, data is being discarded",col,maxlen)
							tb_atom = None
						else:
							tb_atom = _tb.Atom.from_dtype(col_array.dtype)
				if tb_atom is not None:
					col = make_valid_identifier(col)
					if overwrite and col in self.idco._v_node._v_children:
						# delete the old data
						self.h5f.remove_node(self.idco._v_node, col, recursive=True)
					h5var = self.h5f.create_carray(self.idco._v_node, col, tb_atom, shape=col_array.shape)
					h5var[:] = col_array
					if original_source:
						h5var._v_attrs.ORIGINAL_SOURCE = original_source
			if caseid_column is not None and 'caseids' in self.h5top:
				raise LarchError("caseids already exist, not setting new ones")
			if caseid_column is not None and 'caseids' not in self.h5top:
				if caseid_column not in df.columns:
					for col in df.columns:
						if col.casefold() == caseid_column.casefold():
							caseid_column = col
							break
				if caseid_column not in df.columns:
					raise LarchError("caseid_column '{}' not found in data".format(caseid_column))
				proposed_caseids_node = self.idco._v_children[caseid_column]
				if not isinstance(proposed_caseids_node.atom, _tb.atom.Int64Atom):
					col_array = df[caseid_column].values.astype('int64')
					if not numpy.all(col_array==df[caseid_column].values):
						raise LarchError("caseid_column '{}' does not contain only integer values".format(caseid_column))
					h5var = self.h5f.create_carray(self.idco._v_node, caseid_column+'_int64', _tb.Atom.from_dtype(col_array.dtype), shape=col_array.shape)
					h5var[:] = col_array
					caseid_column = caseid_column+'_int64'
					proposed_caseids_node = self.idco._v_children[caseid_column]
				self.h5f.create_soft_link(self.h5top, 'caseids', target=self.idco._v_pathname+'/'+caseid_column)
			if caseid_column is None and 'caseids' not in self.h5top:
				h5var = self.h5f.create_carray(self.h5top, 'caseids', obj=numpy.arange(1, len(df)+1, dtype=numpy.int64))
		except:
			self._df_exception = df
			raise
		return self

