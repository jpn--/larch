

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


from .core import Fountain





class DT(Fountain):

	def _clear_cached_values(self):
		del self._nCases
		del self._nAlts

	def _try_read_attrib(self, h5name, defaultvalue):
		attrib = "_"+h5name
		try:
			a = getattr(self._h5larchnode._v_attrs, h5name)
		except AttributeError:
			# not available in h5, use default value and try to write that to h5
			a = defaultvalue
			try:
				setattr(self._h5larchnode._v_attrs, h5name, a)
			except _tb.exceptions.FileModeError:
				pass
		setattr(self, attrib, a)

	def _try_write_attrib(self, h5name, value):
		setattr(self, "_"+h5name, value)
		try:
			setattr(self._h5larchnode._v_attrs, h5name, value)
		except _tb.exceptions.FileModeError:
			pass

	def __init__(self, filename, mode='a', ipath='/larch', complevel=None, complib='zlib', h5f=None):
		if not _tb_success: raise ImportError("pytables not available")
		super().__init__()
		if h5f is not None:
			self.h5f = h5f
			self._h5f_own = False
		else:
			if complevel is not None:
				self.h5f = _tb.open_file(filename, mode, filters=_tb.Filters(complib=complib, complevel=complevel))
			else:
				self.h5f = _tb.open_file(filename, mode)
			self._h5f_own = True
		self.source_filemode = mode
		self.source_filename = filename
		self._h5larchpath = ipath
		self._h5larchnode = self.h5f._getOrCreatePath(ipath, True)
		self.h5idca = self.h5f._getOrCreatePath(ipath+'/idca', True)
		self.h5idco = self.h5f._getOrCreatePath(ipath+'/idco', True)
#		self._try_read_attrib('larch_idca_tablename',   "larch_idca")
#		self._try_read_attrib('larch_idco_tablename',   "larch_idco")
#		self._try_read_attrib('larch_alts_tablename',   "larch_alts")
#		self._try_read_attrib('larch_idco_weightcolumn',"1")
#		self._try_read_attrib('larch_idca_choicecolumn',"none")
#		self._try_read_attrib('larch_idca_availcolumn', "1")

	def __del__(self):
		print("DEL",repr(self))
		if self._h5f_own:
			self.h5f.close()

	def __repr__(self):
		return "<larch.DT mode '{1}' at {0}>".format(self.source_filename, self.source_filemode)

#	@property
#	def idca_tablename(self):
#		return self._larch_idca_tablename
#
#	@idca_tablename.setter
#	def idca_tablename(self, x):
#		self._try_write_attrib('larch_idca_tablename', x)
#
#	@property
#	def idco_tablename(self):
#		return self._larch_idco_tablename
#
#	@idco_tablename.setter
#	def idco_tablename(self, x):
#		self._try_write_attrib('larch_idco_tablename', x)
#
#	@property
#	def alts_tablename(self):
#		return self._larch_alts_tablename
#
#	@alts_tablename.setter
#	def alts_tablename(self, x):
#		self._try_write_attrib('larch_alts_tablename', x)
#
#
#	@property
#	def idco_weightcolumn(self):
#		return self._larch_idco_weightcolumn
#
#	@idco_weightcolumn.setter
#	def idco_weightcolumn(self, x):
#		self._try_write_attrib('larch_idco_weightcolumn', x)
#
#
#	@property
#	def idca_choicecolumn(self):
#		return self._larch_idca_choicecolumn
#
#	@idca_choicecolumn.setter
#	def idca_choicecolumn(self, x):
#		self._try_write_attrib('larch_idca_choicecolumn', x)
#
#
#	@property
#	def idca_availcolumn(self):
#		return self._larch_idca_availcolumn
#
#	@idca_availcolumn.setter
#	def idca_availcolumn(self, x):
#		self._try_write_attrib('larch_idca_availcolumn', x)

	def _alternative_codes(self):
		h5node = self.h5f.get_node(self._h5larchpath+"/alts",'altids')
		return h5node[:]

	def _alternative_names(self):
		h5node = self.h5f.get_node(self._h5larchpath+"/alts",'names')
		return h5node[:]

	def alternative_codes(self):
		h5node = self.h5f.get_node(self._h5larchpath+"/alts",'altids')
		return tuple(int(i) for i in h5node[:])

	def alternative_names(self):
		h5node = self.h5f.get_node(self._h5larchpath+"/alts",'names')
		return tuple(str(i) for i in h5node[:])

	def alternative_name(self, code):
		codes = self._alternative_codes()
		idx = numpy.where(codes==code)[0][0]
		return self.h5f.get_node(self._h5larchpath+"/alts",'names')[idx]

	def alternative_code(self, name):
		names = self._alternative_names()
		idx = numpy.where(names==name)[0][0]
		return self.h5f.get_node(self._h5larchpath+"/alts",'altids')[idx]

	def nCases(self):
		try:
			return self._nCases
		except AttributeError:
			tbl = self.h5f.get_node(self._h5larchpath, 'caseids')
			self._nCases = int(tbl.shape[0])
			return self._nCases

	def nAlts(self):
		try:
			return self._nAlts
		except AttributeError:
			tbl = self.h5f.get_node(self._h5larchpath+'/alts', 'altids')
			self._nAlts = int(tbl.shape[0])
			return self._nAlts


	def _remake_command(self, cmd, screen, dims):
		## Whoa nelly!
		from tokenize import tokenize, untokenize, NAME, OP, STRING
		from io import BytesIO
		from .util import allowed_math
		recommand = []
		g = tokenize(BytesIO(cmd.encode('utf-8')).readline)
		for toknum, tokval, _, _, _  in g:
			if toknum == NAME and tokval not in allowed_math:
				# replace NAME tokens
				partial = [ (NAME, 'h5node'), (OP, '.'), (NAME, tokval), (OP, '['), ]
				if screen is None:
					partial = partial+[(OP, ':'),]
				else:
					partial = partial+[(NAME, 'screen'),]
				if dims>1:
					partial = partial+[(OP, ','),(OP, ':'),]
				if dims>2:
					partial = partial+[(OP, ','),(OP, ':'),]
				if dims>3:
					partial = partial+[(OP, ','),(OP, ':'),]
				recommand.extend(partial+[(OP, ']'), ])
			else:
				recommand.append((toknum, tokval))
		return untokenize(recommand).decode('utf-8')



	def array_idca(self, *vars, dtype=numpy.float64, screen=None):
		"""Extract a set of idca values from the DB based on preset queries.
		
		Generally you won't need to specify any parameters to this method beyond the
		variables to include in the array, as
		most values are determined automatically from the preset queries.
		However, if you need to override things for this array without changing
		the queries more permanently, you can use the input parameters to do so.
		Note that all override parameters must be called by keyword, not as positional
		arguments.
		
		Parameters
		----------
		vars : tuple of str
			A tuple giving the expressions (often column names, but any valid
			SQLite expression works) to extract as :ref:`idca` format variables.
		
		Other Parameters
		----------------
		table : str
			The idca data will be found in this table.
		ipath : str
			If given, override the usual internal node path with ipath when looking up the table.
		caseid : str
			This sets the column name where the caseids can be found.
		altid : str
			This sets the column name where the altids can be found.
		altcodes : tuple of int
			This is the set of alternative codes used in the data. The second (middle) dimension 
			of the result array will match these codes in length and order.
		dtype : str or dtype
			Describe the data type you would like the output array to adopt, probably 
			'int64', 'float64', or 'bool'.
		
		Returns
		-------
		data : ndarray
			An array with specified dtype, of shape (n_cases,len(altcodes),len(vars)).
		caseids : ndarray
			An int64 array of shape (n_cases,1).
			
		"""
		h5node = self._h5larchnode.idca
		h5caseid = self._h5larchnode.caseids
		if screen is None:
			n_cases = h5caseid.shape[0]
		else:
			n_cases = screen.shape[0]
		n_vars = len(vars)
		n_alts = self.nAlts()
		result = numpy.zeros([n_cases,n_alts,n_vars], dtype=dtype)
		for varnum,v in enumerate(vars):
			try:
				if dtype in (bool, numpy.bool, numpy.bool_):
					if isinstance(v,str) and v.lower() in ('true','false'):
						float_v = dtype(v)
					elif isinstance(v,(bool, numpy.bool, numpy.bool_)):
						float_v = dtype(v)
					else:
						raise ValueError
				else:
					float_v = dtype(v)
			except ValueError:
				## Whoa nelly!
				from numpy import log, exp, log1p
				command = self._remake_command(v,screen,2)
				try:
					result[:,:,varnum] = eval( command )
				except:
					print("command=",command)
					raise
			except:
				raise
			else:
				result[:,:,varnum] = float_v
		result = numpy.nan_to_num(result)
		from .array import Array
		result = result.view(Array)
		result.vars = vars
		return result



	def array_idco(self, *vars, dtype=numpy.float64, screen=None):
		"""Extract a set of idco values from the DB based on preset queries.
		
		Generally you won't need to specify any parameters to this method beyond the
		variables to include in the array, as
		most values are determined automatically from the preset queries.
		However, if you need to override things for this array without changing
		the queries more permanently, you can use the input parameters to do so.
		Note that all override parameters must be called by keyword, not as positional
		arguments.
		
		Parameters
		----------
		vars : tuple of str
			A tuple (or other iterable) giving the expressions (often column names, but any valid
			SQLite expression works) to extract as :ref:`idco` format variables.
		
		Other Parameters
		----------------
		dtype : str or dtype
			Describe the data type you would like the output array to adopt, probably 
			numpy.int64, numpy.float64, or numpy.bool_.
		
		Returns
		-------
		data : ndarray
			An array with specified dtype, of shape (n_cases,len(vars)).
		"""
		h5node = self._h5larchnode.idco
		h5caseid = self._h5larchnode.caseids
		if screen is None:
			n_cases = h5caseid.shape[0]
		else:
			n_cases = screen.shape[0]
		n_vars = len(vars)
		result = numpy.zeros([n_cases,n_vars], dtype=dtype)
		for varnum,v in enumerate(vars):
			try:
				float_v = dtype(v)
			except ValueError:
				from numpy import log, exp, log1p
				command = self._remake_command(v,screen,1)
				try:
					result[:,varnum] = eval( command )
				except:
					print("command=",command)
					raise
			except:
				raise
			else:
				result[:,varnum] = float_v
		result = numpy.nan_to_num(result)
		from .array import Array
		result = result.view(Array)
		result.vars = vars
		return result

	def array_weight(self, *, var=None, **kwargs):
		try:
			self.h5f.get_node(self._h5larchpath+"/idco", '_weight_')
		except _tb.exceptions.NoSuchNodeError:
			return self.array_idco('1', **kwargs)
		else:
			return self.array_idco('_weight_', **kwargs)

	def array_choice(self, *, var=None, **kwargs):
		return self.array_idca('_choice_', **kwargs)

	def array_avail(self, *, var=None, dtype=numpy.bool_, **kwargs):
		try:
			self.h5f.get_node(self._h5larchpath+"/idca", '_avail_')
		except _tb.exceptions.NoSuchNodeError:
			return self.array_idca('1', dtype=dtype, **kwargs)
		else:
			return self.array_idca('_avail_', dtype=dtype, **kwargs)


	def provision(self, needs):
		from . import Model
		if isinstance(needs,Model):
			m = needs
			needs = m.needs()
		else:
			m = None
		import numpy
		provide = {}
		cases = None
		screen = None
		if '_screen_' in self._h5larchnode:
			screen = numpy.nonzero(self._h5larchnode._screen_[:])[0]
		matched_cases = []
		#log = self.logger()
		log = None
		for key, req in needs.items():
			if log:
				log.info("Provisioning {} data...".format(key))
			if key=="Avail":
				provide[key] = self.array_avail(screen=screen)
			elif key=="Weight":
				provide[key] = self.array_weight(screen=screen)
			elif key=="Choice":
				provide[key] = self.array_choice(screen=screen)
			elif key[-2:]=="CA":
				provide[key] = self.array_idca(*req.get_variables(), screen=screen)
			elif key[-2:]=="CO":
				provide[key] = self.array_idco(*req.get_variables(), screen=screen)
			elif key=="Allocation":
				provide[key] = self.array_idco(*req.get_variables(), screen=screen)
		if screen is None:
			provide['caseids'] = self.h5f.get_node(self._h5larchpath+"/caseids")[:]
		else:
			provide['caseids'] = self.h5f.get_node(self._h5larchpath+"/caseids")[screen]
		if len(provide['caseids'].shape) == 1:
			provide['caseids'] = provide['caseids'][:,numpy.newaxis]
		if m is not None:
			return m.provision(provide)
		else:
			return provide


	def _check_ca_natural(self, column):
		return column in self.h5idca._v_leaves

	def _check_co_natural(self, column):
		return column in self.h5idco._v_leaves

	def check_ca(self, column):
		return True

	def check_co(self, column):
		return True

	def variables_ca(self):
		return tuple(i for i in self.h5f.get_node(self._h5larchpath+'/idca')._v_leaves)

	def variables_co(self):
		return tuple(i for i in self.h5f.get_node(self._h5larchpath+'/idco')._v_leaves)

	@staticmethod
	def Example(dataset='MTC', filename='{}.h5', temp=True):
		'''Generate an example data object in memory.
		
		Larch comes with a few example data sets, which are used in documentation
		and testing. This function copies the data into a HDF5 file, which you can
		freely edit without damaging the original data.
		
		Parameters
		----------
		dataset : {'MTC', 'SWISSMETRO', 'MINI', 'ITINERARY'}
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
		from .db import DB
		edb = DB.Example(dataset)

		h5filters = _tb.Filters(complevel=5)
		
		try:
			filename_ = filename.format(dataset)
		except:
			pass
		else:
			filename = filename_

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
		from .util.filemanager import next_stack
		n=0
		while 1:
			try:
				tryname = next_stack(filename, plus=n, allow_natural=True)
				h5f = _tb.open_file(tryname, 'w', filters=h5filters, driver="H5FD_CORE", driver_core_backing_store=0 if temp else 1)
			except ValueError:
				n += 1
			else:
				break
		larchnode = h5f._getOrCreatePath("/larch", True)
		larchidca = h5f._getOrCreatePath("/larch/idca", True)
		larchidco = h5f._getOrCreatePath("/larch/idco", True)
		larchalts = h5f._getOrCreatePath("/larch/alts", True)

		for var_ca in vars_ca:
			h5var = h5f.create_carray(larchidca, var_ca, _tb.Float64Atom(), shape=(edb.nCases(), edb.nAlts()), filters=h5filters)
			arr, caseids = edb.array_idca(var_ca)
			h5var[:,:] = arr.squeeze()

		for var_co in vars_co:
			h5var = h5f.create_carray(larchidco, var_co, _tb.Float64Atom(), shape=(edb.nCases(),), filters=h5filters)
			arr, caseids = edb.array_idco(var_co)
			h5var[:] = arr.squeeze()

		h5avail = h5f.create_carray(larchidca, '_avail_', _tb.BoolAtom(), shape=(edb.nCases(), edb.nAlts()), filters=h5filters)
		arr, caseids = edb.array_avail()
		h5avail[:,:] = arr.squeeze()

		h5caseids = h5f.create_carray(larchnode, 'caseids', _tb.Int64Atom(), shape=(edb.nCases(),), filters=h5filters)
		h5caseids[:] = caseids.squeeze()

		h5scrn = h5f.create_carray(larchnode, '_screen_', _tb.BoolAtom(), shape=(edb.nCases(),), filters=h5filters)
		h5scrn[:] = True

		h5altids = h5f.create_carray(larchalts, 'altids', _tb.Int64Atom(), shape=(edb.nAlts(),), filters=h5filters, title='elemental alternative code numbers')
		h5altids[:] = edb.alternative_codes()

		h5altnames = h5f.create_vlarray(larchalts, 'names', _tb.VLUnicodeAtom(), filters=h5filters, title='elemental alternative names')
		for an in edb.alternative_names():
			h5altnames.append( an )
		
		try:
			ch_ca = edb.queries.get_choice_ca()
			h5f.create_soft_link(larchidca, '_choice_', target='/larch/idca/'+ch_ca)
		except AttributeError:
			h5ch = h5f.create_carray(larchidca, '_choice_', _tb.Float64Atom(), shape=(edb.nCases(), edb.nAlts()), filters=h5filters)
			arr, caseids = edb.array_choice()
			h5ch[:,:] = arr.squeeze()

		wgt = edb.queries.weight
		if wgt:
			h5f.create_soft_link(larchidca, '_weight_', target='/larch/idco/'+wgt)

		return DT(filename, 'w', h5f=h5f)

	def validate_hdf5(self, log=print):
		nerrs = 0
		def zzz(message, valid):
			global nerrs
			vld = "ok" if valid else "ERROR"
			log("{!s:^5}| {}".format(vld, "\n     |  ".join(message.split("\n"))))
			if not valid:
				nerrs += 1
		title = "{0} (with mode '{1}')".format(self.source_filename, self.source_filemode)
		log("========================"+"="*len(title))
		log("larch.DT Validation for {}".format( title ))
		log("------------------------"+"-"*len(title))
		zzz("There should be a designated `larch` group node under which all other nodes reside.",
			isinstance(self._h5larchnode, _tb.group.Group))
		zzz("Under the top node, there should be a group named `idco` to hold that data.",
			isinstance(self._h5larchnode.idco, _tb.group.Group))
		zzz("Under the top node, there should be a group named `idca` to hold that data.",
			isinstance(self._h5larchnode.idca, _tb.group.Group))
		zzz("Under the top node, there should be an array node named `caseids` containing an\n"
			"Int64 1 dimensional array.",
			isinstance(self._h5larchnode.caseids, _tb.array.Array) and isinstance(self._h5larchnode.caseids.atom, _tb.atom.Int64Atom))
		log("------------------------"+"-"*len(title))
		return nerrs
