

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


from .core import Fountain, LarchError

class IncompatibleShape(LarchError):
	pass



class DT(Fountain):

	def clear_cached_values(self):
		try:
			del self._nCases
		except AttributeError:
			pass
		try:
			del self._nAlts
		except AttributeError:
			pass

	def _try_read_attrib(self, h5name, defaultvalue):
		attrib = "_"+h5name
		try:
			a = getattr(self.h5top._v_attrs, h5name)
		except AttributeError:
			# not available in h5, use default value and try to write that to h5
			a = defaultvalue
			try:
				setattr(self.h5top._v_attrs, h5name, a)
			except _tb.exceptions.FileModeError:
				pass
		setattr(self, attrib, a)

	def _try_write_attrib(self, h5name, value):
		setattr(self, "_"+h5name, value)
		try:
			setattr(self.h5top._v_attrs, h5name, value)
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
		self.h5top = self.h5f._getOrCreatePath(ipath, True)
		self.h5idca = self.h5f._getOrCreatePath(ipath+'/idca', True)
		self.h5idco = self.h5f._getOrCreatePath(ipath+'/idco', True)
		self.h5alts = self.h5f._getOrCreatePath(ipath+'/alts', True)

	def __del__(self):
		if self._h5f_own:
			self.h5f.close()

	def __repr__(self):
		return "<larch.DT mode '{1}' at {0}>".format(self.source_filename, self.source_filemode)


	def _alternative_codes(self):
		return self.h5alts.altids[:]

	def _alternative_names(self):
		return self.h5alts.names[:]

	def alternative_codes(self):
		return tuple(int(i) for i in self.h5alts.altids[:])

	def alternative_names(self):
		return tuple(str(i) for i in self.h5alts.names[:])

	def alternative_name(self, code):
		codes = self._alternative_codes()
		idx = numpy.where(codes==code)[0][0]
		return self.h5alts.names[idx]

	def alternative_code(self, name):
		names = self._alternative_names()
		idx = numpy.where(names==name)[0][0]
		return self.h5alts.altids[idx]

	def nCases(self):
		if 'screen' in self.h5top:
			screen = numpy.nonzero(self.h5top.screen[:])[0]
			return int(screen.shape[0])
		else:
			return int(self.h5top.caseids.shape[0])

	def nAlts(self):
		return int(self.h5alts.altids.shape[0])


	def _remake_command(self, cmd, screen, dims):
		## Whoa nelly!
		from tokenize import tokenize, untokenize, NAME, OP, STRING
		from io import BytesIO
		recommand = []
		g = tokenize(BytesIO(cmd.encode('utf-8')).readline)
		for toknum, tokval, _, _, _  in g:
			if toknum == NAME and tokval in self.h5idca:
				if dims==1:
					raise IncompatibleShape("cannot use idca.{} in an idco expression".format(tokval))
				# replace NAME tokens
				partial = [ (NAME, 'self'), (OP, '.'), (NAME, 'h5idca'), (OP, '.'), (NAME, tokval), (OP, '['), ]
				if screen is None:
					partial += [(OP, ':'),]
				else:
					partial += [(NAME, 'screen'),]
				if dims>1:
					partial += [(OP, ','),(OP, ':'),]
				partial += [(OP, ']'), ]
				recommand.extend(partial)
			elif toknum == NAME and tokval in self.h5idco:
				# replace NAME tokens
				partial = [ (NAME, 'self'), (OP, '.'), (NAME, 'h5idco'), (OP, '.'), (NAME, tokval), (OP, '['), ]
				if screen is None:
					partial += [(OP, ':'),]
				else:
					partial += [(NAME, 'screen'),]
				partial += [(OP, ']'), ]
				if dims>1:
					partial += [(OP, '['),(OP, ':'),(OP, ','),(NAME, 'None'),(OP, ']'),]
				recommand.extend(partial)
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
		from numpy import log, exp, log1p, absolute, fabs, sqrt
		h5node = self.h5top.idca
		h5caseid = self.h5top.caseids
		if screen is None and 'screen' not in self.h5top:
			n_cases = h5caseid.shape[0]
		elif screen is None:
			screen = self.get_screen_indexes()
			n_cases = screen.shape[0]
		else:
			n_cases = screen.shape[0]
		n_vars = len(vars)
		n_alts = self.nAlts()
		result = numpy.zeros([n_cases,n_alts,n_vars], dtype=dtype)
		for varnum,v in enumerate(vars):
			command = self._remake_command(v,screen,2)
			try:
				result[:,:,varnum] = eval( command )
			except Exception as exc:
				args = exc.args
				if not args:
					arg0 = ''
				else:
					arg0 = args[0]
				arg0 = arg0 + '\nwithin parsed command: "{!s}"'.format(command)
				exc.args = (arg0,) + args[1:]
				raise
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
		from numpy import log, exp, log1p, absolute, fabs, sqrt
		h5node = self.h5top.idco
		h5caseid = self.h5top.caseids
		if screen is None and 'screen' not in self.h5top:
			n_cases = h5caseid.shape[0]
		elif screen is None:
			screen = self.get_screen_indexes()
			n_cases = screen.shape[0]
		else:
			n_cases = screen.shape[0]
		n_vars = len(vars)
		result = numpy.zeros([n_cases,n_vars], dtype=dtype)
		for varnum,v in enumerate(vars):
			command = self._remake_command(v,screen,1)
			try:
				result[:,varnum] = eval( command )
			except Exception as exc:
				args = exc.args
				if not args:
					arg0 = ''
				else:
					arg0 = args[0]
				arg0 = arg0 + '\nwithin parsed command: "{!s}"'.format(command)
				exc.args = (arg0,) + args[1:]
				raise
		result = numpy.nan_to_num(result)
		from .array import Array
		result = result.view(Array)
		result.vars = vars
		return result

	def array_weight(self, *, var=None, **kwargs):
		try:
			w = self.h5idco._weight_
		except _tb.exceptions.NoSuchNodeError:
			return self.array_idco('1', **kwargs)
		else:
			return self.array_idco('_weight_', **kwargs)

	def array_choice(self, *, var=None, **kwargs):
		return self.array_idca('_choice_', **kwargs)

	def array_avail(self, *, var=None, dtype=numpy.bool_, **kwargs):
		try:
			av = self.h5idca._avail_
		except _tb.exceptions.NoSuchNodeError:
			return self.array_idca('1', dtype=dtype, **kwargs)
		else:
			return self.array_idca('_avail_', dtype=dtype, **kwargs)

	def get_screen_indexes(self):
		if 'screen' not in self.h5top:
			return None
		return numpy.nonzero(self.h5top.screen[:])[0]

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
		screen = self.get_screen_indexes()
		matched_cases = []
		#log = self.logger()
		log = None
		for key, req in needs.items():
			if log:
				log.info("Provisioning {} data...".format(key))
			if key=="Avail":
				provide[key] = numpy.require(self.array_avail(screen=screen), requirements='C')
			elif key=="Weight":
				provide[key] = numpy.require(self.array_weight(screen=screen), requirements='C')
			elif key=="Choice":
				provide[key] = numpy.require(self.array_choice(screen=screen), requirements='C')
			elif key[-2:]=="CA":
				provide[key] = numpy.require(self.array_idca(*req.get_variables(), screen=screen), requirements='C')
			elif key[-2:]=="CO":
				provide[key] = numpy.require(self.array_idco(*req.get_variables(), screen=screen), requirements='C')
			elif key=="Allocation":
				provide[key] = numpy.require(self.array_idco(*req.get_variables(), screen=screen), requirements='C')
		if screen is None:
			provide['caseids'] = numpy.require(self.h5top.caseids[:], requirements='C')
		else:
			provide['caseids'] = numpy.require(self.h5top.caseids[screen], requirements='C')
		if len(provide['caseids'].shape) == 1:
			provide['caseids'].shape = provide['caseids'].shape + (1,)
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
		return tuple(i for i in self.h5idca._v_leaves)

	def variables_co(self):
		return tuple(i for i in self.h5idco._v_leaves)

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
				tryname = next_stack(filename, plus=n, allow_natural=(n==0))
				h5f = _tb.open_file(tryname, 'w', filters=h5filters, driver="H5FD_CORE", driver_core_backing_store=0 if temp else 1)
			except ValueError:
				n += 1
				if n>1000:
					raise RuntimeError("cannot open HDF5 at {}".format(filename))
			else:
				break
		larchnode = h5f._getOrCreatePath("/larch", True)
		larchidca = h5f._getOrCreatePath("/larch/idca", True)
		larchidco = h5f._getOrCreatePath("/larch/idco", True)
		larchalts = h5f._getOrCreatePath("/larch/alts", True)

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

		h5avail = h5f.create_carray(larchidca, '_avail_', _tb.BoolAtom(), shape=(edb.nCases(), edb.nAlts()), filters=h5filters)
		arr, caseids = edb.array_avail()
		h5avail[:,:] = arr.squeeze()

		h5caseids = h5f.create_carray(larchnode, 'caseids', _tb.Int64Atom(), shape=(edb.nCases(),), filters=h5filters)
		h5caseids[:] = caseids.squeeze()

		h5scrn = h5f.create_carray(larchnode, 'screen', _tb.BoolAtom(), shape=(edb.nCases(),), filters=h5filters)
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
		import keyword
		nerrs = 0
		isok = None
		import textwrap
		blank_wrapper = textwrap.TextWrapper(
			width=80,
			tabsize=4,
			initial_indent=   '     │ ',
			subsequent_indent='     │ ',
			)
		ok_wrapper = textwrap.TextWrapper(
			width=80,
			tabsize=4,
			initial_indent=   ' ok  │ ',
			subsequent_indent='     │ ',
			)
		na_wrapper = textwrap.TextWrapper(
			width=80,
			tabsize=4,
			initial_indent=   ' n/a │ ',
			subsequent_indent='     │ ',
			)
		errmsg_wrapper = textwrap.TextWrapper(
			width=80,
			tabsize=4,
			initial_indent=   'ERROR│ ',
			subsequent_indent=' ┃   │ ',
			)
		errval_wrapper = textwrap.TextWrapper(
			width=80,
			tabsize=4,
			initial_indent=   ' ┗━► │ ',
			subsequent_indent=' ┗━► │ ',
			)
		def zzz(message, invalid, make_na=False):
			if make_na:
				log(na_wrapper.fill(message))
			elif invalid:
				log(errmsg_wrapper.fill(message))
				if invalid is True:
					invalid_str = "Nope"
				else:
					invalid_str = str(invalid)
				log(errval_wrapper.fill(invalid_str))
			else:
				log(ok_wrapper.fill(message))
			return 0 if not invalid and not make_na else 1

		def category(catname):
			log('─────┼'+'─'*74)
			log(blank_wrapper.fill(catname))

		## Top lines of display
		title = "{0} (with mode '{1}')".format(self.source_filename, self.source_filemode)
		#log("\u2550"*90)
		log("═"*80)
		log("larch.DT Validation for {}".format( title ))
		log("─────┬"+"─"*74)
		
		## TOP
		nerrs+= zzz("There should be a designated `larch` group node under which all other nodes reside.",
					not isinstance(self.h5top, _tb.group.Group))
		
		## CASEIDS
		category('CASES')
		try:
			caseids_node = self.h5top.caseids
			caseids_nodeatom = caseids_node.atom
			caseids_node_shape = caseids_node.shape
			caseids_node_dim = len(caseids_node.shape)
			caseids_node_len = caseids_node.shape[0]
		except _tb.exceptions.NoSuchNodeError:
			caseids_node = None
			caseids_nodeatom = None
			caseids_node_shape = ()
			caseids_node_dim = 0
			caseids_node_len = 0

		nerrs+= zzz("Under the top node, there must be an array node named `caseids`.",
					'missing caseids node' if caseids_node is None else
					isok if isinstance(caseids_node, _tb.array.Array) else
					'caseids is not an array node')

		nerrs+= zzz("The `caseids` array dtype should be Int64.",
					isok if isinstance(caseids_nodeatom, _tb.atom.Int64Atom) else "caseids dtype is {!s}".format(caseids_nodeatom))

		nerrs+= zzz("The `caseids` array should be 1 dimensional.",
					caseids_node_dim!=1)
		
		nerrs+= zzz("If there may be some data cases that are not to be included in the processing of "
					"the discrete choice model, there should be a node named `screen` under the top "
					"node.",
					None if 'screen' in self.h5top else None,
					'screen' not in self.h5top)

		# default failure values for screen checking
		screen_is_array = False
		screen_is_bool_array = False
		screen_shape = ()
		if 'screen' in self.h5top:
			screen_is_array = isinstance(self.h5top.screen, _tb.array.Array)
			if screen_is_array:
				screen_is_bool_array = isinstance(self.h5top.screen.atom, _tb.atom.BoolAtom)
				screen_shape = self.h5top.screen.shape

		nerrs+= zzz("If it exists `screen` must be a Bool array.",
					not screen_is_array or not screen_is_bool_array,
					'screen' not in self.h5top)

		nerrs+= zzz("And `screen` must be have the same shape as `caseids`.",
					None if screen_shape == caseids_node_shape else "screen is {} while caseids is {}".format(screen_shape, caseids_node_shape),
					'screen' not in self.h5top
					)


		## IDCO
		category('ALTERNATIVES')
		nerrs+= zzz("Under the top node, there should be a group named `alts` to hold alternative data.",
					not isinstance(self.h5top.alts, _tb.group.Group))
		try:
			altids_node = self.h5top.alts.altids
			altids_nodeatom = altids_node.atom
			altids_node_dim = len(altids_node.shape)
			altids_node_len = altids_node.shape[0]
		except _tb.exceptions.NoSuchNodeError:
			altids_node = None
			altids_nodeatom = None
			altids_node_dim = 0
			altids_node_len = 0
		nerrs+= zzz("Within the `alts` node, there should be an array node named `altids` to hold the "
					"identifying code numbers of the alternatives.",
					not isinstance(altids_node, _tb.array.Array) )
		nerrs+= zzz("The `altids` array dtype should be Int64.",
					None if isinstance(altids_nodeatom, _tb.atom.Int64Atom) else "altids dtype is {!s}".format(altids_nodeatom))
		nerrs+= zzz("The `altids` array should be one dimensional.",
					None if (altids_node_dim==1) else "it has {} dimensions".format(altids_node_dim))

		try:
			altnames_node = self.h5top.alts.names
			altnames_nodeatom = altnames_node.atom
			altnames_node_len = altnames_node.shape[0]
		except _tb.exceptions.NoSuchNodeError:
			altnames_node = None
			altnames_nodeatom = None
			altnames_node_len = 0

		nerrs+= zzz("Within the `alts` node, there should also be a VLArray node named `names` to hold "
					"the names of the alternatives.",
					not isinstance(altnames_node, _tb.vlarray.VLArray))
		nerrs+= zzz("The `names` node should hold unicode values.",
					not isinstance(altnames_nodeatom, _tb.atom.VLUnicodeAtom))
		nerrs+= zzz("The `altids` and `names` arrays should be the same length, and this will be the "
					"number of elemental alternatives represented in the data.",
					altnames_node_len!=altids_node_len)

		## IDCO
		category('IDCO FORMAT DATA')
		nerrs+= zzz("Under the top node, there should be a group named `idco` to hold that data.",
					not isinstance(self.h5top.idco, _tb.group.Group))
		nerrs+= zzz("Every child node name in `idco` must be a valid Python identifer (i.e. starts "
					"with a letter or underscore, and only contains letters, numbers, and underscores) "
					"and not a Python reserved keyword.",
					[i for i in self.h5idco._v_children.keys() if not i.isidentifier() or keyword.iskeyword(i)])

		## IDCA
		category('IDCA FORMAT DATA')
		nerrs+= zzz("Under the top node, there should be a group named `idca` to hold that data.",
					not isinstance(self.h5top.idca, _tb.group.Group))
		nerrs+= zzz("Every child node name in `idca` must be a valid Python identifer (i.e. starts "
					"with a letter or underscore, and only contains letters, numbers, and underscores) "
					"and not a Python reserved keyword.",
					[i for i in self.h5idca._v_children.keys() if not i.isidentifier() or keyword.iskeyword(i)])
		nerrs+= zzz("If there may be some alternatives that are unavailable in some cases, there should "
					"be a node named `_avail_` under `idca` that contains an appropriately sized Bool "
					"array indicating the availability status for each alternative.",
					'_avail_' in self.h5idca
					and (not isinstance(self.h5idca._avail_, _tb.array.Array)
						or not isinstance(self.h5idca._avail_.atom, _tb.atom.BoolAtom)))

		## TECHNICAL
		category('OTHER TECHNICAL DETAILS')
		nerrs+= zzz("The set of child node names within `idca` and `idco` should not overlap (i.e. "
					"there should be no node names that appear in both).",
					set(self.h5idca._v_children.keys()).intersection(self.h5idco._v_children.keys()))
		
		## Bottom line of display
		log('═════╧'+'═'*74)
		return nerrs

	validate = validate_hdf5