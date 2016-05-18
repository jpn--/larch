

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
import warnings

class IncompatibleShape(LarchError):
	pass

class HDF5BadFormat(LarchError):
	pass

class HDF5Warning(UserWarning):
    pass


class LocalAttributeSet(object):
	_hdf5_attrs = ('CLASS','FILTERS','TITLE','VERSION')
	def __init__(self, *args, **kwargs):
		self._h5_expr = _tb.attributeset.AttributeSet(*args, **kwargs)
		self._local_expr = dict()
	def __setattr__(self, attr, value):
		if attr in ('_h5_expr','_local_expr'):
			object.__setattr__(self, attr, value)
		else:
			try:
				self._h5_expr.__setattr__(attr,value)
			except _tb.exceptions.FileModeError:
				self._local_expr[attr] = value
				warnings.warn("The HDF5 is not writable so '{}' will not be saved beyond this session.".format(attr), HDF5Warning, stacklevel=2)
	def __getattr__(self, attr):
		if attr in self._local_expr:
			return self._local_expr[attr]
		else:
			if attr in self._hdf5_attrs:
				raise AttributeError("'{}' is a HDF5 attribute and cannot used by larch.".format(attr))
			return self._h5_expr.__getattr__(attr)
	def __delattr__(self, attr):
		try:
			del self.local_expr[attr]
		except KeyError:
			pass
		self._h5_expr.__delattr__(attr)
	def __repr__(self):
		h5attr = self._h5_expr._f_list()
		loc_attr = self._local_expr.keys()
		ret = "{} attributes:".format(self._h5_expr._v__nodepath)
		for i in h5attr:
			ret += "\n  {!s}".format(i)
		for i in loc_attr:
			ret += "\n  {!s} (local only)".format(i)
		return ret
	def __contains__(self, key):
		return key in self._local_expr or (key in self._h5_expr and key not in self._hdf5_attrs)
	def __iter__(self):
		return iter(sorted(self._local_expr.keys()) + sorted(i for i in self._h5_expr._v_attrnames if i not in self._hdf5_attrs))
	def __len__(self):
		return len(self._local_expr) + len([i for i in self._h5_expr._v_attrnames if i not in self._hdf5_attrs])
	def __getitem__(self, key):
		return self.__getattr__(key)
	def __setitem__(self, key, value):
		return self.__setattr__(key,value)
	def __delitem__(self, key):
		return self.__delattr__(key)



class DT(Fountain):
	"""A wrapper for a pytables File used to get data for models.

	This object wraps a :class:`tables.File`, adding a number of methods designed
	specifically for working with choice-based data used in Larch.

	Parameters
	----------
	filename : str or None
		The filename of the HDF5/pytables to open. If None (the default) a 
		named temporary file is created to serve as the backing for an in-memory 
		HDF5 file, which is very fast as long as you've got enough
		memory to store the whole thing.
	mode : str
		The mode used to open the H5F file.  Common values are 'a' for append and 'r' 
		for read only.  See pytables for more detail.
	complevel : int
		The compression level to use for new objects created.  By default no compression
		is used, but substantial disk savings may be available by using it.
	inmemory : bool
		If True (defaults False), the H5FD_CORE driver is used and data will not in general be written
		to disk until the file is closed, when all accumulated changes will be written
		in a single batch.  This can be fast if you have sufficent memory but if an error 
		occurs all your intermediate changes can be lost.
	temp : bool
		If True (defaults False), the inmemory switch is activated and no changes will be
		written to disk when the file is closed. This is automatically set to true if
		the `filename` is None.

	.. warning::
		The normal constructor creates a :class:`DT` object linked to an existing 
		HDF5 file. Editing the object edits the file as well. 

	"""

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

	def _refresh_alts(self):
		self._refresh_dna(self.alternative_names(), self.alternative_codes())

	def __init__(self, filename=None, mode='a', ipath='/larch', complevel=None, complib='zlib', h5f=None, inmemory=False, temp=False):
		if not _tb_success: raise ImportError("pytables not available")
		super().__init__()
		if filename is None:
			temp = True
			from .util.temporaryfile import TemporaryFile
			self._TemporaryFile = TemporaryFile(suffix='.h5f')
			filename = self._TemporaryFile.name
		if h5f is not None:
			self.h5f = h5f
			self._h5f_own = False
		else:
			kwd = {}
			if inmemory or temp:
				kwd['driver']="H5FD_CORE"
			if temp:
				kwd['driver_core_backing_store']=0
			if complevel is not None:
				kwd['filters']=_tb.Filters(complib=complib, complevel=complevel)
			self.h5f = _tb.open_file(filename, mode, **kwd)
			self._h5f_own = True
		self.source_filemode = mode
		self.source_filename = filename
		self._h5larchpath = ipath
		try:
			self.h5top = self.h5f._getOrCreatePath(ipath, True)
		except _tb.exceptions.FileModeError:
			raise HDF5BadFormat("the larch root node at '{}' does not exist and cannot be created".format(ipath))
		try:
			self.h5idca = self.h5f._getOrCreatePath(ipath+'/idca', True)
		except _tb.exceptions.FileModeError:
			raise HDF5BadFormat("the node at '{}/idca' does not exist and cannot be created".format(ipath))
		try:
			self.h5idco = self.h5f._getOrCreatePath(ipath+'/idco', True)
		except _tb.exceptions.FileModeError:
			raise HDF5BadFormat("the node at '{}/idco' does not exist and cannot be created".format(ipath))
		try:
			self.h5alts = self.h5f._getOrCreatePath(ipath+'/alts', True)
		except _tb.exceptions.FileModeError:
			raise HDF5BadFormat("the node at '{}/alts' does not exist and cannot be created".format(ipath))
		try:
			self.h5expr = self.get_or_create_group(self.h5top, 'expr')._v_attrs
		except _tb.exceptions.FileModeError:
			raise HDF5BadFormat("the node at '{}/expr' does not exist and cannot be created".format(ipath))
		self.expr = LocalAttributeSet(self.h5top.expr)
		self._refresh_alts()


	def __del__(self):
		if self._h5f_own:
			self.h5f.close()

	def __repr__(self):
		return "<larch.DT mode '{1}' at {0}>".format(self.source_filename, self.source_filemode)

	def create_group(self, *arg, **kwargs):
		return self.h5f.create_group(*arg, **kwargs)
	def create_array(self, *arg, **kwargs):
		return self.h5f.create_array(*arg, **kwargs)
	def create_carray(self, *arg, **kwargs):
		return self.h5f.create_carray(*arg, **kwargs)
	def create_earray(self, *arg, **kwargs):
		return self.h5f.create_earray(*arg, **kwargs)
	def create_external_link(self, *arg, **kwargs):
		return self.h5f.create_external_link(*arg, **kwargs)
	def create_hard_link(self, *arg, **kwargs):
		return self.h5f.create_hard_link(*arg, **kwargs)
	def create_soft_link(self, *arg, **kwargs):
		return self.h5f.create_soft_link(*arg, **kwargs)

	def get_or_create_group(self, where, name=None, title='', filters=None, createparents=False):
		try:
			return self.h5f.get_node(where, name=name)
		except _tb.NoSuchNodeError:
			if name is not None:
				return self.h5f.create_group(where, name, title=title, filters=filters, createparents=createparents)
			else:
				raise

	def _is_larch_array(self, where, name=None):
		n = self.h5f.get_node(where, name)
		try:
			n = n.dereference()
		except AttributeError:
			pass
		if isinstance(n, _tb.array.Array):
			return True
		if isinstance(n, _tb.group.Group):
			if '_index_' in n and '_values_' in n:
				return True
		return False

	def _is_mapped_larch_array(self, where, name=None):
		n = self.h5f.get_node(where, name)
		try:
			n = n.dereference()
		except AttributeError:
			pass
		if isinstance(n, _tb.group.Group):
			if '_index_' in n and '_values_' in n:
				return True
		return False

	def _alternative_codes(self):
		try:
			return self.h5alts.altids[:]
		except _tb.exceptions.NoSuchNodeError:
			return numpy.empty(0, dtype=numpy.int64)

	def _alternative_names(self):
		try:
			return self.h5alts.names[:]
		except _tb.exceptions.NoSuchNodeError:
			return numpy.empty(0, dtype=str)

	def alternative_codes(self):
		try:
			return tuple(int(i) for i in self.h5alts.altids[:])
		except _tb.exceptions.NoSuchNodeError:
			return ()

	def alternative_names(self):
		try:
			return tuple(str(i) for i in self.h5alts.names[:])
		except _tb.exceptions.NoSuchNodeError:
			return ()

	def alternative_name(self, code):
		codes = self._alternative_codes()
		idx = numpy.where(codes==code)[0][0]
		return self.h5alts.names[idx]

	def alternative_code(self, name):
		names = self._alternative_names()
		idx = numpy.where(names==name)[0][0]
		return self.h5alts.altids[idx]

	def alternatives(self, format=list):
		'''The alternatives of the data.
		
		When format==list or 'list', returns a list of (code,name) tuples.
		When format==dict or 'dict', returns a dictionary with codes as keys
		and names as values
		'''
		if format==list or format=='list':
			return list(zip(self.alternative_codes(), self.alternative_names()))
		if format==dict or format=='dict':
			return {i:j for i,j in zip(self.alternative_codes(), self.alternative_names())}
		if format=='reversedict':
			return {j:i for i,j in zip(self.alternative_codes(), self.alternative_names())}
		raise TypeError('only allows list or dict')

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
		DOT = (OP, '.')
		COLON = (OP, ':')
		COMMA = (OP, ',')
		OBRAC = (OP, '[')
		CBRAC = (OP, ']')
		from io import BytesIO
		recommand = []
		g = tokenize(BytesIO(cmd.encode('utf-8')).readline)
		for toknum, tokval, _, _, _  in g:
			if toknum == NAME and tokval in self.h5idca:
				if dims==1:
					raise IncompatibleShape("cannot use idca.{} in an idco expression".format(tokval))
				if self._is_mapped_larch_array(self.h5idca, tokval):
					# replace NAME tokens
					partial = [ (NAME, 'self'), DOT, (NAME, 'h5idca'), DOT, (NAME, tokval), DOT, (NAME, '_values_'),
								OBRAC,
								(NAME, 'self'), DOT, (NAME, 'h5idca'), DOT, (NAME, tokval), DOT, (NAME, '_index_'),OBRAC,COLON,CBRAC,
								CBRAC, ]
					if screen is None:
						partial += [OBRAC,COLON,]
					else:
						partial += [OBRAC,(NAME, 'screen'),]
					if dims>1:
						partial += [COMMA,COLON,]
					partial += [CBRAC, ]
					recommand.extend(partial)
				else:
					# replace NAME tokens
					partial = [ (NAME, 'self'), DOT, (NAME, 'h5idca'), DOT, (NAME, tokval), OBRAC, ]
					if screen is None:
						partial += [COLON,]
					else:
						partial += [(NAME, 'screen'),]
					if dims>1:
						partial += [COMMA,COLON,]
					partial += [CBRAC, ]
					recommand.extend(partial)
			elif toknum == NAME and tokval in self.h5idco:
				if self._is_mapped_larch_array(self.h5idco, tokval):
					# replace NAME tokens
					partial = [ (NAME, 'self'), DOT, (NAME, 'h5idco'), DOT, (NAME, tokval), DOT, (NAME, '_values_'),
								OBRAC,
								(NAME, 'self'), DOT, (NAME, 'h5idco'), DOT, (NAME, tokval), DOT, (NAME, '_index_'),OBRAC,COLON,CBRAC,
								CBRAC, ]
					if screen is None:
						partial += [OBRAC,COLON,CBRAC,]
					else:
						partial += [OBRAC,(NAME, 'screen'),CBRAC,]
					if dims>1:
						partial += [OBRAC,COLON,COMMA,(NAME, 'None'),CBRAC,]
					recommand.extend(partial)
				else:
					# replace NAME tokens
					partial = [ (NAME, 'self'), DOT, (NAME, 'h5idco'), DOT, (NAME, tokval), ]
					if screen is None:
						partial += [OBRAC,COLON,CBRAC,]
					else:
						partial += [OBRAC,(NAME, 'screen'),CBRAC,]
					if dims>1:
						partial += [OBRAC,COLON,COMMA,(NAME, 'None'),CBRAC,]
					recommand.extend(partial)
			elif toknum == NAME and tokval in self.expr:
				partial = [ (NAME, 'self'), DOT, (NAME, 'expr'), DOT, (NAME, tokval), ]
				recommand.extend(partial)
			else:
				recommand.append((toknum, tokval))
		ret = untokenize(recommand).decode('utf-8')
		return ret



	def array_idca(self, *vars, dtype=numpy.float64, screen=None, strip_nan=True):
		"""Extract a set of idca values from the DT.
		
		Generally you won't need to specify any parameters to this method beyond the
		variables to include in the array, as
		most values are determined automatically from the preset queries.
		
		Parameters
		----------
		vars : tuple of str
			A tuple giving the expressions (often column names, but any valid
			SQLite expression works) to extract as :ref:`idca` format variables.
		
		Other Parameters
		----------------
		dtype : str or dtype
			Describe the data type you would like the output array to adopt, probably 
			'int64', 'float64', or 'bool'.
		screen : None or array of bool
			If given, use this bool array to screen the caseids used to build 
			the array.
		strip_nan : bool
			If True (the default) then NAN values are converted to 0, and INF
			values are converted to the largest magnitude real number representable
			in the selected dtype.
		
		
		Returns
		-------
		data : ndarray
			An array with specified dtype, of shape (n_cases,n_alts,len(vars)).
			
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
		if strip_nan:
			result = numpy.nan_to_num(result)
		from .array import Array
		result = result.view(Array)
		result.vars = vars
		return result



	def array_idco(self, *vars, dtype=numpy.float64, screen=None, strip_nan=True):
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
		screen : None or array of bool
			If given, use this bool array to screen the caseids used to build 
			the array.
		strip_nan : bool
			If True (the default) then NAN values are converted to 0, and INF
			values are converted to the largest magnitude real number representable
			in the selected dtype.
		
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
		if strip_nan:
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

	def provision(self, needs, **kwargs):
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
		if self._check_ca_natural(column):
			return True
		if self._check_co_natural(column):
			return True
		try:
			command = self._remake_command(column,None,2)
			eval( command )
		except:
			return False
		return True

	def check_co(self, column):
		if self._check_co_natural(column):
			return True
		try:
			command = self._remake_command(column,None,1)
			eval( command )
		except:
			return False
		return True

	def variables_ca(self):
		return tuple(i for i in self.h5idca._v_children)

	def variables_co(self):
		return tuple(i for i in self.h5idco._v_children)


	def import_db(self, db, ignore_ca=('caseid','altid'), ignore_co=('caseid')):

		descrip_larch = {}
		descrip_alts = {
			'altid': _tb.Int64Col(pos=1, dflt=-999),
			'name': _tb.StringCol(itemsize=127, pos=2, dflt=""),
		}
		descrip_co = {}
		descrip_ca = {}
		vars_co = db.variables_co()
		vars_ca = db.variables_ca()
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

		for var_ca in vars_ca:
			if var_ca not in ignore_ca:
				h5var = self.h5f.create_carray(self.h5idca, var_ca, _tb.Float64Atom(), shape=(db.nCases(), db.nAlts()), )
				arr, caseids = db.array_idca(var_ca)
				h5var[:,:] = arr.squeeze()

		for var_co in vars_co:
			if var_co not in ignore_co:
				h5var = self.h5f.create_carray(self.h5idco, var_co, _tb.Float64Atom(), shape=(db.nCases(),), )
				arr, caseids = db.array_idco(var_co)
				h5var[:] = arr.squeeze()

		h5avail = self.h5f.create_carray(self.h5idca, '_avail_', _tb.BoolAtom(), shape=(db.nCases(), db.nAlts()), )
		arr, caseids = db.array_avail()
		h5avail[:,:] = arr.squeeze()

		h5caseids = self.h5f.create_carray(self.h5top, 'caseids', _tb.Int64Atom(), shape=(db.nCases(),), )
		h5caseids[:] = caseids.squeeze()

		h5scrn = self.h5f.create_carray(self.h5top, 'screen', _tb.BoolAtom(), shape=(db.nCases(),), )
		h5scrn[:] = True

		h5altids = self.h5f.create_carray(self.h5alts, 'altids', _tb.Int64Atom(), shape=(db.nAlts(),), title='elemental alternative code numbers')
		h5altids[:] = db.alternative_codes()

		h5altnames = self.h5f.create_vlarray(self.h5alts, 'names', _tb.VLUnicodeAtom(), title='elemental alternative names')
		for an in db.alternative_names():
			h5altnames.append( an )
		
		try:
			ch_ca = db.queries.get_choice_ca()
			self.h5f.create_soft_link(self.h5idca, '_choice_', target='/larch/idca/'+ch_ca)
		except AttributeError:
			h5ch = self.h5f.create_carray(self.h5idca, '_choice_', _tb.Float64Atom(), shape=(db.nCases(), db.nAlts()), )
			arr, caseids = db.array_choice()
			h5ch[:,:] = arr.squeeze()

		wgt = db.queries.weight
		if wgt:
			self.h5f.create_soft_link(self.h5idco, '_weight_', target='/larch/idco/'+wgt)



	@staticmethod
	def ExampleDirectory():
		'''Returns the directory location of the example data files.
		
		Larch comes with a few example data sets, which are used in documentation
		and testing. It is important that you do not edit the original data.
		'''
		import os.path
		TEST_DIR = os.path.join(os.path.split(__file__)[0],"data_warehouse")
		if not os.path.exists(TEST_DIR):
			uplevels = 0
			uppath = ""
			while uplevels < 20 and not os.path.exists(TEST_DIR):
				uppath = uppath+ ".."+os.sep
				uplevels += 1
				TEST_DIR = os.path.join(os.path.split(__file__)[0],uppath+"data_warehouse")
		if os.path.exists(TEST_DIR):
			return TEST_DIR	
		raise LarchError("cannot locate 'data_warehouse' examples directory")

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

		import os.path
		example_h5files = {
		  'MTC':os.path.join(DT.ExampleDirectory(),"MTCWork.h5"),
		  }

		h5filters = _tb.Filters(complevel=5)

		try:
			filename_ = filename.format(dataset)
		except:
			pass
		else:
			filename = filename_

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

		if dataset.upper() in example_h5files:

			h5f_orig = _tb.open_file(example_h5files[dataset.upper()])
			h5f_orig.get_node('/larch')._f_copy_children(h5f._getOrCreatePath("/larch", True), overwrite=True, recursive=True, createparents=False)
		else:

			from .db import DB
			edb = DB.Example(dataset)

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
				h5f.create_soft_link(larchidco, '_weight_', target='/larch/idco/'+wgt)

		return DT(filename, 'w', h5f=h5f)

	def validate_hdf5(self, log=print, errlog=None):
		"""Generate a validation report for this DT.
		
		The generated report is fairly detailed and describes each requirement
		for a valid DT file and whether or not it is met.
		
		Parameters
		----------
		log : callable
			Typically "print", but can be replaced with a different callable 
			to accept a series of unicode strings for each line in the report.
		errlog : callable or None
			By default, None.  If not none, the report will print as with `log`
			but only if there are errors.
		
		"""
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
			initial_indent=   ' ┣━► │ ',
			subsequent_indent=' ┣━► │ ',
			)
		def rreplace(_s, _old, _new):
			_li = _s.rsplit(_old, 1)
			return _new.join(_li)
		def zzz(message, invalid, make_na=False):
			if make_na:
				log(na_wrapper.fill(message))
			elif invalid:
				log(errmsg_wrapper.fill(message))
				if invalid is True:
					invalid_str = "Nope"
				else:
					invalid_str = str(invalid)
				log(rreplace(errval_wrapper.fill(invalid_str),'┣','┗'))
			else:
				log(ok_wrapper.fill(message))
			return 0 if (not invalid) or make_na else 1

		def category(catname):
			log('─────┼'+'─'*74)
			log(blank_wrapper.fill(catname))

		def subcategory(catname):
			log('     ├ '+'{:┄<73}'.format(catname+' '))

		## Top lines of display
		title = "{0} (with mode '{1}')".format(self.source_filename, self.source_filemode)
		#log("\u2550"*90)
		log("═"*80)
		log("larch.DT Validation for {}".format( title ))
		log("─────┬"+"─"*74)
		
		
		def isinstance_(obj, things):
			try:
				obj = obj.dereference()
			except AttributeError:
				pass
			return isinstance(obj, things)
		
		
		## TOP
		nerrs+= zzz("There should be a designated `larch` group node under which all other nodes reside.",
					not isinstance_(self.h5top, _tb.group.Group))
		
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
					isok if isinstance_(caseids_node, _tb.array.Array) else
					'caseids is not an array node')

		nerrs+= zzz("The `caseids` array dtype should be Int64.",
					isok if isinstance(caseids_nodeatom, _tb.atom.Int64Atom) else "caseids dtype is {!s}".format(caseids_nodeatom))

		nerrs+= zzz("The `caseids` array should be 1 dimensional.",
					caseids_node_dim!=1)
		
		subcategory('Case Filtering')
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
			screen_is_array = isinstance_(self.h5top.screen, _tb.array.Array)
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


		## ALTS
		category('ALTERNATIVES')
		nerrs+= zzz("Under the top node, there should be a group named `alts` to hold alternative data.",
					not isinstance_(self.h5top.alts, _tb.group.Group))
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
					not isinstance_(altids_node, _tb.array.Array) )
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
					not isinstance_(altnames_node, _tb.vlarray.VLArray))
		nerrs+= zzz("The `names` node should hold unicode values.",
					not isinstance(altnames_nodeatom, _tb.atom.VLUnicodeAtom))
		nerrs+= zzz("The `altids` and `names` arrays should be the same length, and this will be the "
					"number of elemental alternatives represented in the data.",
					altnames_node_len!=altids_node_len)

		## IDCO
		category('IDCO FORMAT DATA')
		nerrs+= zzz("Under the top node, there should be a group named `idco` to hold that data.",
					not isinstance_(self.h5top.idco, _tb.group.Group))
		nerrs+= zzz("Every child node name in `idco` must be a valid Python identifer (i.e. starts "
					"with a letter or underscore, and only contains letters, numbers, and underscores) "
					"and not a Python reserved keyword.",
					[i for i in self.h5idco._v_children.keys() if not i.isidentifier() or keyword.iskeyword(i)])
		
		idco_child_incorrect_sized = {}
		for idco_child in self.h5idco._v_children.keys():
			if isinstance_(self.h5idco._v_children[idco_child], _tb.group.Group):
				if '_index_' not in self.h5idco._v_children[idco_child] or '_values_' not in self.h5idco._v_children[idco_child]:
					idco_child_incorrect_sized[idco_child] = 'invalid group'
			else:
				try:
					if self.h5idco._v_children[idco_child].shape[0] != caseids_node_len:
						idco_child_incorrect_sized[idco_child] = self.h5idco._v_children[idco_child].shape
				except:
					idco_child_incorrect_sized[idco_child] = 'exception'
		nerrs+= zzz("Every child node in `idco` must be (1) an array node with shape the same as `caseids`, "
					"or (2) a group node with child nodes `_index_` as an array with the correct shape and "
					"an integer dtype, and `_values_` such that _values_[_index_] reconstructs the desired "
					"data array.",
					idco_child_incorrect_sized)
		

		## WEIGHT
		subcategory('Case Weights')
		try:
			weightnode = self.h5idco._weight_
		except _tb.exceptions.NoSuchNodeError:
			weightnode = None
			weightnode_atom = None
		else:
			weightnode_atom = weightnode.atom
		nerrs+= zzz("If the cases are to have non uniform weights, then there should a `_weight_` node "
					"(or a name link to a node) within the `idco` group.",
					isok if weightnode else None,
					'_weight_' not in self.h5idco)
		nerrs+= zzz("If weights are given, they should be of Float64 dtype.",
					isok if isinstance(weightnode_atom, _tb.atom.Float64Atom) else "_weight_ dtype is {!s}".format(weightnode_atom),
					'_weight_' not in self.h5idco)


		## IDCA
		category('IDCA FORMAT DATA')
		nerrs+= zzz("Under the top node, there should be a group named `idca` to hold that data.",
					not isinstance_(self.h5top.idca, _tb.group.Group))
		nerrs+= zzz("Every child node name in `idca` must be a valid Python identifer (i.e. starts "
					"with a letter or underscore, and only contains letters, numbers, and underscores) "
					"and not a Python reserved keyword.",
					[i for i in self.h5idca._v_children.keys() if not i.isidentifier() or keyword.iskeyword(i)])

		idca_child_incorrect_sized = {}
		for idca_child in self.h5idca._v_children.keys():
			if isinstance_(self.h5idca._v_children[idca_child], _tb.group.Group):
				if '_index_' not in self.h5idca._v_children[idca_child] or '_values_' not in self.h5idca._v_children[idca_child]:
					idca_child_incorrect_sized[idca_child] = 'invalid group'
				else:
					if self.h5idca._v_children[idca_child]._values_.shape[1] != altids_node_len:
						idca_child_incorrect_sized[idca_child] = self.h5idca._v_children[idca_child]._values_.shape
			else:
				try:
					if self.h5idca._v_children[idca_child].shape[0] != caseids_node_len or \
					   self.h5idca._v_children[idca_child].shape[1] != altids_node_len:
						idca_child_incorrect_sized[idca_child] = self.h5idca._v_children[idca_child].shape
				except:
					idca_child_incorrect_sized[idca_child] = 'exception'
		nerrs+= zzz("Every child node in `idca` must be (1) an array node with the first dimension the "
					"same as the length of `caseids`, and the second dimension the same as the length "
					"of `altids`, or (2) a group node with child nodes `_index_` as a 1-dimensional array "
					"with the same length as the length of `caseids` and "
					"an integer dtype, and a 2-dimensional `_values_` with the second dimension the same as the length "
					"of `altids`, such that _values_[_index_] reconstructs the desired "
					"data array.",
					idca_child_incorrect_sized)

		subcategory('Alternative Availability')
		if '_avail_' in self.h5idca:
			_av_exists = True
			_av_is_array = isinstance_(self.h5idca._avail_, _tb.array.Array)
			_av_atom_bool = isinstance(self.h5idca._avail_.atom, _tb.atom.BoolAtom)
		else:
			_av_exists = False
			_av_is_array = None
			_av_atom_bool = None

		nerrs+= zzz("If there may be some alternatives that are unavailable in some cases, there should "
					"be a node named `_avail_` under `idca`.",
					isok if _av_exists else 'node is missing',
					not _av_exists)
		nerrs+= zzz("If given, it should contain an appropriately sized Bool "
					"array indicating the availability status for each alternative.",
					isok if _av_is_array and _av_atom_bool else
					'not an array' if not _av_is_array else
					'not a bool array',
					not _av_exists)

		subcategory('Chosen Alternatives')
		if '_choice_' in self.h5idca:
			_ch_exists = True
			_ch_is_array = isinstance_(self.h5idca._choice_, _tb.array.Array)
			_ch_atom_float = isinstance(self.h5idca._choice_.atom, _tb.atom.Float64Atom)
		else:
			_ch_exists = False
			_ch_is_array = None
			_ch_atom_float = None
		
		nerrs+= zzz("There should be a node named `_choice_` under `idca`.",
					isok if _ch_exists else 'the node is missing')
		nerrs+= zzz("It should be a Float64 "
					"array indicating the chosen-ness for each alternative. "
					"Typically, this will take a value of 1.0 for the alternative that is "
					"chosen and 0.0 otherwise, although it is possible to have other values, "
					"including non-integer values, in some applications.",
					isok if _ch_is_array and _ch_atom_float else
					'not an array' if not _ch_is_array else
					'not a Float64 array')

		## TECHNICAL
		category('OTHER TECHNICAL DETAILS')
		nerrs+= zzz("The set of child node names within `idca` and `idco` should not overlap (i.e. "
					"there should be no node names that appear in both).",
					set(self.h5idca._v_children.keys()).intersection(self.h5idco._v_children.keys()))
		
		## Bottom line of display
		log('═════╧'+'═'*74)
		
		if errlog is not None and nerrs>0:
			self.validate_hdf5(log=errlog)
		return nerrs

	validate = validate_hdf5



	def import_idco(self, filepath_or_buffer, caseid_column=None, *args, **kwargs):
		"""Import an existing CSV or similar file in idco format into this HDF5 file.
		
		This function relies on :func:`pandas.read_csv` to read and parse the input data.
		All arguments other than those described below are passed through to that function.
		
		Parameters
		----------
		filepath_or_buffer : str or buffer
			This argument will be fed directly to the :func:`pandas.read_csv` function.
		caseid_column : None or str
			If given, this is the column of the input data file to use as caseids.  It must be 
			given if the caseids do not already exist in the HDF5 file.  If it is given and
			the caseids do already exist, a `LarchError` is raised.
		
		Raises
		------
		LarchError
			If caseids exist and are also given; or if caseids do not exist and are not given;
			or if the caseids are no integer values.
		"""
		import pandas
		df = pandas.read_csv(filepath_or_buffer, *args, **kwargs)
		try:
			for col in df.columns:
				col_array = df[col].values
				try:
					tb_atom = _tb.Atom.from_dtype(col_array.dtype)
				except ValueError:
					maxlen = int(df[col].str.len().max())
					maxlen = max(maxlen,8)
					col_array = df[col].astype('S{}'.format(maxlen)).values
					tb_atom = _tb.Atom.from_dtype(col_array.dtype)
				h5var = self.h5f.create_carray(self.h5idco, col, tb_atom, shape=col_array.shape)
				h5var[:] = col_array
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
				proposed_caseids_node = self.h5idco._v_children[caseid_column]
				if not isinstance(proposed_caseids_node.atom, _tb.atom.Int64Atom):
					col_array = df[caseid_column].values.astype('int64')
					if not numpy.all(col_array==df[caseid_column].values):
						raise LarchError("caseid_column '{}' does not contain only integer values".format(caseid_column))
					h5var = self.h5f.create_carray(self.h5idco, caseid_column+'_int64', _tb.Atom.from_dtype(col_array.dtype), shape=col_array.shape)
					h5var[:] = col_array
					caseid_column = caseid_column+'_int64'
					proposed_caseids_node = self.h5idco._v_children[caseid_column]
				self.h5f.create_soft_link(self.h5top, 'caseids', target=self.h5idco._v_pathname+'/'+caseid_column)
		except:
			self._df_exception = df
			raise



	def import_idca(self, filepath_or_buffer, caseid_col, altid_col, choice_col=None, force_int_as_float=True, chunksize=1e1000):
		"""Import an existing CSV or similar file in idca format into this HDF5 file.
		
		This function relies on :func:`pandas.read_csv` to read and parse the input data.
		All arguments other than those described below are passed through to that function.
		
		Parameters
		----------
		filepath_or_buffer : str or buffer
			This argument will be fed directly to the :func:`pandas.read_csv` function.
		caseid_column : None or str
			If given, this is the column of the input data file to use as caseids.  It must be 
			given if the caseids do not already exist in the HDF5 file.  If it is given and
			the caseids do already exist, a `LarchError` is raised.
		altid_col : None or str
			If given, this is the column of the input data file to use as altids.  It must be
			given if the altids do not already exist in the HDF5 file.  If it is given and
			the altids do already exist, a `LarchError` is raised.
		choice_col : None or str
			If given, use this column as the choice indicator.
		force_int_as_float : bool
			If True, data columns that appear to be integer values will still be stored as 
			double precision floats (defaults to True).
		chunksize : int
			The number of rows of the source file to read as a chunk.  Reading a giant file in moderate sized
			chunks can be much faster and less memory intensive than reading the entire file.
		
		Raises
		------
		LarchError
			Various errors.
			
		Notes
		-----
		Chunking may not work on Mac OS X due to a `known bug <http://github.com/pydata/pandas/issues/11793>`_
		in the pandas.read_csv function.
		"""
		import pandas
		casealtreader = pandas.read_csv(filepath_or_buffer, chunksize=chunksize, usecols=[caseid_col,altid_col])
		caseids = numpy.array([], dtype='int64')
		altids = numpy.array([], dtype='int64')
		for chunk in casealtreader:
			caseids = numpy.union1d(caseids, chunk[caseid_col].values)
			altids = numpy.union1d(altids, chunk[altid_col].values)

		if caseids.dtype != numpy.int64:
			from .util.arraytools import labels_to_unique_ids
			case_labels, caseids = labels_to_unique_ids(caseids)
			caseids = caseids.astype('int64')

		if 'caseids' not in self.h5top:
			self.h5f.create_carray(self.h5top, 'caseids', obj=caseids)
		else:
			if not numpy.all(caseids==self.h5top.caseids[:]):
				raise LarchError ('caseids exist but do not match the imported data')

		alt_labels = None
		if 'altids' not in self.h5alts:
			if altids.dtype != numpy.int64:
				from .util.arraytools import labels_to_unique_ids
				alt_labels, altids = labels_to_unique_ids(altids)
			h5altids = self.h5f.create_carray(self.h5alts, 'altids', obj=altids, title='elemental alternative code numbers')
		else:
			if not numpy.all(numpy.in1d(altids, self.h5alts.altids[:], True)):
				raise LarchError ('altids exist but do not match the imported data')
			else:
				altids = self.h5alts.altids[:]
		if 'names' not in self.h5alts:
			h5altnames = self.h5f.create_vlarray(self.h5alts, 'names', _tb.VLUnicodeAtom(), title='elemental alternative names')
			if alt_labels is not None:
				for an in alt_labels:
					h5altnames.append( str(an) )
			else:
				for an in self.h5alts.altids[:]:
					h5altnames.append( 'a'+str(an) )

		caseidmap = {i:n for n,i in enumerate(caseids)}
		altidmap = {i:n for n,i in enumerate(altids)}
		if alt_labels is not None:
			# if the altids are not integers, we replace the altid map with a labels map
			altidmap = {i:n for n,i in enumerate(alt_labels)}

		try:
			filepath_or_buffer.seek(0)
		except AttributeError:
			pass

		colreader = pandas.read_csv(filepath_or_buffer, nrows=1000 )
		force_float_columns = {}
		h5arr = {}
		for col in colreader.columns:
			if col in (caseid_col, altid_col): continue
			if force_int_as_float and colreader[col].dtype == numpy.int64:
				atom_dtype = _tb.atom.Float64Atom()
				force_float_columns[col] = numpy.float64
			else:
				atom_dtype = _tb.Atom.from_dtype(colreader[col].dtype)
			h5arr[col] = self.h5f.create_carray(self.h5idca, col, atom_dtype, shape=(caseids.shape[0], altids.shape[0]))
		if '_present_' not in colreader.columns:
			h5arr['_present_'] = self.h5f.create_carray(self.h5idca, '_present_', _tb.atom.BoolAtom(), shape=(caseids.shape[0], altids.shape[0]))

		try:
			filepath_or_buffer.seek(0)
		except AttributeError:
			pass

		reader = pandas.read_csv(filepath_or_buffer, chunksize=chunksize, dtype=force_float_columns, engine='c')
		try:
			for chunk in reader:
				casemap = chunk[caseid_col].map(caseidmap)
				altmap = chunk[altid_col].map(altidmap)
				for col in chunk.columns:
					if col in (caseid_col, altid_col): continue
					h5arr[col][casemap.values,altmap.values] = chunk[col].values
				if '_present_' not in chunk.columns:
					h5arr['_present_'][casemap.values,altmap.values] = True
		except:
			self._chunk = chunk
			self._casemap = casemap
			self._altmap = altmap
			self._altidmap = altidmap
			raise
		
		self.h5f.create_soft_link(self.h5idca, '_avail_', target=self.h5idca._v_pathname+'/_present_')

		if choice_col:
			if isinstance(self.h5idca._v_children[choice_col].atom, _tb.atom.Float64Atom):
				self.h5f.create_soft_link(self.h5idca, '_choice_', target=self.h5idca._v_pathname+'/'+choice_col)
			else:
				self.h5f.create_carray(self.h5idca, '_choice_', obj=self.h5idca._v_children[choice_col][:].astype('Float64'))


	def check_if_idca_is_idco(self, idca_var, return_data=False):
		if idca_var not in self.h5idca:
			raise LarchError("'{}' is not an idca variable".format(idca_var))
		arr = self.h5idca._v_children[idca_var][:]
		if '_avail_' in self.h5idca:
			av = self.h5idca._avail_[:]
			arr[~av] = numpy.nan
		result = (numpy.nanstd(arr, axis=1).sum()==0)
		if return_data:
			return result, arr, av
		return result
		
	def crack_idca(self, idca_var=None):
		if idca_var is None:
			return
		result, arr, av = self.check_if_idca_is_idco(idca_var, return_data=True)
		if result:
			newarr = numpy.nanmean(arr, axis=1)
			self.h5f.create_carray(self.h5idco, idca_var, obj=newarr)
			self.h5idca._v_children[idca_var]._f_remove()


	def info(self, log=print):
		log("Variables:")
		log("  idco:")
		for i in sorted(self.variables_co()):
			log("    {}".format(i))
		log("  idca:")
		for i in sorted(self.variables_ca()):
			log("    {}".format(i))
		if len(self.expr):
			log("Expr:")
			for i in self.expr:
				log("    {}".format(i))


	@property
	def namespace(self):
		space = {}
		space.update(self.h5idco._v_children.iteritems())
		space.update(self.h5idca._v_children.iteritems())
		space.update({i:self.expr[i] for i in self.expr})
		return space

	def Expr(self, expression):
		return _tb.Expr(expression, uservars=self.namespace)

