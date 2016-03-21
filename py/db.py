
from . import utilities

try:
	from . import apsw
	apsw_Connection = apsw.Connection
except ImportError:
	from .mock_module import Mock
	apsw = Mock()
	apsw_Connection = utilities.Dummy

import numpy


from .core import SQLiteDB, Facet, FacetError, LarchError, QuerySetSimpleCO, QuerySetTwoTable
from .exceptions import NoResultsError, TooManyResultsError
from . import logging
import time
import os
import sys


_docstring_sql_alts =\
"An SQL query that evaluates to a larch_alternatives table.\n\
\n\
Column 1: id (integer) a key for every alternative observed in the sample\n\
Column 2: name (text) a name for each alternative\n\
";


_docstring_sql_idco =\
"An SQL query that evaluates to an larch_idco table.\n\
\n\
Column 1: caseid (integer) a key for every case observed in the sample\n\
Column 2+ can contain any explanatory data, typically numeric data, although non-numeric data is allowable.";


_docstring_sql_idca =\
"An SQL query that evaluates to an larch_idca table.\n\
\n\
Column 1: caseid (integer) a key for every case observed in the sample\n\
Column 2: altid (integer) a key for each alternative available in this case\n\
Column 3+ can contain any explanatory data, typically numeric data, although non-numeric data is allowable.\n\
If no columns have the name 'caseid' and 'altid', elm will use the first two columns, repsectively.\n\
A query with less than two columns will raise an exception.";


_docstring_sql_choice =\
"An SQL query that evaluates to an larch_choice table.\n\
\n\
Column 1: caseid (integer) a key for every case observed in the sample\n\
Column 2: altid (integer) a key for each alternative available in this case\n\
Column 3: choice (numeric, typically 1.0 but could be other values)\n\
\n\
If an alternative is not chosen for a given case, it can have a zero choice value or \
it can simply be omitted from the result.";


_docstring_sql_avail =\
"An SQL query that evaluates to an larch_avail table.\n\
\n\
Column 1: caseid (integer) a key for every case observed in the sample\n\
Column 2: altid (integer) a key for each alternative available in this case\n\
Column 3: avail (boolean) evaluates as 1 or true when the alternative is available, 0 otherwise\n\
\n\
If an alternative is not available for a given case, it can have a zero avail value or \
it can simply be omitted from the result. If no query is given, it is assumed that \
all alternatives are available in all cases.";


_docstring_sql_weight =\
"An SQL query that evaluates to an larch_weight table.\n\
\n\
Column 1: caseid (integer) a key for every case observed in the sample\n\
Column 2: weight (numeric) a weight associated with each case\n\
\n\
If no weights are given, they are assumed to be all equal.";







class DB(utilities.FrozenClass, Facet, apsw_Connection):
	"""An SQLite database connection used to get data for models.

	This object wraps a :class:`apsw.Connection`, adding a number of methods designed
	specifically for working with choice-based data used in Larch.

	Parameters
	----------
	filename : str or None
		The filename or `URI <https://www.sqlite.org/c3ref/open.html#urifilenameexamples>`_
		of the database to open. It must be encoded as a UTF-8 string. (If your
		string contains only usual English characters you probably don't need
		to worry about it.) The default is an in-memory database opened with a URI of
		``file:larchdb?mode=memory``, which is very fast as long as you've got enough
		memory to store the whole thing.
	readonly : bool
		If true, the database connection is opened with a read-only flag set. If the file
		does not already exist, an exception is raised.


	.. warning::
		The normal constructor creates a :class:`DB` object linked to an existing SQLite
		database file. Editing the object edits the file as well. There is currently no
		"undo" so be careful when manipulating the database.

	"""

	sql_alts   = property(Facet.qry_alts  , None, None, _docstring_sql_alts   )
	sql_idco   = property(Facet.qry_idco  , None, None, _docstring_sql_idco   )
	sql_idca   = property(Facet.qry_idca  , None, None, _docstring_sql_idca   )
	sql_choice = property(Facet.qry_choice, None, None, _docstring_sql_choice )
	sql_avail  = property(Facet.qry_avail , None, None, _docstring_sql_avail  )
	sql_weight = property(Facet.qry_weight, None, None, _docstring_sql_weight )

	from .util.numbering import recode_alts

	def __init__(self, filename=None, readonly=False, load_queries=True):
		import os.path
		if filename is None:
			filename="file:larchdb?mode=memory"
		# Use apsw to open a SQLite connection
		if readonly:
			apsw.Connection.__init__(self, filename, flags=apsw.SQLITE_OPEN_URI|apsw.SQLITE_OPEN_READONLY)
		else:
			apsw.Connection.__init__(self, filename, flags=apsw.SQLITE_OPEN_URI|apsw.SQLITE_OPEN_READWRITE|apsw.SQLITE_OPEN_CREATE)
		# Init classes
		Facet.__init__(self,self)
		# easy logging
		from .logging import easy_logging_active
		if easy_logging_active():
			self.logger(True)
		# Load SQLite extended math functions
		trypath = os.path.split(__file__)[0]
		trypath = utilities.path_shrink_until_exists(trypath)
		dir_files = os.listdir(trypath)
		if self.source_filename == "":
			self.source_filename = filename
		self.working_name = self.source_filename
		self._reported_changes = 0
		self.setcommithook(self._commithook)
		# Set Window Title
		facts = []
		if self.readonly("main"):
			facts.append("read only")
		if "mode=memory" in self.working_name:
			facts.append("in memory")
		if len(facts)>0:
			fact = " ["+", ".join(facts)+"]"
		else:
			fact = ""
		self.window_title = "{0:s}{1:s}".format(os.path.basename(self.source_filename),fact)
		self.store = utilities.storage(self)
		try: self._freeze()
		except: pass
		# load queries if available
		if load_queries:
			try:
				self.load_queries()
			except apsw.SQLError:
				pass

	def _commithook(self):
		if self.totalchanges() > self._reported_changes:
			net = self.totalchanges() - self._reported_changes
			self._reported_changes = self.totalchanges()
			log = self.logger()
			if log is not None and net > 0:
				log.debug("committing changes to %i database rows", net)
		return 0

	def operating_name(self):
		facts = []
		if self.readonly("main"):
			facts.append("read only")
		if "mode=memory" in self.working_name:
			facts.append("in memory")
		if len(facts)>0:
			fact = " ["+", ".join(facts)+"]"
		else:
			fact = ""
		return "{0:s}{1:s}".format(os.path.basename(self.source_filename),fact)

	def __del__(self):
		apsw.Connection.close(self)

	def __repr__(self):
		if self.filename == self.source_filename:
			return "<larch.DB at {0}>".format(self.filename)
		elif self.filename == "" and self.source_filename != "":
			return "<larch.DB from {0}>".format(self.source_filename)
		else:
			return "<larch.DB at {1} from {0}>".format(self.source_filename,self.filename)

	@property
	def info(self):
		s = repr(self)[1:-1]+"\n"
		s += "-----------------------------------\n"
		if self.active_facet != "":
			s += "active facet: %s\n"%self.active_facet
		s += "sql_idco:     %s\n"%(self.sql_idco   if self.sql_idco   else "<blank>")
		s += "sql_idca:     %s\n"%(self.sql_idca   if self.sql_idca   else "<blank>")
		s += "sql_alts:     %s\n"%(self.sql_alts   if self.sql_alts   else "<blank>")
		s += "sql_choice:   %s\n"%(self.sql_choice if self.sql_choice else "<blank>")
		s += "sql_avail:    %s\n"%(self.sql_avail  if self.sql_avail  else "<blank>")
		s += "sql_weight:   %s\n"%(self.sql_weight if self.sql_weight else "<blank>")
		s += "-----------------------------------\n"
		s += "nCases:       %s\n"%(self.nCases()   )
		s += "nAlts:        %s\n"%(self.nAlts()    )
		s += "Alts:         "
		s += "\n              ".join(["{!s:5}\t{!s}".format(code,name)
		                            for name,code in
									zip(self.alternative_names()[0:9],self.alternative_codes()[0:9])])
		if self.nAlts()>10:
			s += "\n            ... and {} more".format(self.nAlts()-10)
		s += "\n-----------------------------------"
		return s

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

	@staticmethod
	def Copy(source, destination="file:larchdb?mode=memory"):
		'''Create a copy of a database and link it to a DB object.

		It is often desirable to work on a copy of your data, instead of working
		with the original file. If you data file is not very large and you are 
		working with multiple models, there can be significant speed advantages
		to copying the entire database into memory first, and then working on it
		there, instead of reading from disk every time you want data.
		
		Parameters
		----------
		source : str
			The source SQLite database from which the contents will be copied.
			Can be given as a plain filename or a 
			`URI <https://www.sqlite.org/c3ref/open.html#urifilenameexamples>`_.
		destination : str
			The destination SQLite database to which the contents will be copied.
			Can be given as a plain filename or a 
			`URI <https://www.sqlite.org/c3ref/open.html#urifilenameexamples>`_.
			If it does not exist it will be created. If the destination is not
			given, an in-memory database will be opened with a URI of
			``file:larchdb?mode=memory``.

		Returns
		-------
		DB
			An open connection to destination database.
		
		'''
		d = DB(destination)
		d.copy_from_db(source)
		d.source_filename = source
		try:
			pass
			#d.load_facet()
		except FacetError:
			d.clear_facet()
		#d.refresh()
		try:
			d.load_queries()
		except apsw.SQLError:
			pass
		return d

	@staticmethod
	def NewConnection(source):
		'''Create a new DB object with a new SQLite connection to the same underlying database.
		
		The source connection must be sharable (i.e., have a shared cache).
		
		Parameters
		----------
		source : str
			The source DB from which the connection will be derived.

		Returns
		-------
		DB
			An open connection to destination database.
		
		'''
		d = DB(source.working_name)
		try:
			d.load_queries()
		except apsw.SQLError:
			pass
		return d

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
	def Example(dataset='MTC', shared=False):
		'''Generate an example data object in memory.
		
		Larch comes with a few example data sets, which are used in documentation
		and testing. It is important that you do not edit the original data, so
		this function copies the data into an in-memory database, which you can
		freely edit without damaging the original data.
		
		Parameters
		----------
		dataset : {'MTC', 'SWISSMETRO', 'MINI', 'ITINERARY'}
			Which example dataset should be used.
		shared : bool
			If True, the new copy of the database is opened with a shared cache,
			so additional database connections can share the same in-memory data.
			Defaults to False.
			
		Returns
		-------
		DB
			An open connection to the in-memory copy of the example database.
		
		'''
		import os.path
		TEST_DIR = DB.ExampleDirectory()
		TEST_DATA = {
		  'MTC':os.path.join(TEST_DIR,"MTCWork.sqlite"),
		  'TWINCITY':os.path.join(TEST_DIR,"TwinCityQ.elmdata"),
		  'SWISSMETRO':os.path.join(TEST_DIR,"swissmetro.sqlite"),
		  'ITINERARY':os.path.join(TEST_DIR,"airmini.sqlite"),
		  'MINI':os.path.join(TEST_DIR,"mini.sqlite"),
		  }
		if dataset.upper() not in TEST_DATA:
			raise LarchError("Example data set %s not found"%dataset)
		if shared:
			return DB.Copy(TEST_DATA[dataset.upper()], destination="file:{}?mode=memory&cache=shared".format(dataset.lower()))
		return DB.Copy(TEST_DATA[dataset.upper()], destination="file:{}?mode=memory".format(dataset.lower()))


	@staticmethod
	def CSV_idco(filename, caseid="_rowid_", choice=None, weight=None, tablename="data", savename=None, alts={}, safety=True):
		'''Creates a new larch DB based on an :ref:`idco` CSV data file.

		The input data file should be an :ref:`idco` data file, with the first line containing the column headings.
		The reader will attempt to determine the format (csv, tab-delimited, etc) automatically. 

		Parameters
		----------
		filename : str
			File name (absolute or relative) for CSV (or other text-based delimited) source data.
		caseid : str      
			Column name that contains the unique case id's. If the data is in idco format, case id's can
			be generated automatically based on line numbers, by using the reserved keyword '_rowid_'.
		choice : str or None
			Column name that contains the id of the alternative that is selected (if applicable). If not
			given, no sql_choice table will be autogenerated, and it will need to be set manually later.
		weight : str or None
			Column name of the weight for each case. If None, defaults to equal weights.
		tablename : str  
			The name of the sql table into which the data is to be imported. Do not give a reserved name
			(i.e. any name beginning with *sqlite* or *larch*).
		savename : str or None
			If not None, the name of the location to save the SQLite database file that is created.
		alts : dict
			A dictionary with keys of alt codes, and values of (alt name, avail column, choice column) tuples.
			If `choice` is given, the third item in the tuple is ignored and can be omitted.
		safety : bool     
			If true, all alternatives that are chosen, even if not given in `alts`, will be
			automatically added to the alternatives table.

		Returns
		-------
		DB
			An open connection to the database.
		'''
		eL = logging.getScriber("db")
		d = DB(filename=savename)
		d.queries = QuerySetSimpleCO(d)
		heads = d.import_csv(filename, table=tablename)
		if caseid not in heads: caseid = "_rowid_"
		if "caseid" in heads and caseid!="caseid":
			raise LarchError("If the imported file has a column called caseid, it must be the case identifier")
		d.execute("CREATE TABLE csv_alternatives (id PRIMARY KEY, name TEXT);")
		d.execute("BEGIN TRANSACTION;")
		for code,info in alts.items():
			d.execute("INSERT INTO csv_alternatives VALUES (?,?)",(code,info[0]))
		d.execute("END TRANSACTION;")
		d.queries.set_alts_query("SELECT * FROM csv_alternatives")
		d.queries.set_idco_query("SELECT {1} AS caseid, * FROM {0}".format(tablename,caseid))
		if choice is None:
			d.queries.set_choice_column_map({code:info[2] for code,info in alts.items()})
		else:
			d.queries.set_choice_column(choice)
		d.queries.set_avail_column_map({code:info[1] for code,info in alts.items()})
		if weight is None or weight == "_equal_":
			weight = "_equal_"
		else:
			#d.sql_weight = "SELECT caseid, {0} AS weight FROM larch_idco;".format(weight)
			d.queries.set_weight_column(weight)
		if safety:
			missing_codes = set()
			for row in d.execute("SELECT DISTINCT altid FROM "+d.queries.tbl_choice()+" WHERE altid NOT IN (SELECT id FROM csv_alternatives);"):
				missing_codes.add(row[0])
			for code in missing_codes:
				d.execute("INSERT INTO csv_alternatives VALUES (?,?)",(code,str(code)))
		#d.refresh()
		d.refresh_queries()
		assert( d.qry_alts() == "SELECT * FROM csv_alternatives" )
		return d


	@staticmethod
	def CSV_idca(filename, caseid=None, altid=None, choice=None, weight=None, avail=None, tablename="data",
				 tablename_co="_co", savename=None, alts={}, safety=True, index=False):
		'''Creates a new larch DB based on an :ref:`idca` CSV data file.

		The input data file should be an :ref:`idca` data file, with the first line containing the column headings.
		The reader will attempt to determine the format (csv, tab-delimited, etc) automatically. 

		Parameters
		----------
		filename : str
			File name (absolute or relative) for CSV (or other text-based delimited) source data.
		caseid : str or None
			Column name that contains the caseids. Because multiple rows will share the same
			caseid, caseid's *cannot* be generated automatically based on line numbers by using the 
			reserved keyword '_rowid_'. If None, either the columns titled 'caseid' will be used if it
			exists, and if not then the first column of data in the file will be used.
		altid : str or None
			Column name that contains the altids. If None, the second column of data in the file will be used.
		choice : str or None
			Column name that contains the id of the alternative that is selected (if applicable). If None, 
			the third column of data in the file will be used.
		weight : str or None
			Column name of the weight for each case. If None, defaults to equal weights. Note that the weight
			needs to be identical for all altids sharing the same caseid.
		avail : str or None
			Column name of the availability indicator. If None, it is assumed that unavailable alternatives
			have the entire row of data missing from the table.
		tablename : str  
			The name of the sql table into which the data is to be imported. Do not give a reserved name
			(i.e. any name beginning with *sqlite* or *larch*).
		tablename_co : str or None
			The name of the sql table into which idco format data is to be imported. Do not give a reserved name
			(i.e. any name beginning with *sqlite* or *larch*). If None, then no automatic cracking 
			will be attempted and all data will be imported into the idca table. If the given name begins with
			an underscore, it will be used as a suffix added onto `tablename`.
		savename : str or None
			If not None, the name of the location to save the SQLite database file that is created.
		alts : dict
			A dictionary with integer keys of alt codes, and string values of alt names.
		safety : bool
			If true, all alternatives that appear in the altid column, even if not given in `alts`, will be
			automatically added to the alternatives table.
		index : bool
			If true, automatically create indexes for caseids and altids on the :ref:`idca` table,
			and (if it is created) caseids on the :ref:`idco` table.

		Returns
		-------
		DB
			An open connection to the database.
		'''
		if weight is not None and tablename_co is None:
			raise LarchError("Weight is only allowed in the idco table, but there will be no such table")
		if tablename_co is not None and tablename_co[0]=="_":
			tablename_co = tablename+tablename_co
		eL = logging.getScriber("db")
		d = DB(filename=savename)
		d.queries = QuerySetTwoTable(d)
		if tablename_co:
			heads = d.import_csv(filename, table="larch_temp_import_table", temp=True)
		else:
			heads = d.import_csv(filename, table=tablename)
		heads = [h.rsplit(" ",1)[0] for h in heads]
		lower_heads = {h.lower():h for h in heads}
		if len(heads) != len(lower_heads):
			raise LarchError("The imported data file must have unique (case-insensitive) names for each column")
		# caseid
		if caseid is None:
			if "caseid" in lower_heads:
				caseid = lower_heads["caseid"]
			else:
				caseid = heads[0]
		if caseid.lower() not in lower_heads:
			from pprint import pprint
			pprint(lower_heads)
			raise LarchError("The caseid column '"+caseid+"' was not found in the data file")
		if "caseid" in lower_heads and caseid.lower()!="caseid":
			raise LarchError("If the imported file has a column called caseid, it must be the case identifier")
		# altid
		if altid is None:
			if "altid" in lower_heads:
				altid = lower_heads["altid"]
			else:
				if len(heads)<2:
					raise LarchError("The imported file has less than two columns")
				altid = heads[1]
		if altid.lower() not in lower_heads:
			raise LarchError("The altid column '"+altid+"' was not found in the data file")
		if "altid" in lower_heads and altid.lower()!="altid":
			raise LarchError("If the imported file has a column called altid, it must be the alt identifier")
		# choice
		if choice is None:
			if "choice" in lower_heads:
				choice = lower_heads["choice"]
			else:
				if len(heads)<3:
					raise LarchError("The imported file has less than three columns")
				choice = heads[2]
		if choice.lower() not in lower_heads:
			raise LarchError("The choice column '"+choice+"' was not found in the data file")
		if "choice" in lower_heads and choice.lower()!="choice":
			raise LarchError("If the imported file has a column called choice, it must be the choice identifier")
		### CRACK
		if tablename_co:
			d.crack_idca("larch_temp_import_table", caseid=caseid, ca_tablename=tablename, co_tablename=tablename_co)
		d.execute("CREATE TABLE csv_alternatives (id PRIMARY KEY, name TEXT);")
		d.execute("BEGIN TRANSACTION;")
		for code,info in alts.items():
			d.execute("INSERT INTO csv_alternatives VALUES (?,?)",(code,str(info)))
		d.execute("END TRANSACTION;")
		d.queries.set_alts_query("SELECT * FROM csv_alternatives")
		idca = "SELECT "
		if caseid.lower()!='caseid':
			idca += "{caseid} AS caseid, "
		if altid.lower()!='altid':
			idca += "{altid} AS altid, "
		idca += "* FROM {0}"
		d.queries.set_idca_query(idca.format(tablename,caseid=caseid,altid=altid))
		if tablename_co:
			idco = "SELECT "
			if caseid.lower()!='caseid':
				idco += "{caseid} AS caseid, "
			idco += "* FROM {0}"
			d.queries.set_idco_query(idco.format(tablename_co,caseid=caseid))
		d.queries.choice = choice
		d.queries.avail = avail
		if weight: d.queries.weight = weight
		if safety:
			missing_codes = set()
			for row in d.execute("SELECT DISTINCT altid FROM "+d.queries.tbl_choice()+" WHERE altid NOT IN (SELECT id FROM csv_alternatives);"):
				missing_codes.add(row[0])
			for code in missing_codes:
				d.execute("INSERT INTO csv_alternatives VALUES (?,?)",(code,str(code)))
		d.refresh_queries()
		assert( d.qry_alts() == "SELECT * FROM csv_alternatives" )
		d.save_queries()
		if index:
			d.execute("CREATE INDEX IF NOT EXISTS larch_autoindex_ca_{0} ON {0} ({1},{2})".format(tablename,caseid,altid))
			if tablename_co is not None:
				d.execute("CREATE INDEX IF NOT EXISTS larch_autoindex_co_{0} ON {0} ({1})".format(tablename_co,caseid))
		return d



	def crack_idca(self, tablename, caseid=None, ca_tablename=None, co_tablename=None):
		"""Crack an existing |idca| table into |idca| and |idco| component tables.
		
		This method will automatically analyze an existing |idca| table and
		identify columns of data that are invariant within individual cases. Those 
		variables will be segregated into a new |idco| table, and the remaining
		variables will be put into a new |idca| table.
		
		Parameters
		----------
		tablename : str
			The name of the existing |idca| table
		caseid : str or None
			The name of the column representing the caseids in the existing table. 
			If not given, it is assumed these are in the first column.
		ca_tablename : str or None
			The name of the table that will be created to hold the new (with fewer columns)
			|idca| table.
		co_tablename : str or None
			The name of the table that will be created to hold the new
			|idco| table.
			
		Raises
		------
		apsw.SQLError
			If the name of one of the tables to be created already exists in the database.
		
		"""
		if ca_tablename is None: ca_tablename=tablename+"_ca"
		if co_tablename is None: co_tablename=tablename+"_co"
		
		heads = self.column_names(tablename)
		lower_heads = {h.lower():h for h in heads}
		# caseid
		if caseid is None:
			if "caseid" in lower_heads:
				caseid = lower_heads["caseid"]
			else:
				caseid = heads[0]
		if caseid.lower() not in lower_heads:
			raise LarchError("The caseid column '"+caseid+"' was not found in the source table")
		if "caseid" in lower_heads and caseid.lower()!="caseid":
			raise LarchError("If the source table has a column called caseid, it must be the case identifier")

		v_ca = []
		v_co = []
		cmd = "SELECT SUM(ExcessRows) FROM (SELECT COUNT(*)-1 AS ExcessRows, %(caseid)s FROM (SELECT DISTINCT %(caseid)s, %(varcol)s FROM %(tablename)s) GROUP BY %(caseid)s);"
		for v in heads:
			if v.lower() != caseid.lower():
				total_dev = self.value(cmd % {'caseid':caseid, 'varcol':v, 'tablename':tablename})
				if total_dev>0:
					v_ca.append(v)
				else:
					v_co.append(v)
		if len(v_ca) > 0:
			c_ca = "CREATE TABLE %(ca_tablename)s AS SELECT %(caseid)s"
			for v in v_ca:
				c_ca += ", " + v
			c_ca += " FROM %(tablename)s"
			self.execute(c_ca % locals())
		if len(v_co) >= 0:
			c_co = "CREATE TABLE %(co_tablename)s AS SELECT DISTINCT %(caseid)s"
			for v in v_co:
				c_co += ", " + v
			c_co += " FROM %(tablename)s"
			self.execute(c_co % {'tablename':tablename, 'co_tablename':co_tablename, 'caseid':caseid})






	def import_csv(self, rawdata, table="data", drop_old=False, progress_callback=None, temp=False, column_names=None):
		'''Import raw csv or tab-delimited data into SQLite.

		Parameters
		----------
		rawdata : str
			The filename (relative or absolute) of the raw csv or tab delimited data file.
			If the filename has a .gz extension, it is assumed to be in gzip format instead
			of plain text.
		table : str
			The name of the table into which the data is to be imported
		drop_old : bool
			If true and the `table` already exists in the SQLite database, then the
			pre-existing `table` is deleted.
		progress_callback : callback function
			If given, this callback function takes a single integer
			as an argument and is called periodically while loading
			with the current precentage complete.
		temp : bool
			If true, the data is imported into a temporary table in the database, which
			will be deleted automatically when the connection is closed.
		column_names : list, optional
			If given, use these column names and assume the first line of the data file is
			data, not headers.
		
		Returns
		-------
		list
			A list of column headers from the imported csv file
		'''
		eL = logging.getScriber("db")
		eL.debug("Importing Data...")
		from .utilities import prepare_import_headers
		import csv
		headers, csvReader, smartFile = prepare_import_headers(rawdata, column_names)
		eL.debug("Importing data with headers:")
		for h in headers:
			eL.debug("  {0}".format(h))
		# fix header items for type
		num_cols = len(headers)
		# drop table in DB
		if drop_old: self.drop(table)
		# create table in DB	
		if temp:
			stmt = "CREATE TEMPORARY TABLE IF NOT EXISTS "+table+" (\n  "+",\n  ".join(headers)+")"
		else:
			stmt = "CREATE TABLE IF NOT EXISTS "+table+" (\n  "+",\n  ".join(headers)+")"
		self.execute(stmt)
		#.......
		# insert rows
		num_rows = 0
		num_errors = 0
		logLevel = eL.level
		eL.setLevel(logging.INFO)
		stmt = "INSERT INTO "+table+" VALUES ("+(",?"*len(headers))[1:]+")"
		lastlogupdate = time.time()
		lastscreenupdate = time.time()
		self.execute("BEGIN TRANSACTION;")
		while True:
			try:
				i = next(csvReader)
			except csv.Error:
				num_rows = num_rows+1
				num_errors = num_errors+1
				eL.warning("csv.Error on row %d",num_rows)
				continue
			except StopIteration:
				break
			else:
				if len(i) == len(headers):
					self.execute(stmt,tuple(i))
					num_rows = num_rows+1
					if (time.time()-lastlogupdate > 2):
						eL.info("%i rows imported", num_rows)
						lastlogupdate = time.time()
					if progress_callback is not None and (time.time()-lastscreenupdate > 0.1):
						lastscreenupdate = time.time()
						progress_callback(int(smartFile.percentread()))
				else:
					eL.warning("Incorrect Length (have %d, need %d) Data Row: %s",len(i),len(headers),str(i))
		self.execute("END TRANSACTION;")
		eL.setLevel(logLevel)
		# clean up
		eL.info("Imported %i rows to %s", num_rows, table)
		if num_errors>0:
			eL.info("with %i errors", num_errors)
		return headers

	def import_dbf(self, rawdata, table="data", drop_old=False):
		'''Imports data from a DBF file into an existing larch DB.

		Parameters
		----------
		rawdata : str
			The filename (relative or absolute) of the raw DBF data file.
		table : str
			The name of the table into which the data is to be imported
		drop_old : bool
			If true and the `table` already exists in the SQLite database, then the
			pre-existing `table` is deleted.
		
		Returns
		-------
		list
			A list of column headers from the imported DBF file


		.. note::
			This method requires the dbfpy module (available using pip).
		'''
		eL = logging.getScriber("db")
		eL.debug("Importing DBF Data...")
		try:
			from dbfpy import dbf_open
		except ImportError:
			eL.fatal("importing dbf files requires the dbfpy module (available using pip)")
			raise
		dbff = dbf_open(rawdata)
		headers = dbff.fieldNames()
		if drop_old: self.drop(table)
		eL.debug("Importing data with headers:")
		for h in headers:
			eL.debug("  {0}".format(h))
		# fix header items for type
		num_cols = len(headers)
		# drop table in DB
		if drop_old: self.drop(table)
		# create table in DB
		stmt = "CREATE TABLE IF NOT EXISTS "+table+" (\n  "+",\n  ".join(headers)+")"
		self.execute(stmt)
		#.......
		# insert rows
		num_rows = 0
		logLevel = eL.level
		eL.setLevel(logging.INFO)
		stmt = "INSERT INTO "+table+" VALUES ("+(",?"*len(headers))[1:]+")"
		lastlogupdate = time.time()
		self.execute("BEGIN TRANSACTION;")
		for i in range(len(dbff)):
			row = dbff[i].fieldData
			if len(row) == len(headers):
				self.execute(stmt,row)
				num_rows = num_rows+1
				if (time.time()-lastlogupdate > 2):
					eL.info("%i rows imported, %0.2f%%", num_rows, 100.0*num_rows/len(dbff))
					lastlogupdate = time.time()
			else:
				eL.warning("Incorrect Length (have %d, need %d) Data Row: %s",len(row),len(headers),str(row))
		self.execute("END TRANSACTION;")
		eL.setLevel(logLevel)
		# clean up
		eL.info("Imported %i rows to %s", num_rows, table)
		return headers


	def import_dataframe(self, rawdataframe, table="data", if_exists='fail'):
		'''Imports data from a pandas dataframe into an existing larch DB.
			
		Parameters
		----------
		rawdataframe : :class:`pandas.DataFrame`
			The filename (relative or absolute) of the raw DataFrame.
		table : str
			The name of the table into which the data is to be imported
		if_exists : {'fail', 'replace', 'append'}
			If the table does not exist this parameter is ignored, otherwise,
			*fail*: If table exists, raise a ValueError exception.
			*replace*: If table exists, drop it, recreate it, and insert data.
			*append*: If table exists, insert data.

		Returns
		-------
		list
			A list of column headers from the imported DBF file
		'''
		if if_exists not in ('fail', 'replace', 'append'):
			raise ValueError("'%s' is not valid for if_exists" % if_exists)
		exists = table in self.all_table_names()
		if if_exists == 'fail' and exists:
			raise ValueError("Table '%s' already exists." % table)
		eL = logging.getScriber("db")
		eL.info("Importing pandas dataframe with {} rows and {} columns...".format(len(rawdataframe),len(rawdataframe.columns)))
		import pandas
		# creation/replacement dependent on the table existing and if_exist criteria
		create = None
		if exists:
			if if_exists == 'fail':
				raise ValueError("Table '%s' already exists." % table)
			elif if_exists == 'replace':
				cur = self.cursor()
				cur.execute("DROP TABLE %s;" % table)
				cur.close()
				create = pandas.io.sql.get_schema(rawdataframe, table, 'sqlite')
		else:
			create = pandas.io.sql.get_schema(rawdataframe, table, 'sqlite')
		if create is not None:
			cur = self.cursor()
			cur.execute(create)
			cur.close()
		cur = self.cursor()
		# Replace spaces in DataFrame column names with _.
		safe_names = [s.replace(' ', '_').strip() for s in rawdataframe.columns]
		bracketed_names = ['[' + column + ']' for column in safe_names]
		col_names = ','.join(bracketed_names)
		wildcards = ','.join(['?'] * len(safe_names))
		insert_query = 'INSERT INTO %s (%s) VALUES (%s)' % (table, col_names, wildcards)
		num_rows = 0
		lastlogupdate = time.time()
		cur.execute("BEGIN TRANSACTION;")
		if not len(rawdataframe.columns) == 1:
			sourcedata = rawdataframe.values
		else:
			sourcedata = rawdataframe.values.tolist()
		for x in sourcedata:
			cur.execute(insert_query, tuple(x))
			num_rows = num_rows+1
			if (time.time()-lastlogupdate > 2):
				eL.info("%i rows imported", num_rows)
				lastlogupdate = time.time()
		cur.execute("END TRANSACTION;")
		eL.info("%i rows imported", num_rows)
		cur.close()
		return [s.replace(' ', '_').strip() for s in rawdataframe.columns]

	def import_xlsx(self, io, sheetname=0, table="data", if_exists='fail', **kwargs):
		'''Imports data from an Excel spreadsheet into an existing larch DB.
		
		Parameters
		----------
		io : string, file-like object, or xlrd workbook.
			The string could be a URL. Valid URL schemes include http, ftp, s3, and file.
			For file URLs, a host is expected. For instance, a local file could be
			file://localhost/path/to/workbook.xlsx
		sheetname : string or int, default 0
			Name of Excel sheet or the page number of the sheet
		table : str
			The name of the table into which the data is to be imported
		if_exists : {'fail', 'replace', 'append'}
			If the table does not exist this parameter is ignored, otherwise,
			*fail*: If table exists, raise a ValueError exception.
			*replace*: If table exists, drop it, recreate it, and insert data.
			*append*: If table exists, insert data.

		Returns
		-------
		list
			A list of column headers from the imported DBF file

		Notes
		-----
		This method uses a :class:`pandas.DataFrame` as an intermediate step, first calling
		:func:`pandas.io.excel.read_excel` and then calling :meth:`import_dataframe`. All
		keyword arguments other than those listed here are simply passed to 
		:func:`pandas.io.excel.read_excel`.
		
		'''
		import pandas
		df = pandas.io.excel.read_excel(io, sheetname, **kwargs)
		return self.import_dataframe(df, table=table, if_exists=if_exists)
	

	def export_idca(self, file, include_idco='intersect', exclude=[], **formats):
		'''Export the :ref:`idca` data to a csv file.
		
		Parameters
		----------
		file : str or file-like
			If a string, this is the file name to give to the `open` command. Otherwise,
			this object is passed to :class:`csv.writer` directly.
		include_idco : {'intersect', 'all', 'none'}
			Unless this is 'none', the idca and idco tables are joined on caseids before exporting.
			For 'intersect', a natural join is used, so that all columns with the same name
			are used for the join. This may cause problems if columns in the idca and idco 
			tables have the same name but different data.  For 'all', the join is made on caseids only, and
			every column in both tables is included in the output.
			When 'none', only the idca table is exported and the idco table is ignored.
		exclude : set or list
			A list of variables names to exclude from the output.  This could be useful
			in shrinking the file size if you don't need all the output columns,
			or suppressing duplicate copies of caseid and altid columns.
			
		Notes
		-----
		This method uses a :class:`csv.writer` object to write the output file. Any keyword
		arguments not listed here are passed through to the writer.
		'''
		if include_idco=='all':
			qry = "SELECT * FROM larch_idca, larch_idco ON larch_idca.caseid==larch_idco.caseid"
		elif include_idco=='intersect':
			qry = "SELECT * FROM larch_idca NATURAL JOIN larch_idco"
		elif include_idco=='none' or include_idco is None:
			qry = "SELECT * FROM larch_idca"
		else:
			raise TypeError("include_idco must be one of {'intersect', 'all', 'none'}")
		import csv
		
		if isinstance(file, str):
			if file[-3:].lower()=='.gz':
				import gzip
				csvfile = gzip.open(file, 'wt')
			else:
				csvfile = open(file, 'w', newline='')
		else:
			csvfile = file
		writer = csv.writer(csvfile, **formats)
		cursor = self.execute(qry, cte=True)
		names = [i[0] for i in cursor.getdescription()]
		if len(exclude)>0:
			cols_to_keep = [ num for num,name in enumerate(names) if name not in exclude ]
			writer.writerow(tuple(names[i] for i in cols_to_keep))
			for row in cursor:
				writer.writerow(tuple(row[i] for i in cols_to_keep))
		else:
			writer.writerow(names)
			for row in cursor:
				writer.writerow(row)
		if isinstance(file, str):
			csvfile.close()
		else:
			# cleanup file-like object here
			pass
		return

	def export_idco(self, file, exclude=[], **formats):
		'''Export the :ref:`idco` data to a csv file.
		
		Only the :ref:`idco` table is exported, the :ref:`idca` table is ignored.  Future versions
		of Larch may provide a facility to export idco and idca data together in a 
		single idco output file.
		
		Parameters
		----------
		file : str or file-like
			If a string, this is the file name to give to the `open` command. Otherwise,
			this object is passed to :class:`csv.writer` directly.
		exclude : set or list
			A list of variables names to exclude from the output.  This could be useful
			in shrinking the file size if you don't need all the output columns,
			or suppressing duplicate copies of caseid and altid columns.
			
		Notes
		-----
		This method uses a :class:`csv.writer` object to write the output file. Any keyword
		arguments not listed here are passed through to the writer.
		'''
		qry = "SELECT * FROM larch_idco"
		import csv
		if isinstance(file, str):
			if file[-3:].lower()=='.gz':
				import gzip
				csvfile = gzip.open(file, 'wt')
			else:
				csvfile = open(file, 'w', newline='')
		else:
			csvfile = file
		writer = csv.writer(csvfile, **formats)
		cursor = self.execute(qry, cte=True)
		names = [i[0] for i in cursor.getdescription()]
		exclude = {i.lower() for i in exclude}
		if len(exclude)>0:
			cols_to_keep = [ num for num,name in enumerate(names) if name.lower() not in exclude ]
			writer.writerow(tuple(names[i] for i in cols_to_keep))
			for row in cursor:
				writer.writerow(tuple(row[i] for i in cols_to_keep))
		else:
			writer.writerow(names)
			for row in cursor:
				writer.writerow(row)
		if isinstance(file, str):
			csvfile.close()
		else:
			# cleanup file-like object here
			pass

	def export_query(self, file, query, cte=False, **formats):
		'''Export an arbitrary query to a csv file.
			
		Parameters
		----------
		file : str or file-like
			If a string, this is the file name to give to the `open` command. Otherwise,
			this object is passed to :class:`csv.writer` directly.
		query : str
			A SQL query (generally a SELECT query) to export
		cte : bool
			Should the standard larch queries be available as CTE to the query.
			
		Notes
		-----
		This method uses a :class:`csv.writer` object to write the output file. Any keyword
		arguments not listed here are passed through to the writer.
		'''
		import csv
		if isinstance(file, str):
			if file[-3:].lower()=='.gz':
				import gzip
				csvfile = gzip.open(file, 'wt')
			else:
				csvfile = open(file, 'w', newline='')
		else:
			csvfile = file
		writer = csv.writer(csvfile, **formats)
		cursor = self.execute(query, cte=cte)
		names = [i[0] for i in cursor.getdescription()]
		writer.writerow(names)
		for row in cursor:
			writer.writerow(row)
		if isinstance(file, str):
			csvfile.close()
		else:
			# cleanup file-like object here
			pass

	def execute(self, command, arguments=(), fancy=False, explain=False, fail_silently=False, echo=False, cte=False):
		'''A convenience function wrapping cursor generation and command 
		   execution for simple queries.
		
		Parameters
		----------
		command : str
			An SQLite command to evaulate
		arguments : tuple
			Values to bind to the SQLite command.
		fancy : bool
			If true, return rows as dict-type objects that can be indexed
		    by row headings instead of integer index positions.
		explain : bool
			If true, print the EXPLAIN QUERY PLAN results before executing.
		fail_silently : bool
			If false, log (at level critical) the SQL command and arguments
			to the larch.db logger before returning an SQLError 
		echo : bool
			If true, log (at level critical) the SQL command and arguments
			to the larch.db logger before executing
		cte : bool
			If true, prepend all :attr:`queries` tables (larch_caseids, larch_idco, larch_idca,
			larch_choice, larch_weight, and larch_avail) as common table expression
			in a 'WITH' clause before the command. This allows these names to
			be referenced in the command as if they were actual tables in the 
			:class:`DB`.
		'''
		try:
			if echo:
				l = logging.getScriber("db")
				l.critical("(execute) SQL:\n%s"%command)
				if arguments is not None and arguments!=():
					l.critical("Bindings:")
					l.critical(str(arguments))
			if cte:
				prefix = "WITH \n"
				prefix += "  larch_caseids AS ("+self.qry_caseids()+"), \n"
				prefix += "  larch_idco AS ("+self.qry_idco_()+"), \n"
				prefix += "  larch_idca AS ("+self.qry_idca_()+"), \n"
				prefix += "  larch_choice AS ("+self.qry_choice()+"), \n"
				prefix += "  larch_weight AS ("+self.qry_weight()+"), \n"
				prefix += "  larch_avail AS ("+self.qry_avail()+") \n"
				command = prefix + command
			if explain:
				self.query_plan(command, arguments)
			cur = self.cursor()
			if fancy:
				cur.setrowtrace(_apsw_row_tracer)
			ret = []
			if arguments is ():
				return cur.execute(command)
			else:
				return cur.execute(command, arguments)
		except apsw.SQLError as apswerr:
			if 'no such table: larch_' in str(apswerr):
				return self.execute(command, arguments=arguments, fancy=fancy, explain=explain, fail_silently=fail_silently, echo=echo, cte=True)
			else:
				if not fail_silently:
					l = logging.getScriber("db")
					l.critical("(execute) SQL:\n%s"%command)
					if arguments is not None:
						l.critical("Bindings:")
						l.critical(str(arguments))
				raise

	def dict(self, command, arguments=()):
		'''A convenience function for extracting a single row from an SQL query.
			
		:param command: A SQLite query. If the returned result is other than 1 row, 
		               an LarchError is raised.
		:param arguments: Values to bind to the SQLite command.
		'''
		try:
			if arguments is ():
				cur = self.execute(command,fancy=True)
			else:
				cur = self.execute(command, arguments, fancy=True)
			x = next(cur,None)
			if x is None:
				raise NoResultsError('query returned no rows, expected one row')
			y = next(cur,None)
			if y is not None:
				raise TooManyResultsError('query returned multiple rows, expected only one row')
			return x
		except apsw.SQLError as apswerr:
			l = logging.getScriber("db")
			l.critical("SQL:\n%s"%command)
			if arguments is not None:
				l.critical("Bindings:")
				l.critical(str(arguments))
			raise

	#def value(self, command, arguments=(), *, fail_silently=False, **kwargs):
	def value(self, *args, **kwargs):
		'''A convenience function for extracting a single value from an SQL query.
		
		:param command: A SQLite query.  If there is more than one result column
		                on the query, only the first column will be returned. If
						the returned result is other than 1 row, an LarchError is 
						raised.
		:param arguments: Values to bind to the SQLite command.
		'''
		cur = self.execute(*args, **kwargs)
		try:
			ret = next(cur)
		except StopIteration:
			raise NoResultsError('query returned no rows, expected one row')
		try:
			ret2 = next(cur)
		except StopIteration:
			return ret[0]
		else:
			raise TooManyResultsError('query returned multiple rows, expected only one row')
		#
		#		try:
		#			cur = self.cursor()
		#			ret = []
		#			if arguments is ():
		#				i = [j[0] for j in cur.execute(command, **kwargs)]
		#			else:
		#				i = [j[0] for j in cur.execute(command, arguments, **kwargs)]
		#			if len(i)>1:
		#				raise TooManyResultsError('query returned multiple rows, expected only one row')
		#			if len(i)==0:
		#			return i[0]
		#		except apsw.SQLError as apswerr:
		#			if not fail_silently:
		#				l = logging.getScriber("db")
		#				l.critical("(value) SQL:\n%s"%command)
		#				if arguments is not None:
		#					l.critical("Bindings:")
		#					l.critical(str(arguments))
		#			raise

	def row(self, command, arguments=(), *, fail_silently=False, fancy=False):
		'''A convenience function for extracting a single row from an SQL query.
			
			:param command:   A SQLite query.  If there is more than one result row
			                  on the query, only the first row will be returned. If
			                  the returned result is other than 1 row, an LarchError is
			                  raised.
			:param arguments: Values to bind to the SQLite command.
			:param fail_silently: If False, a summary of any failure is logged to the db logger
			:param fancy: If True, the a dict-like row tracer is built for the returned row
			
			'''
		try:
			if arguments is ():
				i = [j for j in self.execute(command, fancy=fancy)]
			else:
				i = [j for j in self.execute(command, arguments, fancy=fancy)]
			if len(i)>1:
				raise TooManyResultsError('query returned multiple rows, expected only one row')
			if len(i)==0:
				raise NoResultsError('query returned no rows, expected one row')
			return i[0]
		except apsw.SQLError as apswerr:
			if not fail_silently:
				l = logging.getScriber("db")
				l.critical("(value) SQL:\n%s"%command)
				if arguments is not None:
					l.critical("Bindings:")
					l.critical(str(arguments))
			raise

	def values(self, command, arguments=(), column_number=0, *, fail_silently=False, result_type=list):
		'''A convenience function for extracting a list from an SQL query.
			
			:param command: A SQLite query. If there is more than one result column
			                on the query, only a single column will be returned.
			:param arguments: Values to bind to the SQLite command.
			:param column_number: The zero-indexed column number to return, defaults to
								  zero (first column). Specify 'None' to get all columns
								  
			'''
		if isinstance(result_type, str):
			if result_type=='list':
				result_type=list
			elif result_type=='set':
				result_type=set
			elif result_type=='tuple':
				result_type=tuple
			else:
				raise larch.LarchError("incompatible result_type")
		if column_number is None:
			column_number = slice(0,None)
		try:
			cur = self.cursor()
			ret = []
			if arguments is ():
				i = result_type((j[column_number] for j in cur.execute(command)))
			else:
				i = result_type((j[column_number] for j in cur.execute(command, arguments)))
			return i
		except apsw.SQLError as apswerr:
			if not fail_silently:
				l = logging.getScriber("db")
				l.critical("(values) SQL:\n%s"%command)
				if arguments is not None:
					l.critical("Bindings:")
					l.critical(str(arguments))
			raise

	def array(self, command, arguments=(), *, n_rows=None, n_cols=None, fail_silently=False, cte=False):
		'''A convenience function for extracting an array from an SQL query.
			
			:param command: A SQLite query.
			:param arguments: Values to bind to the SQLite command.
			:param n_cols: The number of columns to return in the array, defaults to all.
			:param n_rows: The number of rows to return in the array, defaults to 1.
			'''
		import numpy
		try:
			cur = self.execute(command, arguments, cte=cte)
			if n_rows is None:
				n_rows = self.value("SELECT count(*) FROM ({})".format(command),arguments, cte=cte)
			if n_cols is None:
				try:
					n_cols = len(cur.description)
				except apsw.ExecutionCompleteError:
					return numpy.zeros([n_rows, 0])
			ret = numpy.zeros([n_rows, n_cols])
			n = 0
			for row in cur:
				if n>=n_rows:
					return ret
				try:
					ret[n,:] = row[:n_cols]
				except ValueError:
					ret[n,:] = [(None if isinstance(i,str) else i) for i in row[:n_cols]]
				n += 1
			return ret
		except apsw.SQLError as apswerr:
			if not fail_silently:
				l = logging.getScriber("db")
				l.critical("(values) SQL:\n%s"%command)
				if arguments is not None:
					l.critical("Bindings:")
					l.critical(str(arguments))
			raise

	def array_caseids(self, *, table=None, caseid=None, sort=True, n_cases=None):
		"""Extract the caseids from the DB based on preset queries.
		
		Generaly you won't need to specify any parameters to this method, as
		most values are determined automatically from the preset queries.
		However, if you need to override things for this array without changing
		the queries more permanently, you can use the input parameters to do so.
		Note that all parameters must be called by keyword, not as positional 
		arguments.
		
		Parameters
		----------
		tablename : str
			The caseids will be found in this table.
		caseid : str
			This sets the column name where the caseids can be found.
		sort : bool
			If true (the default) the resulting array will sorted in 
			ascending order.
		n_cases : int
			If you know the number of cases, you can specify it here to speed up
			the return of the results, particularly if the caseids query is complex.
			You can safely ignore this and the number of cases will be calculated
			for you. If you give the wrong number, an exception will be raised.
		
		Returns
		-------
		ndarray
			An int64 array of shape (n_cases,1).
		"""
		import numpy
		if table is None:
			table = self.queries.tbl_caseids()
		co_cols = [i.lower() for i in self.column_names(table)]
		if caseid is None:
			if 'caseid' in co_cols:
				caseid = 'caseid'
			else:
				caseid = co_cols[0]
		qry = "SELECT {} AS caseid FROM {}".format(caseid, table)
		if n_cases is None:
			n_cases = self.value("SELECT count(*) FROM {}".format(table))
		n_vars = 1
		case_slots = dict()
		caseids = numpy.zeros([n_cases,1], dtype='int64')
		n = 0
		self._array_idco_reader(qry, None, caseids)
		if sort:
			order = numpy.argsort(caseids[:,0])
			caseids = caseids[order,:]
		return caseids


	def array_idca(self, *vars, table=None, caseid=None, altid=None, altcodes=None, dtype='float64', sort=True, n_cases=None):
		"""Extract a set of idca values from the DB based on preset queries.
		
		Generaly you won't need to specify any parameters to this method beyond the
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
			The idca data will be found in this table, view, or self contained query (if
			the latter, it should be surrounded by parentheses).
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
		sort : bool
			If true (the default) the resulting arrays (both of them) will sorted in
			ascending order by caseid.
		n_cases : int
			If you know the number of cases, you can specify it here to speed up
			the return of the results, particularly if the caseids query is complex.
			You can safely ignore this and the number of cases will be calculated
			for you. If you give the wrong number, an exception will be raised.
		
		Returns
		-------
		data : ndarray
			An array with specified dtype, of shape (n_cases,len(altcodes),len(vars)).
		caseids : ndarray
			An int64 array of shape (n_cases,1).
			
		Examples
		--------
		Extract a cost and time array from the MTC example data:
		
		>>> import larch
		>>> db = larch.DB.Example()
		>>> x, c = db.array_idca('totcost','tottime')
		>>> x.shape
		(5029, 6, 2)
		>>> x[0]
		Array([[  70.63,   15.38],
		       [  35.32,   20.38],
		       [  20.18,   22.38],
		       [ 115.64,   41.1 ],
		       [   0.  ,   42.5 ],
		       [   0.  ,    0.  ]])

		"""
		import numpy
		if table is None:
			table = self.queries.tbl_idca()
		if altcodes is None:
			altcodes = self.alternative_codes()
		ca_cols = [i.lower() for i in self.column_names(table)]
		if caseid is None:
			if 'caseid' in ca_cols:
				caseid = 'caseid'
			else:
				caseid = ca_cols[0]
		if altid is None:
			if 'altid' in ca_cols:
				altid = 'altid'
			elif ca_cols[1]!=caseid:
				altid = ca_cols[1]
			else:
				altid = ca_cols[0]
		cols = "{} AS caseid, {} AS altid, ".format(caseid,altid) + ", ".join(vars)
		qry = "SELECT {} FROM {}".format(cols, table)
		if n_cases is None:
			n_cases = self.value("SELECT count(distinct {}) FROM {}".format(caseid,table))
		n_alts = len(altcodes)
		n_vars = len(vars)
		case_slots = dict()
		caseids = numpy.zeros([n_cases,1], dtype='int64')
		alt_slots = {a:n for n,a in enumerate(altcodes)}
		n = 0	
		result = numpy.zeros([n_cases,n_alts,n_vars], dtype=dtype)
		self._array_idca_reader(qry, result, caseids, altcodes)
		#for row in self.execute(qry):
		#	if row[0] not in case_slots:
		#		c = case_slots[row[0]] = n
		#		caseids[c] = row[0]
		#		n +=1
		#	else:
		#		c = case_slots[row[0]]
		#	try:
		#		a = alt_slots[row[1]]
		#	except KeyError:
		#		raise KeyError("alt {} appears in data but is not defined".format(row[1]))
		#	result[c,a,:] = row[2:]
		if sort:
			order = numpy.argsort(caseids[:,0])
			result = numpy.ascontiguousarray( result[order,:,:] )
			caseids = caseids[order,:]
		result = numpy.nan_to_num(result)
		from .array import Array
		result = result.view(Array)
		result.vars = vars
		return result, caseids


	def arraymap_idce(self, *vars):
		from .util.arraytools import label_to_index
		# load data and ids
		caseids = self.array_caseids().squeeze()
		ca_vars_str = ", ".join(vars)
		result_ce = self.array("SELECT {} FROM larch_idca".format(ca_vars_str), cte=True)
		ce_altids = self.array("SELECT altid FROM larch_idca", cte=True).astype(int).squeeze()
		ce_caseids = self.array("SELECT caseid FROM larch_idca", cte=True).astype(int).squeeze()
		# convert ids to indexes
		result_ce_caseindex = label_to_index(caseids, ce_caseids)
		result_ce_altindex = label_to_index(self.alternative_codes(), ce_altids)
		try:
			len_caseids = len(caseids)
		except TypeError:
			len_caseids = 0
		return (result_ce_caseindex, result_ce_altindex, result_ce, len_caseids, len(self.alternative_codes()))


	def array_idco(self, *vars, table=None, caseid=None, dtype='float64', sort=True, n_cases=None):
		"""Extract a set of idco values from the DB based on preset queries.
		
		Generaly you won't need to specify any parameters to this method beyond the
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
		tablename : str
			The idco data will be found in this table, view, or self contained query (if
			the latter, it should be surrounded by parentheses).
		caseid : str
			This sets the column name where the caseids can be found.
		dtype : str or dtype
			Describe the data type you would like the output array to adopt, probably 
			'int64', 'float64', or 'bool'.
		sort : bool
			If true (the default) the resulting arrays (both of them) will sorted in
			ascending order by caseid.
		n_cases : int
			If you know the number of cases, you can specify it here to speed up
			the return of the results, particularly if the caseids query is complex.
			You can safely ignore this and the number of cases will be calculated
			for you. If you give the wrong number, an exception will be raised.
		
		Returns
		-------
		data : ndarray
			An array with specified dtype, of shape (n_cases,len(vars)).
		caseids : ndarray
			An int64 array of shape (n_cases,1).
		"""
		import numpy
		if table is None:
			table = self.queries.tbl_idco()
		co_cols = [i.lower() for i in self.column_names(table)]
		if caseid is None:
			if 'caseid' in co_cols:
				caseid = 'caseid'
			else:
				caseid = co_cols[0]
		cols = "{} AS caseid, ".format(caseid) + ", ".join(vars)
		qry = "SELECT {} FROM {}".format(cols, table)
		if n_cases is None:
			n_cases = self.value("SELECT count(*) FROM {}".format(table))
		n_vars = len(vars)
		case_slots = dict()
		caseids = numpy.zeros([n_cases,1], dtype='int64')
		n = 0
		result = numpy.zeros([n_cases,n_vars], dtype=dtype)
		self._array_idco_reader(qry, result, caseids)
		#for row in self.execute(qry):
		#	if row[0] not in case_slots:
		#		c = case_slots[row[0]] = n
		#		caseids[c] = row[0]
		#		n +=1
		#	else:
		#		c = case_slots[row[0]]
		#	result[c,:] = row[1:]
		if sort:
			order = numpy.argsort(caseids[:,0])
			result = numpy.ascontiguousarray( result[order,:] )
			caseids = caseids[order,:]
		result = numpy.nan_to_num(result)
		from .array import Array
		result = result.view(Array)
		result.vars = vars
		return result, caseids

	def array_weight(self, *, table=None, caseid=None, var=None, dtype='float64', sort=True, n_cases=None):
		import numpy
		if table is None:
			table = self.queries.tbl_weight()
		co_cols = [i.lower() for i in self.column_names(table)]
		if caseid is None:
			if 'caseid' in co_cols:
				caseid = 'caseid'
			else:
				caseid = co_cols[0]
		if var is None:
			if 'weight' in co_cols:
				var = 'weight'
			elif 'wgt' in co_cols:
				var = 'wgt'
			elif co_cols[1]!=caseid:
				var = co_cols[1]
			else:
				var = co_cols[0]
		cols = "{} AS caseid, {} AS weight".format(caseid,var)
		qry = "SELECT {} FROM {}".format(cols, table)
		if n_cases is None:
			n_cases = self.value("SELECT count(*) FROM {}".format(table))
		n_vars = 1
		case_slots = dict()
		caseids = numpy.zeros([n_cases,1], dtype='int64')
		n = 0
		result = numpy.zeros([n_cases,n_vars], dtype=dtype)
		try:
			self._array_idco_reader(qry, result, caseids)
		except:
			print("result.shape=",result.shape)
			print("caseids.shape=",caseids.shape)
			print("qry=",qry)
			print("count_cases=",self.value("SELECT count(*) FROM {}".format(table)))
			raise
		#for row in self.execute(qry):
		#	if row[0] not in case_slots:
		#		c = case_slots[row[0]] = n
		#		caseids[c] = row[0]
		#		n +=1
		#	else:
		#		c = case_slots[row[0]]
		#	result[c,:] = row[1:]
		if sort:
			order = numpy.argsort(caseids[:,0])
			result = numpy.ascontiguousarray( result[order,:] )
			caseids = caseids[order,:]
		result = numpy.nan_to_num(result)
		from .array import Array
		result = result.view(Array)
		result.vars = [var,]
		return result, caseids

	def array_choice(self, *, var=None, table=None, caseid=None, altid=None, altcodes=None, dtype='float64', sort=True, n_cases=None):
		import numpy
		if table is None:
			table = self.queries.tbl_choice()
		if altcodes is None:
			altcodes = self.alternative_codes()
		ca_cols = [i.lower() for i in self.column_names(table)]
		if caseid is None:
			if 'caseid' in ca_cols:
				caseid = 'caseid'
			else:
				caseid = ca_cols[0]
		if altid is None:
			if 'altid' in ca_cols:
				altid = 'altid'
			elif ca_cols[1]!=caseid:
				altid = ca_cols[1]
			else:
				altid = ca_cols[0]
		if var is None:
			if 'choice' in ca_cols:
				var = 'choice'
			elif 'chosen' in ca_cols:
				var = 'chosen'
			elif ca_cols[2]!=caseid and ca_cols[2]!=altid:
				var = ca_cols[2]
			elif ca_cols[1]!=caseid and ca_cols[1]!=altid:
				var = ca_cols[1]
			else:
				var = ca_cols[0]
		cols = "{} AS caseid, {} AS altid, {} AS choice".format(caseid,altid,var)
		qry = "SELECT {} FROM {}".format(cols, table)
		if n_cases is None:
			n_cases = self.value("SELECT count(distinct {}) FROM {}".format(caseid,table))
		n_alts = len(altcodes)
		n_vars = 1
		case_slots = dict()
		caseids = numpy.zeros([n_cases,1], dtype='int64')
		alt_slots = {a:n for n,a in enumerate(altcodes)}
		n = 0	
		result = numpy.zeros([n_cases,n_alts,n_vars], dtype=dtype)
		try:
			self._array_idca_reader(qry, result, caseids, altcodes)
		except:
			print("result.shape=",result.shape)
			print("caseids.shape=",caseids.shape)
			print("qry=",qry)
			print("count_cases=",self.value("SELECT count(distinct {}) FROM {}".format(caseid,table)))
			raise
		#for row in self.execute(qry):
		#	if row[0] not in case_slots:
		#		c = case_slots[row[0]] = n
		#		caseids[c] = row[0]
		#		n +=1
		#	else:
		#		c = case_slots[row[0]]
		#	try:
		#		a = alt_slots[row[1]]
		#	except KeyError:
		#		raise KeyError("alt {} appears in data but is not defined".format(row[1]))
		#	result[c,a,:] = row[2:]
		if sort:
			order = numpy.argsort(caseids[:,0])
			result = numpy.ascontiguousarray( result[order,:,:] )
			caseids = caseids[order,:]
		result = numpy.nan_to_num(result)
		from .array import Array
		result = result.view(Array)
		result.vars = [var,]
		return result, caseids

	def array_avail(self, *, var=None, table=None, caseid=None, altid=None, altcodes=None, dtype='bool', sort=True, n_cases=None):
		import numpy
		if altcodes is None:
			altcodes = self.alternative_codes()
		if table is None:
			table = self.queries.tbl_avail()
			if table=="":
				if n_cases is None:
					n_cases = self.value("SELECT count(*) FROM {}".format(self.queries.tbl_caseids()))
				n_alts = len(altcodes)
				n_vars = 1
				result = numpy.ones([n_cases,n_alts,n_vars], dtype=dtype)
				return result, None
		ca_cols = [i.lower() for i in self.column_names(table)]
		if caseid is None:
			if 'caseid' in ca_cols:
				caseid = 'caseid'
			else:
				caseid = ca_cols[0]
		if altid is None:
			if 'altid' in ca_cols:
				altid = 'altid'
			elif ca_cols[1]!=caseid:
				altid = ca_cols[1]
			else:
				altid = ca_cols[0]
		if var is None:
			if 'avail' in ca_cols:
				var = 'avail'
			elif 'available' in ca_cols:
				var = 'available'
			elif ca_cols[2]!=caseid and ca_cols[2]!=altid:
				var = ca_cols[2]
			elif ca_cols[1]!=caseid and ca_cols[1]!=altid:
				var = ca_cols[1]
			else:
				var = ca_cols[0]
		cols = "{} AS caseid, {} AS altid, {} AS avail".format(caseid,altid,var)
		qry = "SELECT {} FROM {}".format(cols, table)
		if n_cases is None:
			n_cases = self.value("SELECT count(distinct {}) FROM {}".format(caseid,table))
		n_alts = len(altcodes)
		n_vars = 1
		case_slots = dict()
		caseids = numpy.zeros([n_cases,1], dtype='int64')
		alt_slots = {a:n for n,a in enumerate(altcodes)}
		n = 0	
		result = numpy.zeros([n_cases,n_alts,n_vars], dtype=dtype)
		try:
			self._array_idca_reader(qry, result, caseids, altcodes)
		except:
			print("result.shape=",result.shape)
			print("caseids.shape=",caseids.shape)
			print("qry=",qry)
			print("count_cases=",self.value("SELECT count(distinct {}) FROM {}".format(caseid,table)))
			raise
		#for row in self.execute(qry):
		#	if row[0] not in case_slots:
		#		c = case_slots[row[0]] = n
		#		caseids[c] = row[0]
		#		n +=1
		#	else:
		#		c = case_slots[row[0]]
		#	try:
		#		a = alt_slots[row[1]]
		#	except KeyError:
		#		raise KeyError("alt {} appears in data but is not defined".format(row[1]))
		#	result[c,a,:] = row[2:]
		if sort:
			order = numpy.argsort(caseids[:,0])
			result = numpy.ascontiguousarray( result[order,:,:] )
			caseids = caseids[order,:]
		from .array import Array
		result = result.view(Array)
		result.vars = [var,]
		return result, caseids

	def array_avail_blind(self, *, var=None, table=None, caseid=None, altid=None, altcodes=None, dtype='bool', sort=True, n_cases=None):
		"""
		Notes
		-----
		The blind version of this function does not attempt to find the number of cases before reading the data. 
		This results in an extra copy step and greater (double plus) memory usage, but may be faster if the larch_avail table
		is a complex query.
		"""
		import numpy
		if altcodes is None:
			altcodes = self.alternative_codes()
		if table is None:
			table = self.queries.tbl_avail()
			if table=="":
				if n_cases is None:
					n_cases = self.value("SELECT count(*) FROM {}".format(self.queries.tbl_caseids()))
				n_alts = len(altcodes)
				n_vars = 1
				result = numpy.ones([n_cases,n_alts,n_vars], dtype=dtype)
				return result, None
		ca_cols = [i.lower() for i in self.column_names(table)]
		if caseid is None:
			if 'caseid' in ca_cols:
				caseid = 'caseid'
			else:
				caseid = ca_cols[0]
		if altid is None:
			if 'altid' in ca_cols:
				altid = 'altid'
			elif ca_cols[1]!=caseid:
				altid = ca_cols[1]
			else:
				altid = ca_cols[0]
		if var is None:
			if 'avail' in ca_cols:
				var = 'avail'
			elif 'available' in ca_cols:
				var = 'available'
			elif ca_cols[2]!=caseid and ca_cols[2]!=altid:
				var = ca_cols[2]
			elif ca_cols[1]!=caseid and ca_cols[1]!=altid:
				var = ca_cols[1]
			else:
				var = ca_cols[0]
		cols = "{} AS caseid, {} AS altid, {} AS avail".format(caseid,altid,var)
		qry = "SELECT {} FROM {}".format(cols, table)
		try:
			result, caseids = self._array_idca_reader_blind(qry, numpy.dtype('bool').num, altcodes)
		except:
			print("qry=",qry)
			print("count_cases=",self.value("SELECT count(distinct {}) FROM {}".format(caseid,table)))
			raise
		#for row in self.execute(qry):
		#	if row[0] not in case_slots:
		#		c = case_slots[row[0]] = n
		#		caseids[c] = row[0]
		#		n +=1
		#	else:
		#		c = case_slots[row[0]]
		#	try:
		#		a = alt_slots[row[1]]
		#	except KeyError:
		#		raise KeyError("alt {} appears in data but is not defined".format(row[1]))
		#	result[c,a,:] = row[2:]
		if sort:
			order = numpy.argsort(caseids[:,0])
			result = numpy.ascontiguousarray( result[order,:,:] )
			caseids = caseids[order,:]
		from .array import Array
		result = result.view(Array)
		result.vars = [var,]
		return result, caseids

	def provision(self, needs, *, idca_avail_ratio_floor=0.1):
		from . import Model
		if isinstance(needs,Model):
			m = needs
			needs = m.needs()
		else:
			m = None
		import numpy
		provide = {}
		cases = None
		n_cases = None
		matched_cases = []
		log = self.logger()
		avail_ratio = 1.0
		
		# do avail first, to evaluate the benefit to use IDCE format
		if "Avail" in needs:
			if log:
				log.info("Provisioning Avail data...")
			provide["Avail"], c = self.array_avail_blind(n_cases=n_cases)
			if cases is None and c is not None:
				cases = c
				matched_cases += ["Avail",]
				n_cases = cases.shape[0]
			avail_ratio = numpy.mean(provide["Avail"])
		
		for key, req in needs.items():
			if log:
				log.info("Provisioning {} data...".format(key))
			if key=="Avail":
				continue
			elif key=="Weight":
				provide[key], c = self.array_weight(n_cases=n_cases)
			elif key=="Choice":
				provide[key], c = self.array_choice(n_cases=n_cases)
			elif key[-2:]=="CA":
				if avail_ratio>idca_avail_ratio_floor:
					provide[key], c = self.array_idca(*req.get_variables(),n_cases=n_cases)
				else:
					provide[key[:-2]+"CE"] = self.arraymap_idce(*req.get_variables())
			elif key[-2:]=="CO":
				provide[key], c = self.array_idco(*req.get_variables(),n_cases=n_cases)
			elif key=="Allocation":
				provide[key], c = self.array_idco(*req.get_variables(),n_cases=n_cases)
			if cases is None and c is not None:
				cases = c
				matched_cases += [key,]
				n_cases = cases.shape[0]
			elif c is not None:
				if numpy.all(cases==c):
					matched_cases += [key,]
				else:
					print("cases.shape=",cases.shape)
					print("c[{}].shape=".format(key),c.shape)
					raise LarchError("mismatched caseids in provisioning, ["+",".join(matched_cases)+"] do not match "+key)
		provide['caseids'] = cases
		if m is not None:
			return m.provision(provide)
		else:
			return provide



	def all_index_names(self):
		z = set()
		qry = "SELECT name FROM (SELECT * FROM sqlite_master UNION ALL SELECT * FROM sqlite_temp_master) WHERE type='index' ORDER BY name"
		for row in self.execute(qry):
			z.add(row[0])
		return z

	def all_sql_names(self):
		z = set()
		qry = "SELECT name FROM (SELECT * FROM sqlite_master UNION ALL SELECT * FROM sqlite_temp_master) ORDER BY name"
		for row in self.execute(qry):
			z.add(row[0])
		return z


	def display(self, stmt, arguments=(), file=None, header=True, format=None, countrows=False, shell=False, w=None, explode=False, **kwargs):
		if format is None and explode:
			format = "explode"
		if shell:
			sh = apsw.Shell(db=self)
			if header:
				sh.process_command(".header on")
			sh.process_command(".mode column")
			if w is not None:
				sh.process_command(".width "+" ".join([str(wx) for wx in w]))
			sh.process_sql(stmt, arguments if arguments!=() else None,)
		else:
			rows = 0
			try:
				if format=="html":
					print("<table>", file=file)
				cur = self.execute(stmt, arguments, **kwargs)
				iter = cur
				if header and format!="explode":
					try:
						descrip = cur.getdescription()
						if format=="html":
							print("<tr>", file=file)
							print("".join(["<th>{0!s:<11}</th>".format(j[0]) for j in descrip]), file=file)
							print("</tr>", file=file)
						else:
							print("\t".join(["{0!s:<11}".format(j[0]) for j in descrip]), file=file)
					except apsw.ExecutionCompleteError:
						print("empty table")
				if format=="explode":
					print("-"*80, file=file)
				for i in iter:
					if format=="html":
						print("<tr>", file=file)
						print("".join(["<td>{0!s:<11}</td>".format(j) for j in i]), file=file)
						print("</tr>", file=file)
					elif format=="explode":
						for j,jh in zip(i,cur.getdescription()):
							print("{1!s:>40}: {0!s:<20}".format(j,jh[0]), file=file)
						print("-"*80, file=file)
					else:
						print("\t".join(["{0!s:<11}".format(j) for j in i]), file=file)
					rows = rows+1
				if format=="html":
					print("</table>", file=file)
			except apsw.SQLError as apswerr:
				l = logging.getScriber("db")
				l.critical("SQL:\n%s"%stmt)
				if arguments is not None:
					l.critical("Bindings:")
					l.critical(str(arguments))
				raise
			if countrows:
				return rows

	def table_info(self,table,schema='main',**kwargs):
		'''A convenience function to replace `display('PRAMGA <schema>.table_info(<table>);', **kwargs)`.'''
		if len(kwargs)==0:
			df = self.dataframe('PRAGMA {}.table_info({});'.format(schema,table))
			max_lens = dict(
				ci=max(3,max(len(str(i)) for i in df['cid'])),
				nm=max(4,max(len(str(i)) for i in df['name'])),
				ty=max(4,max(len(str(i)) for i in df['type'])),
				nn=max(7,max(len(str(i)) for i in df['notnull'])),
				df=max(7,max(len(str(i)) for i in df['dflt_value'])),
				pk=max(2,max(len(str(i)) for i in df['pk'])),
				)
			print("{0:-<{ci}s} {1:-<{nm}s} {2:-<{ty}s} {3:-<{nn}s} {4:-<{df}s} {5:-<{pk}s}".format('','','','','','',**max_lens))
			print("{0!s:{ci}s} {1!s:{nm}s} {2!s:{ty}s} {3!s:{nn}s} {4!s:{df}s} {5!s:{pk}s}".format('cid','name','type','notnull','default','pk',**max_lens))
			print("{0:-<{ci}s} {1:-<{nm}s} {2:-<{ty}s} {3:-<{nn}s} {4:-<{df}s} {5:-<{pk}s}".format('','','','','','',**max_lens))
			for rownum,row in df.iterrows():
				print("{0!s:{ci}s} {1!s:{nm}s} {2!s:{ty}s} {3!s:{nn}s} {4!s:{df}s} {5!s:{pk}s}".format(*row,**max_lens))
			print("{0:-<{ci}s} {1:-<{nm}s} {2:-<{ty}s} {3:-<{nn}s} {4:-<{df}s} {5:-<{pk}s}".format('','','','','','',**max_lens))
		else:
			return self.display('PRAGMA {}.table_info({});'.format(schema,table), **kwargs)

	def index_info(self,index,schema='main',**kwargs):
		'''A convenience function to replace `display('PRAMGA <schema>.index_info(<index>);', **kwargs)`.'''
		return self.display('PRAGMA {}.index_info({});'.format(schema,index), **kwargs)

	def table_schema(self,table,**kwargs):
		return self.value("SELECT sql FROM sqlite_master WHERE name='{0}' UNION ALL SELECT sql FROM sqlite_temp_master WHERE name='{0}';".format(table), **kwargs)

	def table_columns(self,table,with_affinity=False, schema='main'):
		if with_affinity:
			names = self.values('PRAGMA {}.table_info({});'.format(schema,table), column_number=1)
			affin = self.values('PRAGMA {}.table_info({});'.format(schema,table), column_number=2)
			return ["{} {}".format(i,j) for i,j in zip(names,affin)]
		else:
			return self.values('PRAGMA {}.table_info({});'.format(schema,table), column_number=1)

	def table_indexes(self, table, schema='main'):
		return self.values('PRAGMA {}.index_list({});'.format(schema,table), column_number=1)

	def database_list(self,**kwargs):
		'''A convenience function to replace `display('PRAMGA database_list;', **kwargs)`.'''
		if 'type' not in kwargs:
			return self.display('PRAGMA database_list;', **kwargs)
		if kwargs['type'] in (list, 'list'):
			return self.values('PRAGMA database_list;', column_number=1)
		if kwargs['type'] in (dict, 'dict'):
			x = {}
			for row in self.execute('PRAGMA database_list;'):
				x[row[1]] = row[2]
			return x
		raise TypeError("type not valid")


	def seer(self, file=None, counts=False, **kwargs):
		'''Display a variety of information about the DB connection in an HTML report.
		
		Parameters
		----------
		file : str, optional
			A name for the HTML file that will be created. If not given, a temporary
			file will automatically be created.
		counts : bool, optional
			If true, the number of rows in each table is calculated. This may take a 
			long time if the database is large.
			
		Notes
		-----
		The report will pop up in Chrome or a default browser after it is generated.
		
		'''
		import time
		css = """
		table {border-collapse:collapse;} 
		table, th, td {border: 1px solid #999999; padding:2px;}
		.greysmallcap {color:#888888; font-variant: small-caps;}
		.indented {padding-left:20px; margin-top:10px; padding-bottom:10px; background-color:rgba(0,0,255,0.05);}
		.header { color:#888888; font-style:italic; text-align: right; }
		"""
		if file is None:
			file = utilities.TemporaryHtml(css)
		file.write("<body>\n")

		file.write('<div class="header">')
		file.write('DB.seer() at {}'.format(time.strftime("%A %B %d %Y - %I:%M:%S %p")))
		file.write('</div>')

		try:
			qrys = self.queries.info(format='html')
		except:
			pass
		else:
			file.write("<h1>Queries</h1>\n")
			file.write(qrys)
			if counts:
				file.write("\nQueries represent {} cases\n".format(self.nCases()))
		file.write("<h1>Databases</h1>\n")
		self.database_list(file=file, format='html', **kwargs)
		dbs = self.database_list(type=list)
		for dbname in dbs:
			db_tables = self.table_names(dbname, views=False)
			db_views = self.view_names(dbname)
			file.write('<div class="indented">\n')
			file.write('<h2>{} <span class="greysmallcap">database</span></h2>\n'.format(dbname))
			for dbtable in db_tables:
				file.write('<div class="indented">\n')
				file.write('<h3>{} <span class="greysmallcap">table</span></h3>\n'.format(dbtable))
				self.table_info(dbtable, schema=dbname, file=file, format='html', **kwargs)
				file.write('<pre class="sql">')
				file.write( self.value("SELECT sql FROM {}.sqlite_master WHERE name=?".format(dbname),(dbtable,)) )
				file.write('</pre>')
				if counts:
					nrows = self.value("SELECT count(*) FROM {}.{}".format(dbname,dbtable))
					file.write('<br/>Table contains {} rows'.format(nrows))
				indexes = self.table_indexes(dbtable, schema=dbname)
				if len(indexes):
					file.write('<div class="indented">\n')
					for idx in indexes:
						file.write('<h4>{} <span class="greysmallcap">index on {}</span></h4>\n'.format(idx, dbtable))
						self.index_info(idx, schema=dbname, file=file, format='html', **kwargs)
					file.write('</div>\n')
				file.write('</div>\n')
			if len(db_tables)==0:
				file.write('<h3><span class="greysmallcap">no tables in this database</span></h3>\n')
			for dbview in db_views:
				file.write('<div class="indented">\n')
				file.write('<h3>{} <span class="greysmallcap">view</span></h3>\n'.format(dbview))
				self.table_info(dbview, schema=dbname, file=file, format='html', **kwargs)
				file.write('</div>\n')
			file.write('</div>\n')
		file.write("</body>\n")
		file.flush()
		try:
			file.view()
		except AttributeError:
			pass
		return file
	

	def attach(self, sqlname, filename):
		'''Attach another SQLite database.
		
		Parameters
		----------
		sqlname : str
			The name SQLite will use to reference the other database.
		filename : str
			The filename or URI to attach.

		Notes
		-----
		If the other database is already attached, or if the name is already taken by another
		attached database, the command will be ignored. Otherwise, this command is the
		equivalent of executing::

			ATTACH filename AS sqlname;

		.. seealso:: :meth:`DB.detach`
		'''
		if sqlname in self.values('PRAGMA database_list;', column_number=1):
			return
		if filename in self.values('PRAGMA database_list;', column_number=2):
			return
		self.execute("ATTACH '{}' AS {};".format(filename,sqlname))

	def detach(self, sqlname):
		'''Detach another SQLite database.

		Parameters
		----------
		sqlname : str
			The name SQLite uses to reference the other database.

		Notes
		-----
		If the name is not an attached database, the command will be ignored. Otherwise,
		this command is the equivalent of executing::

			DETACH sqlname;

		.. seealso:: :meth:`DB.attach`
	
		'''
		if sqlname not in self.values('PRAGMA database_list;', column_number=1):
			return
		self.execute("DETACH {};".format(sqlname))

	def table_names(self, dbname="main", views=True):
		qry = "SELECT name FROM "
		if dbname.lower()=='temp' or dbname.lower()=='temporary':
			qry += "sqlite_temp_master"
		elif dbname.lower()!='main':
			qry += dbname+".sqlite_master"
		else:
			qry += "sqlite_master"
		qry += " WHERE type='table'"
		if views:
			qry += " OR type='view';"
		else:
			qry += ";"
		return self.values(qry)

	def view_names(self, dbname="main"):
		qry = "SELECT name FROM "
		if dbname.lower()=='temp' or dbname.lower()=='temporary':
			qry += "sqlite_temp_master"
		elif dbname.lower()!='main':
			qry += dbname+".sqlite_master"
		else:
			qry += "sqlite_master"
		qry += " WHERE type='view';"
		return self.values(qry)

	def webpage(self, stmt, arguments=(), *, file=None, title=None):
		'''A convenience function for extracting a single value from an SQL query.
			
		:param stmt:      A SQL query to be evaluated.
		:param arguments: Values to bind to the SQLite query.
		:param file:      A file-like object whereupon to write the HTML table. If None
		                  (the default), a temporary named html file is created.
		:param file:      An optional title to put at the top of the table as an <h1>.
		'''
		if file is None:
			file = utilities.TemporaryHtml("table {border-collapse:collapse;} table, th, td {border: 1px solid #999999; padding:2px;}")
		file.write("<body>\n")
		if title:
			file.write("<h1>{}</h1>\n".format(title))
		self.display(stmt, arguments, file=file, format='html')
		file.write("</body>\n")
		file.flush()
		try:
			file.view()
		except AttributeError:
			pass
		return file

	def slick(self, stmt, arguments=(), *, file=None, title="Untitled"):
		'''A convenience function for extracting a table as a viewable html page from an SQL query.
			
		:param stmt:      A SQL query to be evaluated.
		:param arguments: Values to bind to the SQLite query.
		:param file:      A file-like object whereupon to write the table. If None
		                  (the default), a temporary named html file is created.
		:param title:     An optional title for the page.
		'''
		from .util.slickgrid import display_slickgrid
		cur = self.execute(stmt, arguments)
		try:
			descrip = cur.getdescription()
		except apsw.ExecutionCompleteError:
			return "no results\n{0!s}\n{1}".format(arguments,stmt)
		colnames = ["{0!s}".format(j[0]) for j in descrip]
		datarows = []
		for i in cur:
			datarows.append(["{0!s}".format(j).replace("\n"," ") for j in i])
		display_slickgrid(filename=file, title=title, column_names=colnames, datarows=datarows)


	def dataframe(self, stmt, arguments=(), index_col=None, coerce_float=True):
		"""Reads data from a query into a pandas DataFrame.
					
		:param self:        An existing larch.DB object.
		:param stmt:        The query to be read.
		:param arguments:   Any arguments to the query.
		:param index_col:   Provide an index_col parameter to use one of the
							columns as the index. Otherwise will be 0 to len(results) - 1.
		:param coerce_float:Attempt to convert values to non-string, non-numeric objects (like
							decimal.Decimal) to floating point.
		:result DataFrame:  A pandas DataFrame with the query results table.
		"""
		try:
			import pandas
		except ImportError:
			raise LarchError("creating a dataframe requires the pandas module")
		cur = self.cursor()
		try:
			cur.execute(stmt, arguments)
		except apsw.SQLError as apswerr:
			l = logging.getScriber("db")
			l.critical("SQL:\n%s"%stmt)
			if arguments!=():
				l.critical("Bindings:")
				l.critical(str(arguments))
			raise
		try:
			columns = [col_desc[0] for col_desc in cur.description]
		except apsw.ExecutionCompleteError:
			return None
		rows = list(cur)
		cur.close()
		self.commit()
		result = pandas.DataFrame.from_records(rows, columns=columns, coerce_float=coerce_float)
		if index_col is not None:
			result = result.set_index(index_col)
		return result

	def add_column(self,table,column,extra="",default_value=None):
		"""A convenience method for adding a column to a table. 
		If the column already exists, this command will be silently ignored."""
		if extra != "":
			sp = " "
		else:
			sp = ""
		try:
			self.execute("ALTER TABLE {} ADD COLUMN {}{}{};".format(table,column,sp,extra), fail_silently=True)
		except apsw.SQLError as err:
			if "duplicate" not in str(err): raise
		else:
			if default_value is not None:
				self.execute("UPDATE {} SET {}={};".format(table,column,default_value), fail_silently=True)

	def query_plan(self, stmt, arguments=(), **kwds):
		self.display("EXPLAIN QUERY PLAN "+stmt, arguments=arguments, **kwds)
	
	
#	def descriptive_statistics_co(self, *columns):
#		"""
#		Generate a dataframe of descriptive statistics on the idco data.
#		"""
#		try:
#			import pandas
#		except ImportError:
#			raise LarchError("creating a dataframe requires the pandas module")
#		keys = set()
#		stats = None
#		for u in columns:
#			if u in keys:
#				continue
#			else:
#				keys.add(u)
#			qry = """
#				SELECT
#				'{0}' AS DATA,
#				min({0}) AS MINIMUM,
#				max({0}) AS MAXIMUM,
#				avg({0}) AS MEAN,
#				stdev({0}) AS STDEV
#				FROM larch_idco
#				""".format(u)
#			s = self.dataframe(qry)
#			s = s.set_index('DATA')
#			if stats is None:
#				stats = s
#			else:
#				stats = pandas.concat([stats,s])
#		return stats

	def matrix_library(self, display=None):
		x = []
		n = 0
		while True:
			m = Facet.matrix_library(self,n)
			if m is None: break
			x.append(m)
			if display: display(str(m))
			n += 1
		return x

	queries = property(Facet._get_queries,lambda self,x: self._set_queries(x,x,self))

	def load_queries(self, facetname=None):
		if facetname is None:
			f = self.list_queries()
			if len(f)==1:
				facetname = f[0]
			else:
				for i in f:
					if i.upper()=='DEFAULT':
						facetname = i
						break
		if facetname is not None:
			assert(isinstance(facetname,str))
			self.queries = self.store["queries:{}".format(facetname)]
			self.queries.set_validator(self)
			self.active_facet = facetname

	def save_queries(self, facetname=None):
		if facetname is None:
			facetname = self.active_facet
		assert(isinstance(facetname,str))
		if facetname=='':
			facetname = 'default'
		self.store["queries:{}".format(facetname)] = self.queries

	def del_queries(self, facetname):
		assert(isinstance(facetname,str))
		del self.store["queries:{}".format(facetname)]

	def list_queries(self):
		return self.store.keys_like("queries")


def _count(start=0):
    n = start
    while 1:
        yield n
        n += 1


class _apsw_row_tracer(dict):
	'''A class to use in the setrowtrace function of apsw'''
	def __init__(self, cursor, values):
		self.values = list(values)
		dict.__init__(self, zip((t[0][t[0].find('.')+1:].lower() for t in cursor.getdescription()), _count()))
	def __contains__ (self, k):
		if isinstance(k,int):
			return True if len(self.values)>k and -len(self.values)<=k else False
		return dict.__contains__(self, k.lower())
	def __getitem__ (self, k):
		if isinstance(k,int):
			return self.values[k]
		return self.values[ dict.__getitem__(self, k.lower()) ]
	def __setitem__ (self, k, v):
		if k=="values":
			dict.__setattr__(self, k, v)
		if isinstance(k,int):
			self.values[k] = v
			return
		if k.lower() in self:
			self.values[ dict.__getitem__(self, k.lower()) ] = v
		else:
			dict.__setitem__(self, k.lower(), len(self.values))
			self.values.append(v)
	__getattr__ = __getitem__
	__setattr__ = __setitem__

