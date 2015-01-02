
from . import apsw
from . import utilities
from .core import SQLiteDB, Facet, FacetError, LarchError, QuerySetSimpleCO
from .exceptions import NoResultsError, TooManyResultsError
from . import logging
import time
import os
import sys


_docstring_sql_alts =\
"An SQL query that evaluates to an elm_alternatives table.\n\
\n\
Column 1: id (integer) a key for every alternative observed in the sample\n\
Column 2: name (text) a name for each alternative\n\
";


_docstring_sql_idco =\
"An SQL query that evaluates to an elm_idco table.\n\
\n\
Column 1: caseid (integer) a key for every case observed in the sample\n\
Column 2+ can contain any explanatory data, typically numeric data, although non-numeric data is allowable.";


_docstring_sql_idca =\
"An SQL query that evaluates to an elm_idca table.\n\
\n\
Column 1: caseid (integer) a key for every case observed in the sample\n\
Column 2: altid (integer) a key for each alternative available in this case\n\
Column 3+ can contain any explanatory data, typically numeric data, although non-numeric data is allowable.\n\
If no columns have the name 'caseid' and 'altid', elm will use the first two columns, repsectively.\n\
A query with less than two columns will raise an exception.";


_docstring_sql_choice =\
"An SQL query that evaluates to an elm_choice table.\n\
\n\
Column 1: caseid (integer) a key for every case observed in the sample\n\
Column 2: altid (integer) a key for each alternative available in this case\n\
Column 3: choice (numeric, typically 1.0 but could be other values)\n\
\n\
If an alternative is not chosen for a given case, it can have a zero choice value or \
it can simply be omitted from the result.";


_docstring_sql_avail =\
"An SQL query that evaluates to an elm_avail table.\n\
\n\
Column 1: caseid (integer) a key for every case observed in the sample\n\
Column 2: altid (integer) a key for each alternative available in this case\n\
Column 3: avail (boolean) evaluates as 1 or true when the alternative is available, 0 otherwise\n\
\n\
If an alternative is not available for a given case, it can have a zero avail value or \
it can simply be omitted from the result. If no query is given, it is assumed that \
all alternatives are available in all cases.";


_docstring_sql_weight =\
"An SQL query that evaluates to an elm_weight table.\n\
\n\
Column 1: caseid (integer) a key for every case observed in the sample\n\
Column 2: weight (numeric) a weight associated with each case\n\
\n\
If no weights are given, they are assumed to be all equal.";







class DB(utilities.FrozenClass, Facet, apsw.Connection):

	sql_alts   = property(Facet.qry_alts  , None, None, _docstring_sql_alts   )
	sql_idco   = property(Facet.qry_idco  , None, None, _docstring_sql_idco   )
	sql_idca   = property(Facet.qry_idca  , None, None, _docstring_sql_idca   )
	sql_choice = property(Facet.qry_choice, None, None, _docstring_sql_choice )
	sql_avail  = property(Facet.qry_avail , None, None, _docstring_sql_avail  )
	sql_weight = property(Facet.qry_weight, None, None, _docstring_sql_weight )

	def __init__(self, filename=None, readonly=False):
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
		# Load SQLite extended math functions
		trypath = os.path.split(__file__)[0]
		trypath = utilities.path_shrink_until_exists(trypath)
		dir_files = os.listdir(trypath)
#		for i in dir_files:
#			if "elmsqlite3extension" in i or "elmsqlhaversine" in i:
#				extend_path = os.path.join(trypath,i)
#				try:
#					apsw.Connection.loadextension(self,extend_path)
#				except:
#					self._attempted_loadextension = extend_path
#					print("failed to load sqlite extension:", extend_path)
		if self.source_filename == "":
			self.source_filename = filename
		self.working_name = self.source_filename
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
		try:
			#self.load_queries()
			pass
		except apsw.SQLError:
			pass

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
		raise TypeError('only allows list or dict')

	@staticmethod
	def Copy(source, destination="file:larchdb?mode=memory"):
		'''Create a copy of a database and link it to a DB object.

		:param source: The source database.
		:type source:  str
		:param destination: The destination database.
		:type destination: str
		:returns: A DB object with an open connection to the destination DB.
		
		It is often desirable to work on a copy of your data, instead of working
		with the original file. If you data file is not very large and you are 
		working with multiple models, there can be significant speed advantages
		to copying the entire database into memory first, and then working on it
		there, instead of reading from disk every time you want data.
		
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
	def ExampleDirectory():
		'''Returns the directory location of the example data files.
		
		ELM comes with a few example data sets, which are used in documentation
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
	def Example(dataset='MTC'):
		'''Generate an example data object in memory.
		
		Larch comes with a few example data sets, which are used in documentation
		and testing. It is important that you do not edit the original data, so
		this function copies the data into an in-memory database, which you can
		freely edit without damaging the original data.
		'''
		import os.path
		TEST_DIR = DB.ExampleDirectory()
		TEST_DATA = {
		  'MTC':os.path.join(TEST_DIR,"MTCWork.elmdata"),
		  'TWINCITY':os.path.join(TEST_DIR,"TwinCityQ.elmdata"),
		  'SWISSMETRO':os.path.join(TEST_DIR,"swissmetro.elmdata"),
		  }
		if dataset.upper() not in TEST_DATA:
			raise LarchError("Example data set %s not found"%dataset)
		return DB.Copy(TEST_DATA[dataset.upper()])


	@staticmethod
	def CSV_idco(filename, caseid="_rowid_", choice=None, weight="_equal_", tablename="data", savename=None, alts={}, safety=True):
		'''Creates a new larch DB based on a CSV data file.
		
		The input data file should be an idco data file, with the first line containing the column headings.
		
		:param filename:    File name (absolute or relative) for CSV (or other text-based delimited) source data.
		:param caseid:      Column name that contains the unique case id's. If the data is in idco format, case id's can
		                    be generated automatically based on line numbers, by using the reserved keyword '_rowid_'.
		:param choice:      Column name that contains the id of the alternative that is selected (if applicable). If not
		                    given, no sql_choice table will be autogenerated.
		:param weight:      Column name of the weight for each case. Use the keyword '_equal_' to defaults to equal weights.
		:param tablename:   The name of the sql table into which the data is to be imported. Do not give a reserved name 
							(i.e. any name beginning with *sqlite* or *elm*).
		:param savename:    If not None, the name of the location to save the elmdb file that is created.
		:param alts:        A dictionary with keys of alt codes, and values of (alt name, avail column, choice column) tuples.
		                    If *choice* is given, the third item in the tuple is ignored and can be omitted.
		:param safety:      If true, all alternatives that are chosen, even if not given in *alts*, will be
							automatically added to the alternatives table.
		
		:result:            A larch DB object
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
		#d.sql_alts = "SELECT * FROM csv_alternatives"
		d.queries.set_alts_query("SELECT * FROM csv_alternatives")
		#d.sql_idco = "SELECT {1} AS caseid, * FROM {0}".format(tablename,caseid)
		d.queries.set_idco_query("SELECT {1} AS caseid, * FROM {0}".format(tablename,caseid))
		#d.sql_idca = ""
		if choice is None:
			#choice_subquerys = ["SELECT caseid, {0} AS altid, {1} AS choice FROM elm_idco".format(code,info[2])
			#				   for code,info in alts.items()]
			#d.sql_choice = " UNION ALL ".join(choice_subquerys)
			d.queries.set_choice_column_map({code:info[2] for code,info in alts.items()})
		else:
			d.queries.set_choice_column(choice)
			#d.sql_choice = "SELECT caseid, {0} AS altid, 1 AS choice FROM elm_idco;".format(choice)
		#avail_subquerys = ["SELECT caseid, {0} AS altid, {1} AS avail FROM elm_idco".format(code,info[1])
		#                   for code,info in alts.items()]
		#d.sql_avail = " UNION ALL ".join(avail_subquerys)
		d.queries.set_avail_column_map({code:info[1] for code,info in alts.items()})
		if weight=="_equal_":
			#d.sql_weight = ""
			pass
		else:
			#d.sql_weight = "SELECT caseid, {0} AS weight FROM elm_idco;".format(weight)
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
	
	def import_csv(self, rawdata, table="data", drop_old=False, progress_callback=None):
		'''Import raw csv or tab-delimited data into SQLite.
		
		:param rawdata:     The absolute path to the raw csv or tab delimited data file.
		:param table:       The name of the table into which the data is to be imported
		:param drop_old:    Bool= drop old data table if it already exists?
		:param progress_callback: If given, this callback function takes a single integer
		                    as an argument and is called periodically while loading
		                    with the current precentage complete.
		
		:result:            A list of column headers from the imported csv file
		'''
		eL = logging.getScriber("db")
		eL.debug("Importing Data...")
		from .utilities import prepare_import_headers
		headers, csvReader, smartFile = prepare_import_headers(rawdata)
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
		lastscreenupdate = time.time()
		self.execute("BEGIN TRANSACTION;")
		for i in csvReader:
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
		return headers

	def import_dbf(self, rawdata, table="data", drop_old=False):
		'''Imports data from a DBF file into an existing larch DB.
			
			:param rawdata:     The path to the raw DBF data file.
			:param table:       The name of the table into which the data is to be imported
			:param drop_old:    Bool= drop old data table if it already exists?
			
			:result:            A list of column headers from imported csv file
			
			Note: this method requires the dbfpy module (available using pip).
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
			
			:param rawdataframe: An existing pandas dataframe.
			:param table:        The name of the table into which the data is to be imported
			:param if_exists:    Should be one of {‘fail’, ‘replace’, ‘append’}. If the table
								 does not exist this parameter is ignored, otherwise,
								 *fail*: If table exists, raise a ValueError exception.
								 *replace*: If table exists, drop it, recreate it, and insert data.
								 *append*: If table exists, insert data.
			
			:result:             A list of column headers from imported pandas dataframe
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

	def execute(self, command, arguments=(), fancy=False, explain=False, fail_silently=False, echo=False):
		'''A convenience function wrapping cursor generation and command 
		   execution for simple queries.
		
		:param command: An SQLite command.
		:param arguments: Values to bind to the SQLite command.
		:param fancy: If true, return rows as dict-type objects that can be indexed
		              by row headings instead of integer index positions.
		:param explain: If true, print the EXPLAIN QUERY PLAN results before executing.
		'''
		try:
			if echo:
				l = logging.getScriber("db")
				l.critical("(execute) SQL:\n%s"%command)
				if arguments is not None and arguments!=():
					l.critical("Bindings:")
					l.critical(str(arguments))
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

	def value(self, command, arguments=(), *, fail_silently=False):
		'''A convenience function for extracting a single value from an SQL query.
		
		:param command: A SQLite query.  If there is more than one result column
		                on the query, only the first column will be returned. If
						the returned result is other than 1 row, an LarchError is 
						raised.
		:param arguments: Values to bind to the SQLite command.
		'''
		try:
			cur = self.cursor()
			ret = []
			if arguments is ():
				i = [j[0] for j in cur.execute(command)]
			else:
				i = [j[0] for j in cur.execute(command, arguments)]
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

	def array(self, command, arguments=(), *, n_rows=None, n_cols=None, fail_silently=False):
		'''A convenience function for extracting an array from an SQL query.
			
			:param command: A SQLite query.
			:param arguments: Values to bind to the SQLite command.
			:param n_cols: The number of columns to return in the array, defaults to all.
			:param n_rows: The number of rows to return in the array, defaults to 1.
			'''
		import numpy
		try:
			cur = self.cursor().execute(command, arguments)
			if n_cols is None:
				try:
					n_cols = len(cur.description)
				except apsw.ExecutionCompleteError:
					return numpy.zeros([n_rows, 0])
			if n_rows is None:
				n_rows = self.value("SELECT count(*) FROM ({})".format(command),arguments)
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

	def array_idca(self, vars, table=None, caseid=None, altid=None, altcodes=None, dtype='float64', sort=True, n_cases=None):
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
			result = result[order,:,:]
			caseids = caseids[order,:]
		result = numpy.nan_to_num(result)
		from .array import Array
		result = result.view(Array)
		result.vars = vars
		return result, caseids

	def array_idco(self, vars, table=None, caseid=None, dtype='float64', sort=True, n_cases=None):
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
			result = result[order,:]
			caseids = caseids[order,:]
		result = numpy.nan_to_num(result)
		from .array import Array
		result = result.view(Array)
		result.vars = vars
		return result, caseids

	def array_weight(self, table=None, caseid=None, var=None, dtype='float64', sort=True, n_cases=None):
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
			result = result[order,:]
			caseids = caseids[order,:]
		result = numpy.nan_to_num(result)
		from .array import Array
		result = result.view(Array)
		result.vars = [var,]
		return result, caseids

	def array_choice(self, var=None, table=None, caseid=None, altid=None, altcodes=None, dtype='float64', sort=True, n_cases=None):
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
			result = result[order,:,:]
			caseids = caseids[order,:]
		result = numpy.nan_to_num(result)
		from .array import Array
		result = result.view(Array)
		result.vars = [var,]
		return result, caseids

	def array_avail(self, var=None, table=None, caseid=None, altid=None, altcodes=None, dtype='bool', sort=True, n_cases=None):
		import numpy
		if table is None:
			table = self.queries.tbl_avail()
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
			result = result[order,:,:]
			caseids = caseids[order,:]
		from .array import Array
		result = result.view(Array)
		result.vars = [var,]
		return result, caseids

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
		n_cases = None
		matched_cases = []
		log = self.logger()
		for key, req in needs.items():
			if log:
				log.info("Provisioning {} data...".format(key))
			if key=="Avail":
				provide[key], c = self.array_avail(n_cases=n_cases)
			elif key=="Weight":
				provide[key], c = self.array_weight(n_cases=n_cases)
			elif key=="Choice":
				provide[key], c = self.array_choice(n_cases=n_cases)
			elif key[-2:]=="CA":
				provide[key], c = self.array_idca(vars=req.get_variables(),n_cases=n_cases)
			elif key[-2:]=="CO":
				provide[key], c = self.array_idco(vars=req.get_variables(),n_cases=n_cases)
			if cases is None:
				cases = c
				matched_cases += [key,]
				n_cases = cases.shape[0]
			else:
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


	def display(self, stmt, arguments=(), file=None, header=True, format=None, countrows=False, shell=False, w=None):
		if shell:
			sh = apsw.Shell(db=self)
			if header:
				sh.process_command(".header on")
			sh.process_command(".mode column")
			if w is not None:
				sh.process_command(".width "+" ".join(w))
			sh.process_sql(stmt, arguments if arguments!=() else None,)
		else:
			rows = 0
			try:
				if format=="html":
					print("<table>", file=file)
				cur = self.cursor()
				iter = cur.execute(stmt, arguments)
				if header:
					try:
						descrip = cur.getdescription()
						if format=="html":
							print("<tr>", file=file)
							print("".join(["<th>{0!s:<11}</th>".format(j[0]) for j in descrip]), file=file)
							print("</tr>", file=file)
						else:
							print("\t".join(["{0!s:<11}".format(j[0]) for j in descrip]), file=file)
					except apsw.ExecutionCompleteError:
						pass
				for i in iter:
					if format=="html":
						print("<tr>", file=file)
						print("".join(["<td>{0!s:<11}</td>".format(j) for j in i]), file=file)
						print("</tr>", file=file)
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

	def table_info(self,table,**kwargs):
		'''A convenience function to replace `display('PRAMGA table_info(<table>);', **kwargs)`.'''
		return self.display('PRAGMA table_info({});'.format(table), **kwargs)

	def table_schema(self,table,**kwargs):
		return self.value("SELECT sql FROM sqlite_master WHERE name='{0}' UNION ALL SELECT sql FROM sqlite_temp_master WHERE name='{0}';".format(table), **kwargs)

	def table_columns(self,table,with_affinity=False):
		if with_affinity:
			names = self.values('PRAGMA table_info({});'.format(table), column_number=1)
			affin = self.values('PRAGMA table_info({});'.format(table), column_number=2)
			return ["{} {}".format(i,j) for i,j in zip(names,affin)]
		else:
			return self.values('PRAGMA table_info({});'.format(table), column_number=1)

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

	def attach(self, sqlname, filename):
		if sqlname in self.values('PRAGMA database_list;', column_number=1):
			return
		if filename in self.values('PRAGMA database_list;', column_number=2):
			return
		self.execute("ATTACH '{}' AS {};".format(filename,sqlname))

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
#				FROM elm_idco
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

	queries = property(Facet._get_queries,lambda self,x: self._set_queries(x,x))

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
		import pickle
		assert(isinstance(facetname,str))
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

