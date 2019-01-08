# cython: language_level=3

import numpy
import pandas




cdef extern from "sqlite3.h":
	ctypedef struct sqlite3:
		int busyTimeout
	# ctypedef struct sqlite3_backup
	# ctypedef struct sqlite3_context
	# ctypedef struct sqlite3_value
	# ctypedef long long sqlite3_int64

	# Encoding.
	cdef int SQLITE_UTF8 = 1

	ctypedef struct sqlite3_stmt

	# Return values.
	cdef int SQLITE_OK = 0
	cdef int SQLITE_ERROR = 1
	cdef int SQLITE_NOMEM = 7

	# sqlite_value_type.
	cdef int SQLITE_INTEGER = 1
	cdef int SQLITE_FLOAT   = 2
	cdef int SQLITE3_TEXT   = 3
	cdef int SQLITE_TEXT    = 3
	cdef int SQLITE_BLOB    = 4
	cdef int SQLITE_NULL    = 5

	cdef int SQLITE_ROW     = 100  # sqlite3_step() has another row ready
	cdef int SQLITE_DONE    = 101  # sqlite3_step() has finished executing

	cdef int sqlite3_prepare_v2(
		sqlite3 *db,            # Database handle
		const char *zSql,       # SQL statement, UTF-8 encoded
		int nByte,              # Maximum length of zSql in bytes.
		sqlite3_stmt **ppStmt,  # OUT: Statement handle
		const char **pzTail     # OUT: Pointer to unused portion of zSql
	)

	cdef int sqlite3_step(
		sqlite3_stmt *ppStmt
	)

	cdef int sqlite3_finalize(sqlite3_stmt *pStmt)

	cdef int sqlite3_column_count( sqlite3_stmt *ppStmt )

	cdef int sqlite3_column_bytes(sqlite3_stmt*, int iCol)
	cdef double sqlite3_column_double(sqlite3_stmt*, int iCol)
	cdef long long int sqlite3_column_int64(sqlite3_stmt*, int iCol)
	cdef const char* sqlite3_column_name(sqlite3_stmt*, int iCol)
	cdef const unsigned char *sqlite3_column_text(sqlite3_stmt*, int iCol)



# quick functions

cdef const unsigned char* get_bytes(sqlite3_stmt* stmt, int column):
	if (sqlite3_column_bytes(stmt, column)):
		return sqlite3_column_text(stmt, column)
	else:
		return b""

cdef int count_columns(sqlite3_stmt* _statement):
	if _statement == NULL:
		return 0
	return sqlite3_column_count(_statement)






class SQLPrepError(ValueError):
	pass


class SQLStepError(ValueError):
	pass



from libc.stdint cimport uintptr_t

def all_table_names(uintptr_t connection_handle_as_voidptr):

	cdef sqlite3* connection_handle = <sqlite3*> connection_handle_as_voidptr

	names = []
	qry = "SELECT name, type FROM (SELECT type, name FROM sqlite_master UNION ALL SELECT type, 'temp.'||name AS name FROM sqlite_temp_master) WHERE type='table' OR type='view' ORDER BY name"

	cdef sqlite3_stmt*	_statement = NULL

	cdef int _status = sqlite3_prepare_v2(
		connection_handle,
		qry.encode(),
		-1,               # Length of zSql in bytes. -1 use all
		&_statement,      # OUT: SQLiteStmt handle */
		NULL              # OUT: Pointer to unused portion of zSql */
	)

	if  _status != SQLITE_OK:
		_statement = NULL
		raise SQLPrepError(qry)

	while sqlite3_step( _statement ) == SQLITE_ROW:
		names.append(get_bytes(_statement, 0).decode())

	sqlite3_finalize(_statement)

	return names

cpdef long long int single_integer_result(uintptr_t connection_handle_as_voidptr, str query):

	cdef sqlite3* connection_handle = <sqlite3*> connection_handle_as_voidptr
	cdef sqlite3_stmt*	_statement = NULL
	cdef int _status = sqlite3_prepare_v2(
		connection_handle,
		query.encode(),
		-1,               # Length of zSql in bytes. -1 use all
		&_statement,      # OUT: SQLiteStmt handle */
		NULL              # OUT: Pointer to unused portion of zSql */
	)

	cdef long long int result = -1

	if  _status != SQLITE_OK:
		_statement = NULL
		raise SQLPrepError(query)

	if sqlite3_step( _statement ) == SQLITE_ROW:
		result = sqlite3_column_int64(_statement, 0)
		sqlite3_finalize(_statement)
		return result
	else:
		sqlite3_finalize(_statement)
		raise SQLStepError(query)





def _sqlite_array_2d_float64(
		uintptr_t connection_handle_as_voidptr,
		str query,
		int n_rows=-1,
		out=None,
		index_col=None,
):
	"""

	Parameters
	----------
	connection_handle_as_voidptr
	query
	n_rows : int, optional
		How many rows of data should appear in the output. If the value is given but does not
		match the query results, an underflow or overflow warning will be raised.
	out : ndarray, optional
		If given, this ndarray must have the correct shape and dtype. This is an optimization feature for
		loading data into an existing array in memory.

	Returns
	-------
	pandas.DataFrame
	"""


	cdef int i =0
	cdef sqlite3* connection_handle = <sqlite3*> connection_handle_as_voidptr

	cdef sqlite3_stmt*	_statement = NULL

	if n_rows <= 0:
		# find the correct number of rows
		nrow_query= "SELECT count(*) FROM ({})".format(query)
		n_rows = single_integer_result(connection_handle_as_voidptr, nrow_query)


	cdef int _status = sqlite3_prepare_v2(
		connection_handle,
		query.encode(),
		-1,               # Length of zSql in bytes. -1 use all
		&_statement,      # OUT: SQLiteStmt handle */
		NULL              # OUT: Pointer to unused portion of zSql */
	)


	if  _status != SQLITE_OK:
		_statement = NULL
		raise SQLPrepError(query)

	cdef double[:,:] result
	cdef long long[:] indexarray
	cdef int n_cols
	cdef int col_num_for_index = -1

	try:

		n_cols = count_columns(_statement)

		col_names = []
		# const char *sqlite3_column_name(sqlite3_stmt*, int N);
		for i in range(n_cols):
			colname = sqlite3_column_name(_statement, i).decode()
			if index_col is not None and colname == index_col:
				col_num_for_index = i
			else:
				col_names.append( colname )


		if col_num_for_index >= 0:
			indexarray = numpy.zeros([n_rows, ], dtype=numpy.int64)
			if out is None:
				result = numpy.zeros([n_rows, n_cols-1], dtype=numpy.float64)
			else:
				assert( tuple(out.shape) == tuple(n_rows, n_cols-1) )
				result = out
		else:
			if out is None:
				result = numpy.zeros([n_rows, n_cols], dtype=numpy.float64)
			else:
				assert( tuple(out.shape) == tuple(n_rows, n_cols) )
				result = out


		for rownum in range(n_rows):

			_status = sqlite3_step( _statement )
			if _status == SQLITE_DONE:
				import warnings
				warnings.warn("underfilled array, only {} of {} rows loaded".format(rownum,n_rows))
			elif _status != SQLITE_ROW:
				raise SQLStepError("on row {}:\n{}".format(rownum, query))
			else:
				if col_num_for_index>=0:
					# pull the index from the data
					for i in range(n_cols):
						if i==col_num_for_index:
							indexarray[rownum] = sqlite3_column_int64(_statement, i)
						elif i>col_num_for_index:
							# account for missing column in result array
							result[rownum,i-1] = sqlite3_column_double(_statement, i)
						else:
							result[rownum,i] = sqlite3_column_double(_statement, i)
				else:
					# no index in the data, don't check every time
					for i in range(n_cols):
						result[rownum,i] = sqlite3_column_double(_statement, i)


		# check next step is done, if done was not already triggered
		if _status != SQLITE_DONE:
			_status = sqlite3_step( _statement )

		# if not done, warn overflow
		if _status == SQLITE_ROW:
			import warnings
			warnings.warn("overflowed array, only {} rows loaded".format(n_rows))

	finally:
		sqlite3_finalize(_statement)

	if col_num_for_index>=0:
		return pandas.DataFrame(data=result.base, columns=col_names, index=indexarray)
	else:
		return pandas.DataFrame(data=result.base, columns=col_names)





def _rip_array_idco(connection, query, n_rows, dtype=numpy.float64, out=None):
	with connection:
		cur = connection.cursor().execute(query)
		n_cols = len(cur.description)
		if out is None:
			result = numpy.zeros([n_rows, n_cols], dtype=dtype)
		else:
			assert( tuple(out.shape) == tuple(n_rows, n_cols) )
			result = out
		for rownum, row in enumerate(cur):
			result[rownum,:] = row[:]
	return result



class SQLitePodCO():

	def __init__(self, connection, tabledef, nrows=0, ordering=None):
		self.tabledef = tabledef
		self.connection = connection
		self.ordering = ordering
		if nrows:
			self.nrows = nrows
		else:
			with connection:
				cur = connection.cursor().execute(f"SELECT count(*) FROM ({self.tabledef})")
				self.nrows = next(cur)[0]

	def get_array(self, vars, selector=None, out=None):
		if isinstance(vars, str):
			vars = [vars,]
		qry_vars = ", ".join(vars)
		qry = f"SELECT {qry_vars} FROM ({self.tabledef})"
		if self.ordering:
			qry += f" ORDER BY {self.ordering}"
		if selector is None:
			return _rip_array_idco(self.connection, qry, self.nrows, out=out)
		elif isinstance(selector, slice) and selector.step in (None, 1):
			qry += f" LIMIT {selector.stop-selector.start} OFFSET {selector.start}"
			return _rip_array_idco(self.connection, qry, self.nrows, out=out)
		else:
			result = _rip_array_idco(self.connection, qry, self.nrows)
			out[:] = result[selector]
			return out