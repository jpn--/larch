/*
 *  elm_sql_connect.h
 *
 *  Copyright 2007-2015 Jeffrey Newman
 *
 *  Larch is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  Larch is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with Larch.  If not, see <http://www.gnu.org/licenses/>.
 *  
 */


#ifndef __ELM2_SQL_CONNECT_H__
#define __ELM2_SQL_CONNECT_H__


#ifdef SWIG
////////////////////////////////////////////////////////////////////////////////
// !!!: HELP TEXT

%feature("docstring") elm::SQLiteDB::all_table_names
"A list of all table names in the SQL database, including both data tables and
administrative tables";

%feature("docstring") elm::SQLiteDB::logger 
"The SQLiteDB object contains a link to the standard python logging service. By default,
no logger is connected. Use this function to assign a logger, and then use the usual
python logging controls to set the quantity/destination of output. To stop logging,
call this function without any arguments.";

%feature("docstring") elm::SQLiteDB::commit
"Commit to SQLite database. \n\n\
:param raise_on_fail: Raise an exception if the commit fails for any reason.";

%feature("docstring") elm::SQLiteDB::backup
"Save the current SQLite database to another DB. \n\n\
:param filename: The URI for where to save the DB.";

%feature("docstring") elm::SQLiteDB::eval_integer
"Evaluate a SQL statement that should return a single integer. \n\n\
:param sql: The SQL statement to evaluate.\n\
:param default_value: A default value to return if the statement does not return a single value.";

%feature("docstring") elm::SQLiteDB::eval_float
"Evaluate a SQL statement that should return a single double precision floating point number. \n\n\
:param sql: The SQL statement to evaluate.\n\
:param default_value: A default value to return if the statement does not return a single value.";

%feature("docstring") elm::SQLiteDB::eval_integer
"Evaluate a SQL statement that should return a single text string. \n\n\
:param sql: The SQL statement to evaluate.\n\
:param default_value: A default value to return if the statement does not return a single value.";


#endif // SWIG





#ifdef SWIG
%{
	// In SWIG, these headers are available to the c++ wrapper,
	// but are not themselves wrapped

	#include "etk.h"
	#include "etk_python.h"

	#ifdef BOOST_TOOLS
	#include "boost/shared_ptr.hpp"
	#endif
%}
#endif // SWIG

#ifndef SWIG
	// In not SWIG, these headers are treated normally

	#include "etk.h"
	#include "etk_python.h"

	#ifdef BOOST_TOOLS
	#include "boost/shared_ptr.hpp"
	#endif
#endif // not SWIG








namespace elm {
		
	
	
	
	
//////// MARK: SQLiteDB //////////////////////////////////////////////////////////////////	
#ifndef SWIG

	class SQLiteDB;
	
	class SQLiteStmt
	{
		friend class SQLiteDB;
	
		PyObject*   _parent_db_pyobj;
	
	protected:
		SQLiteStmt(      SQLiteDB* db);
		SQLiteStmt(const SQLiteDB* db);
	public:
		SQLiteStmt(const SQLiteStmt& s);

	public:
		~SQLiteStmt();
		void oops(std::string message="", std::string sql="", int errorcode=0);
		
	protected:
		SQLiteDB*       _sqldb;
		const bool      _readonly;
		sqlite3_stmt*	_statement;		// sql statement context pointer
		int				_status;
		void            db_check();
		
	public:
		void clear();	
		int	 reset();	
		const int& status() const;

		SQLiteStmt* prepare			(std::string sql);
		SQLiteStmt* execute			();
		SQLiteStmt* execute_until_done();
		
		int         count_columns ();
		etk::strvec column_names  ();
		std::string column_name   (int n);
		
		std::string sql() const;
		
		// BIND
		// Attach things to the sql statement before execution.
		SQLiteStmt* bind_blob  (int position, const void* blob_ptr, int blob_bytes);
		SQLiteStmt* bind_text  (int position, std::string text_ptr);
		SQLiteStmt* bind_int   (int position, int value);
		SQLiteStmt* bind_int64 (int position, long long value);
		SQLiteStmt* bind_double(int position, double value);	
		
		// GET
		// Get things from the database, zero-based indexing of items.
		// These work only when execute == SQLITE_ROW
		int			getInt		(int column);
		long long	getInt64	(int column);
		double		getDouble	(int column);
		bool		getBool		(int column);
		std::string getText		(int column);
		const void*	getBlob		(int column);
		int			getByteCount(int column);
		void        getDoubles  (int startColumn, const int& endColumn,
								         double* pushLocation, const unsigned& pushIncrement=1);
		void        getBools (int startColumn, const int& endColumn,
								        bool* pushLocation, const unsigned& pushIncrement=1);
	
		int         simpleInteger(const std::string& sql, const int& defaultvalue);
		long long   simpleInt64  (const std::string& sql, const int& defaultvalue);
		double      simpleDouble (const std::string& sql, const double& defaultvalue);
		std::string simpleText   (const std::string& sql, const std::string& defaultvalue);

		int         simpleInteger(const std::string& sql);
		long long   simpleInt64  (const std::string& sql);
		double      simpleDouble (const std::string& sql);
		std::string simpleText   (const std::string& sql);
		
	};

	typedef boosted::shared_ptr<elm::SQLiteStmt> SQLiteStmtPtr;

#endif // ndef SWIG



		
	class SQLiteDB
	#ifndef SWIG
	:	public etk::object
	#endif // not SWIG
	{
	public:
	
		#ifndef SWIG ///////////////////////////////////// MARK: BEGIN UNSWIGGED

		friend class SQLiteStmt;
		PyObject*       apsw_connection;// apsw python object pointer (Non-ref counted pointer to self)
	
	protected:	
		sqlite3*		_db;			// sqlite3 database pointer
	
	public:
		// INFO
		virtual bool is_open() { return (bool(_db)); }
		

	public:
		std::string what_is_it(std::string name, std::string* temp_prefix=NULL) const;

	public:
		// CURSORS
		SQLiteStmtPtr sql_statement(std::string sql);
		SQLiteStmtPtr sql_statement_readonly(std::string sql);
		SQLiteStmtPtr sql_statement(std::string sql) const;

		SQLiteStmtPtr sql_statement(std::ostringstream& sql);
		SQLiteStmtPtr sql_statement_readonly(std::ostringstream& sql);
		SQLiteStmtPtr sql_statement(std::ostringstream& sql) const;


		#endif // not SWIG ///////////////////////////////// MARK: END UNSWIGGED

	public:	
		// CONSTRUCT & DESTRUCT
		SQLiteDB (PyObject* pylong_ptr_to_db);
		~SQLiteDB();
		
		void close();

	public:
		void copy_from_db(const std::string& file_name_);
		void backup(const std::string& filename);
		
		void commit(int raise_on_fail=1);
		void drop(std::string name);
		
	public:
		std::vector<std::string> all_table_names() const;
		std::vector<std::string> column_names(std::string query) const;
		std::string              column_name (std::string query, int n) const;

		#ifndef SWIG
		PyObject* logger(std::string log_name);
		PyObject* logger(bool log_on);
		PyObject* logger(int log_on);
		#endif // ndef SWIG
		PyObject* logger(PyObject* log=nullptr);
		
		int         error_code();
		std::string error_msg();

		int         eval_integer(const std::string& sql, const int& defaultvalue) const;
		long long   eval_int64  (const std::string& sql, const int& defaultvalue) const;
		double      eval_float  (const std::string& sql, const double& defaultvalue) const;
		std::string eval_text   (const std::string& sql, const std::string& defaultvalue) const;
		
		int         eval_integer(const std::string& sql) const;
		long long   eval_int64  (const std::string& sql) const;
		double      eval_float  (const std::string& sql) const;
		std::string eval_text   (const std::string& sql) const;
		std::vector<long long>          eval_int64_tuple(const std::string& sql) const;
		std::vector<std::string>        eval_string_tuple(const std::string& sql) const;

		
	};
	
	#ifdef SWIG
	%extend SQLiteDB {
	%pythoncode %{
	def __str__(self):
		return "<SQLiteDB: %s>"%self.working_name
	def __repr__(self):
		return "<SQLiteDB: %s>"%self.working_name
	exec_integer = eval_integer
	exec_float   = eval_float
	exec_text    = eval_text
	def Shell(self):
		'''Enter an SQLite shell for accessing the contents of the database object.
		
		The shell is based on the usual SQLite3 command shell, and is not a Python
		interpreted environment. Use Ctrl-C (Windows) or Ctrl-D (Mac OS X) to escape
		back to the usual Python environment.'''
		from . import apsw
		apsw.Shell(db=self).cmdloop()
	%}
	};
	#endif // def SWIG



	
	
	
} // end namespace elm
#endif // __ELM2_SQL_CONNECT_H__

