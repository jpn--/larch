/*
 *  etk_sqlite.h
 *
 *  Copyright 2007-2017 Jeffrey Newman
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


#ifndef __TOOLBOX_SQLITE__
#define __TOOLBOX_SQLITE__

#include "sqlite3.h"
#include <string>
#include "etk.h"


namespace etk {
	
class database_service {
	
	database_service* daba;
	
public:
	database_service(std::string file_name, etk::object* parent=NULL, logging_service* m=NULL);
	database_service(database_service* m=NULL): daba (m) { }
	database_service(database_service& m): daba (&m) { }
	void assign_database (database_service* m=NULL) { daba = m; }
	void assign_database (database_service& m     ) { daba = &m; }
	virtual ~database_service() { }
	
	virtual void assign_messenger(messenger* m=NULL);
	
	// INFO
	virtual bool is_open() ;
	virtual const std::string& get_error_message() const ;
	virtual std::string file_path() const ;
	virtual std::string file_name() const ;
	virtual int status() const ;
	
	// SQL
	// SQLite3 Command Handlers
	virtual int prepare			(std::string sql);
	virtual int execute			(logging_service* msg=NULL);
	virtual int direct_execute	(std::string sql);
	virtual int direct_execute_table (std::string sql);
	virtual int count_columns	();
	
	// BIND
	// Attach things to the sql statement before execution.
	// Uses a ONE-BASED indexing
	virtual int		bind_blob  (int position, const void* blob_ptr, int blob_bytes);
	virtual int		bind_text  (int position, std::string text_ptr);
	virtual int		bind_int   (int position, int value);
	virtual int		bind_double(int position, double value);	
	
	// GET
	// Get things from the database, zero-based indexing of items.
	// These work only when execute == SQLITE_ROW
	virtual int			getInt		(int column);
	virtual double		getDouble	(int column);
	virtual std::string		getText		(int column);
	virtual const void*	getBlob		(int column);
	virtual int			getByteCount(int column);
	
	virtual void        getDoubles (int startColumn, const int& endColumn,
									double* pushLocation, const unsigned& pushIncrement=1);

//	void define_caseids (string tablesource, string casevar, bool temp=false);
//	void define_otherkey (string tablesource, string casevar, string otherkey, string renameotherkey="");
//	void define_alts (std::string tablesource="", std::string altvar="", bool temp=false);
//	void auto_maketablelist (bool temp=false);
//	void make_setup_table();
//	void define_table (const std::string& table_name,
//							std::string table_kind,
//							const std::string& master_key, 
//							const std::string& table_key,
//							const std::string& altfield="");
	
	int simpleInteger(const std::string& sql);
	std::string simpleText(const std::string& sql);
};

	class database
	: public database_service 
	{
	
	const char*		_file_path;		// path to database
	const char*		_file_name;
	
	sqlite3*		_db;			// sqlite3 database
	sqlite3_stmt*	_statement;		// sql statement context pointer
	void			_clear_statement ();	
	
	std::string			_error_message;	// an error message holder
	int				_status;
	
	logging_service _msgr;
	
public:	
	// CONSTRUCT	
	database (std::string file_name, etk::object* parent=NULL, logging_service* m=NULL);
private:
	void _initializer(std::string& file_name);
	void _assimiliate(const std::string& file_name);
public:
	~database();
//	virtual void assign_messenger(logging_service& m);
	
	// INFO
	virtual bool is_open() { return (bool(_db)); }
	virtual const std::string& get_error_message() const { return _error_message; }
	virtual std::string file_path() const ;
	virtual std::string file_name() const { return std::string(_file_name); }
	virtual int status() const { return _status; }
	
	// SQL
	// SQLite3 Command Handlers
	virtual int prepare			(std::string sql);
	virtual int execute			(logging_service* msg=NULL);
	virtual int direct_execute	(std::string sql);
	virtual int direct_execute_table (std::string sql);
	virtual int count_columns	();
	
	// BIND
	// Attach things to the sql statement before execution.
	virtual int		bind_blob  (int position, const void* blob_ptr, int blob_bytes);
	virtual int		bind_text  (int position, std::string text_ptr);
	virtual int		bind_int   (int position, int value);
	virtual int		bind_double(int position, double value);	
	
	// GET
	// Get things from the database, zero-based indexing of items.
	// These work only when execute == SQLITE_ROW
	virtual int			getInt		(int column);
	virtual double		getDouble	(int column);
	virtual std::string getText		(int column);
	virtual const void*	getBlob		(int column);
	virtual int			getByteCount(int column);
	
	virtual void        getDoubles (int startColumn, const int& endColumn,
							double* pushLocation, const unsigned& pushIncrement=1);
	
};

} // end namespace toolbox


#endif

