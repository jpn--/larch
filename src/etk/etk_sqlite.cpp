/*
 *  etk_sqlite.cpp
 *
 *  Copyright 2007-2016 Jeffrey Newman
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


#include "etk_sqlite.h"
#include "etk_exception.h"
#include "etk_vectors.h"
#include <iostream>
#include <cstring>

using namespace etk;
using namespace std;

database::database (string file_name_, object* parent, logging_service* m)
:	database_service ()
,	_file_path	(new char [file_name_.length()+1])
,	_db			(0)
,	_statement	(0)
,	_status		(0)
,   _msgr       (m)
{
	_initializer(file_name_);
}


void database::_initializer(string& file_name_)
{	
	
	
	if (file_name_.length() >= 8 && file_name_.substr(0, 8)==":memory:") {
		
		
		_status = sqlite3_open(":memory:", &_db);
		if (_status) {
			_error_message = sqlite3_errmsg(_db);
			INFO(_msgr) << "FAIL: " << _error_message;
			sqlite3_close(_db);
			_db = NULL;
		} else {
			INFO(_msgr) << "OK";
		}
		
		if (file_name_.length() > 8) {
			file_name_ = file_name_.substr(8);
			strcpy(const_cast<char*>(_file_path),file_name_.c_str());
			_file_name = strrchr( _file_path, DIRSEP );
			if ( _file_name != NULL ) _file_name++;
			_assimiliate(file_name_);
		}

		return;
	} 
	
	// Normal file operation
	strcpy(const_cast<char*>(_file_path),file_name_.c_str());
	_file_name = strrchr( _file_path, DIRSEP );
	if ( _file_name != NULL ) _file_name++;
	BUGGER(_msgr) << "Opening database '" << _file_path << "'... ";
	
	_status = sqlite3_open(_file_path, &_db);
	if (_status) {
		_error_message = sqlite3_errmsg(_db);
		BUGGER(_msgr) << "FAIL: " << _error_message << '\n';
		sqlite3_close(_db);
		_db = NULL;
	} else {
		BUGGER(_msgr) << "OK\n";
	}
	
}


void database::_assimiliate(const string& file_name_)
{
	//  ATTACH "NewData.db" AS ND;
	//  SELECT name FROM ND.SQLITE_MASTER WHERE type="table";
	//  CREATE TABLE x AS SELECT * from ND.x;

	ostringstream sql;
	int z;
	
	// ATTACH
	sql << "ATTACH '"<<file_name_<<"' AS OTHERDB";
	MONITOR(_msgr) << sql.str() ;
	z=direct_execute(sql.str());
	if (z != SQLITE_DONE) OOPS("error in attaching: ",get_error_message());
	sql.str(""); sql.clear();
	
	// GET TABLEs
	sql << "SELECT name FROM OTHERDB.SQLITE_MASTER WHERE type='table'";
	strvec otherTableNames;
	MONITOR(_msgr) << sql.str() ;
	z = direct_execute(sql.str());
	while (z==SQLITE_ROW) {
		otherTableNames.push_back(getText(0));
		z = execute();
	}
	if (z != SQLITE_DONE) OOPS("error in getting other's table names: ",get_error_message());
	sql.str(""); sql.clear();

	// CREATE TABLEs
	for (unsigned i=0; i<otherTableNames.size(); i++) {
		sql << "CREATE TABLE IF NOT EXISTS "<<otherTableNames[i]<<" AS SELECT * from OTHERDB."<<otherTableNames[i];
		MONITOR(_msgr) << sql.str() ;
		z=direct_execute(sql.str());
		if (z != SQLITE_DONE) OOPS("error in copying other's table ",otherTableNames[i],": ",get_error_message());
		sql.str(""); sql.clear();
	}
	
	// GET VIEWs and INDEXs
	sql << "SELECT sql FROM OTHERDB.SQLITE_MASTER WHERE type='view' OR type='index'";
	strvec otherSQLs;
	MONITOR(_msgr) << sql.str() ;
	z = direct_execute(sql.str());
	while (z==SQLITE_ROW) {
		otherSQLs.push_back(getText(0));
		MONITOR(_msgr) << "RECEIVED ["<< otherSQLs.back() <<"]" ;
		z = execute();
	}
	if (z != SQLITE_DONE) OOPS("error in getting other's table views and indexes: ",get_error_message());
	sql.str(""); sql.clear();

	// DETACH
	sql << "DETACH OTHERDB";
	MONITOR(_msgr) << sql.str() ;
	z=direct_execute(sql.str());
	if (z != SQLITE_DONE) OOPS("error in detaching: ",get_error_message());
	sql.str(""); sql.clear();
		
	// CREATE VIEW and INDEX
	for (unsigned i=0; i<otherSQLs.size(); i++) {
		MONITOR(_msgr) << otherSQLs[i];
		if (!otherSQLs[i].empty()) {
			z=direct_execute(otherSQLs[i]);
		}
		if (z != SQLITE_DONE) OOPS("error in copying other's table view or index: ",get_error_message());
		sql.str(""); sql.clear();
	}
	
}

database::~database()
{
	_clear_statement();
	if (_db ) sqlite3_close(_db);	
	delete [] _file_path;
}

/*
void database::assign_messenger(logging_service& m) 
{ 
	_msgr.assign_messenger(m);
}
*/

string database::file_path() const 
{
	if (!_file_path) OOPS("error: database file_path is empty");
	return string(_file_path); 
}


int database::prepare (string sql)
{
	char *unused;
	
	if (!_db) OOPS("the database \"",_file_path,"\" is not open");
	
	_error_message.clear();
	_clear_statement();
	
	_status = sqlite3_prepare( _db,
							  sql.c_str(),
							  -1,               /* Length of zSql in bytes. -1 use all */
							  &_statement,           /* OUT: Statement handle */
							  (const char**)&unused  /* OUT: Pointer to unused portion of zSql */
							  );
	
	if( _status != SQLITE_OK ) {
		_error_message = sqlite3_errmsg(_db);
		_statement = 0;
	}
	return _status;	
}

int database::execute (logging_service* mes)
{
	if (!_db) OOPS("the database \"",_file_path,"\" is not open");
	if (!_statement) {
		OOPS("the SQL statement is not ready to be executed");
	}
	
//	if (mes) cerr << "EXECUTE...";// << messengerPing(false,true);
	_status = sqlite3_step( _statement );
//	if (mes) cerr << "COMPLETE\n";// << messengerPing(true,true);

	switch( _status ){
		case SQLITE_DONE:
			break;
		case SQLITE_ROW:
			break;
		default:
			_error_message = sqlite3_errmsg(_db);
	}
	return _status;
}

int database::direct_execute (string sql)
{
	_status=prepare(sql);
	if (_status==SQLITE_OK) return execute();
	return _status;
}

int database::direct_execute_table (string sql)
{
	char** results;
	int nrow, ncol, r,c,i=0;
	char* errmsg;
	
	int z= sqlite3_get_table(_db,              /* An open database */
							 sql.c_str(),       /* SQL to be executed */
							 &results,       /* Result written to a char *[]  that this points to */
							 &nrow,             /* Number of result rows written here */
							 &ncol,          /* Number of result columns written here */
							 &errmsg          /* Error msg written here */
							 );
	if (z==SQLITE_OK) {
		nrow++;
		for (r=0; r<nrow; r++) {
			INFO_BUFFER(_msgr) << results[i++];
			for (c=1; c<ncol; c++) {
				INFO_BUFFER(_msgr) << "\t" << results[i++];
			}
			INFO(_msgr) << "";
		}
	} else {
		INFO(_msgr) << "SQLite Error: " << errmsg << "\n";
	}
	sqlite3_free_table(results);
	return _status;	
}


int database::count_columns()
{ return sqlite3_column_count(_statement); }

int database::bind_blob(int position, const void* blob_ptr, int blob_bytes)
{ return sqlite3_bind_blob(_statement, position, blob_ptr, blob_bytes, SQLITE_STATIC); }

int database::bind_text(int position, string text)
{ return sqlite3_bind_text(_statement, position, text.c_str(), -1, SQLITE_STATIC); }

int database::bind_int (int position, int value)
{ return sqlite3_bind_int(_statement, position, value); }

int database::bind_double(int position, double value)
{ return sqlite3_bind_double(_statement, position, value); }

int database::getInt (int column) 
{ return sqlite3_column_int(_statement, column); }

double database::getDouble(int column)
{ return sqlite3_column_double(_statement, column); }

void database::getDoubles (int startColumn, const int& endColumn,
						double* pushLocation, const unsigned& pushIncrement)
{
	for (; startColumn<endColumn; startColumn++) {
		*pushLocation = sqlite3_column_double(_statement, startColumn);
		pushLocation += pushIncrement;
	}
}

string database::getText(int column)
{ 
	if (sqlite3_column_bytes(_statement, column))
		return string ( (const char*) sqlite3_column_text(_statement, column));
	else return "";
}

const void* database::getBlob(int column)
{ return sqlite3_column_blob(_statement, column); }

int database::getByteCount(int column)
{ return sqlite3_column_bytes(_statement, column); }



// PRIVATE //

void database::_clear_statement ()
{
	if ( _statement ) {
		sqlite3_finalize( _statement );
		_statement = NULL;
		_status = 0;
	}	
}



// MARK: DATABASE SERVICE

etk::database_service::database_service(string file_name_, object* parent, logging_service* m)
: daba ()
{
	daba = new database (file_name_,parent,m);
}



#define Repoint(x)  if (!daba) OOPS("no database"); return daba->x;
#define RepointV(x)  if (!daba) OOPS("no database"); daba->x;


void etk::database_service::assign_messenger(messenger* m) { RepointV(assign_messenger(m)) }

// INFO
bool etk::database_service::is_open() { Repoint(is_open()) }
const string& etk::database_service::get_error_message() const { Repoint(get_error_message()) }
string etk::database_service::file_path() const { Repoint(file_path()) }
string etk::database_service::file_name() const { Repoint(file_name()) }
int etk::database_service::status() const { Repoint(status()) }

// SQL
// SQLite3 Command Handlers
int etk::database_service::prepare			(string sql)     { Repoint(prepare(sql)) }
int etk::database_service::execute			(logging_service* msg) { Repoint(execute(msg)) }
int etk::database_service::direct_execute	(string sql)     { Repoint(direct_execute(sql)) }
int etk::database_service::direct_execute_table (string sql) { Repoint(direct_execute_table(sql)) }
int etk::database_service::count_columns	()               { Repoint(count_columns()) }

// BIND
// Attach things to the sql statement before execution.
int		etk::database_service::bind_blob  (int position, const void* blob_ptr, int blob_bytes) { Repoint(bind_blob(position,blob_ptr,blob_bytes)) }
int		etk::database_service::bind_text  (int position, string text_ptr) { Repoint(bind_text(position,text_ptr)) }
int		etk::database_service::bind_int   (int position, int value) { Repoint(bind_int(position,value)) }
int		etk::database_service::bind_double(int position, double value) { Repoint(bind_double(position,value)) }

// GET
// Get things from the database, zero-based indexing of items.
// These work only when execute == SQLITE_ROW
int			etk::database_service::getInt		(int column) { Repoint(getInt(column)) }
double		etk::database_service::getDouble	(int column) { Repoint(getDouble(column)) }
string		etk::database_service::getText		(int column) { Repoint(getText(column)) }
const void*	etk::database_service::getBlob		(int column) { Repoint(getBlob(column)) }
int			etk::database_service::getByteCount(int column) { Repoint(getByteCount(column)) }

void        etk::database_service::getDoubles (int startColumn, const int& endColumn,
								double* pushLocation, const unsigned& pushIncrement) { Repoint(getDoubles(startColumn,endColumn,pushLocation,pushIncrement)) }

/*
void etk::database_service::define_caseids (string tablesource, string casevar, bool temp)
{
	*
	CREATE TABLE IF NOT EXISTS elm_case_ids (caseid INTEGER UNIQUE);
	INSERT OR IGNORE INTO elm_case_ids (caseid) SELECT CASENUM FROM STOPS;
	UPDATE elm_case_ids SET casenum=ROWID-1;
	 *
	ostringstream sql;
	int z;
	
	sql << "CREATE ";
	if (temp) sql << "TEMPORARY ";
	sql << "TABLE IF NOT EXISTS elm_case_ids (caseid INTEGER UNIQUE)";
	z=direct_execute(sql.str());
	if (z != SQLITE_DONE) OOPS("error in making elm_case_ids: ",get_error_message(),"\nActive database file: ",file_path());
	sql.str(""); sql.clear();
	
	sql << "INSERT OR IGNORE INTO elm_case_ids (caseid) SELECT "<<casevar<<" FROM "<<tablesource;
	z=direct_execute(sql.str());
	if (z != SQLITE_DONE) OOPS("error in inserting elm_case_ids: ",get_error_message(),"\nActive database file: ",file_path());
	sql.str(""); sql.clear();

}*/


/*
void etk::database_service::define_otherkey (string tablesource, string casevar, string otherkey, string renameotherkey)
{
	ostringstream sql;
	int z;
	if (renameotherkey=="") renameotherkey = otherkey;

	sql << "ALTER TABLE elm_case_ids ADD "<<renameotherkey<<" INTEGER";
	z=direct_execute(sql.str());
	if (z != SQLITE_DONE && get_error_message().substr(0, 21) != "duplicate column name") 
		OOPS("error in adding other key column: ",get_error_message());
	sql.str(""); sql.clear();
	
	sql << "UPDATE elm_case_ids SET "<<renameotherkey<<" = (SELECT "<<otherkey
		<<" FROM "<<tablesource<<" WHERE "<<tablesource<<"."<<casevar<<"=elm_case_ids.caseid)";
	z=direct_execute(sql.str());
	if (z != SQLITE_DONE) OOPS("error in updating other keys: ",get_error_message());
	sql.str(""); sql.clear();
}
 */

//void etk::database_service::define_alts (string tablesource, string altvar, bool temp)
//{
//	/* 
//	 CREATE TABLE IF NOT EXISTS elm_alternatives (\n  id INT UNIQUE,\n  name TEXT,\n  upcodes TEXT DEFAULT NULL,\n  dncodes TEXT DEFAULT NULL);
//	 INSERT OR IGNORE INTO elm_alternatives (id) SELECT ALTNUM FROM STOPS;
//	 UPDATE elm_alternatives SET name="alt"||id WHERE name="";
//	 */
//	
//	ostringstream sql;
//	int z;
//	
//	sql << "CREATE ";
//	if (temp) sql << "TEMPORARY ";
//	sql << "TABLE IF NOT EXISTS elm_alternatives (\n  id INT UNIQUE,\n  name TEXT,\n  upcodes TEXT DEFAULT NULL,\n  dncodes TEXT DEFAULT NULL)";
//	z=direct_execute(sql.str());
//	if (z != SQLITE_DONE) OOPS("error in making elm_alternatives: ",get_error_message());
//	sql.str(""); sql.clear();
//	
//	if (tablesource.empty() || altvar.empty()) return;
//	
//	sql << "INSERT OR IGNORE INTO elm_alternatives (id) SELECT "<<altvar<<" FROM "<<tablesource;
//	z=direct_execute(sql.str());
//	if (z != SQLITE_DONE) OOPS("error in inserting elm_alternatives: ",get_error_message());
//	sql.str(""); sql.clear();
//	
//	sql << "UPDATE elm_alternatives SET name=\"alt\"||id WHERE name=\"\"";
//	z=direct_execute(sql.str());
//	if (z != SQLITE_DONE) OOPS("error in updating elm_alternatives: ",get_error_message());
//	sql.str(""); sql.clear();
//
//	sql << "UPDATE elm_alternatives SET name=\"alt\"||id WHERE name is NULL";
//	z=direct_execute(sql.str());
//	if (z != SQLITE_DONE) OOPS("error in updating elm_alternatives: ",get_error_message());
//	sql.str(""); sql.clear();
//
//}

//void etk::database_service::make_setup_table()
//{
//	ostringstream sql;
//	int z;
//	
//	sql << "CREATE TABLE IF NOT EXISTS elm_setup (\n"
//	<< "  id TEXT UNIQUE,\n"
//	<< "  value)";
//	z=direct_execute(sql.str());
//	if (z != SQLITE_DONE) OOPS("error in making elm_setup: ",get_error_message());
//	sql.str(""); sql.clear();
//}

//void etk::database_service::auto_maketablelist (bool temp)
//{
//	/*		string attach_subtable(const string& table_name,
//	 const string& table_key,
//	 const string& master_key, 
//	 string table_kind,
//	 const string& altfield="");
//	 */
//	ostringstream sql;
//	int z;
//	
//	sql << "CREATE ";
//	if (temp) sql << "TEMPORARY ";
//	sql << "TABLE IF NOT EXISTS elm_tables (\n"
//		<< "  tablename TEXT UNIQUE NOT NULL,\n"
//		<< "  selfkey TEXT NOT NULL,\n"
//		<< "  masterkey TEXT NOT NULL,\n"
//		<< "  kind TEXT NOT NULL,\n"
//		<< "  altfield TEXT)";
//	z=direct_execute(sql.str());
//	if (z != SQLITE_DONE) OOPS("error in making elm_tables: ",get_error_message());
//	sql.str(""); sql.clear();
//	
//}

//void etk::database_service::define_table (const string& table_name,
//											  string table_kind,
//											  const string& master_key, 
//											  const string& table_key,
//											  const string& altfield)
//{
//	/*		string attach_subtable(const string& table_name,
//	 const string& table_key,
//	 const string& master_key, 
//	 string table_kind,
//	 const string& altfield="");
//	 */
//	ostringstream sql;
//	int z;
//	
//	//auto_maketablelist();
//	
//	string tk = uppercase(table_kind);
//	
//	sql << "INSERT OR REPLACE INTO elm_tables (tablename, selfkey, masterkey, kind, altfield)"
//		<< "VALUES (?,?,?,?,?)";
//	z=prepare(sql.str());
//	bind_text(1,table_name);
//	bind_text(2,table_key);
//	bind_text(3,master_key);
//	bind_text(4,tk);
//	bind_text(5,altfield);
//	z=execute();
//	if (z != SQLITE_DONE) OOPS("error in adding to elm_tables: ",get_error_message());
//	sql.str(""); sql.clear();
//	
//}


int etk::database_service::simpleInteger(const std::string& sql)
{
	int z=direct_execute(sql);
	if (z != SQLITE_ROW) OOPS(get_error_message());
	return getInt(0);
}
std::string etk::database_service::simpleText(const std::string& sql)
{
	int z=direct_execute(sql);
	if (z == SQLITE_DONE) return "";
	if (z != SQLITE_ROW) OOPS(get_error_message());
	return getText(0);
}


