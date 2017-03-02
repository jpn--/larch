/*
 *  elm_sql_connect.cpp
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


#include <cstring>
#include "etk.h"
#include "elm_sql_connect.h"

using namespace std;
using namespace etk;

#include <iostream>

#ifdef BOOST_TOOLS
#include "boost/make_shared.hpp"
#endif

elm::SQLiteDB::SQLiteDB (PyObject* apsw_connection)
: _db (NULL)
, apsw_connection (apsw_connection)
{
	Py_XINCREF(apsw_connection);
	PyObject* pylong_ptr_to_db = PyObject_CallMethod(apsw_connection, (char*)"sqlite3pointer", NULL);
	_db = static_cast<sqlite3*>(PyLong_AsVoidPtr(pylong_ptr_to_db));
	sqlite3_enable_load_extension(_db,1);
	Py_CLEAR(pylong_ptr_to_db);
	Py_XDECREF(apsw_connection);
}



elm::SQLiteDB::~SQLiteDB()
{
	close();
//	Py_CLEAR(apsw_connection);
}

void elm::SQLiteDB::close ()
{
	if (_db) {
		_db = NULL;
	}

}



std::vector<std::string> elm::SQLiteDB::all_table_names() const
{
	std::vector<std::string> all_table_names;
	SQLiteStmtPtr s = sql_statement("SELECT name, type FROM (SELECT type, name FROM sqlite_master UNION ALL SELECT type, 'temp.'||name AS name FROM sqlite_temp_master) WHERE type='table' OR type='view' ORDER BY name");
	s->execute();
	while (s->status()==SQLITE_ROW) {
		all_table_names.push_back(s->getText(0));
		s->execute();
	}
	return all_table_names;
}

std::vector<std::string> elm::SQLiteDB::column_names(std::string query) const
{
	std::vector<std::string> x;
	try {
	SQLiteStmtPtr s = sql_statement(query);
	x = s->column_names();
	} catch(etk::SQLiteError& err) {
	SQLiteStmtPtr s = sql_statement("select * from "+query);
	x = s->column_names();
	}
	return x;
}

std::string elm::SQLiteDB::column_name(std::string query, int n) const
{
	std::string x;
	SQLiteStmtPtr s = sql_statement(query);
	x = s->column_name(n);
	return x;
}


PyObject* elm::SQLiteDB::logger(std::string log_name)
{
	if (log_name=="") {
		msg.change_logger_name("");
	} else {
		msg.change_logger_name("larch."+log_name);
	}
	
	return msg.get_logger();
}

PyObject* elm::SQLiteDB::logger(bool z)
{
	if (!z) {
		msg.change_logger_name("");
	} else {
		msg.change_logger_name("larch.Data");
	}
	return msg.get_logger();
}

PyObject* elm::SQLiteDB::logger(int z)
{
	if (z <=0 ) {
		msg.change_logger_name("");
	} else {
		msg.change_logger_name("larch.Data");
	}
	return msg.get_logger();
}

PyObject* elm::SQLiteDB::logger(PyObject* z)
{
	if (PyLong_Check(z)) {
		return logger((int)PyLong_AS_LONG(z));
	}

	if (PyUnicode_Check(z)) {
		return logger(PyString_ExtractCppString(z));
	}

	if (PyBool_Check(z)) {
		return logger(Py_True==z);
	}

	if ( z && z!=Py_None ) {
		msg.set_logger(z);
	} 
	
	return msg.get_logger();
}




std::string elm::SQLiteDB::what_is_it(std::string name, std::string* temp_prefix) const
{
	string t ("");
	SQLiteStmtPtr s = sql_statement("");
	{
		s->prepare("SELECT type FROM sqlite_temp_master WHERE name=?;")->bind_text(1,name);
		t = s->simpleText("","");
		if (!t.empty() && temp_prefix) {
			*temp_prefix = "temp.";
			return t;
		}
	}
	{
		s->prepare("SELECT type FROM sqlite_master WHERE name=?;");
		t = s->bind_text(1,name)->simpleText("","");
		if (!t.empty() && temp_prefix) *temp_prefix = "";
	} 
	return t;
}

void elm::SQLiteDB::drop(std::string name) {
	string tmp ("");
	string t = what_is_it(name,&tmp);
	while (!t.empty()) {
		std::ostringstream sql;
		sql << "DROP "<<t<<" "<<tmp<<name<<";";
		sql_statement(sql)->execute_until_done();
		t = what_is_it(name,&tmp);
	}
}


void elm::SQLiteDB::commit(int raise_on_fail)
{
//	if (raise_on_fail<0) {
//		int z = 0;
//		int trys = 0;
//		while (z != SQLITE_DONE && trys > raise_on_fail) {
//			if (trys<0) {
//				usleep(10000);
//			}
//			trys--;
//			z=sql_statement("commit;")->execute()->status();
//		}
//	} else {
//		int z;
//		z=sql_statement("commit;")->execute()->status();
//		if (z != SQLITE_DONE){
//			if (raise_on_fail>0) OOPS("error in committing: ",get_error_message());
//		}
//	}
}



void elm::SQLiteDB::copy_from_db(const string& file_name_)
{
	int rc;
	sqlite3* copy_from;
	sqlite3_backup *pBackup;  /* Backup object used to copy data */
	
	rc = sqlite3_open(file_name_.c_str(), &copy_from);
	if( rc==SQLITE_OK ){
		/*  Set up the backup procedure to copy from the "main" database of
		 ** connection copy_from to the main database of connection self.
		 ** If something goes wrong, pBackup will be set to NULL and an error
		 ** code and  message left in connection pTo.
		 **
		 ** If the backup object is successfully created, call backup_step()
		 ** to copy data from pFile to pInMemory. Then call backup_finish()
		 ** to release resources associated with the pBackup object.  If an
		 ** error occurred, then  an error code and message will be left in
		 ** connection pTo. If no error occurred, then the error code belonging
		 ** to pTo is set to SQLITE_OK.
		 */
		pBackup = sqlite3_backup_init(_db, "main", copy_from, "main");
		if( pBackup ){
			(void)sqlite3_backup_step(pBackup, -1);
			(void)sqlite3_backup_finish(pBackup);
		}
		rc = sqlite3_errcode(_db);
	}
	(void)sqlite3_close(copy_from);
	if (rc==SQLITE_DONE) return;
	if (rc) OOPS("error in copying database (",rc,")");
}

void elm::SQLiteDB::backup(const std::string& filename)
{
	int rc;
	sqlite3* copy_to;
	sqlite3_backup *pBackup;  /* Backup object used to copy data */
	
	rc = sqlite3_open_v2(filename.c_str(), &copy_to, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_URI, NULL);
	if( rc==SQLITE_OK ){
		/*  Set up the backup procedure to copy from the "main" database of
		 ** connection copy_from to the main database of connection self.
		 ** If something goes wrong, pBackup will be set to NULL and an error
		 ** code and  message left in connection pTo.
		 **
		 ** If the backup object is successfully created, call backup_step()
		 ** to copy data from pFile to pInMemory. Then call backup_finish()
		 ** to release resources associated with the pBackup object.  If an
		 ** error occurred, then  an error code and message will be left in
		 ** connection pTo. If no error occurred, then the error code belonging
		 ** to pTo is set to SQLITE_OK.
		 */
		pBackup = sqlite3_backup_init(copy_to, "main", _db, "main");
		if( pBackup ){
			(void)sqlite3_backup_step(pBackup, -1);
			(void)sqlite3_backup_finish(pBackup);
		}
		rc = sqlite3_errcode(_db);
	}
	(void)sqlite3_close(copy_to);
	if (rc==SQLITE_DONE) return;
	if (rc) OOPS("error in backing up database (",rc,")");
}







elm::SQLiteStmtPtr elm::SQLiteDB::sql_statement(std::string sql)
{
	elm::SQLiteStmtPtr p = boosted::make_shared<elm::SQLiteStmt>(elm::SQLiteStmt(this));
	if (sql!="") {
		p->prepare(sql);
	}
	return p;
}

elm::SQLiteStmtPtr elm::SQLiteDB::sql_statement_readonly(std::string sql)
{
	elm::SQLiteStmtPtr p = boosted::make_shared<elm::SQLiteStmt>(elm::SQLiteStmt(const_cast<const SQLiteDB*>(this)));
	if (sql!="") {
		p->prepare(sql);
	}
	return p;
}

elm::SQLiteStmtPtr elm::SQLiteDB::sql_statement(std::string sql) const
{
	elm::SQLiteStmtPtr p = boosted::make_shared<elm::SQLiteStmt>(elm::SQLiteStmt(this));
	if (sql!="") {
		p->prepare(sql);
	}
	return p;
}

elm::SQLiteStmtPtr elm::SQLiteDB::sql_statement(std::ostringstream& sql)
{
	return sql_statement(sql.str());
}

elm::SQLiteStmtPtr elm::SQLiteDB::sql_statement_readonly(std::ostringstream& sql)
{
	return sql_statement_readonly(sql.str());
}

elm::SQLiteStmtPtr elm::SQLiteDB::sql_statement(std::ostringstream& sql) const
{
	return sql_statement(sql.str());
}



int elm::SQLiteDB::error_code()
{
	return sqlite3_extended_errcode(_db);
}

std::string elm::SQLiteDB::error_msg()
{
	return sqlite3_errmsg(_db);
}



int elm::SQLiteDB::eval_integer(const std::string& sql, const int& defaultvalue) const
{
	return sql_statement(sql)->simpleInteger("",defaultvalue);
}

long long elm::SQLiteDB::eval_int64(const std::string& sql, const int& defaultvalue) const
{
	return sql_statement(sql)->simpleInt64("",defaultvalue);
}

double      elm::SQLiteDB::eval_float  (const std::string& sql, const double& defaultvalue) const
{
	return sql_statement(sql)->simpleDouble("",defaultvalue);
}


std::string elm::SQLiteDB::eval_text   (const std::string& sql, const std::string& defaultvalue) const
{
	return sql_statement(sql)->simpleText("",defaultvalue);
}


int elm::SQLiteDB::eval_integer(const std::string& sql) const
{
	return sql_statement(sql)->simpleInteger("");
}

long long elm::SQLiteDB::eval_int64(const std::string& sql) const
{
	return sql_statement(sql)->simpleInt64("");
}

double      elm::SQLiteDB::eval_float (const std::string& sql) const
{
	return sql_statement(sql)->simpleDouble("");
}


std::string elm::SQLiteDB::eval_text   (const std::string& sql) const
{
	return sql_statement(sql)->simpleText("");
}













elm::SQLiteStmt::SQLiteStmt(SQLiteDB* db)
: _sqldb         (db)
, _readonly      (false)
, _statement     (NULL)
, _status        (0)
//,_parent_db_pyobj(db->apsw_connection)
{
//	Py_INCREF(_parent_db_pyobj);
}

elm::SQLiteStmt::SQLiteStmt(const SQLiteDB* db)
: _sqldb         (const_cast<SQLiteDB*>(db))
, _readonly      (true)
, _statement     (NULL)
, _status        (0)
//,_parent_db_pyobj(const_cast<SQLiteDB*>(db)->apsw_connection)
{
//	Py_INCREF(_parent_db_pyobj);
}

elm::SQLiteStmt::SQLiteStmt(const elm::SQLiteStmt& s)
: _sqldb         (s._sqldb)
, _readonly      (s._readonly)
, _statement     (NULL)
, _status        (0)
//,_parent_db_pyobj(s._sqldb->apsw_connection)
{
//	Py_INCREF(_parent_db_pyobj);
	if (s._statement) {
		prepare(sqlite3_sql(s._statement));
	}
}


elm::SQLiteStmt::~SQLiteStmt()
{
	clear();
//	Py_CLEAR(_parent_db_pyobj);
}

const int& elm::SQLiteStmt::status() const
{
	return _status;
}

void elm::SQLiteStmt::clear()
{
	if ( _statement ) {
		sqlite3_finalize( _statement );
		_statement = NULL;
		_status = 0;
	}	
}

int elm::SQLiteStmt::reset()
{
	if ( _statement ) {
		_status = sqlite3_reset( _statement );
	}	
	return _status;	
}

void elm::SQLiteStmt::db_check()
{
	if (!_sqldb) throw(etk::SQLiteError(SQLITE_ERROR, "there is no database"));
	if (!_sqldb->is_open()) throw(etk::SQLiteError(SQLITE_ERROR, "there is no open database"));
}

std::string elm::SQLiteStmt::sql() const
{
	if (_statement) {
		return sqlite3_sql(_statement);
	}
	return "";
}

void elm::SQLiteStmt::oops(string message, string sql, int errorcode)
{
	db_check();
	if (sql=="") {
		if (_statement) {
			sql = sqlite3_sql(_statement); 
		}
	}
	if (sql!="") {
		sql = "\nSQL: "+sql+"\n";
		
	}
	if (message!="") {
		message = message + ": ";
	}
	if (_sqldb->error_code()) {
		message = cat(message,"(error code ",_sqldb->error_code(),") ",_sqldb->error_msg());
	} 
	if (message=="") {
		message = "Unknown Error";
	}
	if (errorcode==0) errorcode = _sqldb->error_code();
	throw(etk::SQLiteError(errorcode, cat(message, sql)));
}

elm::SQLiteStmt* elm::SQLiteStmt::prepare (string sql)
{
	db_check();
	clear();
	
	BUGGER(_sqldb->msg) << "Preparing SQL: "<<sql;
	
	_status = sqlite3_prepare_v2(_sqldb->_db,
							  sql.c_str(),
							  -1,               /* Length of zSql in bytes. -1 use all */
							  &_statement,           /* OUT: SQLiteStmt handle */
							  NULL  /* OUT: Pointer to unused portion of zSql */
							  );
	
	if( _status != SQLITE_OK ) {
		_statement = nullptr;
		oops(sql);
	}
	return this;	
}

elm::SQLiteStmt* elm::SQLiteStmt::execute ()
{
	if (!_statement) {
		oops();
	}
	
	if (_readonly) {
		if (!sqlite3_stmt_readonly(_statement)) {
			oops("Cursor is Read Only");
		}
	} 

	if (!_statement) {
		oops();
	}
		
	_status = sqlite3_step( _statement );
	
	switch( _status ){
		case SQLITE_DONE:
			break;
		case SQLITE_ROW:
			break;
		default:
			oops();
	}
	return this;
}

elm::SQLiteStmt* elm::SQLiteStmt::execute_until_done()
{
	execute();
	while (_status != SQLITE_DONE) {
		if (_status==SQLITE_ROW) { execute(); } else { oops(); }
	}
	return this;
}


int elm::SQLiteStmt::count_columns()
{ 
	if (!_statement) return 0;
	return sqlite3_column_count(_statement); 
}

strvec elm::SQLiteStmt::column_names()
{
	strvec x;
	if (!_statement) return x;
	int n = sqlite3_column_count(_statement);
	for (int i=0; i<n; i++) {
		x.push_back(sqlite3_column_name(_statement, i));
	}
	return x;
}

std::string elm::SQLiteStmt::column_name(int n)
{
	if (!_statement) oops("no query prepared from which to get a column name");
	if (n>=sqlite3_column_count(_statement)) oops("prepared query does not have that many columns");
	return sqlite3_column_name(_statement, n);
}




elm::SQLiteStmt* elm::SQLiteStmt::bind_blob(int position, const void* blob_ptr, int blob_bytes)
{ 
	_status = sqlite3_bind_blob(_statement, position, blob_ptr, blob_bytes, SQLITE_STATIC);
	if (_status!=SQLITE_OK) oops();
	return this;
}

elm::SQLiteStmt*  elm::SQLiteStmt::bind_text(int position, string text)
{ 
	_status = sqlite3_bind_text(_statement, position, text.c_str(), -1, SQLITE_TRANSIENT); 
	if (_status!=SQLITE_OK) oops();
	return this;
}

elm::SQLiteStmt*  elm::SQLiteStmt::bind_int (int position, int value)
{ 
	_status = sqlite3_bind_int(_statement, position, value); 
	if (_status!=SQLITE_OK) oops();
	return this;
}

elm::SQLiteStmt*  elm::SQLiteStmt::bind_int64 (int position, long long value)
{ 
	_status = sqlite3_bind_int64(_statement, position, sqlite3_int64(value)); 
	if (_status!=SQLITE_OK) oops();
	return this;
}

elm::SQLiteStmt*  elm::SQLiteStmt::bind_double(int position, double value)
{ 
	_status = sqlite3_bind_double(_statement, position, value); 
	if (_status!=SQLITE_OK) oops();
	return this;
}



int elm::SQLiteStmt::getInt (int column) 
{ return sqlite3_column_int(_statement, column); }

long long elm::SQLiteStmt::getInt64 (int column) 
{ return (long long)(sqlite3_column_int64(_statement, column)); }

double elm::SQLiteStmt::getDouble(int column)
{ return sqlite3_column_double(_statement, column); }

bool elm::SQLiteStmt::getBool(int column)
{ return (bool)sqlite3_column_int(_statement, column); }

void elm::SQLiteStmt::getDoubles (int startColumn, const int& endColumn,
						double* pushLocation, const unsigned& pushIncrement)
{
	for (; startColumn<endColumn; startColumn++) {
		*pushLocation = sqlite3_column_double(_statement, startColumn);
		pushLocation += pushIncrement;
	}
}

void elm::SQLiteStmt::getBools (int startColumn, const int& endColumn,
						bool* pushLocation, const unsigned& pushIncrement)
{
	for (; startColumn<endColumn; startColumn++) {
		*pushLocation = (bool)sqlite3_column_int(_statement, startColumn);
		pushLocation += pushIncrement;
	}
}

string elm::SQLiteStmt::getText(int column)
{ 
	if (sqlite3_column_bytes(_statement, column))
		return string ( (const char*) sqlite3_column_text(_statement, column));
	else return "";
}

const void* elm::SQLiteStmt::getBlob(int column)
{ return sqlite3_column_blob(_statement, column); }

int elm::SQLiteStmt::getByteCount(int column)
{ return sqlite3_column_bytes(_statement, column); }

double elm::SQLiteStmt::simpleDouble(const std::string& sql, const double& defaultvalue)
{
	if (sql!="") prepare(sql);
	execute();
	if (_status==SQLITE_DONE) return defaultvalue;
	if (_status!=SQLITE_ROW) oops();
	return getDouble(0);
}

int elm::SQLiteStmt::simpleInteger(const std::string& sql, const int& defaultvalue)
{
	if (sql!="") prepare(sql);
	execute();
	if (_status==SQLITE_DONE) return defaultvalue;
	if (_status!=SQLITE_ROW) oops();
	return getInt(0);
}
long long elm::SQLiteStmt::simpleInt64(const std::string& sql, const int& defaultvalue)
{
	if (sql!="") prepare(sql);
	execute();
	if (_status==SQLITE_DONE) return defaultvalue;
	if (_status!=SQLITE_ROW) oops();
	return getInt64(0);
}
std::string elm::SQLiteStmt::simpleText(const std::string& sql, const std::string& defaultvalue)
{
	if (sql!="") prepare(sql);
	execute();
	if (_status==SQLITE_DONE) return defaultvalue;
	if (_status!=SQLITE_ROW) oops();
	return getText(0);
}

double elm::SQLiteStmt::simpleDouble(const std::string& sql)
{
	if (sql!="") prepare(sql);
	execute();
	if (_status!=SQLITE_ROW) oops();
	return getDouble(0);
}

int elm::SQLiteStmt::simpleInteger(const std::string& sql)
{
	if (sql!="") prepare(sql);
	execute();
	if (_status!=SQLITE_ROW) oops();
	return getInt(0);
}
long long elm::SQLiteStmt::simpleInt64(const std::string& sql)
{
	if (sql!="") prepare(sql);
	execute();
	if (_status!=SQLITE_ROW) oops();
	return getInt64(0);
}
std::string elm::SQLiteStmt::simpleText(const std::string& sql)
{
	if (sql!="") prepare(sql);
	execute();
	if (_status!=SQLITE_ROW) oops();
	return getText(0);
}


std::vector<long long> elm::SQLiteDB::eval_int64_tuple(const std::string& sql) const
{
	std::vector< long long> ret;
	SQLiteStmtPtr s = sql_statement(sql);
	s->execute();
	while (s->status()==SQLITE_ROW) {
		try { 
			ret.push_back(s->getInt64(0));
			s->execute();
		}
		SPOO {
			s->execute();
			continue;
		}
	}
	return ret;

}

std::vector<std::string> elm::SQLiteDB::eval_string_tuple(const std::string& sql) const
{
	std::vector<std::string> ret;
	SQLiteStmtPtr s = sql_statement(sql);
	s->execute();
	while (s->status()==SQLITE_ROW) {
		try { 
			ret.push_back(s->getText(0));
			s->execute();
		}
		SPOO {
			s->execute();
			continue;
		}
	}
	return ret;

}


