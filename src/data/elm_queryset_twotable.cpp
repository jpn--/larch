//
//  elm_queryset_twotable.cpp
//  Hangman
//
//  Created by Jeffrey Newman on 4/24/14.
//  Copyright (c) 2014 Jeffrey Newman. All rights reserved.
//

#include "elm_queryset_twotable.h"
#include "elm_sql_facet.h"

#include "elm_swig_external.h"

//std::shared_ptr<elm::QuerySetTwoTable> elm::QuerySetTwoTable::create(elm::Facet* validator)
//{
//	return std::make_shared<elm::QuerySetTwoTable>(validator);
//}


elm::QuerySetTwoTable::QuerySetTwoTable(elm::Facet* validator)
: QuerySet (validator)
{
	
}

void elm::QuerySetTwoTable::set_validator(elm::Facet* validator)
{
	this->validator = validator;
}


std::string elm::QuerySetTwoTable::__repr__() const
{
	return "<larch.core.QuerySetTwoTable>";
}

std::string elm::QuerySetTwoTable::actual_type() const
{
	return "QuerySetTwoTable";
}

PyObject* elm::QuerySetTwoTable::pickled() const
{
	// base is actually a downcasted pointer to self...
	//elm::QuerySetTwoTable* self = dynamic_cast<elm::QuerySetTwoTable*>(&*base);

	//static swig_type_info _swigt__p_std__shared_ptrT_elm__QuerySetSimpleCO_t = {"_p_std__shared_ptrT_elm__QuerySetSimpleCO_t", "elm::queries1 *|std::shared_ptr< elm::QuerySetSimpleCO > *", 0, 0, (void*)0, 0};
	//static swig_type_info _swigt__p_std__shared_ptrT_elm__QuerySetTwoTable_t = {"_p_std__shared_ptrT_elm__QuerySetTwoTable_t", "elm::queries2 *|std::shared_ptr< elm::QuerySetTwoTable > *", 0, 0, (void*)0, 0};
	//static swig_type_info _swigt__p_elm__QuerySetSimpleCO = {"_p_elm__QuerySetSimpleCO", "elm::QuerySetSimpleCO *", 0, 0, (void*)0, 0};
	//static swig_type_info _swigt__p_elm__QuerySetTwoTable = {"_p_elm__QuerySetTwoTable", "elm::QuerySetTwoTable *", 0, 0, (void*)0, 0};


    PyObject* resultobj = SWIG_NewPointerObj((void*)this, SWIG_TypeQuery("_p_elm__QuerySetTwoTable"), SWIG_POINTER_OWN);
	return resultobj;
}


std::string elm::QuerySetTwoTable::qry_idco   () const
{
	if (!_idco_query.empty()) {
		if (validator) validator->sql_statement(_idco_query);
		return _idco_query;
	}
	
	std::string s = "SELECT DISTINCT "+__alias_caseid_ca()+" AS caseid FROM ("+qry_idca()+")";
	if (validator) validator->sql_statement(s);
	return s;
}

std::string elm::QuerySetTwoTable::qry_idco_   () const
{
	if (!_idco_query.empty()) {
		if (validator) validator->sql_statement(_idco_query);
		
		std::string alias = __alias_caseid_co();
		if (alias != "caseid") {
			return "SELECT "+alias+" AS caseid, * FROM ("+qry_idco()+")";
		}
		
		return _idco_query;
	}
	
	std::string s = "SELECT DISTINCT "+__alias_caseid_ca()+" AS caseid FROM ("+qry_idca()+")";
	if (validator) validator->sql_statement(s);
	return s;
}

std::string elm::QuerySetTwoTable::qry_idca   () const
{
	if (_idca_query.empty()) {
		return "SELECT NULL AS caseid, NULL AS altid LIMIT 0";
	}

	if (validator) validator->sql_statement(_idca_query);
	return _idca_query;
}

std::string elm::QuerySetTwoTable::qry_idca_   () const
{
	if (_idca_query.empty()) {
		return "SELECT NULL AS caseid, NULL AS altid LIMIT 0";
	}
	
	std::string alias_cid = __alias_caseid_ca();
	std::string alias_aid = __alias_altid_ca();
	
	if (alias_cid!="caseid" && alias_aid!="altid") {
		return "SELECT "+alias_cid+" AS caseid, "+alias_aid+" AS altid, * FROM ("+qry_idca()+")";
	}
	if (alias_cid=="caseid" && alias_aid!="altid") {
		return "SELECT "+alias_aid+" AS altid, * FROM ("+qry_idca()+")";
	}
	if (alias_cid!="caseid" && alias_aid=="altid") {
		return "SELECT "+alias_cid+" AS caseid, * FROM ("+qry_idca()+")";
	}
	
	if (validator) validator->sql_statement(_idca_query);
	return _idca_query;
}

std::string elm::QuerySetTwoTable::qry_alts   () const
{
	std::string qry = _alts_query;
	if (qry.empty()) {
		qry = "SELECT NULL as id, NULL as name LIMIT 0";
	}
	
	if (validator) validator->sql_statement(qry);
	return qry;
}

std::string elm::QuerySetTwoTable::__test_query_caseids(const std::string& alias_caseid) const
{
	std::string s = "SELECT ";
	s += alias_caseid;
	s += " AS caseid FROM " + tbl_idco();
	if (validator) validator->sql_statement(s);
	return s;
}

std::string elm::QuerySetTwoTable::__alias_caseid_co() const
{
	try {
		__test_query_caseids("caseid");
		return "caseid";
	} catch (etk::SQLiteError) {
		return validator->sql_statement(qry_idco())->column_name(0);
	}
}

std::string elm::QuerySetTwoTable::__alias_caseid_ca() const
{
	std::string s = "SELECT caseid FROM " + tbl_idca();
	try {
		if (validator) validator->sql_statement(s);
		return "caseid";
	} catch (etk::SQLiteError) {
		return validator->sql_statement(qry_idca())->column_name(0);
	}
}

std::string elm::QuerySetTwoTable::__alias_altid_ca() const
{
	std::string s = "SELECT altid FROM " + tbl_idca();
	try {
		if (validator) validator->sql_statement(s);
		return "altid";
	} catch (etk::SQLiteError) {
		return validator->sql_statement(qry_idca())->column_name(1);
	}
}


std::string elm::QuerySetTwoTable::qry_caseids() const
{
	try {
		return __test_query_caseids("caseid");
	} catch (etk::SQLiteError) {
		return __test_query_caseids( validator->sql_statement(qry_idco())->column_name(0) );
	}
}

std::string elm::QuerySetTwoTable::qry_choice () const
{
	
	if (!_single_choice_column.empty()) {
		std::string s =
		"SELECT "+ __alias_caseid_co()+" AS caseid, "
		+_single_choice_column+" AS altid, 1 AS choice FROM "+tbl_idco();
		
		if (validator) validator->sql_statement(s);
		return s;
	}
	
	if (!_alt_avail_columns.empty()) {
		bool joiner = false;
		std::ostringstream s;
		std::string alias_caseid = __alias_caseid_co();
		for (auto i=_alt_avail_columns.begin(); i!=_alt_avail_columns.end(); i++) {
			if (joiner) {
				s << "\n UNION ALL \n";
			} else {
				joiner = true;
			}
			s << "SELECT "<<alias_caseid<<" AS caseid, " << i->first<<" AS altid, "<< i->second << " AS choice FROM "+tbl_idco();
		}
		if (validator) validator->sql_statement(s.str());
		return s.str();
	}

	if (!_choice_ca_column.empty()) {
		std::string s =
		"SELECT "+ __alias_caseid_ca()+" AS caseid, "+ __alias_altid_ca()+" AS altid, "+_choice_ca_column+" AS choice FROM "+tbl_idca();
		
		if (validator) validator->sql_statement(s);
		return s;
	}
	
	OOPS("choice query not set");
}

std::string elm::QuerySetTwoTable::qry_weight () const
{
	if (!_weight_column.empty()) {
		std::string s =
		"SELECT "+ __alias_caseid_co()+" AS caseid, "+_weight_column+" AS weight FROM "+tbl_idco();
		
		if (validator) validator->sql_statement(s);
		return s;
	}
	
	return
	"SELECT "+ __alias_caseid_co()+" AS caseid, 1 AS weight FROM "+tbl_idco();
}

std::string elm::QuerySetTwoTable::qry_avail  () const
{

	if (!_alt_avail_ca_column.empty()) {
		std::ostringstream s;
		s << "SELECT "<<__alias_caseid_ca()<<" AS caseid, ";
		s << __alias_altid_ca()<<" AS altid, ";
		s << _alt_avail_ca_column << " AS choice FROM "+tbl_idca();
		if (validator) validator->sql_statement(s.str());
		return s.str();
	}


	if (!_alt_avail_columns.empty()) {
		bool joiner = false;
		std::ostringstream s;
		std::string alias_caseid = __alias_caseid_co();
		for (auto i=_alt_avail_columns.begin(); i!=_alt_avail_columns.end(); i++) {
			if (joiner) {
				s << "\n UNION ALL \n";
			} else {
				joiner = true;
			}
			s << "SELECT "<<alias_caseid<<" AS caseid, " << i->first<<" AS altid, "<< i->second << " AS avail FROM "+tbl_idco();
		}
		if (validator) validator->sql_statement(s.str());
		return s.str();
	}
	
	return "";
}


elm::QuerySetTwoTable::~QuerySetTwoTable()
{
	
}

void elm::QuerySetTwoTable::set_idco_query(const std::string& q)
{
	if (_idco_query == q) {
		return;
	}
	
	if (validator) validator->sql_statement(q);
	_idco_query = q;
	
	if (validator) validator->change_in_sql_idco();
}

void elm::QuerySetTwoTable::set_idca_query(const std::string& q)
{
	if (_idca_query == q) {
		return;
	}

	if (validator) validator->sql_statement(q);
	_idca_query = q;
	
	if (validator) validator->change_in_sql_idca();

}

void elm::QuerySetTwoTable::set_choice_co_column(const std::string& col)
{
	if (col.empty()) {
		return;
	}
	
	if (_single_choice_column == col) {
		return;
	}
	
	auto t1 = _alt_choice_columns;
	auto t2 = _single_choice_column;
	auto t3 = _choice_ca_column;
	_alt_choice_columns.clear();
	_single_choice_column = col;
	_choice_ca_column.clear();
	try {
		qry_choice();
	} catch (etk::SQLiteError) {
		_alt_choice_columns = t1;
		_single_choice_column = t2;
		_choice_ca_column = t3;
		throw;
	}

	if (validator) validator->change_in_sql_choice();
	
}

void elm::QuerySetTwoTable::set_choice_co_column_map(const std::map<long long, std::string>& cols)
{
	if (cols.empty()) {
		return;
	}
	
	if (_alt_choice_columns == cols) {
		return;
	}

	auto t1 = _alt_choice_columns;
	auto t2 = _single_choice_column;
	auto t3 = _choice_ca_column;
	_alt_choice_columns = cols;
	_single_choice_column = "";
	_choice_ca_column.clear();
	try {
		qry_choice();
	} catch (etk::SQLiteError) {
		_alt_choice_columns = t1;
		_single_choice_column = t2;
		_choice_ca_column = t3;
		throw;
	}
	
	if (validator) validator->change_in_sql_choice();
}

void elm::QuerySetTwoTable::set_choice_ca_column(const std::string& col)
{
	if (col.empty()) {
		return;
	}
	if (_choice_ca_column == col) {
		return;
	}
	
	auto t1 = _alt_choice_columns;
	auto t2 = _single_choice_column;
	auto t3 = _choice_ca_column;
	_alt_choice_columns.clear();
	_single_choice_column = "";
	_choice_ca_column = col;
	try {
		qry_choice();
	} catch (etk::SQLiteError) {
		_alt_choice_columns = t1;
		_single_choice_column = t2;
		_choice_ca_column = t3;
		throw;
	}
	if (validator) validator->change_in_sql_choice();
}


void elm::QuerySetTwoTable::set_avail_co_column_map(const std::map<long long, std::string>& cols)
{
	if (cols.empty()) {
		return;
	}
	
	if (_alt_avail_columns == cols) {
		return;
	}
	
	auto t1 = _alt_avail_columns;
	auto t2 = _alt_avail_ca_column;
	_alt_avail_columns = cols;
	_alt_avail_ca_column.clear();
	try {
		qry_avail();
	} catch (etk::SQLiteError) {
		_alt_avail_columns = t1;
		_alt_avail_ca_column = t2;
		throw;
	}
	if (validator) validator->change_in_sql_avail();
}

void elm::QuerySetTwoTable::set_avail_ca_column(const std::string& col)
{
	if (col.empty()) {
		return;
	}
	if (_alt_avail_ca_column == col) {
		return;
	}
	
	// save current values to restore in case of failure
	auto t1 = _alt_avail_columns;
	auto t2 = _alt_avail_ca_column;
	
	// set new values
	_alt_avail_columns.clear();
	_alt_avail_ca_column = col;
	
	// test new value
	try {
		qry_avail();
	} catch (etk::SQLiteError) {
		// restore prior values if error
		_alt_avail_columns = t1;
		_alt_avail_ca_column = t2;
		throw;
	}
	
	// Notify the validator that the value has been changed
	if (validator) validator->change_in_sql_avail();
}

void elm::QuerySetTwoTable::set_avail_all()
{
	bool reload = false;
	
	if (!_alt_avail_columns.empty()) {
		reload = true;
	}
	
	_alt_avail_columns.clear();
	
	if (reload) {
		if (validator) {
			validator->change_in_sql_avail();
		}
	}
}


void elm::QuerySetTwoTable::set_weight_co_column(const std::string& col)
{
	if (col.empty()) {
		return;
	}
	if (_weight_column == col) {
		return;
	}

	std::string temp_w = _weight_column;
	_weight_column = col;
	try {
		qry_weight();
	} catch (etk::SQLiteError) {
		_weight_column = temp_w;
		throw;
	}
//	if (validator) validator->change_in_sql_weight();
}

void elm::QuerySetTwoTable::set_alts_query(const std::string& q)
{
	if (q.empty() || _alts_query == q) {
		return;
	}

	if (validator) validator->sql_statement(q);
	_alts_query = q;
	
//	if (validator) validator->change_in_sql_alts();

}

void elm::QuerySetTwoTable::set_alts_values(const std::map<long long, std::string>& alts)
{
	if (alts.empty()) {
		return;
	}
	
	std::ostringstream s;
	bool joiner = false;
	for (auto i=alts.begin(); i!=alts.end(); i++) {
		if (joiner) {
			s << " UNION ALL ";
		} else {
			joiner = true;
		}
		s << "SELECT "<<i->first<<" AS id, \""<< i->second << "\" AS name";
	}
	
	if (s.str() == _alts_query) {
		return;
	}
	
	if (validator) validator->sql_statement(s.str());
	_alts_query = s.str();
	if (validator) validator->change_in_sql_alts();
}






bool elm::QuerySetTwoTable::unweighted() const
{
	if (_weight_column.empty() || _weight_column=="1") {
		return true;
	}
	return false;
}
bool elm::QuerySetTwoTable::all_alts_always_available() const
{
	if (_alt_avail_columns.empty() && _alt_avail_ca_column=="") {
		return true;
	}
	return false;
}










std::string elm::QuerySetTwoTable::get_idco_query() const
{
	return _idco_query;
}

std::string elm::QuerySetTwoTable::get_idca_query() const
{
	return _idca_query;
}



std::string elm::QuerySetTwoTable::get_choice_co_column() const
{
	return _single_choice_column;
}


std::map<long long, std::string> elm::QuerySetTwoTable::get_choice_co_column_map() const
{
	return _alt_choice_columns;
}


std::string elm::QuerySetTwoTable::get_choice_ca_column() const
{
	return _choice_ca_column;
}



std::map<long long, std::string> elm::QuerySetTwoTable::get_avail_co_column_map() const
{
	return _alt_avail_columns;
}


std::string elm::QuerySetTwoTable::get_avail_ca_column() const
{
	return _alt_avail_ca_column;
}



std::string elm::QuerySetTwoTable::get_weight_co_column() const
{
	return _weight_column;
}



std::string elm::QuerySetTwoTable::get_alts_query() const
{
	return _alts_query;
}




