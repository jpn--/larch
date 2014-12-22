//
//  elm_sql_queryset_simpleco.cpp
//  Hangman
//
//  Created by Jeffrey Newman on 4/24/14.
//  Copyright (c) 2014 Jeffrey Newman. All rights reserved.
//

#include "elm_queryset_simpleco.h"
#include "elm_sql_facet.h"


//std::shared_ptr<elm::QuerySetSimpleCO> elm::QuerySetSimpleCO::create(elm::Facet* validator)
//{
//	return std::make_shared<elm::QuerySetSimpleCO>(validator);
//}


elm::QuerySetSimpleCO::QuerySetSimpleCO(elm::Facet* validator)
: QuerySet (validator)
{
	
}

void elm::QuerySetSimpleCO::set_validator(elm::Facet* validator)
{
	if (this->validator != validator) {
		// do something?
	}
	this->validator = validator;
}


std::string elm::QuerySetSimpleCO::__repr__() const
{
	return "<larch.core.QuerySetSimpleCO>";
}

std::string elm::QuerySetSimpleCO::actual_type() const
{
	return "QuerySetSimpleCO";
}


std::string elm::QuerySetSimpleCO::qry_idco   () const
{
	std::string qry = _idco_query;
	if (qry.empty()) {
		qry = "SELECT NULL as caseid LIMIT 0";
	}
	if (validator) validator->sql_statement(qry);
	return qry;
}

std::string elm::QuerySetSimpleCO::qry_idco_  () const
{
	std::string qry = _idco_query;
	if (qry.empty()) {
		qry = "SELECT NULL as caseid LIMIT 0";
	}
	
	std::string alias = __alias_caseid();
	if (alias!="caseid") {
		return "SELECT "+alias+" AS caseid, * FROM ("+qry_idco()+")";
	}
	
	if (validator) validator->sql_statement(qry);
	return qry;
}

std::string elm::QuerySetSimpleCO::qry_idca   () const
{
	return "SELECT NULL AS caseid, NULL AS altid LIMIT 0";
}

std::string elm::QuerySetSimpleCO::qry_idca_  () const
{
	return "SELECT NULL AS caseid, NULL AS altid LIMIT 0";
}

std::string elm::QuerySetSimpleCO::qry_alts   () const
{
	std::string qry = _alts_query;
	if (qry.empty()) {
		qry = "SELECT NULL as id, NULL as name LIMIT 0";
	}
	
	if (validator) validator->sql_statement(qry);
	return qry;
}

std::string elm::QuerySetSimpleCO::__test_query_caseids(const std::string& alias_caseid) const
{
	std::string s = "SELECT ";
	s += alias_caseid;
	s += " AS caseid FROM " + tbl_idco();
	if (validator) validator->sql_statement(s);
	return s;
}

std::string elm::QuerySetSimpleCO::__alias_caseid() const
{
	try {
		__test_query_caseids("caseid");
		return "caseid";
	} catch (etk::SQLiteError) {
		return validator->sql_statement(qry_idco())->column_name(0);
	}
}


std::string elm::QuerySetSimpleCO::qry_caseids() const
{
	try {
		return __test_query_caseids("caseid");
	} catch (etk::SQLiteError) {
		return __test_query_caseids( validator->sql_statement(qry_idco())->column_name(0) );
	}
}

std::string elm::QuerySetSimpleCO::qry_choice () const
{

	if (!_single_choice_column.empty()) {
		std::string s =
		"SELECT "+ __alias_caseid()+" AS caseid, "
		+_single_choice_column+" AS altid, 1 AS choice FROM "+tbl_idco();
		
		if (validator) validator->sql_statement(s);
		return s;
	}

	if (!_alt_avail_columns.empty()) {
		bool joiner = false;
		std::ostringstream s;
		std::string alias_caseid = __alias_caseid();
		for (auto i=_alt_avail_columns.begin(); i!=_alt_avail_columns.end(); i++) {
			if (joiner) {
				s << "\n UNION ALL \n";
			} else {
				joiner = true;
			}
			s << "SELECT "<<alias_caseid<<" AS caseid, " << i->first<<" AS altid, "<< i->second << " AS choice FROM "<<tbl_idco();
		}
		if (validator) validator->sql_statement(s.str());
		return s.str();
	}
	
	OOPS("choice query not set");
}

std::string elm::QuerySetSimpleCO::qry_weight () const
{
	if (!_weight_column.empty()) {
		std::string s =
		"SELECT "+ __alias_caseid()+" AS caseid, "+_weight_column+" AS weight FROM "+tbl_idco();
		
		if (validator) validator->sql_statement(s);
		return s;
	}
	
	return
	"SELECT "+ __alias_caseid()+" AS caseid, 1 AS weight FROM "+tbl_idco();
}

std::string elm::QuerySetSimpleCO::qry_avail  () const
{
	if (!_alt_avail_columns.empty()) {
		bool joiner = false;
		std::ostringstream s;
		std::string alias_caseid = __alias_caseid();
		for (auto i=_alt_avail_columns.begin(); i!=_alt_avail_columns.end(); i++) {
			if (joiner) {
				s << " UNION ALL ";
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


elm::QuerySetSimpleCO::~QuerySetSimpleCO()
{
	
}
	
void elm::QuerySetSimpleCO::set_idco_query(const std::string& q)
{
	bool reload = false;


	if (validator) {
		validator->sql_statement(q);
		if (_idco_query != q) {
			reload = true;
		}
	}
	_idco_query = q;
	
	if (reload) {
//		validator->change_in_sql_idco();
//		validator->change_in_sql_caseids();
	}
}

void elm::QuerySetSimpleCO::set_choice_column(const std::string& col)
{
	if (col.empty()) return;
	
	if (col==_single_choice_column) {
		return;
	}
	
	auto t1 = _alt_choice_columns;
	auto t2 = _single_choice_column;
	_alt_choice_columns.clear();
	_single_choice_column = col;
	try {
		qry_choice();
	} catch (etk::SQLiteError) {
		_alt_choice_columns = t1;
		_single_choice_column = t2;
		throw;
	}

	if (validator) {
//		validator->change_in_sql_choice();
	}

}

void elm::QuerySetSimpleCO::set_choice_column_map(const std::map<long long, std::string>& cols)
{
	if (cols.empty()) {
		return;
	}

	if (cols==_alt_choice_columns) {
		return;
	}

	auto t1 = _alt_choice_columns;
	auto t2 = _single_choice_column;
	_alt_choice_columns = cols;
	_single_choice_column = "";
	try {
		qry_choice();
	} catch (etk::SQLiteError) {
		_alt_choice_columns = t1;
		_single_choice_column = t2;
		throw;
	}
	
	if (validator) {
//		validator->change_in_sql_choice();
	}

}

void elm::QuerySetSimpleCO::set_avail_column_map(const std::map<long long, std::string>& cols)
{
	if (cols.empty()) {
		return;
	}
	
	if (_alt_avail_columns == cols) {
		return;
	}
	
	auto t1 = _alt_avail_columns;
	_alt_avail_columns = cols;
	try {
		qry_avail();
	} catch (etk::SQLiteError) {
		_alt_avail_columns = t1;
		throw;
	}
	
	if (validator) {
//		validator->change_in_sql_avail();
	}
}

void elm::QuerySetSimpleCO::set_avail_all()
{
	bool reload = false;
	
	if (!_alt_avail_columns.empty()) {
		reload = true;
	}
	
	_alt_avail_columns.clear();
	
	if (reload) {
		if (validator) {
//			validator->change_in_sql_avail();
		}
	}
}


void elm::QuerySetSimpleCO::set_weight_column(const std::string& col)
{

	if (col.empty()) {
		if (_weight_column.empty()) {
			return;
		}
		_weight_column.clear();
		if (validator) {
//			validator->change_in_sql_weight();
		}
		return;
	}
	
	if (col == _weight_column) {
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
	
	if (validator) {
//		validator->change_in_sql_weight();
	}
	
	
}

void elm::QuerySetSimpleCO::set_alts_query(const std::string& q)
{
	bool reload = false;

	if (validator) {
		validator->sql_statement(q);
		if (_alts_query != q) {
			reload = true;
		}
	}
	
	_alts_query = q;
	
	if (reload) {
//		validator->change_in_sql_alts();
	}
}

void elm::QuerySetSimpleCO::set_alts_values(const std::map<long long, std::string>& alts)
{
	bool reload = false;

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
	if (validator) {
		validator->sql_statement(s.str());
		if (_alts_query != s.str()) {
			reload = true;
		}
	}
	_alts_query = s.str();

	if (reload) {
//		validator->change_in_sql_alts();
	}
}




bool elm::QuerySetSimpleCO::unweighted() const
{
	if (_weight_column.empty() || _weight_column=="1") {
		return true;
	}
	return false;
}
bool elm::QuerySetSimpleCO::all_alts_always_available() const
{
	if (_alt_avail_columns.empty()) {
		return true;
	}
	return false;
}





std::string elm::QuerySetSimpleCO::get_idco_query() const
{
	return _idco_query;
}

std::string elm::QuerySetSimpleCO::get_choice_column() const
{
	return _single_choice_column;
}


std::map<long long, std::string> elm::QuerySetSimpleCO::get_choice_column_map() const
{
	return _alt_choice_columns;
}


std::map<long long, std::string> elm::QuerySetSimpleCO::get_avail_column_map() const
{
	return _alt_avail_columns;
}


std::string elm::QuerySetSimpleCO::get_weight_column() const
{
	return _weight_column;
}


std::string elm::QuerySetSimpleCO::get_alts_query() const
{
	return _alts_query;
}




