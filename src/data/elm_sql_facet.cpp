/*
 *  elm_sql_facet.cpp
 *
 *  Copyright 2007-2013 Jeffrey Newman
 *
 *  This file is part of ELM.
 *  
 *  ELM is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  ELM is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with ELM.  If not, see <http://www.gnu.org/licenses/>.
 *  
 */

#include <unordered_map>
#include <unordered_set>

#include "elm_sql_facet.h"
#include "elm_sql_scrape.h"

#include "elm_queryset_simpleco.h"
#include "elm_queryset_twotable.h"

#include "elm_swig_external.h"

elm::Facet::Facet(PyObject* pylong_ptr_to_db)
: SQLiteDB(pylong_ptr_to_db)
, Fountain()
, sql_caseids()
, sql_alts()
, sql_idco()
, sql_idca()
, sql_choice()
, sql_avail()
, _nCases(0)
, _nAlts(0)
, window_title()
, source_filename()
, working_name()
, active_facet()
//, _caseindex(nullptr)
, queries(nullptr)
, queries_ptr(nullptr)
{
	try {
//		load_facet();
	} SPOO {
//		clear_facet();
	}
	
}


elm::Facet::~Facet()
{
	Py_CLEAR(queries);
}






//std::string elm::Facet::get_sql_alts() const
//{
//	return sql_alts;
//}
//
//std::string elm::Facet::get_sql_caseids() const
//{
//	if (queries_ptr) {
//		return queries_ptr->qry_caseids();
//	} else {
//		return "-- error: queries_ptr not set --";
//	}
//	return sql_caseids;
//}
//
//std::string elm::Facet::get_sql_idco() const
//{
//	return sql_idco;
//}
//
//
//std::string elm::Facet::get_sql_idca() const
//{
//	return sql_idca;
//}
//
//
//std::string elm::Facet::get_sql_choice() const
//{
//	return sql_choice;
//}
//
//
//std::string elm::Facet::get_sql_avail() const
//{
//	return sql_avail;
//}
//
//
//std::string elm::Facet::get_sql_weight() const
//{
//	return sql_weight;
//}
//
//
//
//
//void elm::Facet::set_sql_caseids(const std::string& qry)
//{
//	if (sql_caseids != qry) {
//		sql_caseids = qry;
//	}
//}
//
//
//void elm::Facet::set_sql_alts(const std::string& qry)
//{
//	if (sql_alts != qry) {
//		sql_alts = qry;
//		change_in_sql_idca();
//		change_in_sql_choice();
//		change_in_sql_avail();
//	}
//}
//
//
//void elm::Facet::set_sql_idco(const std::string& qry)
//{
//	if (sql_idco != qry) {
//		if (etk::to_uppercase(sql_choice).find("ELM_IDCO")!=std::string::npos) {
//			change_in_sql_choice();
//		}
//		if (etk::to_uppercase(sql_weight).find("ELM_IDCO")!=std::string::npos) {
//			change_in_sql_weight();
//		}
//		if (etk::to_uppercase(sql_avail).find("ELM_IDCO")!=std::string::npos) {
//			change_in_sql_avail();
//		}
//		sql_idco = qry;
//		change_in_sql_idco();
//	}
//}
//
//
//void elm::Facet::set_sql_idca(const std::string& qry)
//{
//	if (sql_idca != qry) {
//		if (etk::to_uppercase(sql_choice).find("ELM_IDCA")!=std::string::npos) {
//			change_in_sql_choice();
//		}
//		if (etk::to_uppercase(sql_weight).find("ELM_IDCA")!=std::string::npos) {
//			change_in_sql_weight();
//		}
//		if (etk::to_uppercase(sql_avail).find("ELM_IDCA")!=std::string::npos) {
//			change_in_sql_avail();
//		}
//		sql_idca = qry;
//		change_in_sql_idca();
//	}
//}
//
//
//void elm::Facet::set_sql_choice(const std::string& qry)
//{
//	if (sql_choice != qry) {
//		sql_choice = qry;
//		change_in_sql_choice();
//	}
//}
//
//
//void elm::Facet::set_sql_avail(const std::string& qry)
//{
//	if (sql_avail != qry) {
//		sql_avail = qry;
//		change_in_sql_avail();
//	}
//}
//
//
//void elm::Facet::set_sql_weight(const std::string& qry)
//{
//	if (sql_weight != qry) {
//		sql_weight = qry;
//		change_in_sql_weight();
//	}
//}




void elm::Facet::change_in_sql_caseids()
{
	
	_nCases = eval_integer("SELECT count(*) FROM "+tbl_idco(),0);
	
	SQLiteStmtPtr s = sql_statement("");
	
//	if (_caseindex) {
//		_caseindex.reset();
//	}
	
//	_caseindex = caseindex_t::create();
//	_caseindex->add_caseids(caseids(0,0,1));
}

void elm::Facet::change_in_sql_alts()
{
		
	SQLiteStmtPtr s = sql_statement(qry_alts());
	
	INFO(msg)<< "Reading alternatives..." ;
	_Data_DNA.clear();
	s->execute();
	while (s->status() == SQLITE_ROW) {
		_Data_DNA.add_cell(s->getInt64(0),s->getText(1));
		s->execute();
	}
	_Data_DNA.regrow();
	_nAlts = _Data_DNA.n_elemental();
	INFO(msg)<< "Found "<< _nAlts <<" elemental alternatives." ;
}

void elm::Facet::change_in_sql_idco()
{
//	for (auto i=_extracts.begin(); i!=_extracts.end(); ) {
//		if ( (*i)->dimty==case_var
//		  && (*i)->dtype==mtrx_double
//		  && (*i)->dpurp==purp_vars
//		) {
//			i = _extracts.erase(i);
//		} else {
//			i++;
//		}
//	}
}

void elm::Facet::change_in_sql_idca()
{
//	for (auto i=_extracts.begin(); i!=_extracts.end(); ) {
//		if ( (*i)->dimty==case_alt_var
//			&& (*i)->dtype==mtrx_double
//			&& (*i)->dpurp==purp_vars
//			) {
//			i = _extracts.erase(i);
//		} else {
//			i++;
//		}
//	}
}


void elm::Facet::change_in_sql_choice()
{
//	for (auto i=_extracts.begin(); i!=_extracts.end(); ) {
//		if ( (*i)->dpurp==purp_choice ) {
//			i = _extracts.erase(i);
//		} else {
//			i++;
//		}
//	}
}


void elm::Facet::change_in_sql_avail()
{
//	for (auto i=_extracts.begin(); i!=_extracts.end(); ) {
//		if ( (*i)->dpurp==purp_avail ) {
//			i = _extracts.erase(i);
//		} else {
//			i++;
//		}
//	}
}


void elm::Facet::change_in_sql_weight()
{
//	for (auto i=_extracts.begin(); i!=_extracts.end(); ) {
//		if ( (*i)->dpurp==purp_weight ) {
//			i = _extracts.erase(i);
//		} else {
//			i++;
//		}
//	}
}








void elm::Facet::save_facet(std::string name)
{
	_create_facet_table_if_not_exists();
	sql_statement("INSERT OR REPLACE INTO elm_facets VALUES (?,'alts',?)")->
		bind_text(1, name)->
		bind_text(2, sql_alts)->
		execute_until_done();
	sql_statement("INSERT OR REPLACE INTO elm_facets VALUES (?,'idco',?)")->
		bind_text(1, name)->
		bind_text(2, sql_idco)->
		execute_until_done();
	sql_statement("INSERT OR REPLACE INTO elm_facets VALUES (?,'idca',?)")->
		bind_text(1, name)->
		bind_text(2, sql_idca)->
		execute_until_done();
	sql_statement("INSERT OR REPLACE INTO elm_facets VALUES (?,'avail',?)")->
		bind_text(1, name)->
		bind_text(2, sql_avail)->
		execute_until_done();
	sql_statement("INSERT OR REPLACE INTO elm_facets VALUES (?,'weight',?)")->
		bind_text(1, name)->
		bind_text(2, sql_weight)->
		execute_until_done();
	sql_statement("INSERT OR REPLACE INTO elm_facets VALUES (?,'choice',?)")->
		bind_text(1, name)->
		bind_text(2, sql_choice)->
		execute_until_done();

	active_facet = name;
}



void elm::Facet::load_facet(std::string name)
{
	std::vector<std::string> f (list_facets());

	if (name=="") {
		for (auto i=f.begin(); i!=f.end(); i++) {
			if (etk::to_uppercase(*i)=="DEFAULT") {
				name = *i;
				break;
			}
		}
	}

	if (name=="") {
		if (f.size()!=1) OOPS_FACET("A facet name must be specified unless there is one named 'default' or only one available");
		name = f[0];
	}
	
	std::vector<std::string>::const_iterator f_iter;
	for (f_iter=f.begin(); f_iter!=f.end(); f_iter++) {
		if (*f_iter == name) break;
	}
	if (f_iter!=f.end()) {
		MONITOR_(&msg, "loading "<<name<<" facet");
		active_facet = name;
	}
	
	std::string idco =
	sql_statement("SELECT value FROM elm_facets WHERE facet=? AND component='idco';")->
	bind_text(1, name)->
	execute()->
	getText(0);
	
	std::string idca =
	sql_statement("SELECT value FROM elm_facets WHERE facet=? AND component='idca';")->
	bind_text(1, name)->
	execute()->
	getText(0);
	
//	if (idca.empty()) {
//		queries = elm::QuerySetSimpleCO::create();
//	} else {
//		queries = elm::QuerySetSimpleCO::create();
//	}
	
	

	sql_alts = 
	sql_statement("SELECT value FROM elm_facets WHERE facet=? AND component='alts';")->
		bind_text(1, name)->
		execute()->
		getText(0);
	
	sql_idco = 
	sql_statement("SELECT value FROM elm_facets WHERE facet=? AND component='idco';")->
		bind_text(1, name)->
		execute()->
		getText(0);

	sql_idca =
	sql_statement("SELECT value FROM elm_facets WHERE facet=? AND component='idca';")->
		bind_text(1, name)->
		execute()->
		getText(0);

	sql_avail =
	sql_statement("SELECT value FROM elm_facets WHERE facet=? AND component='avail';")->
		bind_text(1, name)->
		execute()->
		getText(0);

	sql_weight =
	sql_statement("SELECT value FROM elm_facets WHERE facet=? AND component='weight';")->
		bind_text(1, name)->
		execute()->
		getText(0);
	
	sql_choice =
	sql_statement("SELECT value FROM elm_facets WHERE facet=? AND component='choice';")->
		bind_text(1, name)->
		execute()->
		getText(0);
		
//	refresh();
}

void elm::Facet::clear_facet()
{
	sql_alts = SQL_ALTS_BLANK;
	sql_idco = SQL_IDCO_BLANK;
	sql_idca = SQL_IDCA_BLANK;
	sql_avail = SQL_AVAIL_BLANK;
	sql_weight = SQL_WEIGHT_BLANK;
	sql_choice = SQL_CHOICE_BLANK;
//	refresh();
}


PyObject* elm::Facet::_get_queries()
{
	if (!queries) {
		Py_RETURN_NONE;
	}
	Py_XINCREF(queries);
	return queries;
}
void elm::Facet::_set_queries(PyObject* q, QuerySet* qp)
{
	bool time_to_update = false;

	if (queries_ptr != qp && qp) {
		time_to_update=true;
	}

	if (qp) {
		qp->set_validator(this);
	}
	
	Py_XINCREF(q);
	Py_CLEAR(queries);
	queries = q;
	queries_ptr = qp;
	
	if (time_to_update) {
		refresh_queries();
	}
}

void elm::Facet::refresh_queries()
{
	change_in_sql_alts();
	change_in_sql_avail();
	change_in_sql_caseids();
	change_in_sql_choice();
	change_in_sql_idca();
	change_in_sql_idco();
	change_in_sql_weight();
}

void elm::Facet::_create_facet_table_if_not_exists()
{
	sql_statement("\
CREATE TABLE IF NOT EXISTS elm_facets (\n\
  facet TEXT NOT NULL,\n\
  component TEXT NOT NULL,\n\
  value TEXT,\n\
  PRIMARY KEY(facet,component)\n\
)")->execute_until_done();

}

std::vector<std::string> elm::Facet::list_facets() const
{
	std::vector<std::string> f;
	SQLiteStmtPtr stmt = sql_statement("");
	try {
		stmt->prepare("SELECT DISTINCT facet FROM elm_facets ORDER BY facet;");
	} catch (etk::SQLiteError err) {
		return f;
	}
	stmt->execute();
	while (stmt->status()==SQLITE_ROW) {
		f.push_back(stmt->getText(0));
		stmt->execute();
	}
	return f;
}





//void elm::Facet::_init_alts_query()
//{
//	std::string use_query = sql_alts;
//	// If a query for alternatives codes and names is not defined but the idca table is, use that table's values by default.
//	if ((use_query.empty()) && (!sql_idca.empty())) {
//		use_query = "SELECT DISTINCT altid, 'alt'||altid from elm_idca ORDER BY altid";
//	}
//	
//	drop("elm_alternatives");
//	
//	std::ostringstream sql;
//	sql<<"CREATE TEMPORARY VIEW elm_alternatives AS "<<use_query;
//	sql_statement(sql)->execute();
//}

//void elm::Facet::_init_idco_query()
//{
//	drop("elm_idco");
//	std::string use_query = sql_idco;
//	// If a query for idca is not defined but the idco table is, assume all alternatives are available to all cases.
//	if ((use_query.empty()) && (!sql_idca.empty())) {
//		use_query = "SELECT DISTINCT caseid FROM elm_idca";
//	}
//	std::ostringstream sql;
//	sql<<"CREATE TEMPORARY VIEW elm_idco AS "<<use_query;
//	sql_statement(sql)->execute();
//	
//}


//void elm::Facet::_init_idca_query()
//{
//	drop("elm_idca");
//	std::string use_query = sql_idca;
//	// If a query for idca is not defined but the idco table is, assume all alternatives are available to all cases.
//	if ((use_query.empty()) && (!sql_idco.empty()) && (!sql_alts.empty())) {
//		use_query = "SELECT elm_idco."+alias_idco_caseid()+" AS caseid"+
//					", elm_alternatives."+alias_alts_altid()+" AS altid"+
//					" FROM elm_idco, elm_alternatives";
//	}
//	std::ostringstream sql;
//	sql<<"CREATE TEMPORARY VIEW elm_idca AS "<<use_query;
//	sql_statement(sql)->execute();
//	
//}

//void elm::Facet::_init_avail_query()
//{
//	if (!queries_ptr) OOPS_FACET("queries not defined");
//	drop("elm_avail");
//	std::string use_query = sql_avail;
//	// If a query for avail is not defined, assume all alternatives are available to all cases.
//	if ((use_query.empty()) && (!sql_idco.empty()) && (!sql_alts.empty())) {
//		use_query = "SELECT elm_idco."+alias_idco_caseid()+" AS caseid"+
//					", elm_alternatives."+alias_alts_altid()+" AS altid"+
//					", 1 AS avail FROM elm_idco, elm_alternatives";
//	}
//	std::ostringstream sql;
//	sql<<"CREATE TEMPORARY VIEW elm_avail AS "<<use_query;
//	try {
//		sql_statement(sql)->execute();
//	} catch (etk::SQLiteError) {
//		std::string caseid_alias = sql_statement("SELECT * FROM elm_idco")->column_name(0);
//		std::string altid_alias = sql_statement("SELECT * FROM ("+queries_ptr->qry_alts()+") AS elm_alternatives")->column_name(0);
//		use_query = "SELECT elm_idco."+caseid_alias+", elm_alternatives."+altid_alias+", 1 FROM elm_idco, ("+queries_ptr->qry_alts()+") AS elm_alternatives";
//		std::ostringstream sql;
//		sql<<"CREATE TEMPORARY VIEW elm_avail AS "<<use_query;
//		sql_statement(sql)->execute();
//	}
//}






//void elm::Facet::_init_choice_query()
//{
//	drop("elm_choice");
//	std::string use_query = sql_choice;
//	// If a query for choice is not defined, assume nothing is chosen.
//	if ((use_query.empty()) && (!sql_idco.empty()) && (!sql_alts.empty())) {
//		use_query = "SELECT NULL AS caseid, NULL AS altid, NULL AS choice LIMIT 0;";
//	}
//	std::ostringstream sql;
//	sql<<"CREATE TEMPORARY VIEW elm_choice AS "<<use_query;
//	sql_statement(sql)->execute();
//}

//void elm::Facet::_init_weight_query()
//{
//	drop("elm_weight_raw");
//	drop("elm_weight");
//	std::string use_query = sql_weight;
//	// If a query for weight is not defined, assume all equal to 1.
//	if (use_query.empty() || etk::to_uppercase(use_query)=="_EQUAL_") {
//		use_query = etk::cat("SELECT elm_idco.",alias_idco_caseid()," AS caseid, 1 AS weight FROM elm_idco;");
//	}
//	if (read_setup("auto_reweight")=="1") {
//		std::ostringstream sql;
//		sql<<"CREATE TEMPORARY VIEW elm_weight_raw AS "<<use_query;
//		sql_statement(sql)->execute();
//		double totwgt = sql_statement("SELECT sum(weight) FROM elm_weight_raw;")->simpleDouble("");
//		int cntwgt = sql_statement("SELECT count(weight) FROM elm_weight_raw;")->simpleInteger("");
//		sql.str(""); sql.clear();
//		sql << "CREATE TEMPORARY VIEW elm_weight AS SELECT caseid, weight*"<<cntwgt<<"/"<<totwgt<<" AS weight FROM elm_weight_raw;";
//		sql_statement(sql)->execute();
//	} else {
//		std::ostringstream sql;
//		sql<<"CREATE TEMPORARY VIEW elm_weight AS "<<use_query;
//		sql_statement(sql)->execute();
//	}
//}


//void elm::Facet::init_facet_queries()
//{
//	if (sql_idco.empty() && sql_idca.empty())
//		OOPS_FACET("Neither idca nor idco queries are given, you must have at least one of these");
//	if (sql_alts.empty() && sql_idca.empty())
//		OOPS_FACET("Neither idca nor alts queries are given, you must have at least one of these");
//
//	// If IDCO is given, set it first, otherwise set it last.
//	if (!sql_idco.empty()) {
//		_init_idco_query();
//		// If ALTS are not given, set them after setting IDCA, otherwise set them first.
//		if (sql_alts.empty()) {
//			_init_idca_query();
//			_init_alts_query();
//		} else {
//			_init_alts_query();
//			_init_idca_query();
//		}
//	} else {
//		if (sql_alts.empty()) {
//			_init_idca_query();
//			_init_alts_query();
//		} else {
//			_init_alts_query();
//			_init_idca_query();
//		}
//		_init_idco_query();
//	}
//	_init_avail_query();
//	_init_choice_query();
//	_init_weight_query();
//}









//
//std::string& _strip_semicolons(std::string& s)
//{
//	size_t i=s.find(";");
//	while (i != std::string::npos) {
//		s.erase(i,1);
//		i=s.find(";");
//	}
//	return s;
//}
//
//
//
//
//std::string& elm::Facet::build_misc_query(std::string& q) const
//{
//	size_t i=q.find(";");
//	while (i != std::string::npos) {
//		q.erase(i,1);
//		i=q.find(";");
//	}
//	sql_statement(q);
//	q = std::string("(")+q+std::string(")");
//	return q;
//}
//
//
//
//std::string elm::Facet::build_idca_query() const
//{
//
//	std::string use_query = sql_idca;
//	
//	// If a query for idca is not defined but the idco table is, assume all alternatives are available to all cases.
//	if (use_query.empty()) {
//		use_query = "SELECT NULL AS caseid, NULL AS altid";
//	}
//	
//	return build_misc_query(use_query);
//}
//
//std::string elm::Facet::build_idco_query() const
//{
//
//	std::string use_query = sql_idco;
//	
//	// If a query for idca is not defined but the idco table is, assume all alternatives are available to all cases.
//	if ((use_query.empty()) && (!sql_idca.empty())) {
//		use_query = "SELECT NULL AS caseid";
//	}
//
//	return build_misc_query(use_query);
//}
//
//
//std::string elm::Facet::build_alts_query() const
//{
//	std::string use_query = sql_alts;
//	
//	// If a query for alternatives codes and names is not defined but the idca table is, use that table's values by default.
//	if ((use_query.empty()) && (!sql_idca.empty())) {
//		use_query = "SELECT DISTINCT altid, 'alt'||altid from "+build_idca_query()+" AS elm_idca ORDER BY altid";
//	}
//		
//	try {
//		sql_statement(use_query);
//	} catch (etk::SQLiteError) {
//		std::string a = alias_idca_altid();
//		use_query = "SELECT DISTINCT "+a+", 'alt'||"+a+" from "+build_idca_query()+" AS elm_idca ORDER BY "+a;
//	}
//
//	return build_misc_query(use_query);
//}
//
//
//std::string elm::Facet::build_choice_query() const
//{
//	std::string use_query = sql_choice;
//	
//	// If a query for choice is not defined, assume nothing is chosen.
//	if (use_query.empty()) {
//		use_query = "SELECT NULL AS caseid, NULL AS altid, NULL AS choice LIMIT 0;";
//	}
//	
//	return build_misc_query(use_query);
//}
//
//
//std::string elm::Facet::build_weight_query() const
//{
//	std::string use_query = sql_weight;
//	
//	// If a query for weight is not defined, assume all equal to 1.
//	if (use_query.empty() || etk::to_uppercase(use_query)=="_EQUAL_") {
//		use_query = etk::cat("SELECT elm_caseids.",alias_caseids_caseid()," AS caseid, 1 AS weight FROM ",build_caseids_query()," AS elm_caseids;");
//	}
//	
//	return build_misc_query(use_query);
//}
//
//
//std::string elm::Facet::build_caseids_query() const
//{
//	std::string use_query = sql_caseids;
//	
//	// First backup: use idco
//	if (use_query.empty() && !sql_idco.empty()) {
//		use_query = "SELECT caseid FROM "+build_idco_query()+" AS elm_idco ORDER BY caseid";
//		try {
//			sql_statement(use_query);
//		} catch (etk::SQLiteError) {
//			use_query = "SELECT "+alias_idco_caseid()+" AS caseid FROM "+build_idco_query()+" AS elm_idco ORDER BY caseid";
//		}
//	}
//
//	// Second backup: use idca
//	if (use_query.empty() && !sql_idca.empty()) {
//		use_query = "SELECT DISTINCT caseid FROM "+build_idca_query()+" ORDER BY caseid";
//		try {
//			sql_statement(use_query);
//		} catch (etk::SQLiteError) {
//			use_query = "SELECT DISTINCT "+alias_idca_caseid()+" AS caseid FROM "+build_idca_query()+" ORDER BY caseid";
//		}
//	}
//	
//	return build_misc_query(use_query);
//}
//
//
//
//std::string elm::Facet::build_avail_query() const
//{
//	std::string use_query = sql_avail;
//	
//	// If a query for avail is not defined,
//	// but the caseids and alts are defined,
//	// assume all alternatives are available to all cases.
//	if ((use_query.empty()) && (!sql_caseids.empty()) && (!sql_alts.empty())) {
//		use_query = "SELECT elm_caseids."+alias_caseids_caseid()+" AS caseid"+
//		", elm_alternatives."+alias_alts_altid()+" AS altid"+
//		", 1 AS avail FROM "+build_caseids_query()+" AS elm_caseids, "+build_alts_query()+" AS elm_alternatives";
//	}
//	
//	try {
//		sql_statement(use_query);
//	} catch (etk::SQLiteError) {
//		std::string caseid_alias = sql_statement(build_caseids_query())->column_name(0);
//		std::string altid_alias = sql_statement(build_alts_query())->column_name(0);
//		use_query = "SELECT elm_caseids."+caseid_alias+" AS caseid"+
//		", elm_alternatives."+altid_alias+" AS altid"+
//		", 1 AS avail FROM "+build_caseids_query()+" AS elm_caseids, "+build_alts_query()+" AS elm_alternatives";
//		sql_statement(use_query);
//	}
//	
//	return build_misc_query(use_query);
//}
//
//
//
//
//
//
//
//
//








/*
void elm::Facet::refresh()
{
	try {
//		init_facet_queries();
	} catch (etk::FacetError) {
		_nCases = 0;
		_nAlts = 0;
		_Data_DNA.clear();
	}
	
	bool d_alts   = (sql_alts  ==_as_loaded_sql_alts  );
	bool d_idco   = (sql_idco  ==_as_loaded_sql_idco  );
	bool d_idca   = (sql_idca  ==_as_loaded_sql_idca  );
	bool d_choice = (sql_choice==_as_loaded_sql_choice);
	bool d_avail  = (sql_avail ==_as_loaded_sql_avail );
	bool d_weight = (sql_weight==_as_loaded_sql_weight);
	_as_loaded_sql_alts   = sql_alts  ;
	_as_loaded_sql_idco   = sql_idco  ;
	_as_loaded_sql_idca   = sql_idca  ;
	_as_loaded_sql_choice = sql_choice;
	_as_loaded_sql_avail  = sql_avail ;
	_as_loaded_sql_weight = sql_weight;
	
	
//	if (d_idco) {
		_nCases = eval_integer("SELECT count(*) FROM "+tbl_idco(),0);
//	}
	
	SQLiteStmtPtr s = sql_statement("");
	
//	if (d_alts) {
		INFO(msg)<< "Reading alternatives..." ;
		_Data_DNA.clear();
		s->clear();
		s->prepare(query_alts())->execute();
		while (s->status() == SQLITE_ROW) {
			_Data_DNA.add_cell(s->getInt64(0),s->getText(1));
			s->execute();
		}
		_Data_DNA.regrow();
		_nAlts = _Data_DNA.n_elemental();
		INFO(msg)<< "Found "<< _nAlts <<" elemental alternatives." ;
//	}
	

//	if (d_idco || !_caseindex) {
		if (_caseindex) {
			_caseindex.reset();
		}
		
		_caseindex = caseindex_t::create();
		_caseindex->add_caseids(caseids(0,0,1));
//		_nCases = _caseindex->size();
//	}
}
*/


//std::string elm::Facet::read_setup(const std::string& name, const std::string& defaultvalue) const
//{
//	try{
//		SQLiteStmtPtr z = sql_statement("SELECT value FROM elm_setup WHERE id==?");
//		z->bind_text(1,name);
//		return z->simpleText("",defaultvalue);
//	} catch (etk::SQLiteError) {
//		return defaultvalue;
//	}
//}
//void elm::Facet::write_setup(const std::string& name, const std::string& value)
//{
//	sql_statement("CREATE TABLE IF NOT EXISTS elm_setup (\n"
//					  "  id TEXT UNIQUE,\n"
//				      "  value)"  )->execute();
//	SQLiteStmtPtr z = sql_statement("INSERT OR REPLACE INTO elm_setup (id, value) VALUES (?,?)");
//	z->bind_text(1,name);
//	z->bind_text(2,value);
//	z->execute_until_done();
//}






const unsigned& elm::Facet::nCases() const
{
	return _nCases;
}

const unsigned& elm::Facet::nAlts() const
{
	return _nAlts;
}






std::vector<elm::caseid_t> elm::Facet::caseids(const unsigned& firstcasenum, const unsigned& numberofcases, int no_error_checking) const
{
		
	if (firstcasenum >= nCases() && !no_error_checking) {
		OOPS("First case number ",firstcasenum," is out of range. There are only ",nCases()," cases.");
	}
	
	unsigned cs = numberofcases;
	if (numberofcases==0) cs = nCases();
	if (firstcasenum+numberofcases>nCases()) cs = nCases() - firstcasenum;
	
	std::ostringstream sql;
	sql << "SELECT "<<alias_idco_caseid()<<" AS caseid FROM "+tbl_idco()+" ORDER BY caseid "
		<< "LIMIT " << cs << " OFFSET " << firstcasenum << ";";
		
	try {
		sql_statement(sql);
	} catch (etk::SQLiteError) {
		// Should throw an SQLite error when 'caseid' is not a valid column name in elm_idco
		//  in which case use the first column as the caseid.
		sql.str(""); sql.clear();
		std::string caseid_alias = column_name("SELECT * FROM "+tbl_idco(),0);
		sql << "SELECT "<< caseid_alias <<" AS caseid FROM "+tbl_idco()+" ORDER BY caseid "
			<< "LIMIT " << cs << " OFFSET " << firstcasenum << ";";
	}
	
	return eval_int64_tuple(sql.str());

}

std::vector<elm::cellcode> elm::Facet::altids() const
{
	if (!queries_ptr) OOPS_FACET("queries not defined");
		
	std::ostringstream sql;
	sql << "SELECT altid FROM "+queries_ptr->tbl_alts()+" ORDER BY altid;";

	try {
		sql_statement(sql);
	} catch (etk::SQLiteError) {
		// Should throw an SQLite error when 'caseid' is not a valid column name in elm_idco
		//  in which case use the first column as the caseid.
		sql.str(""); sql.clear();
		std::string altid_alias = column_name("SELECT * FROM "+queries_ptr->tbl_alts()+";",0);
		sql << "SELECT "<< altid_alias <<" AS altid FROM "+queries_ptr->tbl_alts()+" ORDER BY altid;";
	}


	return eval_int64_tuple(sql.str());
}

long long elm::Facet::alternative_code(std::string name) const
{
	if (!queries_ptr) OOPS_FACET("queries not defined");
		
	std::ostringstream sql;
	sql << "SELECT altid FROM "+queries_ptr->tbl_alts()+" WHERE name='"<<name<<"';";

	try {
		sql_statement(sql);
	} catch (etk::SQLiteError) {
		// Should throw an SQLite error when 'caseid' is not a valid column name in elm_idco
		//  in which case use the first column as the caseid.
		sql.str(""); sql.clear();
		std::string altid_alias = column_name("SELECT * FROM "+queries_ptr->tbl_alts()+";",0);
		std::string name_alias = column_name("SELECT * FROM "+queries_ptr->tbl_alts()+";",1);
		sql << "SELECT "<< altid_alias <<" AS altid FROM "+queries_ptr->tbl_alts()+" WHERE "<<name_alias<<"='"<<name<<"';";
	}


	return eval_int64(sql.str());
}


std::vector<std::string> elm::Facet::alternative_names() const
{
	if (!queries_ptr) OOPS_FACET("queries not defined");
	std::ostringstream sql;
	sql << "SELECT name FROM "+queries_ptr->tbl_alts()+" ORDER BY altid;";

	try {
		sql_statement(sql);
	} catch (etk::SQLiteError) {
		// Should throw an SQLite error when 'caseid' is not a valid column name in elm_idco
		//  in which case use the first column as the caseid.
		sql.str(""); sql.clear();
		std::string altid_alias = column_name("SELECT * FROM "+queries_ptr->tbl_alts()+";",0);
		std::string name_alias = column_name("SELECT * FROM "+queries_ptr->tbl_alts()+";",1);
		sql << "SELECT "<< name_alias <<" AS name FROM "+queries_ptr->tbl_alts()+" ORDER BY "<<altid_alias<<";";
	}


	return eval_string_tuple(sql.str());
}


std::string elm::Facet::alternative_name(long long code) const
{
	if (!queries_ptr) OOPS_FACET("queries not defined");

	std::ostringstream sql;
	sql << "SELECT name FROM "+queries_ptr->tbl_alts()+" WHERE altid="<<code<<";";

	try {
		sql_statement(sql);
	} catch (etk::SQLiteError) {
		// Should throw an SQLite error when 'caseid' is not a valid column name in elm_idco
		//  in which case use the first column as the caseid.
		sql.str(""); sql.clear();
		std::string altid_alias = column_name("SELECT * FROM "+queries_ptr->tbl_alts()+";",0);
		std::string name_alias = column_name("SELECT * FROM "+queries_ptr->tbl_alts()+";",1);
		sql << "SELECT "<< name_alias <<" AS name FROM "+queries_ptr->tbl_alts()+" WHERE "<<altid_alias<<"="<<code<<";";
	}


	return eval_text(sql.str());
}


std::vector<elm::cellcode> elm::Facet::alternative_codes() const
{
	return altids();
}

elm::VAS_dna elm::Facet::alternatives_dna() const
{
	std::vector<std::string> the_names (alternative_names());
	std::vector<elm::cellcode> the_codes (alternative_codes());
	if (the_names.size() != the_codes.size()) OOPS("vector sizes do not match");
	VAS_dna output;
	for (unsigned i=0; i<the_names.size(); i++) {
		output[the_codes[i]] = VAS_dna_info(the_names[i]);
	}
	return output;
}

bool elm::Facet::check_ca(const std::string& column) const
{
	std::ostringstream sql;
	sql << "SELECT "<<column<<" FROM "+tbl_idca();
	sql_statement(sql.str());
	return true;
}

bool elm::Facet::check_co(const std::string& column) const
{
	std::ostringstream sql;
	sql << "SELECT "<<column<<" FROM "+tbl_idco();
	sql_statement(sql.str());
	return true;
}

std::vector<std::string> elm::Facet::variables_ca() const
{
	return column_names("SELECT * FROM "+tbl_idca());
}

std::vector<std::string> elm::Facet::variables_co() const
{
	return column_names("SELECT * FROM "+tbl_idco());
}


elm::VAS_System* elm::Facet::DataDNA(const long long& c)
{
	// In the future, the basic data structure may be allowed to vary based on the case
	return &_Data_DNA;
}


std::string elm::Facet::alias_caseids_caseid() const
{
	std::string caseid = "caseid";
	try {
		std::ostringstream sql;
		sql << "SELECT "<<caseid<<" FROM "<<tbl_caseids();
		sql_statement(sql.str());
	} catch (etk::SQLiteError) {
		caseid = column_name(qry_caseids(), 0);
	}
	return caseid;
}



std::string elm::Facet::alias_idco_caseid() const
{
	if (!queries_ptr) OOPS_FACET("queries not defined");
	std::string caseid = "caseid";
	try {
		check_co(caseid);
	} catch (etk::SQLiteError) {
		caseid = column_name("SELECT * FROM "+tbl_idco(), 0);
	}
	return caseid;
}

std::string elm::Facet::alias_idca_caseid() const
{
	if (!queries_ptr) OOPS_FACET("queries not defined");
	std::string caseid = "caseid";
	try {
		check_ca(caseid);
	} catch (etk::SQLiteError) {
		caseid = column_name("SELECT * FROM "+tbl_idca(), 0);
	}
	return caseid;
}

std::string elm::Facet::alias_idca_altid() const
{
	if (!queries_ptr) OOPS_FACET("queries not defined");
	std::string altid = "altid";
	try {
		check_ca(altid);
	} catch (etk::SQLiteError) {
		altid = column_name("SELECT * FROM "+tbl_idca(), 1);
	}
	return altid;
}

std::string elm::Facet::alias_alts_altid() const
{
	if (!queries_ptr) OOPS_FACET("queries not defined");
	std::string altid = "altid";
	try {
		std::ostringstream sql;
		sql << "SELECT "<<altid<<" FROM "+tbl_alts()+";";
		sql_statement(sql.str());
	} catch (etk::SQLiteError) {
		try {
			altid = "altid";
			std::ostringstream sql;
			sql << "SELECT "<<altid<<" FROM "+tbl_alts()+";";
			sql_statement(sql.str());
		} catch (etk::SQLiteError) {
			altid = column_name("SELECT * FROM "+tbl_alts()+"", 0);
		}
	}
	return altid;
}

std::string elm::Facet::alias_alts_name() const
{
	if (!queries_ptr) OOPS_FACET("queries not defined");
	std::string altid = "name";
	try {
		std::ostringstream sql;
		sql << "SELECT "<<altid<<" FROM "+tbl_alts()+";";
		sql_statement(sql.str());
	} catch (etk::SQLiteError) {
		try {
			altid = "altname";
			std::ostringstream sql;
			sql << "SELECT "<<altid<<" FROM "+tbl_alts()+";";
			sql_statement(sql.str());
		} catch (etk::SQLiteError) {
			altid = column_name("SELECT * FROM "+tbl_alts()+"", 1);
		}
	}
	return altid;
}



std::string elm::Facet::_query_chopper(long long firstrow, long long numrows) const
{
	std::ostringstream q;
	size_t cs = numrows;
	if (numrows==0) cs = nCases();
	if (firstrow+numrows>nCases()) cs = nCases() - firstrow;
	if (firstrow >= nCases()) OOPS("Asking for first row of ",firstrow," but there are only ",nCases()," rows in the current data");
	if ((cs<nCases())||(firstrow>0)) {
		q << " WHERE caseid IN"
		  << " (SELECT "<<alias_idco_caseid()<<" AS caseid FROM "+tbl_idco()+" ORDER BY caseid"
		  << " LIMIT " << cs << " OFFSET " << firstrow << ")";
	}
	return q.str();
}


//std::string elm::Facet::query_idca(const etk::strvec& columns, long long firstrow, long long numrows, bool validate) const
//{
//	if (!queries_ptr) OOPS_FACET("queries not defined");
//	std::ostringstream q;
//	q << "SELECT "<<alias_idca_caseid()<<" AS caseid, "<<alias_idca_altid()<<" AS altid";
//	for (etk::strvec::const_iterator v= columns.begin(); v!=columns.end(); v++ ) {
//		if (validate) {
//			try {
//				check_ca(*v);
//			} catch (etk::SQLiteError) {
//				continue;
//			}
//		}
//		q << ", " << *v;
//	}
//	q << " FROM "<<tbl_idca();
//
//
//	q << _query_chopper(firstrow, numrows);
//	q << " ORDER BY caseid, altid;";
//	return q.str();
//}

std::string elm::Facet::query_idca(const std::vector<std::string>& columns, bool validate, long long* caseid) const
{
	if (!queries_ptr) OOPS_FACET("queries not defined");
	std::ostringstream q;
	q << "SELECT "<<alias_idca_caseid()<<" AS caseid, "<<alias_idca_altid()<<" AS altid";
	for (etk::strvec::const_iterator v= columns.begin(); v!=columns.end(); v++ ) {
		if (validate) {
			try {
				check_ca(*v);
			} catch (etk::SQLiteError) {
				continue;
			}
		}
		q << ", " << *v;
	}
	q << " FROM "<<tbl_idca();
	
	if (caseid) {
		q << " WHERE caseid="<<*caseid;
	}
	
	q << " ORDER BY caseid, altid;";
	return q.str();
}



//std::string elm::Facet::query_idco(const etk::strvec& columns, long long firstrow, long long numrows, bool validate) const
//{
//	std::ostringstream q;
//	q << "SELECT "<<alias_idco_caseid()<<" AS caseid";
//	for (etk::strvec::const_iterator v= columns.begin(); v!=columns.end(); v++ ) {
//		if (validate) {
//			try {
//				check_co(*v);
//			} catch (etk::SQLiteError) {
//				continue;
//			}
//		}
//		q << ", " << *v;
//	}
//	q << " FROM "<<tbl_idco();
//	q << " ORDER BY caseid";
//	
//	
//	size_t cs = numrows;
//	if (numrows==0) cs = nCases();
//	if (firstrow+numrows>nCases()) cs = nCases() - firstrow;
//	if (firstrow >= nCases()) OOPS("Asking for first row of ",firstrow," but there are only ",nCases()," rows in the current data");
//	if ((cs<nCases())||(firstrow>0)) {
//		q << " LIMIT " << cs << " OFFSET " << firstrow;
//	}
//	q << ";";
//	return q.str();
//}

std::string elm::Facet::query_idco(const std::vector<std::string>& columns, bool validate, long long* caseid) const
{
	std::ostringstream q;
	q << "SELECT "<<alias_idco_caseid()<<" AS caseid";
	for (etk::strvec::const_iterator v= columns.begin(); v!=columns.end(); v++ ) {
		if (validate) {
			try {
				check_co(*v);
			} catch (etk::SQLiteError) {
				continue;
			}
		}
		q << ", " << *v;
	}
	q << " FROM "<<tbl_idco();
	
	if (caseid) {
		q << " WHERE caseid="<<*caseid;
	} else {
		q << " ORDER BY caseid";
	}
	q << ";";
	return q.str();
}


//std::string elm::Facet::query_alts() const
//{
//	if (!queries_ptr) OOPS_FACET("queries not defined");
//	std::ostringstream q;
//	std::string altid_alias = "id";
//	try {
//		sql_statement("SELECT id FROM "+queries_ptr->tbl_alts());
//	} catch (etk::SQLiteError) {
//		altid_alias = sql_statement("SELECT * FROM "+queries_ptr->tbl_alts())->column_name(0);
//	}
//	std::string name_alias = "name";
//	try {
//		sql_statement("SELECT name FROM "+queries_ptr->tbl_alts());
//	} catch (etk::SQLiteError) {
//		name_alias = sql_statement("SELECT * FROM "+queries_ptr->tbl_alts())->column_name(1);
//	}
//	q<<"SELECT "<<altid_alias<<" AS id, "<<name_alias<<" AS name FROM "+queries_ptr->tbl_alts()+";";
//	return q.str();
//}

std::string elm::Facet::query_alts(long long* caseid) const
{
	if (!queries_ptr) OOPS_FACET("queries not defined");
	std::ostringstream q;
	std::string altid_alias = "id";
	try {
		sql_statement("SELECT id FROM "+queries_ptr->tbl_alts());
	} catch (etk::SQLiteError) {
		altid_alias = sql_statement("SELECT * FROM "+queries_ptr->tbl_alts())->column_name(0);
	}
	std::string name_alias = "name";
	try {
		sql_statement("SELECT name FROM "+queries_ptr->tbl_alts());
	} catch (etk::SQLiteError) {
		name_alias = sql_statement("SELECT * FROM "+queries_ptr->tbl_alts())->column_name(1);
	}
	q<<"SELECT "<<altid_alias<<" AS id, "<<name_alias<<" AS name FROM "+queries_ptr->tbl_alts()+";";
	return q.str();
}

//std::string elm::Facet::query_choice(long long firstrow, long long numrows) const
//{
//	if (!queries_ptr) OOPS_FACET("queries not defined");
//	std::ostringstream q;
//	std::string caseid_alias = "caseid";
//	try {
//		sql_statement("SELECT caseid FROM "+queries_ptr->tbl_choice());
//	} catch (etk::SQLiteError) {
//		caseid_alias = sql_statement("SELECT * FROM "+queries_ptr->tbl_choice())->column_name(0);
//	}
//	std::string altid_alias = "altid";
//	try {
//		sql_statement("SELECT altid FROM "+queries_ptr->tbl_choice());
//	} catch (etk::SQLiteError) {
//		altid_alias = sql_statement("SELECT * FROM "+queries_ptr->tbl_choice())->column_name(1);
//	}
//	std::string choice_alias = "choice";
//	try {
//		sql_statement("SELECT choice FROM "+queries_ptr->tbl_choice());
//	} catch (etk::SQLiteError) {
//		choice_alias = sql_statement("SELECT * FROM "+queries_ptr->tbl_choice())->column_name(2);
//	}
//	q<<"SELECT "<<caseid_alias<<" AS caseid, "<<altid_alias<<" AS altid, "<<choice_alias<<" AS choice FROM "+queries_ptr->tbl_choice();
//	q << _query_chopper(firstrow, numrows);
//	q << " ORDER BY caseid, altid;";
//	return q.str();
//}

std::string elm::Facet::query_choice(long long* caseid) const
{
	if (!queries_ptr) OOPS_FACET("queries not defined");
	std::ostringstream q;
	std::string caseid_alias = "caseid";
	try {
		sql_statement("SELECT caseid FROM "+queries_ptr->tbl_choice());
	} catch (etk::SQLiteError) {
		caseid_alias = sql_statement("SELECT * FROM "+queries_ptr->tbl_choice())->column_name(0);
	}
	std::string altid_alias = "altid";
	try {
		sql_statement("SELECT altid FROM "+queries_ptr->tbl_choice());
	} catch (etk::SQLiteError) {
		altid_alias = sql_statement("SELECT * FROM "+queries_ptr->tbl_choice())->column_name(1);
	}
	std::string choice_alias = "choice";
	try {
		sql_statement("SELECT choice FROM "+queries_ptr->tbl_choice());
	} catch (etk::SQLiteError) {
		choice_alias = sql_statement("SELECT * FROM "+queries_ptr->tbl_choice())->column_name(2);
	}
	q<<"SELECT "<<caseid_alias<<" AS caseid, "<<altid_alias<<" AS altid, "<<choice_alias<<" AS choice FROM "+queries_ptr->tbl_choice();
	
	if (caseid) {
		q << " WHERE caseid="<<*caseid;
	}
	q << " ORDER BY caseid, altid;";
	return q.str();
}



std::string elm::Facet::alias_avail_caseid() const
{
	try {
		sql_statement("SELECT caseid FROM "+tbl_avail());
	} catch (etk::SQLiteError) {
		return sql_statement("SELECT * FROM "+tbl_avail())->column_name(0);
	}
	return "caseid";
}
std::string elm::Facet::alias_avail_altid() const
{
	try {
		sql_statement("SELECT altid FROM "+tbl_avail());
	} catch (etk::SQLiteError) {
		return sql_statement("SELECT * FROM "+tbl_avail())->column_name(1);
	}
	return "altid";
}
std::string elm::Facet::alias_avail_avail() const
{
	try {
		sql_statement("SELECT avail FROM "+tbl_avail());
	} catch (etk::SQLiteError) {
		return sql_statement("SELECT * FROM "+tbl_avail())->column_name(2);
	}
	return "avail";
}


//std::string elm::Facet::query_avail(long long firstrow, long long numrows) const
//{
//	std::ostringstream q ("");
//	std::string caseid_alias = alias_avail_caseid();
//	std::string altid_alias = alias_avail_altid();
//	//auto cn =sql_statement("SELECT * FROM elm_avail")->column_names();
//	std::string avail_alias = alias_avail_avail();
//
//	if (queries_ptr) return queries_ptr->qry_avail();
//	
//	OOPS("queries not defined");
//
//	q << "SELECT "<<caseid_alias<<" AS caseid, "<<altid_alias<<" AS altid, "<<avail_alias<<" AS avail ";
//	q << "FROM "<<build_avail_query()<<" AS elm_avail";
//	q << _query_chopper(firstrow, numrows);
//	q << " ORDER BY caseid, altid;";
//	return q.str();
//}
//
//std::string elm::Facet::query_avail(long long firstrow, long long numrows) const
//{
//	if (!queries_ptr) OOPS_FACET("queries not defined");
//	
//	std::string q_avail = queries_ptr->qry_avail();
//	
//	if (q_avail=="") {
//		OOPS("qry_avail not defined");
//	}
//	
//	std::ostringstream q ("");
//	std::string caseid_alias = "caseid";
//	auto cn =sql_statement("SELECT * FROM ("+q_avail+") AS elm_avail")->column_names();
//	try {
//		sql_statement("SELECT caseid FROM ("+q_avail+") AS elm_avail");
//	} catch (etk::SQLiteError) {
//		caseid_alias = sql_statement("SELECT * FROM ("+q_avail+") AS elm_avail")->column_name(0);
//	}
//	std::string altid_alias = "altid";
//	try {
//		sql_statement("SELECT altid FROM ("+q_avail+") AS elm_avail");
//	} catch (etk::SQLiteError) {
//		altid_alias = sql_statement("SELECT * FROM ("+q_avail+") AS elm_avail")->column_name(1);
//	}
//	std::string avail_alias = "avail";
//	try {
//		sql_statement("SELECT avail FROM ("+q_avail+") AS elm_avail");
//	} catch (etk::SQLiteError) {
//		avail_alias = sql_statement("SELECT * FROM ("+q_avail+") AS elm_avail")->column_name(2);
//	}
//	q<<"SELECT "<<caseid_alias<<" AS caseid, "<<altid_alias<<" AS altid, "<<avail_alias<<" AS avail FROM ("+q_avail+") AS elm_avail";
//	q << _query_chopper(firstrow, numrows);
//	q << " ORDER BY caseid, altid;";
//	return q.str();
//}

std::string elm::Facet::query_avail(long long* caseid) const
{
	if (!queries_ptr) OOPS_FACET("queries not defined");
	
	std::string q_avail = queries_ptr->qry_avail();
	
	if (q_avail=="") {
		OOPS("qry_avail not defined");
	}
	
	std::ostringstream q ("");
	std::string caseid_alias = "caseid";
	auto cn =sql_statement("SELECT * FROM ("+q_avail+") AS elm_avail")->column_names();
	try {
		sql_statement("SELECT caseid FROM ("+q_avail+") AS elm_avail");
	} catch (etk::SQLiteError) {
		caseid_alias = sql_statement("SELECT * FROM ("+q_avail+") AS elm_avail")->column_name(0);
	}
	std::string altid_alias = "altid";
	try {
		sql_statement("SELECT altid FROM ("+q_avail+") AS elm_avail");
	} catch (etk::SQLiteError) {
		altid_alias = sql_statement("SELECT * FROM ("+q_avail+") AS elm_avail")->column_name(1);
	}
	std::string avail_alias = "avail";
	try {
		sql_statement("SELECT avail FROM ("+q_avail+") AS elm_avail");
	} catch (etk::SQLiteError) {
		avail_alias = sql_statement("SELECT * FROM ("+q_avail+") AS elm_avail")->column_name(2);
	}
	q<<"SELECT "<<caseid_alias<<" AS caseid, "<<altid_alias<<" AS altid, "<<avail_alias<<" AS avail FROM ("+q_avail+") AS elm_avail";
	if (caseid) {
		q << " WHERE caseid="<<*caseid;
	}
	q << " ORDER BY caseid, altid;";
	return q.str();
}



//std::string elm::Facet::query_weight(long long firstrow, long long numrows) const
//{
//	if (!queries_ptr) OOPS_FACET("queries not defined");
//
//	std::ostringstream q;
//	std::string caseid_alias = "caseid";
//	try {
//		sql_statement("SELECT caseid FROM "+queries_ptr->tbl_weight());
//	} catch (etk::SQLiteError) {
//		caseid_alias = sql_statement("SELECT * FROM "+queries_ptr->tbl_weight())->column_name(0);
//	}
//	std::string weight_alias = "weight";
//	try {
//		sql_statement("SELECT altid FROM "+queries_ptr->tbl_weight());
//	} catch (etk::SQLiteError) {
//		weight_alias = sql_statement("SELECT * FROM "+queries_ptr->tbl_weight())->column_name(1);
//	}
//	q<<"SELECT "<<caseid_alias<<" AS caseid, "<<weight_alias<<" AS weight FROM "+queries_ptr->tbl_weight();
//	q << _query_chopper(firstrow, numrows);
//	q << " ORDER BY caseid;";
//	return q.str();
//}

std::string elm::Facet::query_weight(long long* caseid) const
{
	if (!queries_ptr) OOPS_FACET("queries not defined");
	
	std::ostringstream q;
	std::string caseid_alias = "caseid";
	try {
		sql_statement("SELECT caseid FROM "+queries_ptr->tbl_weight());
	} catch (etk::SQLiteError) {
		caseid_alias = sql_statement("SELECT * FROM "+queries_ptr->tbl_weight())->column_name(0);
	}
	std::string weight_alias = "weight";
	try {
		sql_statement("SELECT altid FROM "+queries_ptr->tbl_weight());
	} catch (etk::SQLiteError) {
		weight_alias = sql_statement("SELECT * FROM "+queries_ptr->tbl_weight())->column_name(1);
	}
	q<<"SELECT "<<caseid_alias<<" AS caseid, "<<weight_alias<<" AS weight FROM "+queries_ptr->tbl_weight();
	if (caseid) {
		q << " WHERE caseid="<<*caseid;
	}
	q << " ORDER BY caseid;";
	return q.str();
}


elm::ScrapePtr elm::Facet::get_scrape_idca()
{
	return elm::Scrape::create(this, IDCA);
}
elm::ScrapePtr elm::Facet::get_scrape_idco()
{
	return elm::Scrape::create(this, IDCO);
}
elm::ScrapePtr elm::Facet::get_scrape_choo()
{
	return elm::Scrape::create(this, CHOO);
}
elm::ScrapePtr elm::Facet::get_scrape_aval()
{
	return elm::Scrape::create(this, AVAL);
}
elm::ScrapePtr elm::Facet::get_scrape_wght()
{
	return elm::Scrape::create(this, WGHT);
}


/*
elm::datamatrix elm::Facet::ask_idca(const std::vector<std::string>& varnames, long long* caseid)
{
	if (!caseid) {
		auto i=_extracts.begin();
		for (; i!=_extracts.end(); i++) {
			if ( (*i)->dtype==mtrx_double
				&& (*i)->dimty==case_alt_var
				&& (*i)->dpurp==purp_vars
				&& (*i)->get_variables()==varnames
				&& nCases()==(*i)->nCases()
				&& nAlts()==(*i)->nAlts()) break;
		}
		if (i!=_extracts.end()) {
			return (*i);
		}
	}
	elm::datamatrix j = elm::datamatrix_t::read_idca(this, varnames, caseid);
	_extracts.push_back(j);
//	if (j) {
//		std::cerr<<"ask_idca::"<<j->__str__()<<"["<<j->refcount()<<"]\n";
//	} else {
//		std::cerr<<"ask_idca::"<<"<None>"<<"\n";
//	}
	return j;
}
elm::datamatrix elm::Facet::ask_idco(const std::vector<std::string>& varnames, long long* caseid)
{
	if (!caseid) {
		auto i=_extracts.begin();
		for (; i!=_extracts.end(); i++) {
			if ( (*i)->dtype==mtrx_double
				&& (*i)->dimty==case_var
				&& (*i)->dpurp==purp_vars
				&& (*i)->get_variables()==varnames
				&& nCases()==(*i)->nCases()) break;
		}
		if (i!=_extracts.end()) {
			return (*i);
		}
	}
	elm::datamatrix j = elm::datamatrix_t::read_idco(this, varnames, caseid);
	_extracts.push_back(j);
//	if (j)
//		std::cerr<<"ask_idco::"<<j->__str__()<<"\n";
//	else
//		std::cerr<<"ask_idco::"<<"<None>"<<"\n";
	return j;
}
elm::datamatrix elm::Facet::ask_choice(long long* caseid)
{
	auto i=_extracts.begin();
	for (; i!=_extracts.end(); i++) {
		if ( (*i)->dtype==mtrx_double
			&& (*i)->dimty==case_alt_var
			&& (*i)->dpurp==purp_choice
			&& nCases()==(*i)->nCases()) break;
	}
	if (i!=_extracts.end()) {
		return (*i);
	}
	elm::datamatrix j = elm::datamatrix_t::read_choo(this, caseid);
	_extracts.push_back(j);
//	if (j)
//		std::cerr<<"ask_choice::"<<j->__str__()<<"\n";
//	else
//		std::cerr<<"ask_choice::"<<"<None>"<<"\n";
	return j;
}
elm::datamatrix elm::Facet::ask_weight(long long* caseid)
{
	auto i=_extracts.begin();
	for (; i!=_extracts.end(); i++) {
		if ( (*i)->dtype==mtrx_double
			&& (*i)->dimty==case_var
			&& (*i)->dpurp==purp_weight
			&& nCases()==(*i)->nCases()) break;
	}
	if (i!=_extracts.end()) {
		return (*i);
	}
	elm::datamatrix j = elm::datamatrix_t::read_wght(this, caseid);
	_extracts.push_back(j);
//	if (j)
//		std::cerr<<"ask_weight::"<<j->__str__()<<"\n";
//	else
//		std::cerr<<"ask_weight::"<<"<None>"<<"\n";
	return j;
}
elm::datamatrix elm::Facet::ask_avail(long long* caseid)
{
	auto i=_extracts.begin();
	for (; i!=_extracts.end(); i++) {
		if ( (*i)->dtype==mtrx_bool
			&& (*i)->dimty==case_alt_var
			&& (*i)->dpurp==purp_avail
			&& nCases()==(*i)->nCases()) break;
	}
	if (i!=_extracts.end()) {
		return (*i);
	}
	elm::datamatrix j = elm::datamatrix_t::read_aval(this, caseid);
	_extracts.push_back(j);
//	if (j)
//		std::cerr<<"ask_avail::"<<j->__str__()<<"\n";
//	else
//		std::cerr<<"ask_avail::"<<"<None>"<<"\n";
	return j;
}


*/





//elm::caseindex elm::Facet::ask_caseids()
//{
//	return _caseindex;
//}

elm::VAS_dna  elm::Facet::ask_dna(const long long& c)
{
	return DataDNA()->genome();
}

/*
elm::datamatrix elm::Facet::matrix_library(size_t n)
{
	auto i = _extracts.begin();
	while (n > 0 && i!=_extracts.end()) {
		i++; n--;
	}
	if (i==_extracts.end()) return nullptr;
	return *i;
}

*/



void elm::Facet::_array_idco_reader(const std::string& qry, elm::darray* array, elm::darray* caseids)
{
	assert(array->nCases()==caseids->nCases());
	assert(caseids->dtype == NPY_INT64);
	
	size_t row = 0;
	clock_t prevmsgtime = clock();
	clock_t timenow;
	
	size_t n_cases = array->_repository.size1();
	size_t n_vars = array->_repository.size2();
	
	auto stmt = sql_statement_readonly(qry);
	
	if (stmt->count_columns()-1 < n_vars) {
		OOPS("(vars underflow ) table has ",stmt->count_columns()-1," variables after the caseid, but the array offers space for ",n_vars," variables");
	}
	if (stmt->count_columns()-1 > n_vars) {
		OOPS("(vars overflow  ) table has ",stmt->count_columns()-1," variables after the caseid, but the array offers space for ",n_vars," variables");
	}

	std::unordered_set<long long> caseid_set;
	
	stmt->execute();
	while ((stmt->status()==SQLITE_ROW) && row<array->nCases()) {

		long long current_row_caseid = stmt->getInt64(0);

		if (caseid_set.find(current_row_caseid) != caseid_set.end()) {
			OOPS("duplicate caseid ",current_row_caseid);
		} else {
			caseid_set.insert(current_row_caseid);
		}

		timenow = clock();
		if (timenow > prevmsgtime + (CLOCKS_PER_SEC * 3)) {
			INFO(msg) << "reading idco row "<<row<<", "
				<<"case "<< caseids->value_int64(row, 0) << ", "
				<< 100.0*double(row)/double(n_cases) << "% ..." ;
			prevmsgtime = clock();
		}

		caseids->value_int64(row, 0) = current_row_caseid;

		if (array->dtype == NPY_DOUBLE) {
			for (size_t i=0; i<n_vars; i++) {
				array->value_double(row, i) = stmt->getDouble(i+1);
			}
		} else if (array->dtype == NPY_INT64) {
			for (size_t i=0; i<n_vars; i++) {
				array->value_int64(row, i) = stmt->getInt64(i+1);
			}
		} else if (array->dtype == NPY_BOOL) {
			for (size_t i=0; i<n_vars; i++) {
				array->value_bool(row, i) = stmt->getBool(i+1);
			}
		} else {
			OOPS("unsupported dtype");
		}
		row++;
		stmt->execute();
	}
	
	if (row < n_cases) {
		OOPS("(cases underflow) completed reading table after ",row," cases, array of ",n_cases," not filled");
	}
	if (stmt->status()==SQLITE_ROW) {
		OOPS("(cases overflow ) not completed reading table but already filled all ",row," cases");
	}
}



void elm::Facet::_array_idca_reader(const std::string& qry, elm::darray* array, elm::darray* caseids, const std::vector<long long>& altids)
{
	assert(array->nCases()==caseids->nCases());
	assert(caseids->dtype == NPY_INT64);
	
	size_t row = 0;
	size_t max_caserow = 0;
	clock_t prevmsgtime = clock();
	clock_t timenow;
	
	size_t n_cases = array->_repository.size1();
	size_t n_alts = array->_repository.size2();
	size_t n_vars = array->_repository.size3();
	
	if (altids.size() < n_alts) {
		OOPS("(alts underflow ) vector has ",altids.size()," altids given, but the array offers space for ",n_alts," alternatives");
	}
	if (altids.size() > n_alts) {
		OOPS("(alts overflow  ) vector has ",altids.size()," altids given, but the array offers space for ",n_alts," alternatives");
	}
	
	auto stmt = sql_statement_readonly(qry);
	
	if (stmt->count_columns()-2 < n_vars) {
		OOPS("(vars underflow ) table has ",stmt->count_columns()-2," variables after the caseid and altid, but the array offers space for ",n_vars," variables");
	}
	if (stmt->count_columns()-2 > n_vars) {
		OOPS("(vars overflow  ) table has ",stmt->count_columns()-2," variables after the caseid and altid, but the array offers space for ",n_vars," variables");
	}
	
	
	std::unordered_map<long long, size_t> caseid_map;
	std::unordered_map<long long, size_t> altid_map;
	
	// initialized the alt map with altids
	size_t j=0;
	for (auto i=altids.begin(); i!=altids.end(); i++) {
		altid_map[*i] = j;
		j++;
	}
	
	size_t c,a;
	
	stmt->execute();
	while ((stmt->status()==SQLITE_ROW)) {

		long long current_row_caseid = stmt->getInt64(0);
		long long current_row_altid  = stmt->getInt64(1);
		
		auto caseiter = caseid_map.find(current_row_caseid);
		if (caseiter == caseid_map.end()) {
			c = caseid_map[current_row_caseid] = max_caserow;
			caseids->value_int64(c, 0) = current_row_caseid;
			max_caserow++;
			if (max_caserow>array->nCases()) {
				OOPS("(cases overflow ) not completed reading table but already filled all ",array->nCases()," cases");
			}
		} else {
			c = caseiter->second;
		}

		auto altiter = altid_map.find(current_row_altid);
		if (altiter == altid_map.end()) {
			OOPS("table contains unknown altid ",current_row_altid);
		} else {
			a = altiter->second;
		}


		timenow = clock();
		if (timenow > prevmsgtime + (CLOCKS_PER_SEC * 3)) {
			INFO(msg) << "reading idca row "<<row<<", "
				<<"case "<< current_row_caseid << ", "
				<< 100.0*double(c)/double(n_cases) << "% ..." ;
			prevmsgtime = clock();
		}


		if (array->dtype == NPY_DOUBLE) {
			for (size_t i=0; i<n_vars; i++) {
				array->value_double(c,a, i) = stmt->getDouble(i+2);
			}
		} else if (array->dtype == NPY_INT64) {
			for (size_t i=0; i<n_vars; i++) {
				array->value_int64(c,a, i) = stmt->getInt64(i+2);
			}
		} else if (array->dtype == NPY_BOOL) {
			for (size_t i=0; i<n_vars; i++) {
				array->value_bool(c,a, i) = stmt->getBool(i+2);
			}
		} else {
			OOPS("unsupported dtype");
		}
		row++;
		stmt->execute();
	}
	
	if (max_caserow < n_cases) {
		OOPS("(cases underflow) completed reading table after ",max_caserow," cases, array of ",n_cases," not filled");
	}
	if (stmt->status()==SQLITE_ROW) {
		OOPS("(cases overflow ) not completed reading table but already filled all ",max_caserow," cases");
	}
}









