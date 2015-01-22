/*
 *  elm_sql_facet.h
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


#ifndef __ELM2_SQL_FACET_H__
#define __ELM2_SQL_FACET_H__

#include <string>


#ifdef SWIG
#define NOSWIG(x) 
#define FOSWIG(x) x
#define SWIGRO(x) %readonly x %readwrite
#else
#define NOSWIG(x) x
#define FOSWIG(x) 
#define SWIGRO(x) x
#endif // def SWIG



#ifdef SWIG
////////////////////////////////////////////////////////////////////////////////
// !!!: HELP TEXT

%feature("docstring") elm::Facet::get_sql_alts
"An SQL query that evaluates to an larch_alternatives table.\n\
\n\
Column 1: id (integer) a key for every alternative observed in the sample\n\
Column 2: name (text) a name for each alternative\n\
";


%feature("docstring") elm::Facet::get_sql_idco
"An SQL query that evaluates to an larch_idco table.\n\
\n\
Column 1: caseid (integer) a key for every case observed in the sample\n\
Column 2+ can contain any explanatory data, typically numeric data, although non-numeric data is allowable.";


%feature("docstring") elm::Facet::get_sql_idca
"An SQL query that evaluates to an larch_idca table.\n\
\n\
Column 1: caseid (integer) a key for every case observed in the sample\n\
Column 2: altid (integer) a key for each alternative available in this case\n\
Column 3+ can contain any explanatory data, typically numeric data, although non-numeric data is allowable.\n\
If no columns have the name 'caseid' and 'altid', elm will use the first two columns, repsectively.\n\
A query with less than two columns will raise an exception.";


%feature("docstring") elm::Facet::get_sql_choice
"An SQL query that evaluates to an larch_choice table.\n\
\n\
Column 1: caseid (integer) a key for every case observed in the sample\n\
Column 2: altid (integer) a key for each alternative available in this case\n\
Column 3: choice (numeric, typically 1.0 but could be other values)\n\
\n\
If an alternative is not chosen for a given case, it can have a zero choice value or \
it can simply be omitted from the result.";


%feature("docstring") elm::Facet::get_sql_avail
"An SQL query that evaluates to an larch_avail table.\n\
\n\
Column 1: caseid (integer) a key for every case observed in the sample\n\
Column 2: altid (integer) a key for each alternative available in this case\n\
Column 3: avail (boolean) evaluates as 1 or true when the alternative is available, 0 otherwise\n\
\n\
If an alternative is not available for a given case, it can have a zero avail value or \
it can simply be omitted from the result. If no query is given, it is assumed that \
all alternatives are available in all cases.";


%feature("docstring") elm::Facet::get_sql_weight
"An SQL query that evaluates to an larch_weight table.\n\
\n\
Column 1: caseid (integer) a key for every case observed in the sample\n\
Column 2: weight (numeric) a weight associated with each case\n\
\n\
If no weights are given, they are assumed to be all equal.";


#endif // SWIG

#ifdef SWIG
%{
	// In SWIG, these headers are available to the c++ wrapper,
	// but are not themselves wrapped
	#include "elm_sql_facet.h"
%}
#endif // SWIG

#ifndef SWIG
// In not SWIG, these headers are treated normally
#include "elm_sql_connect.h"
#include "elm_vascular.h"
#include "elm_datamatrix.h"
#include "elm_fountain.h"
#include "elm_darray.h"

#define SQL_ALTS_BLANK "SELECT NULL AS id, NULL AS name, NULL AS upcodes, NULL AS dncodes LIMIT 0;"
#define SQL_IDCO_BLANK "SELECT NULL AS caseid LIMIT 0;"
#define SQL_IDCA_BLANK "SELECT NULL AS caseid, NULL AS altid LIMIT 0;"
#define SQL_CHOICE_BLANK "SELECT NULL AS caseid, NULL AS altid, NULL AS choice LIMIT 0;"
#define SQL_AVAIL_BLANK "SELECT NULL AS caseid, NULL AS altid, NULL AS avail LIMIT 0;"
#define SQL_WEIGHT_BLANK "SELECT NULL AS caseid, 1 AS weight LIMIT 0;"

#endif // not SWIG


#include "elm_queryset.h"



namespace elm {
		
//////// MARK: SQL FACET //////////////////////////////////////////////////////////////////	

	typedef long long caseid_t;

#ifndef SWIG
	class Scrape;
	typedef boosted::shared_ptr<Scrape> ScrapePtr;
	typedef boosted::weak_ptr<Scrape>   ScrapePtr_weak;
#endif // ndef SWIG
				
	class Facet
	: public SQLiteDB
	, public Fountain
	{
	public:
//		virtual elm::caseindex ask_caseids();
		virtual elm::VAS_dna  ask_dna(const long long& c=0);
//		virtual elm::datamatrix ask_idco(const std::vector<std::string>& varnames, long long* caseid=nullptr);
//		virtual elm::datamatrix ask_idca(const std::vector<std::string>& varnames, long long* caseid=nullptr);
//		virtual elm::datamatrix ask_choice(long long* caseid=nullptr);
//		virtual elm::datamatrix ask_weight(long long* caseid=nullptr);
//		virtual elm::datamatrix ask_avail(long long* caseid=nullptr);
		
		virtual ~Facet();

	public:
		std::string window_title;
		std::string source_filename;
		std::string working_name;
		std::string active_facet;

	private:
		std::string sql_caseids;
		std::string sql_alts;
		std::string sql_idco;
		std::string sql_idca;
		std::string sql_choice;
		std::string sql_avail;
		std::string sql_weight;

		friend class QuerySetSimpleCO;
		friend class QuerySetTwoTable;
		friend class QuerySet;

		void change_in_sql_caseids();
		void change_in_sql_alts   ();
		void change_in_sql_idco   ();
		void change_in_sql_idca   ();
		void change_in_sql_choice ();
		void change_in_sql_avail  ();
		void change_in_sql_weight ();

		std::string _as_loaded_sql_alts  ;
		std::string _as_loaded_sql_idco  ;
		std::string _as_loaded_sql_idca  ;
		std::string _as_loaded_sql_choice;
		std::string _as_loaded_sql_avail ;
		std::string _as_loaded_sql_weight;

	
	private:
		PyObject* queries;
		QuerySet* queries_ptr;
	public:
		PyObject* _get_queries();
		void _set_queries(PyObject* q, QuerySet* qp);
		
		void refresh_queries();
	
	public:
//		std::string get_sql_caseids() const;
//		std::string get_sql_alts() const;
//		std::string get_sql_idco() const;
//		std::string get_sql_idca() const;
//		std::string get_sql_choice() const;
//		std::string get_sql_avail() const;
//		std::string get_sql_weight() const;
//
//		void set_sql_caseids(const std::string& qry);
//		void set_sql_alts(const std::string& qry);
//		void set_sql_idco(const std::string& qry);
//		void set_sql_idca(const std::string& qry);
//		void set_sql_choice(const std::string& qry);
//		void set_sql_avail(const std::string& qry);
//		void set_sql_weight(const std::string& qry);
				
		Facet(PyObject* pylong_ptr_to_db);
		
		void save_facet(std::string name);
		void load_facet(std::string name="");
		void clear_facet();
		
		std::vector<std::string> list_facets() const;

	public:
//		void init_facet_queries();

 
#ifndef SWIG
	private:
//		void _init_alts_query();
//		void _init_idco_query();
//		void _init_idca_query();
//		void _init_avail_query();
//		void _init_choice_query();
//		void _init_weight_query();

		void _create_facet_table_if_not_exists();

#endif // ndef SWIG
		
		
	public:
	//	std::vector<unsigned long long> caseids(const unsigned& firstcasenum=0, const unsigned& numberofcases=0) const;




	public:
//		void refresh();

//		std::string read_setup (const std::string& name, const std::string& defaultvalue="") const;
//		void        write_setup(const std::string& name, const std::string& value);

		
		const unsigned& nCases() const ;
		const unsigned& nAlts() const ;
		
		std::vector<long long> caseids(const unsigned& firstcasenum=0, const unsigned& numberofcases=0, int no_error_checking=0) const;
		std::vector<long long> altids() const;

		std::vector<std::string>    alternative_names() const;
		std::vector<long long>      alternative_codes() const;
		
		std::string    alternative_name(long long) const;
		long long      alternative_code(std::string) const;

		bool check_ca(const std::string& column) const;
		bool check_co(const std::string& column) const;

		std::vector<std::string> variables_ca() const;
		std::vector<std::string> variables_co() const;


#ifndef SWIG

	public:
		ScrapePtr get_scrape_idca();
		ScrapePtr get_scrape_idco();
		ScrapePtr get_scrape_choo();
		ScrapePtr get_scrape_wght();
		ScrapePtr get_scrape_aval();

		VAS_dna alternatives_dna() const;


	protected:
		// Dimensionality
		unsigned	_nCases;
		unsigned	_nAlts;
		
		VAS_System  _Data_DNA;
	public:
		VAS_System* DataDNA(const long long& c=0);
		
	public:
//		caseindex _caseindex;
		
//		std::list<datamatrix> _extracts;
		
	protected:
		// Column names of important keys
		std::string alias_idco_caseid() const;
		std::string alias_idca_caseid() const;
		std::string alias_idca_altid() const;
		std::string alias_alts_altid() const;
		std::string alias_alts_name() const;
		std::string alias_caseids_caseid() const;
		
		std::string alias_avail_caseid() const;
		std::string alias_avail_altid() const;
		std::string alias_avail_avail() const;


	private:
		std::string _query_chopper(long long firstrow, long long numrows) const;
		std::string& build_misc_query(std::string& q) const;

		friend class Scrape;
		
#endif // ndef SWIG

		
	public:
//		std::string query_idca(const etk::strvec& columns, long long firstrow=0, long long numrows=0, bool validate=false) const;
//		std::string query_idco(const etk::strvec& columns, long long firstrow=0, long long numrows=0, bool validate=false) const;
//		std::string query_alts() const;
//		std::string query_choice(long long firstrow=0, long long numrows=0) const;
//		std::string query_avail(long long firstrow=0, long long numrows=0) const;
//		std::string query_weight(long long firstrow=0, long long numrows=0) const;

		std::string query_idca(const std::vector<std::string>& columns, bool validate=false, long long* caseid=nullptr) const;
		std::string query_idco(const std::vector<std::string>& columns, bool validate=false, long long* caseid=nullptr) const;
		std::string query_alts(long long* caseid) const;
		std::string query_choice(long long* caseid=nullptr) const;
		std::string query_avail(long long* caseid=nullptr) const;
		std::string query_weight(long long* caseid=nullptr) const;


//		std::string build_idca_query() const;
//		std::string build_idco_query() const;
//		std::string build_alts_query() const;
//		std::string build_choice_query() const;
//		std::string build_avail_query() const;
//		std::string build_weight_query() const;
//		std::string build_caseids_query() const;

		std::string qry_idca(const bool& corrected=true) const {return queries_ptr ? queries_ptr->qry_idca(corrected) : OOPS_FACET("queries undefined");}
		std::string qry_idco(const bool& corrected=true) const {return queries_ptr ? queries_ptr->qry_idco(corrected) : OOPS_FACET("queries undefined");}
		std::string qry_idca_  () const {return queries_ptr ? queries_ptr->qry_idca_  () : OOPS_FACET("queries undefined");}
		std::string qry_idco_  () const {return queries_ptr ? queries_ptr->qry_idco_  () : OOPS_FACET("queries undefined");}
		std::string qry_alts   () const {return queries_ptr ? queries_ptr->qry_alts   () : OOPS_FACET("queries undefined");}
		std::string qry_choice () const {return queries_ptr ? queries_ptr->qry_choice () : OOPS_FACET("queries undefined");}
		std::string qry_avail  () const {return queries_ptr ? queries_ptr->qry_avail  () : OOPS_FACET("queries undefined");}
		std::string qry_weight () const {return queries_ptr ? queries_ptr->qry_weight () : OOPS_FACET("queries undefined");}
		std::string qry_caseids() const {return queries_ptr ? queries_ptr->qry_caseids() : OOPS_FACET("queries undefined");}
		
		std::string tbl_idca   (const bool& corrected=true) const {return queries_ptr ? queries_ptr->tbl_idca   (corrected) : OOPS_FACET("queries undefined");}
		std::string tbl_idco   (const bool& corrected=true) const {return queries_ptr ? queries_ptr->tbl_idco   (corrected) : OOPS_FACET("queries undefined");}
		std::string tbl_alts   () const {return queries_ptr ? queries_ptr->tbl_alts   () : OOPS_FACET("queries undefined");}
		std::string tbl_choice () const {return queries_ptr ? queries_ptr->tbl_choice () : OOPS_FACET("queries undefined");}
		std::string tbl_avail  () const {return queries_ptr ? queries_ptr->tbl_avail  () : OOPS_FACET("queries undefined");}
		std::string tbl_weight () const {return queries_ptr ? queries_ptr->tbl_weight () : OOPS_FACET("queries undefined");}
		std::string tbl_caseids() const {return queries_ptr ? queries_ptr->tbl_caseids() : OOPS_FACET("queries undefined");}
		
		inline bool unweighted() const { return queries_ptr ? queries_ptr->unweighted() : OOPS_FACET("queries undefined");}
		inline bool all_alts_always_available() const
			{ return queries_ptr ? queries_ptr->all_alts_always_available() : OOPS_FACET("queries undefined");}
		
		
	public:
//		elm::datamatrix matrix_library(size_t n);
		
		
		
		void _array_idco_reader(const std::string& qry, elm::darray* array, elm::darray* caseids);
		void _array_idca_reader(const std::string& qry, elm::darray* array, elm::darray* caseids, const std::vector<long long>& altids);
	};







	#ifdef SWIG
	%extend Facet {
	%pythoncode %{
	def sql(self):
		print("sql_idco:   %s"%(self.sql_idco   if self.sql_idco   else "<blank>"))
		print("sql_idca:   %s"%(self.sql_idca   if self.sql_idca   else "<blank>"))
		print("sql_alts:   %s"%(self.sql_alts   if self.sql_alts   else "<blank>"))
		print("sql_choice: %s"%(self.sql_choice if self.sql_choice else "<blank>"))
		print("sql_avail:  %s"%(self.sql_avail  if self.sql_avail  else "<blank>"))
		print("sql_weight: %s"%(self.sql_weight if self.sql_weight else "<blank>"))
	%}
	};


	#endif // def SWIG






	
} // end namespace elm
#endif // __ELM2_SQL_FACET_H__

