//
//  elm_sql_queryset_simpleco.h
//  Hangman
//
//  Created by Jeffrey Newman on 4/24/14.
//  Copyright (c) 2014 Jeffrey Newman. All rights reserved.
//

#ifndef __Hangman__elm_sql_queryset_simpleco__
#define __Hangman__elm_sql_queryset_simpleco__

#include "elm_queryset.h"
#include <map>

#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif

namespace elm {
	
	
	
	class QuerySetSimpleCO
	: public QuerySet
	{
	public:

//		static std::shared_ptr<QuerySetSimpleCO> create(elm::Facet* validator=nullptr);
		
		virtual std::string qry_idco   () const;
		virtual std::string qry_idco_  () const;
		virtual std::string qry_idca   () const;
		virtual std::string qry_idca_  () const;
		virtual std::string qry_alts   () const;
		virtual std::string qry_caseids() const;
		virtual std::string qry_choice () const;
		virtual std::string qry_weight () const;
		virtual std::string qry_avail  () const;

		virtual bool unweighted() const;
		virtual bool all_alts_always_available() const;
		
		virtual ~QuerySetSimpleCO();

		virtual std::string __repr__() const;
		virtual std::string actual_type() const;
		
	public:

		QuerySetSimpleCO(elm::Facet* validator=nullptr);

		void set_validator(elm::Facet* validator=nullptr);

		void set_idco_query(const std::string& q);
		void set_choice_column(const std::string& col);
		void set_choice_column_map(const std::map<long long, std::string>& cols);
		void set_avail_column_map(const std::map<long long, std::string>& cols);
		void set_avail_all();
		void set_weight_column(const std::string& col);
		void set_alts_query(const std::string& q);
		void set_alts_values(const std::map<long long, std::string>& alts);

		std::string get_idco_query() const;
		std::string get_choice_column() const;
		std::map<long long, std::string> get_choice_column_map() const;
		std::map<long long, std::string> get_avail_column_map() const;
		std::string get_weight_column() const;
		std::string get_alts_query() const;
		
	private:
		std::string _idco_query;
		std::string _single_choice_column;
		std::map<long long, std::string> _alt_choice_columns;
		std::map<long long, std::string> _alt_avail_columns;
		std::string _weight_column;
		std::string _alts_query;
		
		std::string __test_query_caseids(const std::string& alias_caseid) const;
		std::string __alias_caseid() const;

		
	};

	
};




#ifdef SWIG
%pythoncode %{
from . import QuerySetSimpleCO as _morefuncs
del _morefuncs
%}
#endif // def SWIG



#endif /* defined(__Hangman__elm_sql_queryset_simpleco__) */
