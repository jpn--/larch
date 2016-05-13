/*
 *  elm_sql_queryset_simpleco.h
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
		
		virtual std::string qry_idco   (const bool& corrected=true) const;
		virtual std::string qry_idco_  () const;
		virtual std::string qry_idca   (const bool& corrected=true) const;
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
		void set_avail_query(const std::string& q);
		void set_avail_all();
		void set_weight_column(const std::string& col);
		void set_alts_query(const std::string& q);
		void set_alts_values(const std::map<long long, std::string>& alts);

		std::string get_idco_query() const;
		std::string get_choice_column() const;
		std::map<long long, std::string> get_choice_column_map() const;
		std::map<long long, std::string> get_avail_column_map() const;
		std::string get_avail_query() const;
		std::string get_weight_column() const;
		std::string get_alts_query() const;
		std::map<long long, std::string> _get_alts_values() const;
		
	private:
		std::string _idco_query;
		std::string _single_choice_column;
		std::map<long long, std::string> _alt_choice_columns;
		std::map<long long, std::string> _alt_avail_columns;
		std::string _weight_column;
		std::string _alts_query;
		std::string _alt_avail_query;
		
		std::string __test_query_caseids(const std::string& alias_caseid) const;
		std::string __alias_caseid() const;

		
	};

	
};




#ifdef SWIG
%pythoncode %{
from . import _QuerySetSimpleCO_extras as _morefuncs
del _morefuncs
%}
#endif // def SWIG



#endif /* defined(__Hangman__elm_sql_queryset_simpleco__) */
