/*
 *  elm_sql_queryset.h
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

#ifndef __Hangman__elm_sql_queryset__
#define __Hangman__elm_sql_queryset__

#include <string>
#include <vector>

#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif


namespace elm {
	
	class Facet;
	
	class QuerySet {
		
	  protected:
		elm::Facet* validator;
		PyObject* py_validator;
		
	  public:
		
		virtual std::string qry_idco   (const bool& corrected=true) const;
		virtual std::string qry_idco_  () const;
		virtual std::string qry_idca   (const bool& corrected=true) const;
		virtual std::string qry_idca_  () const;
		virtual std::string qry_alts   () const;
		virtual std::string qry_caseids() const;
		virtual std::string qry_choice () const;
		virtual std::string qry_weight () const;
		virtual std::string qry_avail  () const;

		virtual std::string tbl_idco   (const bool& corrected=true) const;
		virtual std::string tbl_idca   (const bool& corrected=true) const;
		virtual std::string tbl_alts   () const;
		virtual std::string tbl_caseids() const;
		virtual std::string tbl_choice () const;
		virtual std::string tbl_weight () const;
		virtual std::string tbl_avail  () const;
		
		virtual bool unweighted() const;
		virtual bool all_alts_always_available() const;
		
		
		virtual ~QuerySet();
		
		QuerySet(elm::Facet* validator=nullptr, PyObject* validator2=nullptr);

		virtual void set_validator_(elm::Facet* validator, PyObject* validator2);
		virtual PyObject* get_validator();

		virtual std::string __repr__() const;
		virtual std::string actual_type() const;

		virtual PyObject* pickled() const;
		
	};
	
#ifdef SWIG
%extend QuerySet {
%pythoncode {
def __getstate__(self):
	args = {}
	for i in dir(self):
		if len(i)>4 and i[:4]=='get_' and i!='get_validator':
			args['set_'+i[4:]] = getattr(self,i)()
	return args
def __setstate__(self, state):
	self.__init__()
	for key, value in state.items():
		getattr(self,key)(value)
def set_validator(self, v):
	self.set_validator_(v,v)
	
}
}
#endif // def SWIG
	
	
};


#endif /* defined(__Hangman__elm_sql_queryset__) */
