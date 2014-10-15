//
//  elm_sql_queryset.h
//  Hangman
//
//  Created by Jeffrey Newman on 4/24/14.
//  Copyright (c) 2014 Jeffrey Newman. All rights reserved.
//

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
		
	  public:
		
		virtual std::string qry_idco   () const;
		virtual std::string qry_idco_  () const;
		virtual std::string qry_idca   () const;
		virtual std::string qry_idca_  () const;
		virtual std::string qry_alts   () const;
		virtual std::string qry_caseids() const;
		virtual std::string qry_choice () const;
		virtual std::string qry_weight () const;
		virtual std::string qry_avail  () const;

		std::string tbl_idco   () const;
		std::string tbl_idca   () const;
		std::string tbl_alts   () const;
		std::string tbl_caseids() const;
		std::string tbl_choice () const;
		std::string tbl_weight () const;
		std::string tbl_avail  () const;
		
		virtual bool unweighted() const;
		virtual bool all_alts_always_available() const;
		
		
		virtual ~QuerySet();
		
		QuerySet(elm::Facet* validator=nullptr);

		virtual void set_validator(elm::Facet* validator);

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
		if len(i)>4 and i[:4]=='get_':
			args['set_'+i[4:]] = getattr(self,i)()
	return args
def __setstate__(self, state):
	self.__init__()
	for key, value in state.items():
		getattr(self,key)(value)
}
}
#endif // def SWIG
	
	
};


#endif /* defined(__Hangman__elm_sql_queryset__) */
