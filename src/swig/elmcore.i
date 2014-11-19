%define DOCSTRING
"The ELM module v3 can be used to estimate discrete choice models. It runs in python v3."
%enddef

%module(docstring=DOCSTRING, package="elm") core



// Fix an error in swig property::setter's
%pythoncode %{

def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "thisown"): return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static):
        object.__setattr__(self, name, value)
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

%}


// Fix an error in swigging Py3:
%{
#define PyInt_FromSize_t PyLong_FromSize_t
%}







%pythoncode %{
class LarchError(Exception):
	def __str__(self):
		return "ELM has encountered an error:" + Exception.__str__(self)
class SQLiteError(Exception):
	def __str__(self):
		return "ELM has encountered an error using SQLite:" + Exception.__str__(self)
class FacetError(Exception):
	def __str__(self):
		return "ELM has encountered an error in the data facet:" + Exception.__str__(self)
pass	
%}



//#ifdef SWIG
//			%feature("pythonappend") elm::datamatrix_t::read_idca %{
//				#val.incref() # when creating in python, python owns the only reference to this object
//				%}
//			%feature("pythonappend") elm::datamatrix_t::read_idco %{
//				#val.incref() # when creating in python, python owns the only reference to this object
//				%}
//			%feature("pythonappend") elm::datamatrix_t::read_wght %{
//				#val.incref() # when creating in python, python owns the only reference to this object
//				%}
//			%feature("pythonappend") elm::datamatrix_t::read_choo %{
//				#val.incref() # when creating in python, python owns the only reference to this object
//				%}
//			%feature("pythonappend") elm::datamatrix_t::read_aval %{
//				#val.incref() # when creating in python, python owns the only reference to this object
//				%}
//#endif // def SWIG
//			
//
//			
//#ifdef SWIG
//			%feature("pythonappend") elm::Facet::ask_idca %{
//				val.incref() # python gets a shared reference to object
//				%}
//			%feature("pythonappend") elm::Facet::ask_idco %{
//				val.incref() # python gets a shared reference to object
//				%}
//			%feature("pythonappend") elm::Facet::ask_choo %{
//				val.incref() # python gets a shared reference to object
//				%}
//			%feature("pythonappend") elm::Facet::ask_wght %{
//				val.incref() # python gets a shared reference to object
//				%}
//			%feature("pythonappend") elm::Facet::ask_aval %{
//				val.incref() # python gets a shared reference to object
//				%}
//#endif // def SWIG
			



/////////////////////////////
// MARK: Exception Handling
%{
#include "etk_exception.h"
#include "etk_test_swig.h"
%}


%{
static PyObject* ptrToLarchError;  /* add this! */
static PyObject* ptrToSQLError;  /* add this! */
static PyObject* ptrToFacetError;  /* add this! */
%}

%init %{
    ptrToLarchError = PyErr_NewException("larch.LarchError", NULL, NULL);
    Py_INCREF(ptrToLarchError);
    PyModule_AddObject(m, "LarchError", ptrToLarchError);
    ptrToSQLError = PyErr_NewException("larch.SQLiteError", NULL, NULL);
    Py_INCREF(ptrToSQLError);
    PyModule_AddObject(m, "SQLiteError", ptrToSQLError);
    ptrToFacetError = PyErr_NewException("larch.FacetError", NULL, NULL);
    Py_INCREF(ptrToFacetError);
    PyModule_AddObject(m, "FacetError", ptrToFacetError);
%}

%pythoncode %{
	LarchError = _core.LarchError
	SQLiteError = _core.SQLiteError
	FacetError = _core.FacetError
%}

%include "exception.i"
%exception {
	try {
		$action
	} catch (const etk::PythonStopIteration& e) {
		PyErr_SetNone(PyExc_StopIteration);
		return NULL;
	} catch (const etk::SQLiteError& e) {
		PyErr_SetString(ptrToSQLError, const_cast<char*>(e.what()));
		return NULL;
	} catch (const etk::FacetError& e) {
		PyErr_SetString(ptrToFacetError, const_cast<char*>(e.what()));
		return NULL;
	} catch (const std::exception& e) {
		PyErr_SetString(ptrToLarchError, const_cast<char*>(e.what()));
		return NULL;
	}
}



%include "std_string.i"
%include "std_vector.i"
%include "std_map.i"

%include "std_shared_ptr.i"
%shared_ptr(elm::datamatrix_t);
%shared_ptr(elm::caseindex_t);
%shared_ptr(elm::QuerySet);
%shared_ptr(elm::QuerySetSimpleCO);
%shared_ptr(elm::QuerySetTwoTable);

namespace elm {
	typedef std::shared_ptr<elm::datamatrix_t> datamatrix;
	typedef std::shared_ptr<elm::caseindex_t> caseindex;
//	typedef std::shared_ptr<elm::QuerySet> queries;
//	typedef std::shared_ptr<elm::QuerySetSimpleCO> queries1;
//	typedef std::shared_ptr<elm::QuerySetTwoTable> queries2;
};

%{
#define PY_ARRAY_UNIQUE_SYMBOL _ETK_PY_ARRAY_UNIQUE_SYMBOL_ 
#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#include <numpy/arrayobject.h>
%}


%init %{

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
import_array();

%}

%pythoncode %{
import numpy
__default_array_type__ = numpy.ndarray


%}


 

// Instantiate templates
namespace std { 
	%template(IntVector) vector<int>;
	%template(DoubleVector) vector<double>;
	%template(ULongLongVector) vector<unsigned long long>;
	%template(LongLongVector) vector<long long>;
	
}


%template(StrVector) std::vector<std::string>;
%template(IntStringDict) std::map<long long, std::string>;

%extend std::map<long long, std::string> {
	%pythoncode {
		def __reduce__(self):
			args = (dict(self), )
			return self.__class__, args
	}
}
 

%{
#include "etk_arraymath.h"
#include "elm_queryset.h"
#include "elm_queryset_simpleco.h"
#include "elm_queryset_twotable.h"
%}
 
namespace elm {
	void set_linalg(PyObject* mod);
};

%include "etk_test_swig.h"
%include "etk_refcount.h"

//%include "etk_vectors.h"

%include "sherpa_pack.h"

%include "elm2array.i"	 

%include "elm_fountain.h" 

%include "elm_queryset.h"
%include "elm_queryset_simpleco.h"
%include "elm_queryset_twotable.h"
	
%include "elm_sql_connect.h"
%include "elm_sql_facet.h"
%pythoncode %{
from .db import DB
%}

%include "etk_resultcodes.h"

%include "elm_parameter2.h"
%include "elm_cellcode.h"
%include "elm_vascular.h"
%include "elm_inputstorage.h"
%include "elm_model2_options.h"
%include "elm_runstats.h"

%include "elm_caseindex.h"
%include "elm_datamatrix.h"

%template(Needs) std::map<std::string, elm::darray_req>;
%extend std::map<std::string, elm::darray_req> {
	%pythoncode {
		def __repr__(self):
			return "<Needs:" + ",".join(["{}({})".format(i,len(j.get_variables())) for i,j in self.items()]) + ">"
	}
}
%include "elm_darray.h"

%include "elm_parameterlist.h"
%include "sherpa_freedom.h"
%include "sherpa.h"
%include "elm_model2.h"
%pythoncode %{
from .model import Model
%}
 

 
