/*
 *  etk_exception.h
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


// ERROR HANDLER //

#ifndef __etk_exception__
#define __etk_exception__

#include <exception>
#include <string>
#include "etk_cat.h"
#include <Python.h>
#include <memory>

#define OOPSCODE_GENERAL   -31200
#define OOPSCODE_ZEROPROB  -31201
#define OOPSCODE_PYTHON    -31202
#define OOPSCODE_CACHE     -31203
#define OOPSCODE_SQLITE    -31204
#define OOPSCODE_PROVISION -31205



namespace etk {

class exception_t: public std::exception {
protected:
	std::string	_description;
	int		_oops_code;
public:
	exception_t (const std::string& d="") throw();
	exception_t (const int& errorcode, const std::string& d) throw();
	virtual ~exception_t() throw() { }
	virtual const char* what() const throw ();
	virtual int code() const throw ();
};

class PythonStandardException: public exception_t {
public:
	PyObject* _PyExc;
	PythonStandardException (PyObject* PyExc, const std::string& d="") throw();
	virtual ~PythonStandardException() throw() { }
	virtual const char* what() const throw ();
};


class LarchCacheError: public exception_t {
public:
	LarchCacheError (const std::string& d="") throw();
	virtual ~LarchCacheError() throw() { }
	virtual const char* what() const throw ();
};

class ProvisioningError: public exception_t {
public:
	ProvisioningError (const std::string& d="") throw();
	virtual ~ProvisioningError() throw() { }
	virtual const char* what() const throw ();
};

class ZeroProbWhenChosen: public exception_t {
public:
	ZeroProbWhenChosen (const std::string& d="") throw();
	virtual ~ZeroProbWhenChosen() throw() { }
	virtual const char* what() const throw ();
};

class FacetError: public exception_t {
public:
	FacetError (const std::string& d="") throw();
	virtual ~FacetError() throw() { }
};

class UserInterrupt: public exception_t {
public:
	UserInterrupt (const std::string& d="") throw();
	virtual ~UserInterrupt() throw() { }
};

class SQLiteError: public exception_t {
public:
	SQLiteError (const int& errorcode, const std::string& d="") throw();

	virtual ~SQLiteError() throw() { }
};

class PythonError: public exception_t {
public:
	PythonError () throw();

	virtual ~PythonError() throw() { }
};

class PythonStopIteration: public exception_t {
public:
	PythonStopIteration () throw();
	virtual ~PythonStopIteration() throw() { }
};


class ParameterNameError: public exception_t {
public:
	ParameterNameError (const std::string& d="") throw();
	virtual ~ParameterNameError() throw() { }
};

class ndarray;

class MatrixInverseError: public exception_t {
	std::shared_ptr<ndarray> the_matrix;
public:
	PyObject* the_object() const;
	MatrixInverseError (ndarray* matrix, const std::string& d="") throw();
	virtual ~MatrixInverseError() throw() { }
};


bool PythonErrorCheck();


}


#define PYTHON_INTERRUPT throw(etk::UserInterrupt("UserInterrupt"))
#define PYTHON_ERRORCHECK if (etk::PythonErrorCheck()) throw(etk::PythonError())
#define PYTHON_STOP_ITERATION throw(etk::PythonStopIteration())

#ifdef __APPLE__
# define OOPS(...)  throw(etk::exception_t(etk::cat(__VA_ARGS__,"\n",__FILE__,":",__LINE__,": from here")))
#else
# define OOPS(...)  PYTHON_ERRORCHECK; else throw(etk::exception_t(etk::cat(__VA_ARGS__," @ ",__LINE__," of ",__FILE__)))
#endif
	
#define SPOO       catch(etk::exception_t& oops)

#define TODO       OOPS("error: this feature has not yet been completed")


#define OOPS_ZEROPROB(...)  throw(etk::ZeroProbWhenChosen(etk::cat(__VA_ARGS__)))

#define OOPS_PROVISIONING(...)  throw(etk::ProvisioningError(etk::cat(__VA_ARGS__)))

#define OOPS_SQLITE(...) throw(etk::SQLiteError(etk::cat("SQLite Error:",get_error_message(),"\n",__VA_ARGS__)))
#define OOPS_FACET(...) throw(etk::FacetError(etk::cat(__VA_ARGS__)))
#define OOPS_CACHE(...) throw(etk::LarchCacheError(etk::cat(__VA_ARGS__)))
#define OOPS_PYSTD(pyexcept,...) throw(etk::PythonStandardException((pyexcept),etk::cat(__VA_ARGS__)))
#define OOPS_MATRIXINVERSE(matrix,...) throw(etk::MatrixInverseError(matrix,etk::cat(__VA_ARGS__)))

#define OOPS_AttributeError(...) throw(etk::PythonStandardException(PyExc_AttributeError,etk::cat(__VA_ARGS__)))
#define OOPS_IndexError(...) throw(etk::PythonStandardException(PyExc_IndexError,etk::cat(__VA_ARGS__)))
#define OOPS_FileExists(...) throw(etk::PythonStandardException(PyExc_FileExistsError,etk::cat(__VA_ARGS__)))
#define OOPS_FileNotFound(...) throw(etk::PythonStandardException(PyExc_FileNotFoundError,etk::cat(__VA_ARGS__)))
#define OOPS_KeyError(...) throw(etk::PythonStandardException(PyExc_KeyError,etk::cat(__VA_ARGS__)))
#define OOPS_ValueError(...) throw(etk::PythonStandardException(PyExc_ValueError,etk::cat(__VA_ARGS__)))
#define OOPS_TypeError(...) throw(etk::PythonStandardException(PyExc_TypeError,etk::cat(__VA_ARGS__)))
#define OOPS_NotImplemented(...) throw(etk::PythonStandardException(PyExc_NotImplementedError,etk::cat(__VA_ARGS__)))


#endif

