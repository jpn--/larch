/*
 *  etk_exception.cpp
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


#include "etk.h"



etk::exception_t::exception_t (const std::string& d) throw()
: std::exception()
, _description (d)
, _oops_code(0) 
{ 
	// int i=0;
}

etk::exception_t::exception_t (const int& errorcode, const std::string& d) throw()
: std::exception()
, _description (d)
, _oops_code(errorcode) 
{ 
	// int i=0;
}

const char* etk::exception_t::what() const throw () { return _description.c_str(); }
int etk::exception_t::code() const throw () { return _oops_code; }




etk::PythonStandardException::PythonStandardException (PyObject* PyExc, const std::string& d) throw()
: etk::exception_t (OOPSCODE_PYTHON, d)
, _PyExc(PyExc)
{ }

const char* etk::PythonStandardException::what() const throw () {
	if (_description.empty()) {
		return "cached value not available";
	} else {
		return _description.c_str();
	}
}






etk::LarchCacheError::LarchCacheError (const std::string& d) throw()
: etk::exception_t (OOPSCODE_CACHE, d)
{ }

const char* etk::LarchCacheError::what() const throw () {
	if (_description.empty()) {
		return "cached value not available";
	} else {
		return _description.c_str();
	}
}


etk::ProvisioningError::ProvisioningError (const std::string& d) throw()
: etk::exception_t (OOPSCODE_PROVISION, d)
{ }

const char* etk::ProvisioningError::what() const throw () {
	if (_description.empty()) {
		return "cached value not available";
	} else {
		return _description.c_str();
	}
}




etk::ZeroProbWhenChosen::ZeroProbWhenChosen (const std::string& d) throw()
: etk::exception_t (OOPSCODE_ZEROPROB, d)
{ }

const char* etk::ZeroProbWhenChosen::what() const throw () {
	if (_description.empty()) {
		return "Zero probability for a chosen alternative";
	} else {
		return _description.c_str();
	}
}




etk::MatrixInverseError::MatrixInverseError (etk::ndarray* matrix, const std::string& d) throw()
: etk::exception_t (OOPSCODE_GENERAL, d)
, the_matrix(std::make_shared<etk::ndarray>(matrix->size1(), matrix->size2()))
{
	auto ii = matrix->size1();
	auto jj = matrix->size2();
//	the_matrix = ;
	for (auto i=0; i<ii; i++) {
		for (auto j=0; j<jj; j++) {
			the_matrix->at(i,j) = matrix->at(i,j);
		}
	}
}

PyObject* etk::MatrixInverseError::the_object() const
{
	std::shared_ptr<etk::ndarray> copy_of_matrix = std::make_shared<etk::ndarray>(the_matrix->size1(), the_matrix->size2());
	for (auto i=0; i<the_matrix->size1(); i++) {
		for (auto j=0; j<the_matrix->size2(); j++) {
			copy_of_matrix->at(i,j) = the_matrix->at(i,j);
		}
	}
	return (*(&(copy_of_matrix)))->get_object();
}




etk::FacetError::FacetError (const std::string& d) throw()
: etk::exception_t (d)
{ }




etk::UserInterrupt::UserInterrupt (const std::string& d) throw()
: etk::exception_t (d)
{
	_oops_code = -8;
}

etk::SQLiteError::SQLiteError (const int& errorcode, const std::string& d) throw()
: etk::exception_t (errorcode, d)
{ }

etk::PythonError::PythonError () throw()
: etk::exception_t (-666, "")
{ }

etk::PythonStopIteration::PythonStopIteration () throw()
: etk::exception_t (-555, "")
{ }

etk::ParameterNameError::ParameterNameError (const std::string& d) throw()
: etk::exception_t (-301, d)
{ }



bool etk::PythonErrorCheck()
{
	return PyErr_Occurred();
}

/*
	
	SQLite Error Codes
	-1	No rows returned, one row expected
	-5	Master table settings not valid
*/
