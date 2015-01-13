/*
 *  sherpa_freedom.cpp
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

#include "etk.h"
#include "sherpa_freedom.h"
#include <iomanip>


freedom_info::freedom_info(const std::string& name,
						   const double& value,
						   const double& null_value,
						   const int& holdfast,
						   const double& initial_value,
						   const double& std_err,
						   const double& robust_std_err,
						   const double& min_value,
						   const double& max_value,
						   PyObject* covariance,
						   PyObject* robust_covariance)
: name (name)
, initial_value (initial_value)
, value (value)
, std_err (std_err)
, robust_std_err(robust_std_err)
, null_value (null_value)
, max_value (max_value)
, min_value (min_value)
, holdfast (holdfast)
, _covar (nullptr)
, _robust_covar (nullptr)
{
	if (isNan(initial_value)) this->initial_value = value;	

	if (covariance) {
		_covar = PyDict_Copy(covariance);
	} else {
		_covar = PyDict_New();
	}
	Py_XINCREF(_covar);

	if (robust_covariance) {
		_robust_covar = PyDict_Copy(covariance);
	} else {
		_robust_covar = PyDict_New();
	}
	Py_XINCREF(_robust_covar);
}


freedom_info freedom_info::copy(const freedom_info& n)
{

	freedom_info x (n.name, n.initial_value, n.value, n.std_err, n.robust_std_err, n.null_value, n.max_value, n.min_value, n.holdfast);
	

	x._covar = PyDict_Copy(n._covar);
	Py_XINCREF(x._covar);

	x._robust_covar = PyDict_Copy(n._robust_covar);
	Py_XINCREF(x._robust_covar);
	
	return x;
}


freedom_info::~freedom_info()
{
	Py_CLEAR(_covar);
	Py_CLEAR(_robust_covar);
}

double freedom_info::t_stat() const
{
	return (value - null_value) / std_err;
}

std::string freedom_info::representation(bool pretty) const
{
	std::ostringstream rep;
	std::string tab ("");
	std::string newline ("");
	if (pretty) {
		tab = "\t";
		newline = "\n";
	} else {
		rep << std::setprecision(16) << std::scientific;
	}
	rep << "Parameter(" << newline
	<< tab << "name='" << name << "', " << newline
	<< tab << "value=" << value << ", " << newline
	<< tab << "std_err=" << std_err << ", " << newline
	<< tab << "robust_std_err=" << robust_std_err << ", " << newline
	<< tab << "null_value=" << null_value << ", " << newline
	<< tab << "initial_value=" << initial_value << ", " << newline
	<< tab << "holdfast=" << holdfast;
	if (!isInf(max_value)) {
		rep << ", " << newline << tab << "max_value=" << max_value;
	}
	if (!isInf(min_value)) {
		rep << ", " << newline << tab << "min_value=" << min_value;
	}
	rep << newline << ")";
	return rep.str();
}


PyObject* freedom_info::getCovariance() const
{	
	Py_XINCREF(_covar);
	return _covar;
}	

void freedom_info::setCovariance(PyObject* covariance)
{
	if (!PyDict_Check(covariance)) OOPS("covariance must be a python dictionary");
	Py_CLEAR(_covar);
	_covar = PyDict_Copy(covariance);
	Py_XINCREF(_covar);
}

PyObject* freedom_info::getRobustCovariance() const
{	
	Py_XINCREF(_robust_covar);
	return _robust_covar;
}	

void freedom_info::setRobustCovariance(PyObject* covariance)
{
	if (!PyDict_Check(covariance)) OOPS("covariance must be a python dictionary");
	Py_CLEAR(_robust_covar);
	_robust_covar = PyDict_Copy(covariance);
	Py_XINCREF(_robust_covar);
}

void freedom_info::update(const double& value_,
						   const double& null_value_,
						   const int& holdfast_,
						   const double& initial_value_,
						   const double& std_err_,
						   const double& robust_std_err_,
						   const double& min_value_,
						   const double& max_value_,
						   PyObject* covariance_,
						   PyObject* robust_covariance_)
{
	if (!isNan(initial_value_)) initial_value = initial_value_;
	if (!isNan(null_value_   )) null_value    = null_value_   ;
	if (!isNan(value_        )) value         = value_        ;
	if (!isNan(std_err_      )) std_err       = std_err_      ;
	if (!isNan(robust_std_err_)) robust_std_err = robust_std_err_;
	if (!isNan(max_value_)) max_value         = max_value_    ;
	if (!isNan(min_value_    )) min_value     = min_value_    ;
	if ((holdfast_ != -1     )) holdfast      = holdfast_     ;
	if (covariance_) setCovariance(covariance_);
	if (robust_covariance_) setRobustCovariance(robust_covariance_);

}

