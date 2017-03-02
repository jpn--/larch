/*
 *  etk_basic.cpp
 *
 *  Copyright 2007-2017 Jeffrey Newman
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
#include <iostream>
#include <numpy/npy_math.h>

/*	class dictionary_sd {
		std::string	_key;
		double		_value;
		bool		_value_changed;
		PyObject*	_pyo;
		
		dictionary_sd(PyObject* dict);
		~dictionary_sd();
		
		double& operator[](const std::string key);
	};
*/

etk::dictionary_sd::dictionary_sd(PyObject* dict)
: _key ("")
, _value (NAN)
, _value_orig (NAN)
, _pyo (dict)
{
	if (!PyDict_Check(_pyo)) OOPS("dictionary_sd requires a python dictionary object");
	Py_XINCREF(_pyo);
}

etk::dictionary_sd::~dictionary_sd()
{
	int err = 999;
	if (_value_orig != _value) {
		PyObject* v = PyFloat_FromDouble(_value);
		if (!v) {
			v = Py_None;
			Py_XINCREF(v);
		}
		err = PyDict_SetItemString(_pyo, _key.c_str(), v);
		Py_CLEAR(v);
	}
	Py_CLEAR(_pyo);
//	if (err<0) {
//		std::cerr << "potential failure, did not add '"<<_key.c_str()<<"' to dict with value "<<_value<<"\n";
//	} else if (err==0) {
//		std::cerr << "ok, did add '"<<_key.c_str()<<"' to dict with value "<<_value<<"\n";
//	}
}

double& etk::dictionary_sd::key(const std::string& key)
{
	_key = key;
	PyObject* v = PyDict_GetItemString(_pyo, _key.c_str());
	// v is null if _key is not in the dict
	if (v) {
		_value = _value_orig = PyFloat_AsDouble(v);
		if (PyErr_Occurred()) OOPS("an error occurred in retriving a double from the dictionary");
	}
	return _value;
}

void etk::dictionary_sd::set_key_nan(const std::string& key)
{
	_key = key;
	_value = _value_orig = 0;
	
	PyObject* v = PyFloat_FromDouble(Py_NAN);
	Py_XINCREF(v);
	int err = PyDict_SetItemString(_pyo, _key.c_str(), v);
	Py_CLEAR(v);

//	if (err!=0) {
//		std::cerr << "potential failure, did not add '"<<_key.c_str()<<"' to dict with value NaN\n";
//	} else {
//		std::cerr << "ok, did add '"<<_key.c_str()<<"' to dict with value NaN\n";
//	}

}




