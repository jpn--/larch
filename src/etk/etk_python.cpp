/*
 *  etk_random.cpp
 *
 *  Copyright 2007-2013 Jeffrey Newman
 *
 *  This file is part of ELM.
 *  
 *  ELM is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  ELM is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with ELM.  If not, see <http://www.gnu.org/licenses/>.
 *  
 */

#include <iostream>
#include <string>
#include "etk_python.h"
#include "larch_portable.h"

#include "sqlite3ext.h"


boosted::mutex etk::python_global_mutex;


PyObject* etk::py_one_item_list(PyObject* item)
{
	PyObject* L = PyList_New(0);
	PyList_Append(L, item);
	Py_CLEAR(item);
	return L;
}


int etk::py_add_to_dict(PyObject* d, const std::string& key, const std::string& value)
{
	PyObject* item = PyString_FromString(value.c_str());
	int success = PyDict_SetItemString(d,key.c_str(),item);
	Py_CLEAR(item);
	return success;
}

int etk::py_add_to_dict(PyObject* d, const std::string& key, const double& value)
{
	PyObject* item = PyFloat_FromDouble(value);
	int success = PyDict_SetItemString(d,key.c_str(),item);
	Py_CLEAR(item);
	return success;
}

int etk::py_add_to_dict(PyObject* d, const std::string& key, const unsigned& value)
{
	PyObject* item = PyLong_FromUnsignedLongLong((unsigned long long)(value));
	int success = PyDict_SetItemString(d,key.c_str(),item);
	Py_CLEAR(item);
	return success;
}

int etk::py_add_to_dict(PyObject* d, const std::string& key, const unsigned long long& value)
{
	PyObject* item = PyLong_FromUnsignedLongLong((unsigned long long)(value));
	int success = PyDict_SetItemString(d,key.c_str(),item);
	Py_CLEAR(item);
	return success;
}

int etk::py_add_to_dict(PyObject* d, const std::string& key, const int& value)
{
	PyObject* item = PyInt_FromLong(long(value));
	int success = PyDict_SetItemString(d,key.c_str(),item);
	Py_CLEAR(item);
	return success;
}

int etk::py_add_to_dict(PyObject* d, const std::string& key, const long& value)
{
	PyObject* item = PyInt_FromLong(value);
	int success = PyDict_SetItemString(d,key.c_str(),item);
	Py_CLEAR(item);
	return success;
}

int etk::py_add_to_dict(PyObject* d, const std::string& key, const bool& value)
{
	PyObject* item = PyInt_FromLong(long(value));
	int success = PyDict_SetItemString(d,key.c_str(),item);
	Py_CLEAR(item);
	return success;
}

int etk::py_add_to_dict(PyObject* d, const std::string& key, PyObject* value)
{
	int success = PyDict_SetItemString(d,key.c_str(),value);
	return success;
}







#define PY_DICT_KEY_NOT_FOUND -2

int etk::py_read_from_dict(PyObject* d, const std::string& key,  std::string& value)
{
	int ret = 0;
	PyObject* item = PyDict_GetItemString(d,key.c_str());
	if (item) {
		std::string v = PyString_ExtractCppString(item);
		if (PyErr_Occurred()) {
			ret = -1;
			PyErr_Print();
		} else {
			value = v;
		}
		Py_CLEAR(item);
		return ret;
	} else {
		ret = PY_DICT_KEY_NOT_FOUND;
	}
	return ret;
}

int etk::py_read_from_dict(PyObject* d, const std::string& key,  double& value)
{
	int ret = 0;
	PyObject* item = PyDict_GetItemString(d,key.c_str());
	if (item) {
		double v = PyFloat_AsDouble(item);
		if (PyErr_Occurred()) {
			ret = -1;
			PyErr_Print();
		} else {
			value = v;
		}
		Py_CLEAR(item);
		return ret;
	} else {
		ret = PY_DICT_KEY_NOT_FOUND;
	}
	return ret;
}

int etk::py_read_from_dict(PyObject* d, const std::string& key,  unsigned& value)
{
	int ret = 0;
	PyObject* item = PyDict_GetItemString(d,key.c_str());
	if (item) {
		unsigned long v = PyLong_AsUnsignedLong(item);
		if (PyErr_Occurred()) {
			ret = -1;
			PyErr_Print();
		} else {
			value = v;
		}
		Py_CLEAR(item);
		return ret;
	} else {
		ret = PY_DICT_KEY_NOT_FOUND;
	}
	return ret;
}

int etk::py_read_from_dict(PyObject* d, const std::string& key,  unsigned long long& value)
{
	int ret = 0;
	PyObject* item = PyDict_GetItemString(d,key.c_str());
	if (item) {
		unsigned long long v = PyLong_AsUnsignedLongLong(item);
		if (PyErr_Occurred()) {
			ret = -1;
			PyErr_Print();
		} else {
			value = v;
		}
		Py_CLEAR(item);
		return ret;
	} else {
		ret = PY_DICT_KEY_NOT_FOUND;
	}
	return ret;
}

int etk::py_read_from_dict(PyObject* d, const std::string& key,  int& value)
{
	int ret = 0;
	PyObject* item = PyDict_GetItemString(d,key.c_str());
	if (item) {
		long v = PyInt_AsLong(item);
		if (PyErr_Occurred()) {
			ret = -1;
			PyErr_Print();
		} else {
			value = v;
		}
		Py_CLEAR(item);
		return ret;
	} else {
		ret = PY_DICT_KEY_NOT_FOUND;
	}
	return ret;
}

int etk::py_read_from_dict(PyObject* d, const std::string& key,  long& value)
{
	int ret = 0;
	PyObject* item = PyDict_GetItemString(d,key.c_str());
	if (item) {
		long v = PyInt_AsLong(item);
		if (PyErr_Occurred()) {
			ret = -1;
			PyErr_Print();
		} else {
			value = v;
		}
		Py_CLEAR(item);
		return ret;
	} else {
		ret = PY_DICT_KEY_NOT_FOUND;
	}
	return ret;
}

int etk::py_read_from_dict(PyObject* d, const std::string& key,  bool& value)
{
	int ret = 0;
	PyObject* item = PyDict_GetItemString(d,key.c_str());
	if (item) {
		long v = PyInt_AsLong(item);
		if (PyErr_Occurred()) {
			ret = -1;
			PyErr_Print();
		} else {
			value = v;
		}
		Py_CLEAR(item);
		return ret;
	} else {
		ret = PY_DICT_KEY_NOT_FOUND;
	}
	return ret;
}





#if PY_MAJOR_VERSION < 3

std::string PyString_ExtractCppString(PyObject* pystr) {
	return PyString_AsString(pystr);
}


#else

std::string PyString_ExtractCppString(PyObject* pystr) {

	PyObject* pyo = PyUnicode_Encode(PyUnicode_AsUnicode(pystr), PyUnicode_GetSize(pystr), "ascii", "ignore");
	std::string x (PyBytes_AsString(pyo));
	Py_CLEAR(pyo);
	return x;
}


#endif


#ifdef __cplusplus 
extern "C" {
#endif

extern const sqlite3_api_routines *sqlite3_api;
void sqlite3_bonus_autoinit();
void sqlite3_haversine_autoinit();

#ifdef __cplusplus
}
#endif



void etk::larch_initialize()
{
	std::cerr << "larch_initialize()\n";
	_larch_init_;
	sqlite3_bonus_autoinit();
	sqlite3_haversine_autoinit();
}


char* etk::larch_openblas_get_config()
{
#ifdef __APPLE__
	return "vecLib";
#else
	return openblas_get_config();
#endif
}

