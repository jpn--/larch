/*
 *  etk_random.cpp
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

#include <iostream>
#include <string>
#include <vector>
#include "etk_python.h"
#include "larch_portable.h"

#include "sqlite3ext.h"


#define time_resolution_to_save std::chrono::milliseconds


boosted::mutex etk::python_global_mutex;

std::string etk::discovered_platform_description;
int etk::number_of_cpu = 1;

PyObject* etk::pickle_module = nullptr;
PyObject* etk::base64_module = nullptr;

void etk::initialize_platform()
{
	PyObject* platform_module = PyImport_ImportModule("platform");
	PyObject* platform = PyObject_GetAttrString(platform_module, "processor");
	PyObject* name = PyObject_CallFunction(platform, "()");
	discovered_platform_description = PyString_ExtractCppString(name);
	if (PyErr_Occurred()) {
		discovered_platform_description = "unidentified processor";
		PyErr_Clear();
	}
	Py_CLEAR(name);
	Py_CLEAR(platform);
	Py_CLEAR(platform_module);

}

void etk::initialize_n_cpu()
{
	PyObject* multiprocessing_module = PyImport_ImportModule("multiprocessing");
	PyObject* cpu_count = PyObject_GetAttrString(multiprocessing_module, "cpu_count");
	PyObject* n_cpu = PyObject_CallFunction(cpu_count, "()");
	int t = PyInt_AsLong(n_cpu);
	if (!PyErr_Occurred()) {
		number_of_cpu = t;
//		std::cerr << "initialize_n_cpu("<<number_of_cpu<<")\n";
	} else {
		PyErr_Clear();
//		std::cerr << "initialize_n_cpu(?:"<<number_of_cpu<<")\n";
	}
	Py_CLEAR(n_cpu);
	Py_CLEAR(cpu_count);
	Py_CLEAR(multiprocessing_module);
}

void etk::initialize_pickle()
{
	etk::pickle_module = PyImport_ImportModule("pickle");
	etk::base64_module = PyImport_ImportModule("base64");
	//pickle = PyObject_CallMethodObjArgs(etk::pickle_module, "dumps", to_dump_object, NULL);
	//if (pickle != NULL) { ... }
	//Py_XDECREF(pickle);
}



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


int etk::py_add_to_dict(PyObject* d, const std::string& key, const std::vector<std::string>& value)
{
	PyObject* item = PyList_New(value.size());
	Py_ssize_t i=0;
	for (auto j=value.begin(); j!=value.end(); j++, i++) {
		PyList_SET_ITEM(item, i, PyString_FromString(j->c_str()));
	}
	int success = PyDict_SetItemString(d,key.c_str(),item);
	Py_CLEAR(item);
	return success;
}

int etk::py_add_to_dict(PyObject* d, const std::string& key, const std::vector< std::chrono::time_point<std::chrono::high_resolution_clock> >& value)
{
	PyObject* item = PyList_New(value.size());
	Py_ssize_t i=0;
	for (auto j=value.begin(); j!=value.end(); j++, i++) {
		auto j_ms = std::chrono::time_point_cast<time_resolution_to_save>(*j);
		PyList_SET_ITEM(item, i, PyLong_FromUnsignedLongLong((unsigned long long)(j_ms.time_since_epoch().count())));
	}
	int success = PyDict_SetItemString(d,key.c_str(),item);
	Py_CLEAR(item);
	return success;
}






//		std::vector<std::string> process_label;
//		std::vector< std::chrono::time_point<std::chrono::high_resolution_clock> > process_starttime;
//		std::vector< std::chrono::time_point<std::chrono::high_resolution_clock> > process_endtime;












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


int etk::py_read_from_dict(PyObject* d, const std::string& key,  std::vector<std::string>& value)
{
	int ret = 0;
	PyObject* list = PyDict_GetItemString(d,key.c_str());
	
	if (list) {
		value.clear();
		
		Py_ssize_t list_size = PyList_Size(list);
		for (Py_ssize_t i=0; i<list_size; i++) {
			PyObject* item = PyList_GetItem(list, i);
			if (item) {
				std::string v = PyString_ExtractCppString(item);
				if (PyErr_Occurred()) {
					ret = -1;
					PyErr_Print();
				} else {
					value.push_back(v);
				}

			}
		}
	} else {
		ret = PY_DICT_KEY_NOT_FOUND;
	}
	return ret;
	
}


int etk::py_read_from_dict(PyObject* d, const std::string& key, std::vector< std::chrono::time_point<std::chrono::high_resolution_clock> >& value)
{
	int ret = 0;
	PyObject* list = PyDict_GetItemString(d,key.c_str());
	
	if (list) {
		value.clear();
		
		Py_ssize_t list_size = PyList_Size(list);
		
		for (Py_ssize_t i=0; i<list_size; i++) {
			PyObject* item = PyList_GetItem(list, i);
			if (item) {
				unsigned long long v = PyLong_AsUnsignedLongLong(item);
				if (PyErr_Occurred()) {
					ret = -1;
					PyErr_Print();
				} else {
					value.push_back(
						std::chrono::time_point<std::chrono::high_resolution_clock>(time_resolution_to_save(v))
					);
				}
			}
		}
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
//	std::cerr << "larch_initialize()\n";
	_larch_init_;
	sqlite3_bonus_autoinit();
	sqlite3_haversine_autoinit();
	initialize_platform();
	initialize_n_cpu();
	initialize_pickle();
}


char* etk::larch_openblas_get_config()
{
#ifdef __APPLE__
	return "vecLib";
#else
	return openblas_get_config();
#endif
}

