/*
 *  etk.h
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


#ifndef __ETK_PYTHON__
#define __ETK_PYTHON__

#include <Python.h>
#include <string>
#include <vector>
#include <chrono>
#include "etk_thread.h"

#if PY_MAJOR_VERSION < 3

std::string PyString_ExtractCppString(PyObject* pystr) ;


#else

// Fix Int to Long
#define PyInt_AsLong(x) PyLong_AsLong(x)
#define PyInt_FromLong(x) PyLong_FromLong(x)
#define PyString_FromString(x) PyUnicode_FromString(x)

std::string PyString_ExtractCppString(PyObject* pystr) ;


#endif

namespace etk {

void initialize_platform();
void initialize_n_cpu();
void initialize_pickle();

PyObject* py_one_item_list(PyObject* item);

int py_add_to_dict(PyObject* d, const std::string& key, const std::string& value);
int py_add_to_dict(PyObject* d, const std::string& key, const double& value);
int py_add_to_dict(PyObject* d, const std::string& key, const int& value);
int py_add_to_dict(PyObject* d, const std::string& key, const long& value);
int py_add_to_dict(PyObject* d, const std::string& key, const bool& value);
int py_add_to_dict(PyObject* d, const std::string& key, const unsigned& value);
int py_add_to_dict(PyObject* d, const std::string& key, const unsigned long long& value);
int py_add_to_dict(PyObject* d, const std::string& key, PyObject* value);
int py_add_to_dict(PyObject* d, const std::string& key, const std::vector<std::string>& value);
int py_add_to_dict(PyObject* d, const std::string& key, const std::vector< std::chrono::time_point<std::chrono::high_resolution_clock> >& value);

int py_read_from_dict(PyObject* d, const std::string& key, std::string& value);
int py_read_from_dict(PyObject* d, const std::string& key, double& value);
int py_read_from_dict(PyObject* d, const std::string& key, int& value);
int py_read_from_dict(PyObject* d, const std::string& key, long& value);
int py_read_from_dict(PyObject* d, const std::string& key, bool& value);
int py_read_from_dict(PyObject* d, const std::string& key, unsigned& value);
int py_read_from_dict(PyObject* d, const std::string& key, unsigned long long& value);
int py_read_from_dict(PyObject* d, const std::string& key, std::vector<std::string>& value);
int py_read_from_dict(PyObject* d, const std::string& key, std::vector< std::chrono::time_point<std::chrono::high_resolution_clock> >& value);

extern boosted::mutex python_global_mutex;
extern std::string discovered_platform_description;
extern int number_of_cpu;
extern PyObject* pickle_module;
extern PyObject* base64_module;

void larch_initialize();

char* larch_openblas_get_config();


	

};

#endif // __ETK_PYTHON__

