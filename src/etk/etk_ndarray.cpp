/*
 *  etk_ndarray.cpp
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
#include <numpy/arrayobject.h> 

using namespace etk;

#include <iostream>
#include <iomanip>
#include <cmath>
#include <climits>
#include <cstring>

#include "etk_arraymath.h"

#define ROWS PyArray_DIM(pool, 0)
#define COLS (PyArray_NDIM(pool)>1 ? PyArray_DIM(pool, 1) : 1)
#define DEPS (PyArray_NDIM(pool)>2 ? PyArray_DIM(pool, 2) : 1)

PyObject* etk::array_module = NULL;

void etk::_set_array_module(PyObject* mod)
{
	Py_CLEAR(array_module);
	array_module = mod;
	Py_XINCREF(array_module);
}


PyObject* etk::get_array_type(const char* type) {
//	PyObject* array_module = PyImport_ImportModule("larch.array");
	if (!array_module) {
		OOPS("there is no larch.array indicated");
	}
	PyObject* type_obj = PyObject_GetAttrString(array_module, type);
//	Py_CLEAR(array_module);
	if (!type_obj) {
		OOPS("Failed to find larch.array type object");
	}
	return type_obj;
}




symmetric_matrix::symmetric_matrix(PyObject* obj)
: ndarray (obj)
{

}

symmetric_matrix::symmetric_matrix(symmetric_matrix& that, bool use_same_memory)
: ndarray (that, use_same_memory)
{

}


symmetric_matrix::symmetric_matrix(const char* arrayType)
: ndarray (0,0,arrayType)
{

}


symmetric_matrix::symmetric_matrix(const int& r, const char* arrayType)
: ndarray (r,r,arrayType)
{

}

symmetric_matrix::symmetric_matrix(const int& r,const int& c, const char* arrayType)
: ndarray (r,r,arrayType)
{

}

symmetric_matrix::symmetric_matrix(const int& r,const int& c,const int& s, const char* arrayType)
: ndarray (r,r,arrayType)
{

}

inline PyObject* Py_XINCREF_INLINE (PyObject* obj) {Py_XINCREF(obj);return obj;}

ndarray::ndarray(PyObject* obj)
: pool ((PyArrayObject*)Py_XINCREF_INLINE(obj))
{
	if (!PyArray_Check(pool)) {
		Py_CLEAR(pool);
		OOPS("Error creating array wrapper, input must be an array. Try reformatting it with larch.array.pack().");
	}
	if (!PyArray_ISCONTIGUOUS(pool)) {
		Py_CLEAR(pool);
		OOPS("Error creating array wrapper, input array must be C-Contiguous. Try reformatting it with larch.array.pack().");
	}
	else if (!PyArray_ISALIGNED(pool)) {
		Py_CLEAR(pool);
		OOPS("Error creating array wrapper, input array must be aligned. Try reformatting it with larch.array.pack().");
	}
//	else if (!PyArray_ISWRITEABLE(pool)) {
//		Py_CLEAR(pool);
//		OOPS("Error creating array wrapper, input array must be writable. Try reformatting it with larch.array.pack().");
//	}
	else if (!PyArray_ISCARRAY_RO(pool)) {
		std::string errmsg = cat("Error creating array wrapper, flags offered are ",std::hex,PyArray_FLAGS(pool),", flags needed are ",std::hex,NPY_ARRAY_CARRAY_RO);
		Py_CLEAR(pool);
		OOPS(errmsg);
	}
}







#include "etk_thread.h"

boosted::mutex ndarray::python_mutex;

void ndarray::same_memory_as(ndarray& x)
{
	boosted::lock_guard<boosted::mutex> LOCK(etk::python_global_mutex);
	Py_CLEAR(pool);
	pool = x.pool;
	Py_XINCREF(pool);
}


//    (PyArrayObject*).descr->type_num
#define ASSERT_ARRAY_WRITEABLE if ( !PyArray_ISWRITEABLE(pool) ) OOPS("assert failure, array not writeable")
#define ASSERT_ARRAY_DOUBLE    if ( PyArray_DESCR(pool)->type_num != NPY_DOUBLE) OOPS("assert failure, not NPY_DOUBLE")
#define ASSERT_ARRAY_BOOL      if ( PyArray_DESCR(pool)->type_num != NPY_BOOL) OOPS("assert failure, not NPY_BOOL")
#define ASSERT_ARRAY_INT64     if ( PyArray_DESCR(pool)->type_num != NPY_INT64) OOPS("assert failure, not NPY_INT64")

void ndarray::quick_new(const int& datatype, const char* arrayClass, const int& r,const int& c,const int& s)
{
	//boosted::lock_guard<boosted::mutex> pylock(python_mutex);
	boosted::lock_guard<boosted::mutex> LOCK(etk::python_global_mutex);
	Py_CLEAR(pool);
	npy_intp dims [3] = {r,c,s};
	int ndims = 3;
	if (s==-1) ndims = 2;
	if (c==-1) ndims = 1;
	if (strncmp(arrayClass, "SymmetricArray", 14)==0) {ndims=2; dims[1]=dims[0];}
	PyObject* subtype = get_array_type(arrayClass);
	pool = (PyArrayObject*)PyArray_New((PyTypeObject*)subtype, ndims, &dims[0], datatype, nullptr, nullptr, 0, 0, nullptr);
	if (!pool) {
		//if (PyErr_Occurred()) PyErr_Print();
		PYTHON_ERRORCHECK;
		OOPS("Unknown error creating array");
	}
	Py_INCREF(pool);
	Py_CLEAR(subtype);
	if (!PyArray_ISCARRAY(pool)) {
		std::cout << "Generated array is not C-Contiguous\n";
		std::cout << "<repr>: ";
		PyObject_Print((PyObject*)pool,stdout,0);
		std::cout << "\n<str>: ";
		PyObject_Print((PyObject*)pool,stdout,0);
		std::cout << "\n";
		std::cout << "dec flags: "<< PyArray_FLAGS(pool) << "\n";
		std::cout << "hex flags: 0x"<< std::hex << PyArray_FLAGS(pool) << "\n";
		std::cout << "desired flags: 0x"<< std::hex << NPY_ARRAY_CARRAY << "\n";
		Py_CLEAR(pool);
		OOPS("Error creating c-contiguous array");
	}
	PyArray_FILLWBYTE(pool, 0);
}

ndarray::ndarray(const char* arrayType)
: pool (nullptr)
{
/*	npy_intp dims [1] = {0};
	//pool = (PyArrayObject*)PyArray_ZEROS(1, &dims[0], NPY_DOUBLE, 0);
	PyObject* subtype = get_array_type(arrayType);
	pool = (PyArrayObject*)PyArray_New((PyTypeObject*)subtype, 1, &dims[0], NPY_DOUBLE, nullptr, nullptr, 0, NPY_CARRAY, nullptr);
	Py_CLEAR(subtype);
	Py_INCREF(pool);
	if (!PyArray_ISCARRAY(pool)) {
		Py_CLEAR(pool);
		OOPS("Error creating array");
	}*/
	quick_new(NPY_DOUBLE, arrayType, 0);
}

ndarray::ndarray(const char* arrayType, const int& datatype, const int& r, const int& c, const int& s)
: pool (nullptr)
{
	quick_new(datatype, arrayType, r,c,s);
}


ndarray::ndarray(const int& r, const char* arrayType)
: pool (nullptr)
{
	quick_new(NPY_DOUBLE, arrayType, r);
}

ndarray::ndarray(const int& r,const int& c, const char* arrayType)
: pool (nullptr)
{
	quick_new(NPY_DOUBLE, arrayType, r, c);
}

ndarray::ndarray(const int& r,const int& c,const int& s, const char* arrayType)
: pool (nullptr)
{
	quick_new(NPY_DOUBLE, arrayType, r, c, s);
}

ndarray::ndarray(const ndarray& that, bool use_same_memory)
: pool (nullptr)
{
	if (use_same_memory) {
		OOPS("cannot use same memory for const array");
	} else {
		if (!that.pool) OOPS("Error copying ndarray, source is null");
		if (!pool || !PyArray_SAMESHAPE(pool, that.pool) || PyArray_DTYPE(pool)!=PyArray_DTYPE(that.pool) ) {
			Py_CLEAR(pool);
			pool = (PyArrayObject*)PyArray_NewCopy((PyArrayObject*)that.pool, NPY_CORDER);
			Py_INCREF(pool);
		} else {
			int z = PyArray_CopyInto((PyArrayObject*)pool, (PyArrayObject*)that.pool);
			if (z) OOPS("Error copying ndarray");
		}
	}
	
	
}

ndarray::ndarray(ndarray& that, bool use_same_memory)
: pool (nullptr)
{
	if (use_same_memory && that.pool) {
		Py_INCREF(that.pool);
		pool = that.pool;
	} else {
		if (!that.pool) OOPS("Error copying ndarray, source is null");
		if (!pool || !PyArray_SAMESHAPE(pool, that.pool) || PyArray_DTYPE(pool)!=PyArray_DTYPE(that.pool) ) {
			Py_CLEAR(pool);
			pool = (PyArrayObject*)PyArray_NewCopy((PyArrayObject*)that.pool, NPY_CORDER);
			Py_INCREF(pool);
		} else {
			int z = PyArray_CopyInto((PyArrayObject*)pool, (PyArrayObject*)that.pool);
			if (z) OOPS("Error copying ndarray");
		}
	}
	
	
}

ndarray::~ndarray()
{
	Py_CLEAR(pool);
}


void ndarray::resize(const int& r)
{
	if (!pool
		|| PyArray_DESCR(pool)->type_num != NPY_DOUBLE
		||	ndim() != 1
		||	size1() != r
		)
	quick_new(NPY_DOUBLE, "Array", r);
}

void ndarray::resize(const int& r,const int& c)
{
	if (!pool
		|| PyArray_DESCR(pool)->type_num != NPY_DOUBLE
		||	ndim() != 2
		||	size1() != r
		||	size2() != c
		)
	quick_new(NPY_DOUBLE, "Array", r,c);
}

void ndarray::resize(const int& r,const int& c,const int& s)
{
	if (!pool
		|| PyArray_DESCR(pool)->type_num != NPY_DOUBLE
		||	ndim() != 3
		||	size1() != r
		||	size2() != c
		||	size3() != s
		)
	quick_new(NPY_DOUBLE, "Array", r,c,s);
}

void ndarray::resize_bool(const int& r)
{
	if (!pool
		|| PyArray_DESCR(pool)->type_num != NPY_BOOL
		||	ndim() != 1
		||	size1() != r
		)
	quick_new(NPY_BOOL, "Array", r);
}

void ndarray::resize_bool(const int& r,const int& c)
{
	if (!pool
		|| PyArray_DESCR(pool)->type_num != NPY_BOOL
		||	ndim() != 2
		||	size1() != r
		||	size2() != c
		)
	quick_new(NPY_BOOL, "Array", r,c);
}

void ndarray::resize_bool(const int& r,const int& c,const int& s)
{
	if (!pool
		|| PyArray_DESCR(pool)->type_num != NPY_BOOL
		||	ndim() != 3
		||	size1() != r
		||	size2() != c
		||	size3() != s
		)
	quick_new(NPY_BOOL, "Array", r,c,s);
}

void ndarray::resize(ndarray& prototype)
{
	Py_CLEAR(pool);
	pool = (PyArrayObject*)PyArray_NewLikeArray(prototype.pool, NPY_CORDER, NULL, 1);
	Py_INCREF(pool);
	PyArray_FILLWBYTE(pool, 0);
}

void ndarray::resize(etk::three_dim& prototype)
{
	quick_new(NPY_DOUBLE, "Array", prototype.size1(),prototype.size2(),prototype.size3());
}

void ndarray::resize_if_needed(etk::three_dim& prototype)
{
	if (ndim()!=3 || size1()!=prototype.size1() || size2()!=prototype.size2() || size3()!=prototype.size3()) {
		quick_new(NPY_DOUBLE, "Array", prototype.size1(),prototype.size2(),prototype.size3());
	}
}



double& ndarray::operator[](const int& i)
{
	ASSERT_ARRAY_DOUBLE;
	if (i<PyArray_SIZE(pool)) return (static_cast<double*>( PyArray_DATA(pool) ))[i];
	OOPS(cat("puddle access out of range: ",i," in ",PyArray_SIZE(pool)));
}

const double& ndarray::operator[](const int& i) const
{
	ASSERT_ARRAY_DOUBLE;
	if (i<PyArray_SIZE(pool)) return (static_cast<double*>( PyArray_DATA(pool) ))[i];
	OOPS(cat("const puddle access out of range: ",i," in ",PyArray_SIZE(pool)));
}

const double& ndarray::operator()(const int& r) const
{
	ASSERT_ARRAY_DOUBLE;
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	return *(const double*)PyArray_GETPTR1(pool, r);
}

const double& ndarray::operator()(const int& r, const int& c) const
{
	ASSERT_ARRAY_DOUBLE;
	if (PyArray_NDIM(pool)<2) {
		if (c==0) {
			return operator()(r);
		}
		OOPS("2 dim location requested in ndarray with ",PyArray_NDIM(pool)," dim ");
	}
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	if (c>=PyArray_DIM(pool, 1)) {
		OOPS("const rectangle col access out of range, asking ",c," but having only ",PyArray_DIM(pool, 1));
	}
	return *(const double*)PyArray_GETPTR2(pool, r, c);
}

const bool& ndarray::bool_at(const int& r) const
{
	ASSERT_ARRAY_BOOL;
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	return *(const bool*)PyArray_GETPTR1(pool, r);
}

const long long& ndarray::int64_at(const int& r) const
{
	ASSERT_ARRAY_INT64;
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	return *(const long long*)PyArray_GETPTR1(pool, r);
}


const bool& ndarray::bool_at(const int& r, const int& c) const
{
	ASSERT_ARRAY_BOOL;
	if (PyArray_NDIM(pool)<2) {
		if (c==0) {
			return bool_at(r);
		}
		OOPS("2 dim location requested in ndarray with ",PyArray_NDIM(pool)," dim ");
	}
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	if (c>=PyArray_DIM(pool, 1)) {
		OOPS("const rectangle col access out of range, asking ",c," but having only ",PyArray_DIM(pool, 1));
	}
	return *(const bool*)PyArray_GETPTR2(pool, r, c);
}

const long long& ndarray::int64_at(const int& r, const int& c) const
{
	ASSERT_ARRAY_INT64;
	if (PyArray_NDIM(pool)<2) {
		if (c==0) {
			return int64_at(r);
		}
		OOPS("2 dim location requested in ndarray with ",PyArray_NDIM(pool)," dim ");
	}
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	if (c>=PyArray_DIM(pool, 1)) {
		OOPS("const rectangle col access out of range, asking ",c," but having only ",PyArray_DIM(pool, 1));
	}
	return *(const long long*)PyArray_GETPTR2(pool, r, c);
}

const double& symmetric_matrix::operator()(const int& r, const int& c) const
{
	ASSERT_ARRAY_DOUBLE;
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	if (c>=PyArray_DIM(pool, 1)) {
		OOPS("const rectangle col access out of range, asking ",c," but having only ",PyArray_DIM(pool, 1));
	}
	if (r>c) return *(double*)PyArray_GETPTR2(pool, c, r);
	return *(double*)PyArray_GETPTR2(pool, r, c);
}
double& symmetric_matrix::operator()(const int& r, const int& c)
{
	ASSERT_ARRAY_DOUBLE;
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	if (c>=PyArray_DIM(pool, 1)) {
		OOPS("const rectangle col access out of range, asking ",c," but having only ",PyArray_DIM(pool, 1));
	}
	if (r>c) return *(double*)PyArray_GETPTR2(pool, c, r);
	return *(double*)PyArray_GETPTR2(pool, r, c);
}

const double& ndarray::operator()(const int& r, const int& c, const int& d) const
{
	ASSERT_ARRAY_DOUBLE;
	if (PyArray_NDIM(pool)<3) {
		if (d==0) {
			return operator()(r,c);
		}
		OOPS("3 dim location requested in ndarray with ",PyArray_NDIM(pool)," dim ");
	}
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	if (c>=PyArray_DIM(pool, 1)) {
		OOPS("const rectangle col access out of range, asking ",c," but having only ",PyArray_DIM(pool, 1));
	} 
	if (d>=PyArray_DIM(pool, 2)) {
		OOPS("const rectangle dep access out of range, asking ",d," but having only ",PyArray_DIM(pool, 2));
	}
	return *(const double*)PyArray_GETPTR3(pool, r, c, d);
}

const bool& ndarray::bool_at(const int& r, const int& c, const int& d) const
{
	ASSERT_ARRAY_BOOL;
	if (PyArray_NDIM(pool)<3) {
		if (d==0) {
			return bool_at(r,c);
		}
		OOPS("3 dim location requested in ndarray with ",PyArray_NDIM(pool)," dim ");
	}
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	if (c>=PyArray_DIM(pool, 1)) {
		OOPS("const rectangle col access out of range, asking ",c," but having only ",PyArray_DIM(pool, 1));
	}
	if (d>=PyArray_DIM(pool, 2)) {
		OOPS("const rectangle dep access out of range, asking ",d," but having only ",PyArray_DIM(pool, 2));
	}
	return *(const bool*)PyArray_GETPTR3(pool, r, c, d);
}

const long long& ndarray::int64_at(const int& r, const int& c, const int& d) const
{
	ASSERT_ARRAY_INT64;
	if (PyArray_NDIM(pool)<3) {
		if (d==0) {
			return int64_at(r,c);
		}
		OOPS("3 dim location requested in ndarray with ",PyArray_NDIM(pool)," dim ");
	}
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	if (c>=PyArray_DIM(pool, 1)) {
		OOPS("const rectangle col access out of range, asking ",c," but having only ",PyArray_DIM(pool, 1));
	}
	if (d>=PyArray_DIM(pool, 2)) {
		OOPS("const rectangle dep access out of range, asking ",d," but having only ",PyArray_DIM(pool, 2));
	}
	return *(const long long*)PyArray_GETPTR3(pool, r, c, d);
}


double& ndarray::operator()(const int& r) 
{
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	return *( double*)PyArray_GETPTR1(pool, r);
}

double& ndarray::operator()(const int& r, const int& c) 
{
	if (PyArray_NDIM(pool)<2) {
		if (c==0) {
			return operator()(r);
		}
		OOPS("2 dim location requested in ndarray with ",PyArray_NDIM(pool)," dim ");
	}
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	if (c>=PyArray_DIM(pool, 1)) {
		OOPS("const rectangle col access out of range, asking ",c," but having only ",PyArray_DIM(pool, 1));
	} 
	return *( double*)PyArray_GETPTR2(pool, r, c);
}

double& ndarray::operator()(const int& r, const int& c, const int& d) 
{
	if (PyArray_NDIM(pool)<3) {
		if (d==0) return operator()(r,c);
		OOPS("3 dim location requested in ndarray with ",PyArray_NDIM(pool)," dim ");
	}
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	if (c>=PyArray_DIM(pool, 1)) {
		OOPS("const rectangle col access out of range, asking ",c," but having only ",PyArray_DIM(pool, 1));
	} 
	if (d>=PyArray_DIM(pool, 2)) {
		OOPS("const rectangle dep access out of range, asking ",d," but having only ",PyArray_DIM(pool, 2));
	}
	return *( double*)PyArray_GETPTR3(pool, r, c, d);
}

void* ndarray::voidptr(const int& r)
{
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	return PyArray_GETPTR1(pool, r);
}

void* ndarray::voidptr(const int& r, const int& c)
{
	if (PyArray_NDIM(pool)<2) {
		if (c==0) {
			return voidptr(r);
		}
	}
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	if (c>=PyArray_DIM(pool, 1)) {
		OOPS("const rectangle col access out of range, asking ",c," but having only ",PyArray_DIM(pool, 1));
	} 
	return PyArray_GETPTR2(pool, r, c);
}

void* ndarray::voidptr(const int& r, const int& c, const int& d)
{
	if (PyArray_NDIM(pool)<3) {
		if (d==0) return voidptr(r,c);
	}
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	if (c>=PyArray_DIM(pool, 1)) {
		OOPS("const rectangle col access out of range, asking ",c," but having only ",PyArray_DIM(pool, 1));
	} 
	if (d>=PyArray_DIM(pool, 2)) {
		OOPS("const rectangle dep access out of range, asking ",d," but having only ",PyArray_DIM(pool, 2));
	}
	return PyArray_GETPTR3(pool, r, c, d);
}



bool& ndarray::bool_at(const int& r)
{
	ASSERT_ARRAY_BOOL;
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	return *( bool*)PyArray_GETPTR1(pool, r);
}

long long& ndarray::int64_at(const int& r)
{
	ASSERT_ARRAY_INT64;
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	return *( long long*)PyArray_GETPTR1(pool, r);
}

bool& ndarray::bool_at(const int& r, const int& c)
{
	ASSERT_ARRAY_BOOL;
	if (PyArray_NDIM(pool)<2) {
		if (c==0) {
			return bool_at(r);
		}
		OOPS("2 dim location requested in ndarray with ",PyArray_NDIM(pool)," dim ");
	}
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	if (c>=PyArray_DIM(pool, 1)) {
		OOPS("const rectangle col access out of range, asking ",c," but having only ",PyArray_DIM(pool, 1));
	}
	return *( bool*)PyArray_GETPTR2(pool, r, c);
}

long long& ndarray::int64_at(const int& r, const int& c)
{
	ASSERT_ARRAY_INT64;
	if (PyArray_NDIM(pool)<2) {
		if (c==0) {
			return int64_at(r);
		}
		OOPS("2 dim location requested in ndarray with ",PyArray_NDIM(pool)," dim ");
	}
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	if (c>=PyArray_DIM(pool, 1)) {
		OOPS("const rectangle col access out of range, asking ",c," but having only ",PyArray_DIM(pool, 1));
	}
	return *( long long*)PyArray_GETPTR2(pool, r, c);
}

bool& ndarray::bool_at(const int& r, const int& c, const int& d)
{
	ASSERT_ARRAY_BOOL;
	if (PyArray_NDIM(pool)<3) {
		if (d==0) return bool_at(r,c);
		OOPS("3 dim location requested in ndarray with ",PyArray_NDIM(pool)," dim ");
	}
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	if (c>=PyArray_DIM(pool, 1)) {
		OOPS("const rectangle col access out of range, asking ",c," but having only ",PyArray_DIM(pool, 1));
	}
	if (d>=PyArray_DIM(pool, 2)) {
		OOPS("const rectangle dep access out of range, asking ",d," but having only ",PyArray_DIM(pool, 2));
	}
	return *( bool*)PyArray_GETPTR3(pool, r, c, d);
}

long long& ndarray::int64_at(const int& r, const int& c, const int& d)
{
	ASSERT_ARRAY_INT64;
	if (PyArray_NDIM(pool)<3) {
		if (d==0) return int64_at(r,c);
		OOPS("3 dim location requested in ndarray with ",PyArray_NDIM(pool)," dim ");
	}
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	if (c>=PyArray_DIM(pool, 1)) {
		OOPS("const rectangle col access out of range, asking ",c," but having only ",PyArray_DIM(pool, 1));
	}
	if (d>=PyArray_DIM(pool, 2)) {
		OOPS("const rectangle dep access out of range, asking ",d," but having only ",PyArray_DIM(pool, 2));
	}
	return *( long long*)PyArray_GETPTR3(pool, r, c, d);
}



void ndarray::operator= (const ndarray& that)
{
	if (!that.pool) OOPS("Error copying ndarray, source is null");
	if (!pool || !PyArray_SAMESHAPE(pool, that.pool) || PyArray_DTYPE(pool)!=PyArray_DTYPE(that.pool) ) {
		Py_CLEAR(pool);
		pool = (PyArrayObject*)PyArray_NewCopy((PyArrayObject*)that.pool, NPY_CORDER);
		Py_INCREF(pool);
	} else {
		int z = PyArray_CopyInto((PyArrayObject*)pool, (PyArrayObject*)that.pool);
		if (z) OOPS("Error copying ndarray");
	}
}
void ndarray::operator= (const symmetric_matrix& that)
{
	if (!that.pool) OOPS("Error copying ndarray, source is null");
	if (!pool || !PyArray_SAMESHAPE(pool, that.pool)) {
		Py_CLEAR(pool);
		pool = (PyArrayObject*)PyArray_NewCopy((PyArrayObject*)that.pool, NPY_CORDER);
		Py_INCREF(pool);
	} else {
		int z = PyArray_CopyInto((PyArrayObject*)pool, (PyArrayObject*)that.pool);
		if (z) OOPS("Error copying ndarray");
	}
	// Copy upper triangle to lower triangle
	for (size_t i=0; i<size1(); i++) {
		for (size_t j=i+1; j<size1(); j++) {
			*(double*)PyArray_GETPTR2(pool, j, i) = *(double*)PyArray_GETPTR2(pool, i, j);
		}
	}

}

//ndarray::ndarray(const etk::memarray_symmetric& that)
//: pool(nullptr)
//{
//	quick_new(NPY_DOUBLE, "SymmetricArray", that.size1(), that.size1());
//	for (size_t i=0; i<size1(); i++) {
//		for (size_t j=0; j<size1(); j++) {
//			*(double*)PyArray_GETPTR2(pool, j, i) = that(i,j);
//		}
//	}
//}

//symmetric_matrix::symmetric_matrix(const etk::memarray_symmetric& that)
//: ndarray(that)
//{
//
//}

//void ndarray::operator= (const memarray_symmetric& that)
//{
//	Py_CLEAR(pool);
//	quick_new(NPY_DOUBLE, "SymmetricArray", that.size1(), that.size1());
//	for (size_t i=0; i<size1(); i++) {
//		for (size_t j=0; j<size1(); j++) {
//			*(double*)PyArray_GETPTR2(pool, j, i) = that(i,j);
//		}
//	}
//}

bool ndarray::operator==(const ndarray& that) const
{
	if (!pool || !that.pool) return false;
	if (!PyArray_SAMESHAPE(pool, that.pool)) return false;
	return !(memcmp(ptr(), that.ptr(), size()*PyArray_DESCR(pool)->elsize));
}

void ndarray::prob_scale_2 (ndarray* out) {
	ASSERT_ARRAY_DOUBLE;
	if (out && out!=this) {
		if ( !out->pool || !PyArray_SAMESHAPE(pool, out->pool) ) {
			Py_CLEAR(out->pool);
			out->pool = (PyArrayObject*)PyArray_NewCopy((PyArrayObject*)pool, NPY_CORDER);
			Py_INCREF(out->pool);
		}
	} else out = this;
	unsigned x1, x2, x3; double temp;
	if (PyArray_NDIM(pool)==3) {
		for ( x1=0; x1<ROWS; x1++ ) {
			for ( x3=0; x3<DEPS; x3++ ) {
				temp = 0;
				for ( x2=0; x2<COLS; x2++ ) { temp += this->operator()(x1,x2,x3); }
				if (!temp) break;
				for ( x2=0; x2<COLS; x2++ ) { out->operator()(x1,x2,x3) /= temp; }
			}
		}
	} else if (PyArray_NDIM(pool)==2) {
		for ( x1=0; x1<ROWS; x1++ ) {
			{
				temp = 0;
				for ( x2=0; x2<COLS; x2++ ) { temp += this->operator()(x1,x2); }
				if (!temp) break;
				for ( x2=0; x2<COLS; x2++ ) { out->operator()(x1,x2) /= temp; }
			}
		}
	}
}

void ndarray::sector_prob_scale_2 (const std::vector<unsigned>& sectors, ndarray* out) {
	ASSERT_ARRAY_DOUBLE;
	if (out && out!=this) {
		if ( !out->pool || !PyArray_SAMESHAPE(pool, out->pool) ) {
			Py_CLEAR(out->pool);
			out->pool = (PyArrayObject*)PyArray_NewCopy((PyArrayObject*)pool, NPY_CORDER);
			Py_INCREF(out->pool);
		}
	} else out = this;
	unsigned x1, x2, x3, i, sectorbegin, sectorend; double temp;
	if (PyArray_NDIM(pool)==3) {
		for ( x1=0; x1<ROWS; x1++ ) {
			for ( x3=0; x3<DEPS; x3++ ) {
				for (i=0; i<sectors.size()-1; i++) {
					sectorbegin = sectors[i];
					sectorend   = sectors[i+1];
					temp = 0;
					for ( x2=sectorbegin; x2<sectorend; x2++ ) { temp += this->operator()(x1,x2,x3); }
					if (!temp) break;
					for ( x2=sectorbegin; x2<sectorend; x2++ ) { out->operator()(x1,x2,x3) /= temp; }
				}
			}
		}
	} else if (PyArray_NDIM(pool)==2) {
		for ( x1=0; x1<ROWS; x1++ ) {
			for (i=0; i<sectors.size()-1; i++) {
				sectorbegin = sectors[i];
				sectorend   = sectors[i+1];
				temp = 0;
				for ( x2=sectorbegin; x2<sectorend; x2++ ) { temp += this->operator()(x1,x2); }
				if (!temp) break;
				for ( x2=sectorbegin; x2<sectorend; x2++ ) { out->operator()(x1,x2) /= temp; }
			}
		}
	}
}


void ndarray::sector_prob_scale_2 (const std::vector<unsigned>& sectors, const unsigned& rowbegin, const unsigned& rowend) {
	ASSERT_ARRAY_DOUBLE;
	ndarray* out = this;
	unsigned x1, x2, x3, i, sectorbegin, sectorend; double temp;
	if (PyArray_NDIM(pool)==3) {
		for ( x1=rowbegin; x1<rowend; x1++ ) {
			for ( x3=0; x3<DEPS; x3++ ) {
				for (i=0; i<sectors.size()-1; i++) {
					sectorbegin = sectors[i];
					sectorend   = sectors[i+1];
					temp = 0;
					for ( x2=sectorbegin; x2<sectorend; x2++ ) { temp += this->operator()(x1,x2,x3); }
					if (!temp) break;
					for ( x2=sectorbegin; x2<sectorend; x2++ ) { out->operator()(x1,x2,x3) /= temp; }
				}
			}
		}
	} else if (PyArray_NDIM(pool)==2) {
		for ( x1=rowbegin; x1<rowend; x1++ ) {
			for (i=0; i<sectors.size()-1; i++) {
				sectorbegin = sectors[i];
				sectorend   = sectors[i+1];
				temp = 0;
				for ( x2=sectorbegin; x2<sectorend; x2++ ) { temp += this->operator()(x1,x2); }
				if (!temp) break;
				for ( x2=sectorbegin; x2<sectorend; x2++ ) { out->operator()(x1,x2) /= temp; }
			}
		}
	}
}



void ndarray::logsums_2 (ndarray* out) {
	ASSERT_ARRAY_DOUBLE;
	if (!out || out==this) {
		OOPS("cannot calculate logsums in place");
	}
	if ( !out->pool || PyArray_NDIM(out->pool)!=1 || PyArray_DIM(out->pool, 0)!=ROWS ) {
		Py_CLEAR(out->pool);
		npy_intp dims [1] = {ROWS};

		out->pool = (PyArrayObject*)PyArray_New((PyTypeObject*)get_array_type("Array"), 1, &dims[0], NPY_DOUBLE, nullptr, nullptr, 0, 0, nullptr);
		Py_INCREF(out->pool);
	}
	unsigned x1, x2; double temp;
	if (PyArray_NDIM(pool)!=2) {
		OOPS("can only calculate logsums on a 2d array");
	} else {
		for ( x1=0; x1<ROWS; x1++ ) {
			{
				temp = 0;
				for ( x2=0; x2<COLS; x2++ ) { temp += this->operator()(x1,x2); }
				out->operator()(x1) = ::log(temp);
			}
		}
	}
}


void ndarray::exp () {
	ASSERT_ARRAY_DOUBLE;
	double* i;
	double* z=static_cast<double*>( PyArray_DATA(pool) );
	for (i=z; i!=z+PyArray_SIZE(pool); i++) {
		*i = ::exp(*i);
	}
}


void ndarray::is_exponential_of (const ndarray& that) {
	ASSERT_ARRAY_DOUBLE;
	if (!PyArray_SAMESHAPE(pool, that.pool)) OOPS("is_exponential_of different sized ndarray");
	double* i;
	double* z=static_cast<double*>( PyArray_DATA(pool) );
	double* j=static_cast<double*>( PyArray_DATA(that.pool) );
	for (i=z; z+PyArray_SIZE(pool); i++,j++) {
		*i = ::exp(*j);
	}
}

void ndarray::log () {
	ASSERT_ARRAY_DOUBLE;
	double* i;
	double* z=static_cast<double*>( PyArray_DATA(pool) );
	for (i=z; i!=z+PyArray_SIZE(pool); i++) {
		*i = ::log(*i);
	}
}

void ndarray::neg () {
	ASSERT_ARRAY_DOUBLE;
	double* i;
	double* z=static_cast<double*>( PyArray_DATA(pool) );
	for (i=z; i!=z+PyArray_SIZE(pool); i++) {
		*i = -(*i);
	}
}

double ndarray::sum () const {
	ASSERT_ARRAY_DOUBLE;
	double s (0.0);
	double* i;
	double* z=static_cast<double*>( PyArray_DATA(pool) );
	for (i=z; i!=z+PyArray_SIZE(pool); i++) {
		s += *i;
	}
	return s;
}


double ndarray::operator*(const ndarray& that) const // dot product
{
	ASSERT_ARRAY_DOUBLE;
	if (size() != that.size()) {
		OOPS("puddle dot-product of different sized puddles, ",size()," vs ",that.size());
	}
	return cblas_ddot(size(), ptr(),1, that.ptr(),1);
}

void ndarray::operator+=(const ndarray& that)
{
	ASSERT_ARRAY_DOUBLE;
	if (size() != that.size()) {
		OOPS("puddle addition of different sized puddles, this is ",size()," while that is ",that.size());
	}
	cblas_daxpy(size(), 1, that.ptr(), 1, ptr(), 1);
}
void ndarray::operator-=(const ndarray& that)
{
	ASSERT_ARRAY_DOUBLE;
	if (size() != that.size()) OOPS("puddle subtraction of different sized puddles");
	cblas_daxpy(size(), -1, that.ptr(), 1, ptr(), 1);
}
void ndarray::operator+=(const memarray_raw& that)
{
	ASSERT_ARRAY_DOUBLE;
	if (size() != that.size()) OOPS("puddle addition of different sized puddles");
	cblas_daxpy(size(), 1, that.ptr(), 1, ptr(), 1);
}
void ndarray::operator-=(const memarray_raw& that)
{
	ASSERT_ARRAY_DOUBLE;
	if (size() != that.size()) OOPS("puddle subtraction of different sized puddles");
	cblas_daxpy(size(), -1, that.ptr(), 1, ptr(), 1);
}

void ndarray::projection(const ndarray& fixture, const ndarray& beam, const double& distance)
{
	ASSERT_ARRAY_DOUBLE;
	if (size() != fixture.size()) OOPS("puddle projection fixture wrong size");
	if (size() != beam.size()) OOPS("puddle beam fixture wrong size");
	operator=(fixture); //memcpy(pool.ptr(), fixture.ptr(), size()*sizeof(double));
	cblas_daxpy(size(), distance, beam.ptr(), 1, ptr(), 1);
}

void ndarray::initialize(const double& init)
{
	if (!init) {
		size_t s = size();
		double* p = ptr();
		memset(ptr(), 0, size()*PyArray_DESCR(pool)->elsize);
	}
	else {
		for (unsigned i=0; i<size(); i++) ptr()[i]=init;
	}
}

void ndarray::bool_initialize(const bool& init)
{
		for (unsigned i=0; i<size(); i++) ptr_bool()[i]=init;
}


void ndarray::scale(const double& scal)
{
	ASSERT_ARRAY_DOUBLE;
	cblas_dscal(size(), scal, ptr(), 1);
}


double ndarray::scale_so_total_is(const double& tot)
{
	double current_total = sum();
	double needed_scale_factor = tot/current_total;
	scale(needed_scale_factor);
	return needed_scale_factor;
}

double ndarray::scale_so_mean_is(const double& mean)
{
	return scale_so_total_is(mean * size());
}

std::vector<double> ndarray::vectorize(unsigned start, unsigned stop) const
{
	ASSERT_ARRAY_DOUBLE;
	std::vector<double> ret;
	if (UINT_MAX == stop) stop = size();
	if (start > stop) return ret;
	while (start < stop) {
		ret.push_back( operator[](start) );
		start++;
	}
	return ret;
}

std::vector<double> ndarray::negative_vectorize(unsigned start, unsigned stop) const
{
	ASSERT_ARRAY_DOUBLE;
	std::vector<double> ret;
	if (UINT_MAX == stop) stop = size();
	if (start > stop) return ret;
	while (start < stop) {
		ret.push_back( - operator[](start) );
		start++;
	}
	return ret;
}





std::string ndarray::printrow(const unsigned& r) const
{
	ASSERT_ARRAY_DOUBLE;
	std::ostringstream ret;
	unsigned x2, x3;
	char depMarker, colMarker, rowMarker;
	if (ROWS==0) {
		return "no rows in array";
	}
	if (DEPS==1) {
		depMarker = ' ';
		colMarker = '\t';
		rowMarker = '\n';
	} else {
		depMarker = '\t';
		colMarker = '\n';
		rowMarker = '\n';
	}
	for ( x2=0; x2<COLS; x2++ ) {
		for ( x3=0; x3<DEPS; x3++ ) {
			ret << operator()(r,x2,x3) << depMarker;
		}
		ret << colMarker;
	}
	ret << rowMarker;
	return ret.str();
}

std::string ndarray::printrow_hex(const unsigned& r) const
{
	ASSERT_ARRAY_DOUBLE;
	std::ostringstream ret;
	ret << std::hexfloat << std::setprecision(15);
	unsigned x2, x3;
	char depMarker, colMarker, rowMarker;
	if (ROWS==0) {
		return "no rows in array";
	}
	if (DEPS==1) {
		depMarker = ' ';
		colMarker = '\t';
		rowMarker = '\n';
	} else {
		depMarker = '\t';
		colMarker = '\n';
		rowMarker = '\n';
	}
	for ( x2=0; x2<COLS; x2++ ) {
		for ( x3=0; x3<DEPS; x3++ ) {
			ret << operator()(r,x2,x3) << depMarker;
		}
		ret << colMarker;
	}
	ret << rowMarker;
	return ret.str();
}

std::string ndarray::printrows(unsigned rstart, const unsigned& rfinish) const
{
	ASSERT_ARRAY_DOUBLE;
	std::ostringstream ret;
	for (; rstart < rfinish; rstart++) {
		ret << printrow(rstart);
	}
	return ret.str();
}

std::string ndarray::printall() const
{
	ASSERT_ARRAY_DOUBLE;
	return printrows(0,ROWS);
}

std::string ndarray::printrows_hex(unsigned rstart, const unsigned& rfinish) const
{
	ASSERT_ARRAY_DOUBLE;
	std::ostringstream ret;
	for (; rstart < rfinish; rstart++) {
		ret << printrow_hex(rstart);
	}
	return ret.str();
}

std::string ndarray::printall_hex() const
{
	ASSERT_ARRAY_DOUBLE;
	return printrows_hex(0,ROWS);
}



void symmetric_matrix::copy_uppertriangle_to_lowertriangle()
{
	ASSERT_ARRAY_DOUBLE;
	for (size_t i=0; i<size1(); i++) {
		for (size_t j=i+1; j<size1(); j++) {
			*(double*)PyArray_GETPTR2(pool, j, i) = *(double*)PyArray_GETPTR2(pool, i, j);
		}
	}
}

void symmetric_matrix::copy_lowertriangle_to_uppertriangle()
{
	ASSERT_ARRAY_DOUBLE;
	for (size_t i=0; i<size1(); i++) {
		for (size_t j=i+1; j<size1(); j++) {
			*(double*)PyArray_GETPTR2(pool, i,j) = *(double*)PyArray_GETPTR2(pool, j,i);
		}
	}
}

void symmetric_matrix::inv(logging_service* msg_)
{
	ASSERT_ARRAY_DOUBLE;
//	BUGGER_(msg_, "inv received matrix =\n" << printSquare() );
	copy_uppertriangle_to_lowertriangle();
//	BUGGER_(msg_, "inv symmetric-ized matrix =\n" << printSquare() );
	PyObject* linalg = elm::elm_linalg_module; // PyImport_ImportModule("larch.linalg");
	Py_XINCREF(linalg);
	if (!linalg) {
		OOPS("Failed to load larch.linalg");
	}
	PyObject* linalg_inv = PyObject_GetAttrString(linalg, "general_inverse");
	if (!linalg_inv) {
		Py_CLEAR(linalg);
		OOPS("Failed to find larch.linalg.general_inverse");
	}
	PyObject* result = PyObject_CallFunctionObjArgs(linalg_inv, pool, NULL);
	Py_CLEAR(linalg_inv);
	Py_CLEAR(linalg);
	if (!result) {
		//OOPS_MATRIXINVERSE(this, "Failed to get inverse");
		OOPS("Failed to get inverse");
	}
	Py_CLEAR(pool);
	pool = (PyArrayObject*)result;
	Py_INCREF(pool);
	Py_CLEAR(result);
}

void symmetric_matrix::inv_bonafide(logging_service* msg_)
{
	ASSERT_ARRAY_DOUBLE;
//	BUGGER_(msg_, "inv received matrix =\n" << printSquare() );
	copy_uppertriangle_to_lowertriangle();
//	BUGGER_(msg_, "inv symmetric-ized matrix =\n" << printSquare() );
	PyObject* linalg = elm::elm_linalg_module; // PyImport_ImportModule("larch.linalg");
	Py_XINCREF(linalg);
	if (!linalg) {
		OOPS("Failed to load larch.linalg");
	}
	PyObject* linalg_inv = PyObject_GetAttrString(linalg, "matrix_inverse");
	if (!linalg_inv) {
		Py_CLEAR(linalg);
		OOPS("Failed to find larch.linalg.matrix_inverse");
	}
	PyObject* result = PyObject_CallFunctionObjArgs(linalg_inv, pool, NULL);
	Py_CLEAR(linalg_inv);
	Py_CLEAR(linalg);
	if (!result) {
		//OOPS_MATRIXINVERSE(this, "Failed to get inverse");
		OOPS("Failed to get inverse");
	}
	Py_CLEAR(pool);
	pool = (PyArrayObject*)result;
	Py_INCREF(pool);
	Py_CLEAR(result);
}

void symmetric_matrix::initialize_identity()
{
	ASSERT_ARRAY_DOUBLE;
	initialize();
	for (size_t i=0; i<size1(); i++) operator()(i,i) = 1;
}


bool symmetric_matrix::all_zero() const
{
	for (size_t i=0; i<size1(); i++) {
		for (size_t j=i; j<size1(); j++) {
			if(this->operator()(i, j)) return false;
		}
	}
	return true;
}



std::string symmetric_matrix::printSquare()
{
	ASSERT_ARRAY_DOUBLE;
    std::ostringstream pr;
	for (unsigned i=0; i<size1(); i++) {
		for (unsigned j=0; j<size1(); j++) {
			pr.width(12);
			pr << this->operator()(i, j) << "\t";
		}
		pr << "\n";
	}
	return pr.str();
}
void symmetric_matrix::operator= (memarray_raw& that)
{
	ASSERT_ARRAY_DOUBLE;
	if (size1()==that.size1() && size2()==that.size2() && that.size3()==1) {
		memcpy(ptr(), *that, sizeof(double)*that.size1()*that.size2());
	} else {
		resize(that.size1(), that.size2());
		memcpy(ptr(), *that, sizeof(double)*that.size1()*that.size2());
	}

	copy_uppertriangle_to_lowertriangle();
}

void symmetric_matrix::operator= (ndarray& that)
{
	ASSERT_ARRAY_DOUBLE;
	PyObject* that_pool = that.get_object();
	if (!that_pool) {
		Py_CLEAR(that_pool);
		OOPS("Error copying ndarray, source is null");
	}
	if (!pool || !PyArray_SAMESHAPE(pool, (PyArrayObject*)that_pool)) {
		Py_CLEAR(pool);
		pool = (PyArrayObject*)PyArray_NewCopy((PyArrayObject*)that_pool, NPY_CORDER);
		Py_INCREF(pool);
		return;
	}
	int z = PyArray_CopyInto((PyArrayObject*)pool, (PyArrayObject*)that_pool);
	Py_CLEAR(that_pool);
	if (z) {
		OOPS("Error copying ndarray");
	}
	copy_uppertriangle_to_lowertriangle();
}
void symmetric_matrix::operator= (const symmetric_matrix& that)
{
	ASSERT_ARRAY_DOUBLE;
	if (!that.pool) OOPS("Error copying ndarray, source is null");
	if (!pool || !PyArray_SAMESHAPE(pool, that.pool)) {
		Py_CLEAR(pool);
		pool = (PyArrayObject*)PyArray_NewCopy((PyArrayObject*)that.pool, NPY_CORDER);
		Py_INCREF(pool);
		return;
	}
	int z = PyArray_CopyInto((PyArrayObject*)pool, (PyArrayObject*)that.pool);
	if (z) OOPS("Error copying ndarray");
}


void symmetric_matrix::resize(const int& r)
{
	quick_new(NPY_DOUBLE, "SymmetricArray", r,r);
}

void symmetric_matrix::resize(const int& r,const int& c)
{
	quick_new(NPY_DOUBLE, "SymmetricArray", r,c);
}

void symmetric_matrix::resize(const int& r,const int& c,const int& s)
{
	quick_new(NPY_DOUBLE, "SymmetricArray", r,c,s);
}

void symmetric_matrix::resize(etk::three_dim& prototype)
{
	quick_new(NPY_DOUBLE, "SymmetricArray", prototype.size1(),prototype.size2(),prototype.size3());
}



