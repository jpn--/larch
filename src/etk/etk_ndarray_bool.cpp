/*
 *  etk_ndarray.cpp
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
#include <numpy/arrayobject.h> 

using namespace etk;

#include <iostream>
#include <cmath>
#include <climits>
#include <cstring>


#define ROWS PyArray_DIM(pool, 0)
#define COLS (PyArray_NDIM(pool)>1 ? PyArray_DIM(pool, 1) : 1)
#define DEPS (PyArray_NDIM(pool)>2 ? PyArray_DIM(pool, 2) : 1)




ndarray_bool::ndarray_bool(PyObject* obj)
: pool ((PyArrayObject*)obj)
, flag (ARRAY_INIT_ON_RESIZE)
{
	Py_INCREF(pool);
	if (!PyArray_ISCARRAY(pool)) {
		Py_CLEAR(pool);
		OOPS("Error creating array");
	}
}




ndarray_bool::ndarray_bool()
: pool (nullptr)
, flag (0)
{
}


ndarray_bool::ndarray_bool(const int& r)
: pool (nullptr)
, flag (0)
{
	npy_intp dims [1] = {r};
	PyObject* subtype = get_array_type("Array");
	pool = (PyArrayObject*)PyArray_New((PyTypeObject*)subtype, 1, &dims[0], NPY_BOOL, nullptr, nullptr, 0, 0, nullptr);
	Py_CLEAR(subtype);
	Py_INCREF(pool);
	if (!PyArray_ISCARRAY(pool)) {
		Py_CLEAR(pool);
		OOPS("Error creating array");
	}
	PyArray_FILLWBYTE(pool, 0);
}

ndarray_bool::ndarray_bool(const int& r,const int& c)
: pool (nullptr)
, flag (0)
{
	npy_intp dims [2] = {r,c};
	PyObject* subtype = get_array_type("Array");
	//pool = (PyArrayObject*)PyArray_ZEROS(2, &dims[0], NPY_DOUBLE, 0);
	pool = (PyArrayObject*)PyArray_New((PyTypeObject*)subtype, 2, &dims[0], NPY_BOOL, nullptr, nullptr, 0, 0, nullptr);
	Py_CLEAR(subtype);
	Py_INCREF(pool);
	if (!PyArray_ISCARRAY(pool)) {
		std::cout << "<repr>: ";
		PyObject_Print((PyObject*)pool,stdout,0);
		std::cout << "\n<str>: ";
		PyObject_Print((PyObject*)pool,stdout,0);
		std::cout << "\n";
		std::cout << "dec flags: "<< PyArray_FLAGS(pool) << "\n";
		std::cout << "hex flags: 0x"<< std::hex << PyArray_FLAGS(pool) << "\n";
		std::cout << "desired flags: 0x"<< std::hex << NPY_ARRAY_CARRAY << "\n";
		Py_CLEAR(pool);
		OOPS("Error creating array");
	}
	PyArray_FILLWBYTE(pool, 0);
}

ndarray_bool::ndarray_bool(const int& r,const int& c,const int& s)
: pool (nullptr)
, flag (0)
{
	npy_intp dims [3] = {r,c,s};
	PyObject* subtype = get_array_type("Array");
	//pool = (PyArrayObject*)PyArray_ZEROS(3, &dims[0], NPY_DOUBLE, 0);
	pool = (PyArrayObject*)PyArray_New((PyTypeObject*)subtype, 3, &dims[0], NPY_BOOL, nullptr, nullptr, 0, 0, nullptr);
	Py_CLEAR(subtype);
	Py_INCREF(pool);
	if (!PyArray_ISCARRAY(pool)) {
		std::cout << "<repr>: ";
		PyObject_Print((PyObject*)pool,stdout,0);
		std::cout << "\n<str>: ";
		PyObject_Print((PyObject*)pool,stdout,0);
		std::cout << "\n";
		std::cout << "dec flags: "<< PyArray_FLAGS(pool) << "\n";
		std::cout << "hex flags: 0x"<< std::hex << PyArray_FLAGS(pool) << "\n";
		std::cout << "desired flags: 0x"<< std::hex << NPY_ARRAY_CARRAY << "\n";
		Py_CLEAR(pool);
		OOPS("Error creating array");
	}
	PyArray_FILLWBYTE(pool, 0);
}

ndarray_bool::~ndarray_bool()
{
	Py_CLEAR(pool);
}


void ndarray_bool::resize(const int& r)
{
	Py_CLEAR(pool);
	if (flag & ARRAY_SYMMETRIC) { resize(r,r); } else {
		npy_intp dims [1] = {r};
		pool = (PyArrayObject*)PyArray_SimpleNew(1, &dims[0], NPY_BOOL);
		Py_INCREF(pool);
	}
	if (flag & ARRAY_INIT_ON_RESIZE) PyArray_FILLWBYTE(pool, 0);
}

void ndarray_bool::resize(const int& r,const int& c)
{
	Py_CLEAR(pool);
	if (flag & ARRAY_SYMMETRIC && r!=c) OOPS("must be square to be symmetric");
	npy_intp dims [2] = {r,c};
	pool = (PyArrayObject*)PyArray_SimpleNew(2, &dims[0], NPY_BOOL);
	Py_INCREF(pool);
	if (flag & ARRAY_INIT_ON_RESIZE) PyArray_FILLWBYTE(pool, 0);
}

void ndarray_bool::resize(const int& r,const int& c,const int& s)
{
	Py_CLEAR(pool);
	npy_intp dims [3] = {r,c,s};
	pool = (PyArrayObject*)PyArray_SimpleNew(3, &dims[0], NPY_BOOL);
	Py_INCREF(pool);
	if (flag & ARRAY_INIT_ON_RESIZE) PyArray_FILLWBYTE(pool, 0);
}



bool& ndarray_bool::operator[](const int& i)
{
	if (i<PyArray_SIZE(pool)) return (static_cast<bool*>( PyArray_DATA(pool) ))[i];
	OOPS(cat("puddle access out of range: ",i," in ",PyArray_SIZE(pool)));
}

const bool& ndarray_bool::operator[](const int& i) const
{
	if (i<PyArray_SIZE(pool)) return (static_cast<bool*>( PyArray_DATA(pool) ))[i];
	OOPS(cat("const puddle access out of range: ",i," in ",PyArray_SIZE(pool)));
}

const bool& ndarray_bool::operator()(const int& r) const
{
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	return *(const bool*)PyArray_GETPTR1(pool, r);
}

const bool& ndarray_bool::operator()(const int& r, const int& c) const
{
	if (PyArray_NDIM(pool)<2) {
		if (c==0) {
			return operator()(r);
		}
		OOPS("2 dim location requested in ndarray_bool with ",PyArray_NDIM(pool)," dim ");
	}
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	if (c>=PyArray_DIM(pool, 1)) {
		OOPS("const rectangle col access out of range, asking ",c," but having only ",PyArray_DIM(pool, 1));
	}
	if (flag & ARRAY_SYMMETRIC && r>c) {
		return *(const bool*)PyArray_GETPTR2(pool, c, r);
	}
	return *(const bool*)PyArray_GETPTR2(pool, r, c);
}

const bool& ndarray_bool::operator()(const int& r, const int& c, const int& d) const
{
	if (PyArray_NDIM(pool)<3) {
		if (d==0) {
			return operator()(r,c);
		}
		OOPS("3 dim location requested in ndarray_bool with ",PyArray_NDIM(pool)," dim ");
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


bool& ndarray_bool::operator()(const int& r) 
{
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	return *( bool*)PyArray_GETPTR1(pool, r);
}

bool& ndarray_bool::operator()(const int& r, const int& c) 
{
	if (PyArray_NDIM(pool)<2) {
		if (c==0) {
			return operator()(r);
		}
		OOPS("2 dim location requested in ndarray_bool with ",PyArray_NDIM(pool)," dim ");
	}
	if (r>=PyArray_DIM(pool, 0)) {
		OOPS("rectangle row access out of range, asking ",r," but having only ",PyArray_DIM(pool, 0));
	}
	if (c>=PyArray_DIM(pool, 1)) {
		OOPS("rectangle col access out of range, asking ",c," but having only ",PyArray_DIM(pool, 1));
	} 
	if (flag & ARRAY_SYMMETRIC && r>c) {
		return *( bool*)PyArray_GETPTR2(pool, c, r);
	}
	return *( bool*)PyArray_GETPTR2(pool, r, c);
}

bool& ndarray_bool::operator()(const int& r, const int& c, const int& d) 
{
	if (PyArray_NDIM(pool)<3) {
		if (d==0) return operator()(r,c);
		OOPS("3 dim location requested in ndarray_bool with ",PyArray_NDIM(pool)," dim ");
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

void ndarray_bool::operator= (const ndarray_bool& that)
{
	if (!that.pool) OOPS("Error copying ndarray_bool, source is null");
	if (!pool || !PyArray_SAMESHAPE(pool, that.pool)) {
		Py_CLEAR(pool);
		pool = (PyArrayObject*)PyArray_NewCopy((PyArrayObject*)that.pool, NPY_CORDER);
		Py_INCREF(pool);
	} else {
		int z = PyArray_CopyInto((PyArrayObject*)pool, (PyArrayObject*)that.pool);
		if (z) OOPS("Error copying ndarray_bool");
	}
	// Copy upper triangle to lower triangle if needed
	if (that.flag & ARRAY_SYMMETRIC) {
		for (size_t i=0; i<size1(); i++) {
			for (size_t j=i+1; j<size1(); j++) {
				*(bool*)PyArray_GETPTR2(pool, j, i) = *(bool*)PyArray_GETPTR2(pool, i, j);
			}
		}
	}
}

bool ndarray_bool::operator==(const ndarray_bool& that) const
{
	if (!pool || !that.pool) return false;
	if (!PyArray_SAMESHAPE(pool, that.pool)) return false;
	return !(memcmp(ptr(), that.ptr(), size()*sizeof(bool)));
}


void ndarray_bool::operator|=(const ndarray_bool& that)
{
	if (size() != that.size()) OOPS("ndarray_bool 'or' of different sized ndarray_bool");
	for (size_t i=0; i<size(); i++) {
		(static_cast<bool*>( PyArray_DATA(pool) ))[i] |= (static_cast<bool*>( PyArray_DATA(that.pool) ))[i];
	}
}
void ndarray_bool::operator&=(const ndarray_bool& that)
{
	if (size() != that.size()) OOPS("ndarray_bool 'and' of different sized ndarray_bool"); 
	for (size_t i=0; i<size(); i++) {
		(static_cast<bool*>( PyArray_DATA(pool) ))[i] &= (static_cast<bool*>( PyArray_DATA(that.pool) ))[i];
	}
}


void ndarray_bool::initialize(const bool& init)
{
	if (!init) memset(ptr(), 0, size()*sizeof(bool));
	else {
		for (unsigned i=0; i<size(); i++) ptr()[i]=init;
	}
}






std::string ndarray_bool::printrow(const unsigned& r) const
{
	std::ostringstream ret;
	unsigned x2, x3;
	char depMarker, colMarker, rowMarker;
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

std::string ndarray_bool::printrows(unsigned rstart, const unsigned& rfinish) const
{
	std::ostringstream ret;
	for (; rstart < rfinish; rstart++) {
		ret << printrow(rstart);
	}
	return ret.str();
}

std::string ndarray_bool::printall() const
{
	return printrows(0,ROWS);
}





