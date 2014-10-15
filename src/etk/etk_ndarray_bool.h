/*
 *  etk_ndarray.h
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


#ifndef __TOOLBOX_NDARRAY_BOOL__
#define __TOOLBOX_NDARRAY_BOOL__

#include "etk_python.h"
#include <climits>
#include <numpy/arrayobject.h> 

namespace etk {

	class ndarray_bool {
	
		friend class ndarray;
	
	public:
		PyArrayObject* pool;
		
	public:
		int flag;
		
	public:
		inline const bool* operator* () const { return static_cast<const bool*>( PyArray_DATA(pool) ); }
		inline bool* operator* () { return static_cast<bool*>( PyArray_DATA(pool) ); }

	// Constructor & Destructor
		ndarray_bool();
		ndarray_bool(const int& r);
		ndarray_bool(const int& r,const int& c);
		ndarray_bool(const int& r,const int& c,const int& s);
		ndarray_bool(PyObject* obj);
		~ndarray_bool();
		void initialize(const bool& init=0);
		void destroy() {Py_CLEAR(pool);}
		void make_symmetric() { flag |= ARRAY_SYMMETRIC; }
		
	// Direct Access
		PyObject*  get_object() {Py_XINCREF(pool); return (PyObject*)pool;}

	// Unfiltered Access
		bool& operator[](const int& i);
		const bool& operator[](const int& i) const;

	// Location Access
		const bool& operator()(const int& i) const;
		const bool& operator()(const int& i, const int& j) const;
		const bool& operator()(const int& i, const int& j, const int& k) const;
		bool& operator()(const int& i) ;
		bool& operator()(const int& i, const int& j) ;
		bool& operator()(const int& i, const int& j, const int& k) ;
		
	// Pointer Location Access
		const bool* ptr() const { return static_cast<bool*>( PyArray_DATA(pool) ); }
		const bool* ptr(const int& i) const {return &(operator()(i));}
		const bool* ptr(const int& i, const int& j) const {return &(operator()(i,j));}
		const bool* ptr(const int& i, const int& j, const int& k) const {return &(operator()(i,j,k));}
		bool* ptr() { return static_cast<bool*>( PyArray_DATA(pool) ); }
		bool* ptr(const int& i)  {return &(operator()(i));}
		bool* ptr(const int& i, const int& j)  {return &(operator()(i,j));}
		bool* ptr(const int& i, const int& j, const int& k)  {return &(operator()(i,j,k));}
		
	// Resizing
		void resize(const int& r);
		void resize(const int& r,const int& c);
		void resize(const int& r,const int& c,const int& s);
	
	// Copying
		void operator= (const ndarray_bool& that);
	
	// Math
		void operator|=(const ndarray_bool& that);
		void operator&=(const ndarray_bool& that);

	// Attributes
		size_t size() const { if (pool) return PyArray_SIZE(pool); else return 0; }
		size_t size1() const { if (pool) return PyArray_DIM(pool,0); else return 0; }
		size_t size2() const { if (pool) {if (PyArray_NDIM(pool)>1) return PyArray_DIM(pool,1); else return 1;} else return 0; }
		size_t size3() const { if (pool) {if (PyArray_NDIM(pool)>2) return PyArray_DIM(pool,2); else return 1;} else return 0;}
		bool operator==(const ndarray_bool& that) const;
		
	// Printing
		std::string printrow(const unsigned& r) const;
		std::string printrows(unsigned rstart, const unsigned& rfinish) const;
		std::string printall() const;

	};









} // end namespace etk

#endif // __TOOLBOX_NDARRAY_BOOL__
