/*
 *  etk_ndarray.h
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


#ifndef __TOOLBOX_NDARRAY__
#define __TOOLBOX_NDARRAY__

#ifndef SWIG

#include "etk_python.h"
#include "etk_logger.h"
#include <climits>
#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#include <numpy/arrayobject.h> 
#include <vector>

#define ARRAY_SYMMETRIC      0x1
#define ARRAY_INIT_ON_RESIZE 0x2

namespace etk {

	class ndarray;
	class ndarray_bool;
	class symmetric_matrix;
	class memarray_raw;
//	class memarray_symmetric;

	PyObject* get_array_type(const char* type);

	class three_dim {
	protected:
		unsigned rows;
		unsigned cols;
		unsigned deps;
	public:
		virtual const unsigned& size1() const { return rows; }
		const unsigned& size2() const { return cols; }
		const unsigned& size3() const { return deps; }
		three_dim(const unsigned& r, const unsigned& c, const unsigned& d)
		: rows(r), cols(c), deps(d){ }

	};

	class ndarray {
	
		static boosted::mutex python_mutex;
	
	public:
		PyArrayObject* pool;
		

	public:
		inline const double* operator* () const { return static_cast<const double*>( PyArray_DATA(pool) ); }
		inline double* operator* () { return static_cast<double*>( PyArray_DATA(pool) ); }

	// Constructor & Destructor
	protected:
		void quick_new(const int& datatype, const char* arrayClass, const int& r,const int& c=-1,const int& s=-1);
	public:
		ndarray(const char* arrayType, const int& datatype, const int& r, const int& c=-1, const int& s=-1);

		ndarray(const char* arrayType="Array");
		ndarray(const int& r, const char* arrayType="Array");
		ndarray(const int& r,const int& c, const char* arrayType="Array");
		ndarray(const int& r,const int& c,const int& s, const char* arrayType="Array");
		ndarray(PyObject* obj);
		ndarray(const ndarray& that) = delete;
		ndarray(ndarray& that, bool use_same_memory);
		ndarray(const ndarray& that, bool use_same_memory);

		~ndarray();
		void initialize(const double& init=0);
		void bool_initialize(const bool& init=false);
		void destroy() {Py_CLEAR(pool);}
		void same_memory_as(ndarray&);
		void same_ccontig_memory_as(ndarray& x);
		
	// Direct Access
		PyObject*  get_object(bool incref=true) {if (incref) Py_XINCREF(pool); return (PyObject*)pool;}

	// Unfiltered Access
		double& operator[](const int& i);
		const double& operator[](const int& i) const;

	// Location Access
		const double& operator()(const int& i) const;
		const double& operator()(const int& i, const int& j) const;
		const double& operator()(const int& i, const int& j, const int& k) const;
		double& operator()(const int& i) ;
		double& operator()(const int& i, const int& j) ;
		double& operator()(const int& i, const int& j, const int& k) ;
		void* voidptr(const int& i) ;
		void* voidptr(const int& i, const int& j) ;
		void* voidptr(const int& i, const int& j, const int& k) ;
		inline const double& at(const int& i) const { return operator()(i); }
		inline const double& at(const int& i, const int& j) const { return operator()(i,j); }
		inline const double& at(const int& i, const int& j, const int& k) const { return operator()(i,j,k); }
		inline double& at(const int& i)  { return operator()(i); }
		inline double& at(const int& i, const int& j)  { return operator()(i,j); }
		inline double& at(const int& i, const int& j, const int& k)  { return operator()(i,j,k); }

	// Location Access (long long)
		const long long& int64_at(const int& i) const ;
		const long long& int64_at(const int& i, const int& j) const ;
		const long long& int64_at(const int& i, const int& j, const int& k) const ;
		long long& int64_at(const int& i)  ;
		long long& int64_at(const int& i, const int& j)  ;
		long long& int64_at(const int& i, const int& j, const int& k)  ;

	// Location Access (bool)
		const bool& bool_at(const int& i) const ;
		const bool& bool_at(const int& i, const int& j) const ;
		const bool& bool_at(const int& i, const int& j, const int& k) const ;
		bool& bool_at(const int& i)  ;
		bool& bool_at(const int& i, const int& j)  ;
		bool& bool_at(const int& i, const int& j, const int& k)  ;

		
	// Pointer Location Access
		const double* ptr() const { return pool? static_cast<double*>( PyArray_DATA(pool) ) :nullptr; }
		const double* ptr(const int& i) const {return pool? &(operator()(i)) :nullptr;}
		const double* ptr(const int& i, const int& j) const {return pool? &(operator()(i,j)) :nullptr;}
		const double* ptr(const int& i, const int& j, const int& k) const {return pool? &(operator()(i,j,k)) :nullptr;}
		double* ptr() { return pool? static_cast<double*>( PyArray_DATA(pool) ) :nullptr; }
		double* ptr(const int& i)  {return pool? &(operator()(i)) :nullptr;}
		double* ptr(const int& i, const int& j)  {return pool? &(operator()(i,j)) :nullptr;}
		double* ptr(const int& i, const int& j, const int& k)  {return pool? &(operator()(i,j,k)) :nullptr;}

	// Pointer Location Access (bool)
		inline const bool* ptr_bool() const { return static_cast<bool*>( PyArray_DATA(pool) ); }
		inline const bool* ptr_bool(const int& i) const {return &(bool_at(i));}
		inline const bool* ptr_bool(const int& i, const int& j) const {return &(bool_at(i,j));}
		inline const bool* ptr_bool(const int& i, const int& j, const int& k) const {return &(bool_at(i,j,k));}
		inline bool* ptr_bool() { return static_cast<bool*>( PyArray_DATA(pool) ); }
		inline bool* ptr_bool(const int& i)  {return &(bool_at(i));}
		inline bool* ptr_bool(const int& i, const int& j)  {return &(bool_at(i,j));}
		inline bool* ptr_bool(const int& i, const int& j, const int& k)  {return &(bool_at(i,j,k));}

		
	// Resizing
		void resize(const int& r);
		void resize(const int& r,const int& c);
		void resize(const int& r,const int& c,const int& s);
		void resize_bool(const int& r);
		void resize_bool(const int& r,const int& c);
		void resize_bool(const int& r,const int& c,const int& s);
		void resize(ndarray& prototype);
		void resize(etk::three_dim& prototype);
		void resize_if_needed(etk::three_dim& prototype);
	
	// Copying
		void operator= (const ndarray& that);
		void operator= (const symmetric_matrix& that);
//		void operator= (const memarray_symmetric& that);
		std::vector<double> vectorize(unsigned start=0, unsigned stop=UINT_MAX) const;
		std::vector<double> negative_vectorize(unsigned start=0, unsigned stop=UINT_MAX) const;
	
	// Math
		void exp ();
		void is_exponential_of (const ndarray& that);
		void log ();
		void neg ();
		double sum() const;
		void scale(const double& scal);
		double scale_so_total_is(const double& tot);
		double scale_so_mean_is(const double& mean);
		void prob_scale_2 (ndarray* out=nullptr);
		void sector_prob_scale_2 (const std::vector<unsigned>& sectors, ndarray* out=nullptr);
		void sector_prob_scale_2 (const std::vector<unsigned>& sectors, const unsigned& rowbegin, const unsigned& rowend);
		void logsums_2 (ndarray* out);
		void operator+=(const ndarray& that);
		void operator-=(const ndarray& that);
		void operator+=(const memarray_raw& that);
		void operator-=(const memarray_raw& that);
		double operator*(const ndarray& that) const; // vector dot product
		void projection(const ndarray& fixture, const ndarray& beam, const double& distance);

	// Attributes
		size_t ndim() const { if (pool) return PyArray_NDIM(pool); else return 0; }
		size_t size() const { if (pool) return PyArray_SIZE(pool); else return 0; }
		size_t size1() const { if (pool) return PyArray_DIM(pool,0); else return 0; }
		size_t size2() const { if (pool) {if (PyArray_NDIM(pool)>1) return PyArray_DIM(pool,1); else return 1;} else return 0; }
		size_t size3() const { if (pool) {if (PyArray_NDIM(pool)>2) return PyArray_DIM(pool,2); else return 1;} else return 0;}
		size_t sizeButLast() const { if (pool) return PyArray_SIZE(pool)/PyArray_DIM(pool,(PyArray_NDIM(pool)-1)); else return 0;}
		size_t sizeLast() const { if (pool) return PyArray_DIM(pool,(PyArray_NDIM(pool)-1)); else return 0;}
		bool operator==(const ndarray& that) const;
		
	// Printing
		std::string printrow(const unsigned& r) const;
		std::string printrows(unsigned rstart, const unsigned& rfinish) const;
		std::string printall() const;

		std::string printrow_hex(const unsigned& r) const;
		std::string printrows_hex(unsigned rstart, const unsigned& rfinish) const;
		std::string printall_hex() const;

	};




	class symmetric_matrix: public ndarray {
	
	public:
		symmetric_matrix(const char* arrayType="SymmetricArray");
		symmetric_matrix(const int& r, const char* arrayType="SymmetricArray");
		symmetric_matrix(const int& r,const int& c, const char* arrayType="SymmetricArray");
		symmetric_matrix(PyObject* obj);
		symmetric_matrix(symmetric_matrix& that, bool use_same_memory);
	private:
		symmetric_matrix(const int& r,const int& c,const int& s, const char* arrayType="SymmetricArray");
	public:
		void copy_uppertriangle_to_lowertriangle();
		void copy_lowertriangle_to_uppertriangle();

	public:
		void inv(logging_service* msg_=NULL);
		void inv_bonafide(logging_service* msg_=NULL);
		void initialize_identity();
		std::string printSquare();

	// Copying
		void operator= (memarray_raw& that);
		void operator= (ndarray& that);
		void operator= (const symmetric_matrix& that);

	// Resizing
		void resize(const int& r);
		void resize(const int& r,const int& c);
		void resize(const int& r,const int& c,const int& s);
		void resize(etk::three_dim& prototype);

	// Location Access
		const double& operator()(const int& i, const int& j) const;
		double& operator()(const int& i, const int& j) ;
		inline const double& at(const int& i, const int& j) const { return operator()(i,j); }
		inline double& at(const int& i, const int& j)  { return operator()(i,j); }
		
	// Pointer Location Access
		const double* ptr() const { return static_cast<double*>( PyArray_DATA(pool) ); }
		const double* ptr(const int& i, const int& j) const {return &(operator()(i,j));}
		double* ptr() { return static_cast<double*>( PyArray_DATA(pool) ); }
		double* ptr(const int& i, const int& j)  {return &(operator()(i,j));}

	// Other
		bool all_zero() const;


	};

	extern PyObject* array_module;
	void _set_array_module(PyObject* mod);

} // end namespace etk

#endif // ndef SWIG

///// !: SWIG :! /////

#ifdef SWIG
%{
#include "etk_ndarray.h"
%}
namespace etk {

	void _set_array_module(PyObject* mod);

	class three_dim {
	public:
		virtual const unsigned& size1() const;
		const unsigned& size2() const ;
		const unsigned& size3() const ;
		three_dim(const unsigned& r, const unsigned& c, const unsigned& d);

	};

}
#endif // def SWIG



#endif // __TOOLBOX_NDARRAY__
