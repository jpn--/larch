/*
 *  etk_math.cpp
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
#include "larch_portable.h"
#include "etk_python.h"
#include "etk_math.h"
#include "etk_exception.h"

using namespace etk;
/*
#ifndef __APPLE__
extern "C" {
	int dpptrf_(char *uplo, integer *n, doublereal *ap, integer *info);
	int dpptri_(char *uplo, integer *n, doublereal *ap, integer *info);
}
#endif

void etk::dppinv (double* A, int N) {
	// LAPACK uses Column Major data format, while xlogit uses Row Major, so
	// this function calls for the lower triangular instead of the usual upper
	char up = 'L';
#ifdef __APPLE__
	__CLPK_integer output = 0;
	// factorize
	dpptrf_(&up, ( __CLPK_integer* )&N, A, &output);
	// invert using factorization
	dpptri_(&up, ( __CLPK_integer* )&N, A, &output);
#else
    integer output = 0;
	// factorize
	dpptrf_(&up, ( integer* )&N, A, &output);
	// invert using factorization
	dpptri_(&up, ( integer* )&N, A, &output);
	
	
	// ATLAS?
	// factorize
	//	clapack_dpptrf(CblasRowMajor, CblasUpper, N, A);
	// invert using factorization
	//	clapack_dpptri(CblasRowMajor, CblasUpper, N, A);
#endif
}
*/
/*
 void dppinv (double* A, int N) {
#ifdef __APPLE__
	// LAPACK uses Column Major data format, while xlogit uses Row Major, so
	// this function calls for the lower triangular instead of the usual upper
	char up = 'L'; 
	__CLPK_integer output = 0;
	// factorize
	dpptrf_(&up, ( __CLPK_integer* )&N, A, &output);
	// invert using factorization
	dpptri_(&up, ( __CLPK_integer* )&N, A, &output);
#else
	// factorize
	clapack_dpptrf(CblasRowMajor, CblasUpper, N, A); 
	// invert using factorization
	clapack_dpptri(CblasRowMajor, CblasUpper, N, A); 
#endif
}
 */

void etk::element_multiply(const int& N, const double& alpha, const double * A, const int& incA, 
					  const double * X, const int& incX, 
					  const double& beta, double * Y, const int& incY) 
{
	cblas_dsbmv(CblasRowMajor, CblasUpper, 
				N, 0, alpha, A, incA, X, incX, beta, Y, incY);
}

void etk::simple_element_multiply(const int& N, const double * A, const int& incA, 
							 const double * X, const int& incX, 
							 double * Y, const int& incY) 
{
	cblas_dsbmv(CblasRowMajor, CblasUpper, 
				N, 0, 1.0, A, incA, X, incX, 0.0, Y, incY);
}


void etk::simple_inplace_element_multiply(const int& N, const double * A, const int& incA, 
									 double * Y, const int& incY)
{
	for (int i=0; i<N; i++) Y[i*incY] *= A[i*incA];
}




void* etk::scipy_dgemm;

void etk::load_scipy_blas_functions()
{
//	etk::scipy_dgemm = nullptr;

//>>> import scipy.linalg.cython_blas as b
//>>> b.__pyx_capi__['dgemm']
//<capsule object "void (char *, char *, int *, int *, int *, __pyx_t_5scipy_6linalg_11cython_blas_d *, __pyx_t_5scipy_6linalg_11cython_blas_d *, int *, __pyx_t_5scipy_6linalg_11cython_blas_d *, int *, __pyx_t_5scipy_6linalg_11cython_blas_d *, __pyx_t_5scipy_6linalg_11cython_blas_d *, int *)" at 0x107435990>

//	etk::scipy_dgemm = PyCapsule_Import("scipy.linalg.cython_blas.dgemm", 0);

	boosted::lock_guard<boosted::mutex> LOCK(etk::python_global_mutex);

	PyObject* cython_blas = PyImport_ImportModule("scipy.linalg.cython_blas");
	PyObject* pyx_capi = PyObject_GetAttrString(cython_blas, "__pyx_capi__");
	PyObject* dgemm_capsule = PyDict_GetItemString(pyx_capi, "dgemm");
	void* t = PyCapsule_GetPointer(dgemm_capsule, PyCapsule_GetName(dgemm_capsule));
	etk::scipy_dgemm = t;

	if (PyErr_Occurred()) OOPS("an error occurred in load_scipy_blas_functions");
	
	Py_CLEAR(dgemm_capsule);
	Py_CLEAR(pyx_capi);
	Py_CLEAR(cython_blas);
	
	
}

