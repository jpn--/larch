/*
 *  etk_math.h
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







#ifndef __TOOLBOX_MATH__
#define __TOOLBOX_MATH__

namespace etk {

//void dppinv (double* A, int N);

// X * A * alpha + Y * beta -> Y
void element_multiply(const int& N, const double& alpha, const double * A, const int& incA, 
					  const double * X, const int& incX, 
					  const double& beta, double * Y, const int& incY) ;

void simple_inplace_element_multiply(const int& N, const double * A, const int& incA, 
									 double * Y, const int& incY) ;

void simple_element_multiply(const int& N, const double * A, const int& incA, 
							 const double * X, const int& incX, 
							 double * Y, const int& incY) ;


#define PYBLAS_dgemm \
 (*(void (*)(const enum CBLAS_ORDER __Order,                                \
        const enum CBLAS_TRANSPOSE __TransA,                                \
        const enum CBLAS_TRANSPOSE __TransB, const int __M, const int __N,  \
        const int __K, const double __alpha, const double *__A,             \
        const int __lda, const double *__B, const int __ldb,                \
        const double __beta, double *__C, const int __ldc))                 \
		etk::scipy_dgemm)

extern void *scipy_dgemm;

//extern void (*scipy_dgemm) (const enum CBLAS_ORDER __Order,
//        const enum CBLAS_TRANSPOSE __TransA,
//        const enum CBLAS_TRANSPOSE __TransB, const int __M, const int __N,
//        const int __K, const double __alpha, const double *__A,
//        const int __lda, const double *__B, const int __ldb,
//        const double __beta, double *__C, const int __ldc);

void load_scipy_blas_functions();



};
#endif

