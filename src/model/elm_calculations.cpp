/*
 *  elm_calculations.cpp
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


#include "elm_calculations.h"

void elm::__logit_utility
( etk::memarray&  U
, elm::darray_ptr   Data_CA
, elm::darray_ptr   Data_CO
, const etk::memarray& Coef_CA
, const etk::memarray& Coef_CO
, const double&   U_premultiplier
)
{
	if (Data_CA->nVars()>0 /*&& Data_CA->fully_loaded()*/) {
		// Fast Linear Algebra		
		if (U.size2()==Data_CA->nAlts()) {
			cblas_dgemv(CblasRowMajor,CblasNoTrans, 
						Data_CA->nCases() * Data_CA->nAlts(), Data_CA->nVars(), 
						1,
						Data_CA->values(0,0),Data_CA->nVars(),
						*Coef_CA,1,
						U_premultiplier, *U,1);
		} else {
			for (unsigned a=0;a<Data_CA->nAlts();a++) {
				cblas_dgemv(CblasRowMajor,CblasNoTrans,
							Data_CA->nCases(),Data_CA->nVars(),
							1, 
							Data_CA->values(0,0)+(a*Data_CA->nVars()), Data_CA->nAlts()*Data_CA->nVars(), 
							*Coef_CA,1, 
							U_premultiplier, *U+a, U.size2() );
			}
		}		
	} else if (Data_CA->nVars()>0) {
		// Slow case-by-case
		for (unsigned c=0; c<Data_CA->nCases(); c++) {
			if (U.size2()==Data_CA->nAlts()) {
				cblas_dgemv(CblasRowMajor,CblasNoTrans, 
							Data_CA->nAlts(), Data_CA->nVars(), 
							1,
							Data_CA->values(c,1),Data_CA->nVars(),
							*Coef_CA,1,
							U_premultiplier, U.ptr(c),1);
			} else {
				for (unsigned a=0;a<Data_CA->nAlts();a++) 
					cblas_dgemv(CblasRowMajor,CblasNoTrans,
								1,Data_CA->nVars(),
								1, 
								Data_CA->values(c,1)+(a*Data_CA->nVars()), Data_CA->nAlts()*Data_CA->nVars(), 
								*Coef_CA,1, 
								U_premultiplier, U.ptr(c)+a, U.size2() );
			}
		}
	}
	if (Data_CA->nVars()==0) {
		if (U_premultiplier) U.scale(U_premultiplier); else U.initialize(0.0);
	}
	
	if (Data_CO->nVars()>0 /*&& Data_CO->fully_loaded()*/) {
		// Fast Linear Algebra		
		cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
					Data_CO->nCases(), Coef_CO.size2(), Data_CO->nVars(),
					1,
					Data_CO->values(0,0), Data_CO->nVars(),
					*Coef_CO, Coef_CO.size2(),
					1,*U,U.size2());
	} else if (Data_CO->nVars()>0) {
		// Slow case-by-case
		for (unsigned c=0; c<Data_CA->nCases(); c++) {
			if (Data_CO->nVars())
				cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
							1,Coef_CO.size2(), Data_CO->nVars(),
							1,
							Data_CO->values(c,1), Data_CO->nVars(),
							*Coef_CO, Coef_CO.size2(),
							1,U.ptr(c),U.size2());
		}
	}
}




void elm::__logit_utility_arrays
( etk::memarray&  U
 , const etk::ndarray*   Data_CA
 , const etk::ndarray*   Data_CO
 , const etk::memarray& Coef_CA
 , const etk::memarray& Coef_CO
 , const double&   U_premultiplier
 )
{
	if (Data_CA && Data_CA->size3()>0) {
		// Fast Linear Algebra
		if (U.size2()==Data_CA->size2()) {
			cblas_dgemv(CblasRowMajor,CblasNoTrans,
						Data_CA->size1() * Data_CA->size2(), Data_CA->size3(),
						1,
						Data_CA->ptr(),Data_CA->size3(),
						*Coef_CA,1,
						U_premultiplier, *U,1);
		} else {
			for (unsigned a=0;a<Data_CA->size2();a++) {
				cblas_dgemv(CblasRowMajor,CblasNoTrans,
							Data_CA->size1(),Data_CA->size3(),
							1,
							Data_CA->ptr()+(a*Data_CA->size3()), Data_CA->size2()*Data_CA->size3(),
							*Coef_CA,1,
							U_premultiplier, *U+a, U.size2() );
			}
		}
	}
	if ((!Data_CA) || Data_CA->size3()==0) {
		if (U_premultiplier) U.scale(U_premultiplier); else U.initialize(0.0);
	}

	
	if (Data_CO->size2()>0 /*&& Data_CO->fully_loaded()*/) {
		// Fast Linear Algebra
		cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
					Data_CO->size1(), Coef_CO.size2(), Data_CO->size2(),
					1,
					Data_CO->ptr(), Data_CO->size2(),
					*Coef_CO, Coef_CO.size2(),
					1,*U,U.size2());
	}


	
}



