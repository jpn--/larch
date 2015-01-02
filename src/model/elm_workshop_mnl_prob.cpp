/*
 *  elm_model.cpp
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


#include <cstring>
#include "etk.h"
#include <iostream>

#include "elm_workshop_mnl_prob.h"


elm::mnl_prob_w::mnl_prob_w(  etk::ndarray* U
							, etk::ndarray* CLL
							, elm::darray_ptr Data_CA
							, elm::darray_ptr Data_CO
							, elm::darray_ptr Data_AV
							, elm::darray_ptr Data_Ch
							, etk::ndarray* Coef_CA
							, etk::ndarray* Coef_CO
							, const double& U_premultiplier
							, etk::logging_service* msgr
							)
: Probability(U)
, CaseLogLike(CLL)
, Data_CA(Data_CA)
, Data_CO(Data_CO)
, Data_AV(Data_AV)
, Data_Ch(Data_Ch)
, Coef_CA(Coef_CA)
, Coef_CO(Coef_CO)
, U_premultiplier(U_premultiplier)
, msg_(msgr)
{
}

elm::mnl_prob_w::~mnl_prob_w()
{
}


void elm::mnl_prob_w::work(size_t firstcase, size_t numberofcases, boosted::mutex* result_mutex)
{
	unsigned nElementals = 0;
	if (Data_CA && Data_CA->nVars()>0) {
		nElementals = Data_CA->nAlts();
	} else if (Data_CO && Coef_CO->size2()>0) {
		nElementals = Coef_CO->size2();
	} else {
		return;
		OOPS("no useful data!");
	};

	// UTILITY //

	if (Data_CA && Data_CA->nVars()>0 /* && Data_CA->fully_loaded()*/) {
		// Fast Linear Algebra		
		if (Probability->size2()==Data_CA->nAlts()) {
			cblas_dgemv(CblasRowMajor,CblasNoTrans, 
						numberofcases * Data_CA->nAlts(), Data_CA->nVars(), 
						1,
						Data_CA->values(firstcase,0),Data_CA->nVars(),
						Coef_CA->ptr(),1,
						U_premultiplier, Probability->ptr(firstcase),1);
		} else {
			for (unsigned a=0;a<Data_CA->nAlts();a++) {
				if (Probability->size2()<=0){
					OOPS("IncY Zero");
				}
				cblas_dgemv(CblasRowMajor,CblasNoTrans,
							numberofcases,Data_CA->nVars(),
							1, 
							Data_CA->values(firstcase,0)+(a*Data_CA->nVars()), Data_CA->nAlts()*Data_CA->nVars(), 
							Coef_CA->ptr(),1, 
							U_premultiplier, Probability->ptr(firstcase)+a, Probability->size2() );
			}
		}		
	} else if (Data_CA && Data_CA->nVars()>0) {
		// Slow case-by-case
		for (unsigned c=firstcase; c<firstcase+numberofcases; c++) {
			if (Probability->size2()==Data_CA->nAlts()) {
				cblas_dgemv(CblasRowMajor,CblasNoTrans, 
							Data_CA->nAlts(), Data_CA->nVars(), 
							1,
							Data_CA->values(c,1),Data_CA->nVars(),
							Coef_CA->ptr(),1,
							U_premultiplier, Probability->ptr(c),1);
			} else {
				for (unsigned a=0;a<Data_CA->nAlts();a++) {
					if (Probability->size2()<=0){
						OOPS("IncY Zero");
					}
					cblas_dgemv(CblasRowMajor,CblasNoTrans,
								1,Data_CA->nVars(),
								1, 
								Data_CA->values(c,1)+(a*Data_CA->nVars()), Data_CA->nAlts()*Data_CA->nVars(), 
								Coef_CA->ptr(),1, 
								U_premultiplier, Probability->ptr(c)+a, Probability->size2() );
				}
			}
		}
	}
	if (!Data_CA || Data_CA->nVars()==0) {
		if (U_premultiplier) {
			cblas_dscal(Probability->size2()*Probability->size3()*numberofcases, U_premultiplier, Probability->ptr(firstcase), 1);
		} else {
			memset(Probability->ptr(firstcase), 0, sizeof(double)*Probability->size2()*Probability->size3()*numberofcases);
		}
	}
	
	if (Data_CO && Data_CO->nVars()>0 /*&& Data_CO->fully_loaded()*/) {
		// Fast Linear Algebra		
		cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
					numberofcases,Coef_CO->size2(), Data_CO->nVars(),
					1,
					Data_CO->values(firstcase,0), Data_CO->nVars(),
					Coef_CO->ptr(), Coef_CO->size2(),
					1,Probability->ptr(firstcase),Probability->size2());
	} else if (Data_CO && Data_CO->nVars()>0) {
		// Slow case-by-case
		for (unsigned c=firstcase; c<firstcase+numberofcases; c++) {
			if (Data_CO->nVars())
				cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
							1,Coef_CO->size2(), Data_CO->nVars(),
							1,
							Data_CO->values(c,1), Data_CO->nVars(),
							Coef_CO->ptr(), Coef_CO->size2(),
							1,Probability->ptr(c),Probability->size2());
		}
	}

	// PROBABILITY //
	
	if (firstcase==0) BUGGER_(msg_, "Util[0]="<<Probability->printrow(0));
	
	for (unsigned c=firstcase; c<firstcase+numberofcases; c++) {
		double sum_prob = 0.0;
		double sum_choice = 0.0;
		CaseLogLike->at(c) = 0.0;
		for (unsigned a=0;a<nElementals;a++) {
			if (!Data_AV->boolvalue(c,a)) {
//				if (c==0) BUGGER_(msg_, "Availability for case 0 alt "<<a<<" is NO");
				Probability->at(c,a) = 0.0;
			} else {
				double* p = Probability->ptr(c,a);
				if (Data_Ch->value(c,a)) {
					CaseLogLike->at(c) += (*p) * Data_Ch->value(c,a);
					sum_choice += Data_Ch->value(c,a);
				}
				*p = exp(*p);
				sum_prob += *p;
//				if (c==0) BUGGER_(msg_, "Availability for case 0 alt "<<a<<" is YES, exp(Utility)=\t"<<*p);
			}
		}
		if (c==0) BUGGER_(msg_, "Data_AV[0]="<<Data_AV->boolvalue(0,0)<<Data_AV->boolvalue(0,1)<<Data_AV->boolvalue(0,2));
		if (c==0) BUGGER_(msg_, "Sum(exp(Util))[0]="<<sum_prob);
		if (sum_prob) {
			for (unsigned a=0;a<nElementals;a++) {
				Probability->at(c,a) /= sum_prob;
			}
			if (sum_choice) {
				CaseLogLike->at(c) -= log(sum_prob) * sum_choice;
			}
		}
		
		
	}
	
	if (firstcase==0) BUGGER_(msg_, "Prob[0]="<<Probability->printrow(0));

}

