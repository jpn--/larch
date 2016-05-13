/*
 *  elm_model.cpp
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


#include <cstring>
#include "elm_model2.h"
#include "elm_parameter2.h"
#include "elm_sql_scrape.h"
#include "elm_names.h"
#include <iostream>

#include "elm_workshop_mnl_gradient.h"


using namespace etk;
using namespace elm;
using namespace std;




elm::workshop_mnl_gradient::workshop_mnl_gradient
( const unsigned&   dF
, const unsigned&   nElementals
, const paramArray& Params_UtilityCA
, const paramArray& Params_UtilityCO
, const paramArray& Params_QuantityCA
, const paramArray& Params_LogSum
, elm::darray_ptr     Data_UtilityCA
, elm::darray_ptr     Data_UtilityCO
, elm::darray_ptr     Data_QuantityCA
, elm::darray_ptr     Data_Choice
, elm::darray_ptr     Data_Weight
, const unsigned&   firstcase
, const unsigned&   numberofcases
)
: dF         (dF)
, nElementals(nElementals)
, Workspace    (nElementals)
, CaseGrad        (dF)
, Grad_UtilityCA (Params_UtilityCA.size1(),Params_UtilityCA.size2(),Params_UtilityCA.size3())
, Grad_UtilityCO (Params_UtilityCO.size1(),Params_UtilityCO.size2(),Params_UtilityCO.size3())
, Grad_QuantityCA(Params_QuantityCA.size1(),Params_QuantityCA.size2(),Params_QuantityCA.size3())
, workshopBHHH    (dF)
, workshopGCurrent(dF)
, multichoices	  (numberofcases,1,1)
, Params_UtilityCA(&Params_UtilityCA)
, Params_UtilityCO(&Params_UtilityCO)
, Params_QuantityCA(&Params_QuantityCA)
, Params_LogSum   (&Params_LogSum)
, Data_UtilityCA  (Data_UtilityCA)
, Data_UtilityCO  (Data_UtilityCO)
, Data_QuantityCA (Data_QuantityCA)
, Data_Choice     (Data_Choice)
, Data_Weight     (Data_Weight)
, firstcase(firstcase)
, numberofcases(numberofcases)
, lastcase(firstcase+numberofcases)
{
//	Data_UtilityCA->incref();
//	Data_UtilityCO->incref();
//	Data_Choice   ->incref();
//	if (Data_Weight) {
//		Data_Weight->incref();
//	}

//	Data_UtilityCA->load_values(firstcase,numberofcases);
//	Data_UtilityCO->load_values(firstcase,numberofcases);
//	Data_Choice   ->load_values(firstcase,numberofcases);
//	if (Data_Weight) {
//		Data_Weight->load_values(firstcase,numberofcases);
//	}
	
	unsigned m,a,found;
	double sum;
	for (unsigned c=firstcase;c<lastcase;c++) {
		m=c-firstcase;
		multichoices.input(false,m);
		found=0;
		sum = 0;
		for (a=0;a<nElementals;a++) {
			if (Data_Choice->value(c,a,0)) {
				found++;
				sum += Data_Choice->value(c,a,0);
			}
		}
		if (found>1 || sum != 1.0) {
			multichoices.input(true,m);
		}
	}
}

elm::workshop_mnl_gradient::~workshop_mnl_gradient()
{
//	Data_UtilityCA->decref();
//	Data_UtilityCO->decref();
//	Data_Choice   ->decref();
//	if (Data_Weight) {
//		Data_Weight->decref();
//	}
}






void elm::workshop_mnl_gradient::case_gradient_mnl
( const unsigned& c
, const etk::memarray& Probability
)
{
	double wgt = 1.0;
	if (Data_Weight) wgt = Data_Weight->value(c,0);
	cblas_dcopy(nElementals,Data_Choice->values(c,1),1,*Workspace,1);
	if (wgt!=1.0) cblas_dscal(nElementals, wgt, *Workspace,1);
	cblas_daxpy(nElementals,-wgt,Probability.ptr(c),1,*Workspace,1);
	// idCA
	//ModelCaseGrad.initialize();
	if (Data_UtilityCA->nVars()) {
		cblas_dgemv(CblasRowMajor,CblasTrans,nElementals,Data_UtilityCA->nVars(),
					-1,Data_UtilityCA->values(c,1),Data_UtilityCA->nVars(),*Workspace,1,0,*Grad_UtilityCA,1);
	}
	// idCO
	if (Data_UtilityCO->nVars()) {
		double* point = *Grad_UtilityCO;
		memset(point, 0, nElementals*Data_UtilityCO->nVars()*sizeof(double));
		cblas_dger(CblasRowMajor,Data_UtilityCO->nVars(),nElementals,-1,
				   Data_UtilityCO->values(c,1),1,*Workspace,1,point,nElementals);
	}

	CaseGrad.initialize();
	elm::push_to_freedoms2(*Params_UtilityCA  , *Grad_UtilityCA  , *CaseGrad);
	elm::push_to_freedoms2(*Params_UtilityCO  , *Grad_UtilityCO  , *CaseGrad);

	// BHHH
	#ifdef SYMMETRIC_PACKED
	cblas_dspr(CblasRowMajor,CblasUpper, dF,1,*CaseGrad, 1, *workshopBHHH);
	#else
	cblas_dsyr(CblasRowMajor,CblasUpper, dF,1,*CaseGrad, 1, *workshopBHHH, workshopBHHH.size1());
	#endif
	
	// ACCUMULATE
	cblas_daxpy(dF,1,*CaseGrad,1,*workshopGCurrent,1);
}

void elm::workshop_mnl_gradient::case_gradient_mnl_multichoice
( const unsigned& c
, const etk::memarray& Probability
)
{
	double thisWgt;
	for (unsigned a=0; a<nElementals; a++) {
		thisWgt = Data_Choice->value(c,a);
		if (Data_Weight) thisWgt *= Data_Weight->value(c,0);
		if (thisWgt==0) {
			continue;
		}
		memset(*Workspace, 0, nElementals*sizeof(double));
		(*Workspace)[a] = thisWgt;
		cblas_daxpy(nElementals,-1/*thisWgt*/,Probability.ptr(c),1,*Workspace,1);
		// idCA
		//ModelCaseGrad.initialize();
		if (Data_UtilityCA->nVars()) {
			cblas_dgemv(CblasRowMajor,CblasTrans,nElementals,Data_UtilityCA->nVars(),
						-1,Data_UtilityCA->values(c,1),Data_UtilityCA->nVars(),*Workspace,1,0,*Grad_UtilityCA,1);
		}
		// idCO
		if (Data_UtilityCO->nVars()) {
			double* point = *Grad_UtilityCO;
			memset(point, 0, nElementals*Data_UtilityCO->nVars()*sizeof(double));
			cblas_dger(CblasRowMajor,Data_UtilityCO->nVars(),nElementals,-1,
					   Data_UtilityCO->values(c,1),1,*Workspace,1,point,nElementals);
		}
		CaseGrad.initialize();
		elm::push_to_freedoms2(*Params_UtilityCA  , *Grad_UtilityCA  , *CaseGrad);
		elm::push_to_freedoms2(*Params_UtilityCO  , *Grad_UtilityCO  , *CaseGrad);
		#ifdef SYMMETRIC_PACKED
		cblas_dspr(CblasRowMajor,CblasUpper, dF,thisWgt/*1*/,*CaseGrad, 1, *workshopBHHH);
		#else
		cblas_dsyr(CblasRowMajor,CblasUpper, dF,thisWgt/*1*/,*CaseGrad, 1, *workshopBHHH, workshopBHHH.size1());
		#endif
		cblas_daxpy(dF,thisWgt/*1*/,*CaseGrad,1,*workshopGCurrent,1);
	}
}



void elm::workshop_mnl_gradient::workshop_mnl_gradient_send
( memarray& GCurrent
, symmetric_matrix& Bhhh)
{
	GCurrent += workshopGCurrent;
	Bhhh += workshopBHHH;
}



void elm::workshop_mnl_gradient::workshop_mnl_gradient_do(const etk::memarray& Probability)  // version 1
{
//	BUGGER(msg)<< "Beginning MNL Gradient Evaluation" ;
	unsigned c;
	workshopGCurrent.initialize(0.0);
	workshopBHHH.initialize(0.0);
	for (c=firstcase;c<lastcase;c++) {
		if (multichoices(c-firstcase)) {
			case_gradient_mnl_multichoice(c,Probability);
		} else {
			case_gradient_mnl(c,Probability);		
		}
	}
//	BUGGER(msg)<< "End MNL Gradient Evaluation" ;
}




