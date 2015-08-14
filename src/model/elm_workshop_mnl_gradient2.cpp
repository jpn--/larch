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
#include "elm_model2.h"
#include "elm_parameter2.h"
#include "elm_sql_scrape.h"
#include "elm_names.h"
#include <iostream>

#include "elm_workshop_mnl_gradient.h"


using namespace etk;
using namespace elm;
using namespace std;




elm::workshop_mnl_gradient2::workshop_mnl_gradient2
(  const unsigned&   dF
 , const unsigned&   nElementals
 , elm::ca_co_packet UtilPK
 , elm::ca_co_packet QuantPK
 , elm::darray_ptr     Data_Choice
 , elm::darray_ptr     Data_Weight
 , const etk::memarray* Probability
 , etk::memarray* GCurrent
 , etk::memarray_symmetric* Bhhh
 , etk::logging_service* msgr
 , const etk::bitarray* _Data_MultiChoice
 )
: dF           (dF)
, nElementals  (nElementals)
, Workspace    (nElementals)
, CaseGrad        (dF)
, Grad_UtilityCA (UtilPK.Params_CA->size1(),UtilPK.Params_CA->size2(),UtilPK.Params_CA->size3())
, Grad_UtilityCO (UtilPK.Params_CO->size1(),UtilPK.Params_CO->size2(),UtilPK.Params_CO->size3())
, Grad_QuantityCA(QuantPK.Params_CA->size1(),QuantPK.Params_CA->size2(),QuantPK.Params_CA->size3())
, workshopBHHH    (dF)
, workshopGCurrent(dF)
, _multichoices	  (_Data_MultiChoice)
, Data_Choice     (Data_Choice)
, Data_Weight     (Data_Weight)
, _Probability    (Probability)
, _GCurrent (GCurrent)
, _Bhhh     (Bhhh)
, msg_     (nullptr)
, nCA (UtilPK.Params_CA->length())
, nCO (UtilPK.Params_CO->length())
, nQ  (QuantPK.Params_CA->length())
, nPar(nCA+nCO+nQ)
, UtilPacket (UtilPK)
, QuantPacket (QuantPK)
{

	size_t s =workshopGCurrent.size();
	size_t s1 = s;
	s1 += 1;
	//BUGGER_(msg_, "elm::workshop_mnl_gradient2:: \n\n\nI AM ALIVE!\n\n" );
}

elm::workshop_mnl_gradient2::~workshop_mnl_gradient2()
{

	Workspace.resize(0);
	CaseGrad.resize(0);
	workshopGCurrent.resize(0);
	workshopBHHH.resize(0);
	Grad_UtilityCA.resize(0);
	Grad_UtilityCO.resize(0);


}





void elm::workshop_mnl_gradient2::case_gradient_mnl
( const unsigned& c
 , const etk::memarray& Probability
 )
{
	double wgt = 1.0;
	if (Data_Weight) wgt = Data_Weight->value(c,0);
	cblas_dcopy(nElementals,Data_Choice->values(c,1),1,*Workspace,1);
//	if (wgt!=1.0) {
//		cblas_dscal(nElementals, wgt, *Workspace,1);
//	}
	cblas_daxpy(nElementals,-1/*wgt*/,Probability.ptr(c),1,*Workspace,1);
	// idCA
	//ModelCaseGrad.initialize();
	if (UtilPacket.Data_CA && UtilPacket.Data_CA->nVars()) {
		cblas_dgemv(CblasRowMajor,CblasTrans,nElementals,UtilPacket.Data_CA->nVars(),
					-1,UtilPacket.Data_CA->values(c,1),UtilPacket.Data_CA->nVars(),*Workspace,1,0,*Grad_UtilityCA,1);
	}
	// idCO
	if (UtilPacket.Data_CO && UtilPacket.Data_CO->nVars()) {
		double* point = *Grad_UtilityCO;
		Grad_UtilityCO.initialize();
		//memset(point, 0, nElementals*UtilPacket.Data_CO->nVars()*sizeof(double));
		cblas_dger(CblasRowMajor,UtilPacket.Data_CO->nVars(),nElementals,-1,
				   UtilPacket.Data_CO->values(c,1),1,*Workspace,1,point,nElementals);
	}
	
	CaseGrad.initialize();
	elm::push_to_freedoms2(*(UtilPacket.Params_CA)  , *Grad_UtilityCA  , *CaseGrad);
	elm::push_to_freedoms2(*(UtilPacket.Params_CO)  , *Grad_UtilityCO  , *CaseGrad);
	
	// BHHH
#ifdef SYMMETRIC_PACKED
	cblas_dspr(CblasRowMajor,CblasUpper, dF,wgt/*1*/,*CaseGrad, 1, *workshopBHHH);
#else
	cblas_dsyr(CblasRowMajor,CblasUpper, dF,wgt/*1*/,*CaseGrad, 1, *workshopBHHH, workshopBHHH.size1());
#endif
	
	// ACCUMULATE
	cblas_daxpy(dF,wgt/*1*/,*CaseGrad,1,*workshopGCurrent,1);
}

void elm::workshop_mnl_gradient2::case_gradient_mnl_multichoice
( const unsigned& c
 , const etk::memarray& Probability
 )
{
	//TODO: change so weight applies to BHHH result after outer product
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
		if (UtilPacket.Data_CA && UtilPacket.Data_CA->nVars()) {
			cblas_dgemv(CblasRowMajor,CblasTrans,nElementals,UtilPacket.Data_CA->nVars(),
						-1,UtilPacket.Data_CA->values(c,1),UtilPacket.Data_CA->nVars(),*Workspace,1,0,*Grad_UtilityCA,1);
		}
		// idCO
		if (UtilPacket.Data_CO && UtilPacket.Data_CO->nVars()) {
			double* point = *Grad_UtilityCO;
			Grad_UtilityCO.initialize();
			//memset(point, 0, nElementals*UtilPacket.Data_CO->nVars()*sizeof(double));
			cblas_dger(CblasRowMajor,UtilPacket.Data_CO->nVars(),nElementals,-1,
					   UtilPacket.Data_CO->values(c,1),1,*Workspace,1,point,nElementals);
		}
		CaseGrad.initialize();
		elm::push_to_freedoms2(*(UtilPacket.Params_CA)  , *Grad_UtilityCA  , *CaseGrad);
		elm::push_to_freedoms2(*(UtilPacket.Params_CO)  , *Grad_UtilityCO  , *CaseGrad);
#ifdef SYMMETRIC_PACKED
		cblas_dspr(CblasRowMajor,CblasUpper, dF,thisWgt/*1*/,*CaseGrad, 1, *workshopBHHH);
#else
		cblas_dsyr(CblasRowMajor,CblasUpper, dF,thisWgt/*1*/,*CaseGrad, 1, *workshopBHHH, workshopBHHH.size1());
#endif
		cblas_daxpy(dF,thisWgt/*1*/,*CaseGrad,1,*workshopGCurrent,1);
	}
}



void elm::workshop_mnl_gradient2::work(size_t firstcase, size_t numberofcases, boosted::mutex* result_mutex)
{
	workshop_mnl_gradient_do(firstcase,numberofcases);
	_lock = result_mutex;
	workshop_mnl_gradient_send();
}



void elm::workshop_mnl_gradient2::workshop_mnl_gradient_send
()
{
	if (_lock) {
		_lock->lock();
		*_GCurrent += workshopGCurrent;
		*_Bhhh += workshopBHHH;
		_lock->unlock();
	} else {
		*_GCurrent += workshopGCurrent;
		*_Bhhh += workshopBHHH;
	}
}



void elm::workshop_mnl_gradient2::workshop_mnl_gradient_do(const unsigned& firstcase, const unsigned& numberofcases)
{


	//BUGGER_(msg_, "Beginning MNL Gradient Evaluation" );
	unsigned c;
	workshopGCurrent.initialize(0.0);
	workshopBHHH.initialize(0.0);
	size_t lastcase = firstcase + numberofcases;
	for (c=firstcase;c<lastcase;c++) {
		if ((*_multichoices)(c-firstcase)) {
			case_gradient_mnl_multichoice(c,*_Probability);
		} else {
			case_gradient_mnl(c,*_Probability);
		}
	}
	//BUGGER_(msg_, "End MNL Gradient Evaluation" );


	
}


