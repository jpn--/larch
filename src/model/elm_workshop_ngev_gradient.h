/*
 *  elm_model.cpp
 *
 *  Copyright 2007-2017 Jeffrey Newman
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

#ifndef __ELM_WORKSHOP_NGEV_GRADIENT_H__
#define __ELM_WORKSHOP_NGEV_GRADIENT_H__


#include <cstring>
#include "elm_model2.h"
#include "elm_parameter2.h"
#include "elm_sql_scrape.h"
#include "elm_names.h"
#include "etk_workshop.h"
#include <iostream>


namespace elm {










class workshop_ngev_gradient
: public etk::workshop
{

public:

	unsigned dF;
	unsigned nNodes;
	
	// Fused Parameter Block Sizes
	size_t nCA;
	size_t nCO;
	size_t nMU;
	size_t nSA;
	size_t nSO;
	size_t nAO;
	size_t nQA;
	size_t nQL;
	size_t nPar;
	
	inline size_t offset_mu() {return nCA+nCO;}
	inline size_t offset_sampadj() {return nCA+nCO+nMU;}
	inline size_t offset_alloc() {return nCA+nCO+nMU+nSA+nSO;}
	inline size_t offset_quant() {return nCA+nCO+nMU+nSA+nSO+nAO;}
	
	etk::memarray_raw dUtil      ;
	etk::memarray_raw dProb      ;

	etk::memarray_raw dSampWgt   ;
	etk::memarray_raw dAdjProb   ;

	etk::memarray_raw Workspace  ;

	etk::memarray_raw GradT_Fused    ;
	etk::memarray_raw CaseGrad       ;
	
	etk::memarray_raw workshopGCurrent;
	etk::symmetric_matrix workshopBHHH   ;

	const paramArray* Params_LogSum;
	const paramArray* Params_QuantLogSum;
	
	elm::darray_ptr Data_Choice;
	elm::darray_ptr Data_Weight;

	const etk::memarray* _Quantity;
	const etk::memarray* _Probability;
	const etk::memarray* _AdjProbability;
	const etk::memarray* _Cond_Prob;
	const VAS_System* _Xylem;
	etk::memarray* _GCurrent;
	etk::symmetric_matrix* _Bhhh;
	PyArrayObject* _GCurrentCasewise;
	boosted::mutex* _lock;

	etk::ndarray* export_dProb;
	
	int threadnumber;
	etk::logging_service* msg_;

	elm::ca_co_packet UtilPacket;
	const elm::darray_export_map* UtilityCE_map;
	elm::ca_co_packet AllocPacket;
	elm::ca_co_packet SampPacket;
	elm::ca_co_packet QuantPacket;
	const double* CoefQuantLogsum;
	
//	workshop_ngev_gradient();
	
	workshop_ngev_gradient(
	   const unsigned&   dF
	 , const unsigned&   nNodes
	 , elm::ca_co_packet UtilPacket
	 , const elm::darray_export_map* UtilCEmap
	 , elm::ca_co_packet AllocPacket
	 , elm::ca_co_packet SampPacket
	 , elm::ca_co_packet QuantPacket
	 , const paramArray& Params_LogSum
	 , const paramArray& Params_QuantLogSum
	 , const double* CoefQuantLogsum
	 , elm::darray_ptr     Data_Choice
	 , elm::darray_ptr     Data_Weight
	 , const etk::memarray* AdjProbability
	 , const etk::memarray* Probability
	 , const etk::memarray* Cond_Prob
	 , const VAS_System* Xylem
	 , etk::memarray* GCurrent
//	 , etk::ndarray* GCurrentCasewise
	 , PyArrayObject* Py_GradientArray
	 , etk::symmetric_matrix* Bhhh
	 , etk::logging_service* msgr
	 , etk::ndarray* export_dProb
	 , boosted::mutex* use_lock
	 );

	void rebuild_local_data(
	   const unsigned&   dF
	 , const unsigned&   nNodes
	 , elm::ca_co_packet UtilPacket
	 , const elm::darray_export_map* UtilCEmap
	 , elm::ca_co_packet AllocPacket
	 , elm::ca_co_packet SampPacket
	 , elm::ca_co_packet QuantPacket
	 , const paramArray& Params_LogSum
	 , const paramArray& Params_QuantLogSum
	 , const double* Coef_QuantLogSum
	 , elm::darray_ptr     Data_Choice
	 , elm::darray_ptr     Data_Weight
	 , const etk::memarray* AdjProbability
	 , const etk::memarray* Probability
	 , const etk::memarray* Cond_Prob
	 , const VAS_System* Xylem
	 , etk::memarray* GCurrent
	 , PyArrayObject* GCurrentCasewise
	 , etk::symmetric_matrix* Bhhh
	 , etk::logging_service* msgr
	 , etk::ndarray* export_dProb
	 , boosted::mutex* use_lock
	 );
	
	virtual ~workshop_ngev_gradient();

	void workshop_ngev_gradient_do(const unsigned& firstcase, const unsigned& numberofcases);
	void workshop_ngev_gradient_send();

	void case_dUtility_dFusedParameters( const unsigned& c );
	void case_dProbability_dFusedParameters( const unsigned& c );

	void case_dSamplingFactor_dFusedParameters( const unsigned& c );
	void case_dAdjProbability_dFusedParameters( const unsigned& c );

	void case_dLogLike_dFusedParameters( const unsigned& c );
	
	virtual void work(size_t firstcase, size_t numberofcases, boosted::mutex* result_mutex);	
};

typedef std::function< void(std::shared_ptr<workshop_ngev_gradient>) >    workshop_ngev_gradient_updater_t;


}
#endif // __ELM_WORKSHOP_NGEV_GRADIENT_H__

