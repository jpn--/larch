/*
 *  elm_model.cpp
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

#ifndef __ELM_WORKSHOP_NL_GRADIENT_H__
#define __ELM_WORKSHOP_NL_GRADIENT_H__


#include <cstring>
#include "elm_model2.h"
#include "elm_parameter2.h"
#include "elm_sql_scrape.h"
#include "elm_names.h"
#include "etk_workshop.h"
#include <iostream>


namespace elm {




void __casewise_nl_dUtility_dParameters
( const double*	     Pr      // node probability, in [N_Nodes] space
, const double*      CPr 	 // conditional probability, in [N_Edges] space
, const double*	     Util 	 // scale-free utility, in [N_Nodes] space 
, const unsigned&    c 		 // Case Index Number
, const VAS_System&  Xylem
, datamatrix      Data_UtilityCA
, datamatrix      Data_UtilityCO
, etk::memarray_raw&     dUtilCA
, etk::memarray_raw&     dUtilCO
, etk::memarray_raw&     dUtilMU
);

void __casewise_nl_dProb_dParam
( etk::memarray_raw&     dProbCA   // [nodes,params]
, etk::memarray_raw&     dProbCO   // [nodes,params]
, etk::memarray_raw&     dProbMU   // [nodes,params]
, const VAS_System&  Xylem     // 
, const etk::memarray_raw&    dUtilCA   // [nodes,params]
, const etk::memarray_raw&    dUtilCO   // [nodes,params]
, const etk::memarray_raw&    dUtilMU   // [nodes,params]
, const double*      Util      // [nodes]
, const double*      CPr       // [edges]
, const double*      Pr        // [nodes]
, double*            scratchCA   // [params]
, double*            scratchCO   // [params]
, double*            scratchMU   // [params]
, const double*      Cho       // [elementals]
);



void __casewise_dLogLike_dParameters
( double*            dLL     // [params]
, const etk::memarray_raw&    dProb   // [nodes, params]
, const double*      Pr      // [nodes]
, const double*      Cho     // [elementals]
, const unsigned&    nA      // number of elementals
, const unsigned&    nP      // number of parameters
);


void __casewise_nl_gradient
( const unsigned& c
, const etk::memarray* Probability
, const etk::memarray* Cond_Prob
, const etk::memarray* Utility
, const VAS_System*    Xylem
, datamatrix        Data_UtilityCA
, datamatrix        Data_UtilityCO
, datamatrix        Data_Choice
, etk::memarray_raw& dUtilCA
, etk::memarray_raw& dUtilCO
, etk::memarray_raw& dUtilMU
, etk::memarray_raw& dProbCA
, etk::memarray_raw& dProbCO
, etk::memarray_raw& dProbMU
, etk::memarray_raw& WorkspaceCA
, etk::memarray_raw& WorkspaceCO
, etk::memarray_raw& WorkspaceMU
, etk::memarray_raw& dLLCA
, etk::memarray_raw& dLLCO
, etk::memarray_raw& dLLMU
);





class workshop_nl_gradient
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
	size_t nPar;
	
	etk::memarray_raw dUtil      ;
	etk::memarray_raw dProb      ;

	etk::memarray_raw dSampWgt   ;
	etk::memarray_raw dAdjProb   ;

	etk::memarray_raw Workspace  ;

	etk::memarray_raw GradT_Fused    ;
	etk::memarray_raw CaseGrad       ;
	
	etk::memarray_raw workshopGCurrent;
	etk::memarray_symmetric workshopBHHH   ;

	const paramArray* Params_LogSum;
	
	datamatrix Data_Choice;
	datamatrix Data_Weight;

	const etk::memarray* _Probability;
	const etk::memarray* _AdjProbability;
	const etk::memarray* _Cond_Prob;
	const VAS_System* _Xylem;
	etk::memarray* _GCurrent;
	etk::memarray_symmetric* _Bhhh;
	boosted::mutex* _lock;
	
	int threadnumber;
	etk::logging_service* msg_;

	elm::ca_co_packet UtilPacket;
	elm::ca_co_packet SampPacket;

//	workshop_nl_gradient();
	
	workshop_nl_gradient(
	   const unsigned&   dF
	 , const unsigned&   nNodes
	 , elm::ca_co_packet UtilPacket
	 , elm::ca_co_packet SampPacket
	 , const paramArray& Params_LogSum
	 , datamatrix     Data_Choice
	 , datamatrix     Data_Weight
	 , const etk::memarray* AdjProbability
	 , const etk::memarray* Probability
	 , const etk::memarray* Cond_Prob
	 , const VAS_System* Xylem
	 , etk::memarray* GCurrent
	 , etk::memarray_symmetric* Bhhh
	 , etk::logging_service* msgr
	 );
	
	virtual ~workshop_nl_gradient();

	void workshop_nl_gradient_do(const unsigned& firstcase, const unsigned& numberofcases);
	void workshop_nl_gradient_send();

	void case_dUtility_dFusedParameters( const unsigned& c );
	void case_dProbability_dFusedParameters( const unsigned& c );

	void case_dSamplingFactor_dFusedParameters( const unsigned& c );
	void case_dAdjProbability_dFusedParameters( const unsigned& c );

	void case_dLogLike_dFusedParameters( const unsigned& c );
	
	virtual void work(size_t firstcase, size_t numberofcases, boosted::mutex* result_mutex);	
};



}
#endif // __ELM_WORKSHOP_NL_GRADIENT_H__

