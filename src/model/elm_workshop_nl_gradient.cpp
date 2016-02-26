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
#include "etk_thread.h"
#include "etk_arraymath.h"

#include "elm_workshop_nl_gradient.h"

using namespace etk;
using namespace elm;
using namespace std;



void elm::workshop_nl_gradient::case_dUtility_dFusedParameters( const unsigned& c )
{

	const double*	   Pr = _Probability->ptr(c);
	const double*      CPr = _Cond_Prob->ptr(c);
	const double*	   Util = UtilPacket.Outcome->ptr(c);
	const VAS_System*  Xylem = _Xylem;
//	datamatrix      Data_UtilityCA = UtilPacket.Data_CA;
//	datamatrix      Data_UtilityCO = UtilPacket.Data_CO;

	dUtil.initialize(0.0);
	// for this case, this will be the derivative of utility at
	//  each node w.r.t. params [N_Nodes, N_Params]
	
	unsigned a,u;
	
	for (a=0; a<Xylem->size()-1; a++) {
		// 'a' is iterated over all the relevant nodes in the network
		//  the last node is not relevant, as it is the root node and has no predecessors

		if (!Pr[a]) continue;
		
		// First, we calculate the effect of various parameters on the utility
		// of 'a' directly. For elemental alternatives, this means beta, gamma,
		// and theta parameters. For other nodes, only mu has a direct effect
		if (a<Xylem->n_elemental()) {
			// BETA for SELF (elemental alternatives)
			if (nCA) {
				if (UtilPacket.Data_CE && UtilPacket.Data_CE->active()) {
					UtilPacket.Data_CE->export_into(dUtil.ptr(a),c,a,nCA);
				} else {
					UtilPacket.Data_CA->ExportData(dUtil.ptr(a),c,a,UtilPacket.Data_CA->nAlts());
				}
			}
			if (nCO) UtilPacket.Data_CO->ExportData(dUtil.ptr(a)+nCA,c,a,UtilPacket.nAlt());
		} else {
			// MU for SELF (adjust the kiddies contributions) /////HERE
			dUtil(a,a-Xylem->n_elemental()+nCA+nCO) += Util[a];
			dUtil(a,a-Xylem->n_elemental()+nCA+nCO) /= (*Xylem)[a]->mu();
		}
		
		u=0; 
		
		// MU for Parent (non-competitive edge)
		dUtil((*Xylem)[a]->upcell(u)->slot(),(*Xylem)[a]->upcell(u)->mu_offset()+nCA+nCO) -= 
			CPr[(*Xylem)[a]->upedge(u)->edge_slot()] * Util[a];
		
		// Finally, roll up secondary effects on parents
		if (CPr[(*Xylem)[a]->upedge(u)->edge_slot()]) {
			cblas_daxpy(dUtil.size2(),CPr[(*Xylem)[a]->upedge(u)->edge_slot()],dUtil.ptr(a),1,dUtil.ptr((*Xylem)[a]->upcell(u)->slot()),1);
		}
		
	}
}


void elm::workshop_nl_gradient::case_dProbability_dFusedParameters( const unsigned& c )
{

	const double*	   Pr = _Probability->ptr(c);
	const double*      CPr = _Cond_Prob->ptr(c);
	const double*	   Util = UtilPacket.Outcome->ptr(c);
	const VAS_System*  Xylem = _Xylem;
//	datamatrix      Data_UtilityCA = UtilPacket.Data_CA;
//	datamatrix      Data_UtilityCO = UtilPacket.Data_CO;

	double*			   scratch = Workspace.ptr();
	const double*	   Cho = Data_Choice->values(c);
	
	dProb.initialize(0.0);
			
	unsigned i,u;
	for (i=Xylem->size()-1; i!=0; ) {
		i--;
		u=0;
		
		if (i<Xylem->n_elemental()) {
			if (Cho) {
				if ((Pr[i]==0)&&(Cho[i]>0)) {
					throw(ZeroProbWhenChosen(cat("Zero probability case_dProbability_dFusedParameters c=",c)));
				}
				if (Cho[i]==0) {
					//continue;
				}
			}
		}
		
		// scratch = dUtil[down] - dUtil[up]
		cblas_dcopy(nPar, dUtil.ptr(i), 1, scratch, 1);
		cblas_daxpy(nPar, -1, dUtil.ptr((*Xylem)[i]->upcell(u)->slot()), 1, scratch, 1);

		// adjust Mu for hierarchical structure
		scratch[(*Xylem)[i]->upcell(u)->mu_offset()+nCA+nCO] += (Util[(*Xylem)[i]->upcell(u)->slot()]
														- Util[i] 
														) / (*Xylem)[i]->upcell(u)->mu();
		
		
		// scratch *= Pr[up]/mu[up]
		cblas_dscal(nPar, Pr[(*Xylem)[i]->upcell(u)->slot()]/(*Xylem)[i]->upcell(u)->mu(), scratch, 1);
		
		// scratch += dProb[up]
		cblas_daxpy(nPar, 1.0, dProb.ptr((*Xylem)[i]->upcell(u)->slot()), 1, scratch, 1);
		
		// dProb += scratch * CPr
		cblas_daxpy(nPar, CPr[(*Xylem)[i]->upedge(u)->edge_slot()], scratch, 1, dProb.ptr(i), 1);
		
	}
}




void elm::workshop_nl_gradient::case_dSamplingFactor_dFusedParameters( const unsigned& c )
{
	
	size_t nalt = _Xylem->n_elemental();
	const double*	   Pr = _Probability->ptr(c);
	elm::darray_ptr      Data_CA = SampPacket.Data_CA;
	elm::darray_ptr      Data_CO = SampPacket.Data_CO;

	for (int a=0; a<nalt; a++) {
		if (!Pr[a]) continue;
		
		// First, we calculate the effect of various parameters on the utility
		// of 'a' directly. For elemental alternatives, this means beta, gamma,
		// and theta parameters. For other nodes, only mu has a direct effect
		if (nSA) Data_CA->ExportData(dSampWgt.ptr(a),c,a,nalt);
		if (nSO) Data_CO->ExportData(dSampWgt.ptr(a)+nSA,c,a,nalt);
				
	}

}

void elm::workshop_nl_gradient::case_dAdjProbability_dFusedParameters( const unsigned& c )
{

	size_t nalt = _Xylem->n_elemental();

	const double* dTerm_dParam = dSampWgt.ptr();
	const double* OldP = _Probability->ptr(c);
	const double* NewP = _AdjProbability->ptr(c);

	dAdjProb.initialize();
	if (dSampWgt.size()) {
		for (int a=0; a<nalt; a++) {
			cblas_dcopy(dSampWgt.size2(), dSampWgt.ptr(a), 1, dAdjProb.ptr(a,nCA+nCO+nMU), 1);
		}
	}
	
	
	for (int a=0; a<nalt; a++) {
		if (OldP[a]) {
			cblas_daxpy(nPar, 1/OldP[a], dProb.ptr(a), 1, dAdjProb.ptr(a), 1);
		}
	}


	//Workspace.initialize();

	cblas_dgemv(CblasRowMajor, CblasTrans, nalt, nPar, 1, dAdjProb.ptr(), nPar, NewP, 1, 0, Workspace.ptr(), 1);

	for (int a=0; a<nalt; a++) {
		cblas_daxpy(nPar, -1, Workspace.ptr(), 1, dAdjProb.ptr(a), 1);
		cblas_dscal(nPar, NewP[a], dAdjProb.ptr(a), 1);
	}	



}





void elm::workshop_nl_gradient::case_dLogLike_dFusedParameters( const unsigned& c )
{
	double*            dLL = GradT_Fused.ptr();    // [params]
	const double*      Pr  = _Probability->ptr(c);   // [nodes]
	const double*	   Cho = Data_Choice->values(c);
	const unsigned     nA  = _Xylem->n_elemental();    // number of elementals
	etk::memarray_raw* dPr = &dProb;

	if (SampPacket.relevant()) {
		Pr = _AdjProbability->ptr(c);
		dPr = &dAdjProb;
	}

	unsigned a;
	for (a=0; a<nA; a++) {
		if (Cho[a]) {
			if (Pr[a]) {		
				cblas_daxpy(nPar, -Cho[a]/Pr[a], dPr->ptr(a), 1, dLL, 1);
			} else {
				std::ostringstream err;
				for (unsigned aa=0; aa<nA; aa++) {
					err << aa << "(ch=" << Cho[aa]<<")pr="<<Pr[aa]<<",";
				}
				throw(ZeroProbWhenChosen(cat("Zero probability case_dLogLike_dFusedParameters c=",c,"\n",err.str())));
			}
		}
	}
	
//	if (c==0) BUGGER_(msg_, " dProb:" << dProb.printall() << "\n dAdjProb"<<dAdjProb.printall());
}


void elm::__casewise_nl_dUtility_dParameters
( const double*	     Pr      // node probability, in [N_Nodes] space
, const double*      CPr 	 // conditional probability, in [N_Edges] space
, const double*	     Util 	 // scale-free utility, in [N_Nodes] space 
, const unsigned&    c 		 // Case Index Number
, const VAS_System&  Xylem
, elm::darray_ptr      Data_UtilityCA
, elm::darray_ptr      Data_UtilityCO
, memarray_raw&          dUtilCA
, memarray_raw&          dUtilCO
, memarray_raw&          dUtilMU
)
{
	dUtilCA.initialize(0.0);
	dUtilCO.initialize(0.0);
	dUtilMU.initialize(0.0);
	// for this case, this will be the derivative of utility at
	//  each node w.r.t. params [N_Nodes, N_Params]
	
	unsigned a,u;
	
	for (a=0; a<Xylem.size()-1; a++) {
		// 'a' is iterated over all the relevant nodes in the network
		//  the last node is not relevant, as it is the root node and has no predecessors

		if (!Pr[a]) continue;
		
		// First, we calculate the effect of various parameters on the utility
		// of 'a' directly. For elemental alternatives, this means beta, gamma,
		// and theta parameters. For other nodes, only mu has a direct effect
		if (a<Xylem.n_elemental()) {
			// BETA for SELF (elemental alternatives)
			if (dUtilCA.size()) Data_UtilityCA->ExportData(dUtilCA.ptr(a),c,a,Data_UtilityCA->nAlts());
			if (dUtilCO.size()) Data_UtilityCO->ExportData(dUtilCO.ptr(a),c,a,Xylem.n_elemental());
		} else {
			// MU for SELF (adjust the kiddies contributions) /////HERE
			dUtilMU(a,a-Xylem.n_elemental()) += Util[a];
			dUtilMU(a,a-Xylem.n_elemental()) /= Xylem[a]->mu();
		}
		
		u=0; 
		
		// MU for Parent (non-competitive edge)
		dUtilMU(Xylem[a]->upcell(u)->slot(),Xylem[a]->upcell(u)->mu_offset()) -= 
			CPr[Xylem[a]->upedge(u)->edge_slot()] * Util[a];
		
		// Finally, roll up secondary effects on parents
		if (CPr[Xylem[a]->upedge(u)->edge_slot()]) {
			if (dUtilCA.size()) cblas_daxpy(dUtilCA.size2(),CPr[Xylem[a]->upedge(u)->edge_slot()],dUtilCA.ptr(a),1,dUtilCA.ptr(Xylem[a]->upcell(u)->slot()),1);
			if (dUtilCO.size()) cblas_daxpy(dUtilCO.size2(),CPr[Xylem[a]->upedge(u)->edge_slot()],dUtilCO.ptr(a),1,dUtilCO.ptr(Xylem[a]->upcell(u)->slot()),1);
			if (dUtilMU.size()) cblas_daxpy(dUtilMU.size2(),CPr[Xylem[a]->upedge(u)->edge_slot()],dUtilMU.ptr(a),1,dUtilMU.ptr(Xylem[a]->upcell(u)->slot()),1);
		}
		
	}
}





void elm::__casewise_nl_dProb_dParam
( memarray_raw&          dProbCA   // [nodes,params]
, memarray_raw&          dProbCO   // [nodes,params]
, memarray_raw&          dProbMU   // [nodes,params]
, const VAS_System&  Xylem     // 
, const memarray_raw&    dUtilCA   // [nodes,params]
, const memarray_raw&    dUtilCO   // [nodes,params]
, const memarray_raw&    dUtilMU   // [nodes,params]
, const double*      Util      // [nodes]
, const double*      CPr       // [edges]
, const double*      Pr        // [nodes]
, double*            scratchCA   // [params]
, double*            scratchCO   // [params]
, double*            scratchMU   // [params]
, const double*      Cho       // [elementals]
)
{
	
	dProbCA.initialize(0.0);
	dProbCO.initialize(0.0);
	dProbMU.initialize(0.0);
		
	unsigned nPCA = dProbCA.size2();
	unsigned nPCO = dProbCO.size2();
	unsigned nPMU = dProbMU.size2();
	
	unsigned i,u;
	for (i=Xylem.size()-1; i!=0; ) {
		i--;
		u=0;
		
		if (i<Xylem.n_elemental()) {
			if (Cho) {
				if ((Pr[i]==0)&&(Cho[i]>0)) {
					throw(ZeroProbWhenChosen("Zero probability __casewise_nl_dProb_dParam"));
				}
				if (Cho[i]==0) {
					//continue;
				}
			}
		}
		
		// scratch = dUtil[down] - dUtil[up]
		if (nPCA){
		cblas_dcopy(nPCA, dUtilCA.ptr(i), 1, scratchCA, 1);
		cblas_daxpy(nPCA, -1, dUtilCA.ptr(Xylem[i]->upcell(u)->slot()), 1, scratchCA, 1);
		}
		if (nPCO){
		cblas_dcopy(nPCO, dUtilCO.ptr(i), 1, scratchCO, 1);
		cblas_daxpy(nPCO, -1, dUtilCO.ptr(Xylem[i]->upcell(u)->slot()), 1, scratchCO, 1);
		}
		if (nPMU){
		cblas_dcopy(nPMU, dUtilMU.ptr(i), 1, scratchMU, 1);
		cblas_daxpy(nPMU, -1, dUtilMU.ptr(Xylem[i]->upcell(u)->slot()), 1, scratchMU, 1);
		}
		// adjust Mu for hierarchical structure
		scratchMU[Xylem[i]->upcell(u)->mu_offset()] += (Util[Xylem[i]->upcell(u)->slot()] 
														- Util[i] 
														) / Xylem[i]->upcell(u)->mu();
		
		
		// scratch *= Pr[up]/mu[up]
		if (nPCA) cblas_dscal(nPCA, Pr[Xylem[i]->upcell(u)->slot()]/Xylem[i]->upcell(u)->mu(), scratchCA, 1);
		if (nPCO) cblas_dscal(nPCO, Pr[Xylem[i]->upcell(u)->slot()]/Xylem[i]->upcell(u)->mu(), scratchCO, 1);
		if (nPMU) cblas_dscal(nPMU, Pr[Xylem[i]->upcell(u)->slot()]/Xylem[i]->upcell(u)->mu(), scratchMU, 1);
		
		// scratch += dProb[up]
		if (nPCA) cblas_daxpy(nPCA, 1.0, dProbCA.ptr(Xylem[i]->upcell(u)->slot()), 1, scratchCA, 1);
		if (nPCO) cblas_daxpy(nPCO, 1.0, dProbCO.ptr(Xylem[i]->upcell(u)->slot()), 1, scratchCO, 1);
		if (nPMU) cblas_daxpy(nPMU, 1.0, dProbMU.ptr(Xylem[i]->upcell(u)->slot()), 1, scratchMU, 1);
		
		// dProb += scratch * CPr
		if (nPCA) cblas_daxpy(nPCA, CPr[Xylem[i]->upedge(u)->edge_slot()], scratchCA, 1, dProbCA.ptr(i), 1);
		if (nPCO) cblas_daxpy(nPCO, CPr[Xylem[i]->upedge(u)->edge_slot()], scratchCO, 1, dProbCO.ptr(i), 1);
		if (nPMU) cblas_daxpy(nPMU, CPr[Xylem[i]->upedge(u)->edge_slot()], scratchMU, 1, dProbMU.ptr(i), 1);
		
	}
}



/////





/////



void elm::__casewise_dLogLike_dParameters
( double*            dLL     // [params]
, const memarray_raw&    dProb   // [nodes, params]
, const double*      Pr      // [nodes]
, const double*      Cho     // [elementals]
, const unsigned&    nA      // number of elementals
, const unsigned&    nP      // number of parameters
)
{
	unsigned a;
	for (a=0; a<nA; a++) {
		if (Cho[a]) {
			if (Pr[a]) {		
				cblas_daxpy(nP, -Cho[a]/Pr[a], dProb.ptr(a), 1, dLL, 1);
			} else {
				throw(ZeroProbWhenChosen("Zero probability __casewise_dLogLike_dParameters"));
			}
		}
	}
}


void elm::__casewise_nl_gradient
( const unsigned& c
, const memarray* Probability
, const memarray* Cond_Prob
, const memarray* Utility
, const VAS_System* Xylem
, elm::darray_ptr  Data_UtilityCA
, elm::darray_ptr  Data_UtilityCO
, elm::darray_ptr  Data_Choice
, memarray_raw& dUtilCA
, memarray_raw& dUtilCO
, memarray_raw& dUtilMU
, memarray_raw& dProbCA
, memarray_raw& dProbCO
, memarray_raw& dProbMU
, memarray_raw& WorkspaceCA
, memarray_raw& WorkspaceCO
, memarray_raw& WorkspaceMU
, memarray_raw& dLLCA
, memarray_raw& dLLCO
, memarray_raw& dLLMU
)
{
	try {
	
	__casewise_nl_dUtility_dParameters
	( Probability->ptr(c)
	, Cond_Prob->ptr(c)
	, Utility->ptr(c) 
	, c 		 
	, *Xylem
	, Data_UtilityCA
	, Data_UtilityCO
	, dUtilCA
	, dUtilCO
	, dUtilMU
	);

//	if (c==0) {
//	BUGGER(*msg) << "dUtilCA\n" << dUtilCA.printall();
//	BUGGER(*msg) << "dUtilCO\n" << dUtilCO.printall();
//	BUGGER(*msg) << "dUtilMU\n" << dUtilMU.printall();
//	}
	
	__casewise_nl_dProb_dParam
	( dProbCA
	, dProbCO
	, dProbMU
	, *Xylem
	, dUtilCA
	, dUtilCO
	, dUtilMU
	, Utility->ptr(c)
	, Cond_Prob->ptr(c)
	, Probability->ptr(c)
	, *WorkspaceCA
	, *WorkspaceCO
	, *WorkspaceMU	 
	, Data_Choice->values(c)
	);

//	if (c==0) {
//	BUGGER(*msg) << "dProbCA\n" << dProbCA.printall();
//	BUGGER(*msg) << "dProbCO\n" << dProbCO.printall();
//	BUGGER(*msg) << "dProbMU\n" << dProbMU.printall();
//	}
	
	dLLCA.initialize();
	dLLCO.initialize();
	dLLMU.initialize();
	
	
	if (dProbCA.size())
	__casewise_dLogLike_dParameters
	( *dLLCA                 // [params]
	, dProbCA                // [nodes, params]
	, Probability->ptr(c)     // [nodes]
	, Data_Choice->values(c) // [elementals]
	, Xylem->n_elemental()    // number of elementals
	, dProbCA.size2()        // number of parameters
	);
	
	if (dProbCO.size())
		__casewise_dLogLike_dParameters
	( *dLLCO                 // [params]
	, dProbCO                // [nodes, params]
	, Probability->ptr(c)     // [nodes]
	, Data_Choice->values(c) // [elementals]
	, Xylem->n_elemental()    // number of elementals
	, dProbCO.size2()        // number of parameters
	);

	if (dProbMU.size())
	__casewise_dLogLike_dParameters
	( *dLLMU                 // [params]
	, dProbMU                // [nodes, params]
	, Probability->ptr(c)     // [nodes]
	, Data_Choice->values(c) // [elementals]
	, Xylem->n_elemental()    // number of elementals
	, dProbMU.size2()        // number of parameters
	);

	} catch (ZeroProbWhenChosen) {
		std::string prob_row = Probability->printrow(c);
		std::string choice_row = Data_Choice->printcase(c);
		
//		FATAL(*msg)
//			<< "\n\nProbability\n"<< Probability.printrow(c)
//			<< "\n\nChoices\n"<< Data_Choice->printcase(c);
	
		OOPS_ZEROPROB( "zero probability found for a chosen alternative in case ", c, "/n",prob_row, "/n",choice_row);
		
	}
}



void __casewise_nl_gradient_with_samp
( const unsigned& c
, const memarray* AdjProbability
, const memarray* Probability
, const memarray* Cond_Prob
, const memarray* Utility
, const VAS_System* Xylem
, elm::darray_ptr  Data_UtilityCA
, elm::darray_ptr  Data_UtilityCO
, elm::darray_ptr  Data_Choice
, memarray_raw& dUtilCA
, memarray_raw& dUtilCO
, memarray_raw& dUtilMU
, memarray_raw& dProbCA
, memarray_raw& dProbCO
, memarray_raw& dProbMU
, memarray_raw& dAdjProbCA
, memarray_raw& dAdjProbCO
, memarray_raw& dAdjProbMU
, memarray_raw& WorkspaceCA
, memarray_raw& WorkspaceCO
, memarray_raw& WorkspaceMU
, memarray_raw& dLLCA
, memarray_raw& dLLCO
, memarray_raw& dLLMU
, memarray_raw& dLLSampCA
, memarray_raw& dLLSampCO
, memarray_raw& dSampCA
, memarray_raw& dSampCO
, memarray_raw& dSampAdjProbCA
, memarray_raw& dSampAdjProbCO
, elm::ca_co_packet& SampPK
)
{
	try {
	__casewise_nl_dUtility_dParameters
	( Probability->ptr(c)
	, Cond_Prob->ptr(c)
	, Utility->ptr(c) 
	, c 		 
	, *Xylem
	, Data_UtilityCA
	, Data_UtilityCO
	, dUtilCA
	, dUtilCO
	, dUtilMU
	);

//	if (c==0) {
//	BUGGER(*msg) << "dUtilCA\n" << dUtilCA.printall();
//	BUGGER(*msg) << "dUtilCO\n" << dUtilCO.printall();
//	BUGGER(*msg) << "dUtilMU\n" << dUtilMU.printall();
//	}
	
	__casewise_nl_dProb_dParam
	( dProbCA
	, dProbCO
	, dProbMU
	, *Xylem
	, dUtilCA
	, dUtilCO
	, dUtilMU
	, Utility->ptr(c)
	, Cond_Prob->ptr(c)
	, Probability->ptr(c)
	, *WorkspaceCA
	, *WorkspaceCO
	, *WorkspaceMU	 
	, Data_Choice->values(c)
	);

//	if (c==0) {
//	BUGGER(*msg) << "dProbCA\n" << dProbCA.printall();
//	BUGGER(*msg) << "dProbCO\n" << dProbCO.printall();
//	BUGGER(*msg) << "dProbMU\n" << dProbMU.printall();
//	}
	
	SampPK.logit_partial_deriv(c, &dSampCA, &dSampCO);
		
	elm::case_logit_add_term_deriv
	(SampPK.nAlt(),
	 dSampCA.size2(),
	 NULL,                 //[ nAlts, nParams ]
	 dSampCA.ptr(),        //[ nAlts, nParams ]
	 dSampAdjProbCA.ptr(), //[ nAlts, nParams ]
	 *WorkspaceCA,         //[ nParams ]
	 Probability->ptr(c),  //[ nAlts ]
	 AdjProbability->ptr(c)//[ nAlts ]
	 );
	elm::case_logit_add_term_deriv
	(SampPK.nAlt(),
	 dSampCO.size2(),
	 NULL,                 //[ nAlts, nParams ]
	 dSampCO.ptr(),        //[ nAlts, nParams ]
	 dSampAdjProbCO.ptr(), //[ nAlts, nParams ]
	 *WorkspaceCO,         //[ nParams ]
	 Probability->ptr(c),  //[ nAlts ]
	 AdjProbability->ptr(c)//[ nAlts ]
	 );
	elm::case_logit_add_term_deriv
	(SampPK.nAlt(),
	 dProbCA.size2(),
	 NULL,                 //[ nAlts, nParams ]
	 dProbCA.ptr(),        //[ nAlts, nParams ]
	 dAdjProbCA.ptr(),     //[ nAlts, nParams ]
	 *WorkspaceCA,         //[ nParams ]
	 Probability->ptr(c),  //[ nAlts ]
	 AdjProbability->ptr(c)//[ nAlts ]
	 );
	elm::case_logit_add_term_deriv
	(SampPK.nAlt(),
	 dProbCO.size2(),
	 NULL,                 //[ nAlts, nParams ]
	 dProbCO.ptr(),        //[ nAlts, nParams ]
	 dAdjProbCO.ptr(),     //[ nAlts, nParams ]
	 *WorkspaceCO,         //[ nParams ]
	 Probability->ptr(c),  //[ nAlts ]
	 AdjProbability->ptr(c)//[ nAlts ]
	 );
	elm::case_logit_add_term_deriv
	(SampPK.nAlt(),
	 dProbMU.size2(),
	 NULL,                 //[ nAlts, nParams ]
	 dProbMU.ptr(),        //[ nAlts, nParams ]
	 dAdjProbMU.ptr(),     //[ nAlts, nParams ]
	 *WorkspaceMU,         //[ nParams ]
	 Probability->ptr(c),  //[ nAlts ]
	 AdjProbability->ptr(c)//[ nAlts ]
	 );
	
	
	
	dLLCA.initialize();
	dLLCO.initialize();
	dLLMU.initialize();
	dLLSampCA.initialize();
	dLLSampCO.initialize();
	
	
	if (dAdjProbCA.size())
	__casewise_dLogLike_dParameters
	( *dLLCA                 // [params]
	, dAdjProbCA                // [nodes, params]
	, Probability->ptr(c)     // [nodes]
	, Data_Choice->values(c) // [elementals]
	, Xylem->n_elemental()    // number of elementals
	, dAdjProbCA.size2()        // number of parameters
	);
	
	if (dAdjProbCO.size())
	__casewise_dLogLike_dParameters
	( *dLLCO                 // [params]
	, dAdjProbCO                // [nodes, params]
	, Probability->ptr(c)     // [nodes]
	, Data_Choice->values(c) // [elementals]
	, Xylem->n_elemental()    // number of elementals
	, dAdjProbCO.size2()        // number of parameters
	);

	if (dAdjProbMU.size())
	__casewise_dLogLike_dParameters
	( *dLLMU                 // [params]
	, dAdjProbMU                // [nodes, params]
	, Probability->ptr(c)     // [nodes]
	, Data_Choice->values(c) // [elementals]
	, Xylem->n_elemental()    // number of elementals
	, dAdjProbMU.size2()        // number of parameters
	);





	if (dSampAdjProbCA.size())
	__casewise_dLogLike_dParameters
	( *dLLSampCA                 // [params]
	, dSampAdjProbCA                // [nodes, params]
	, Probability->ptr(c)     // [nodes]
	, Data_Choice->values(c) // [elementals]
	, Xylem->n_elemental()    // number of elementals
	, dSampAdjProbCA.size2()        // number of parameters
	);
	
	if (dSampAdjProbCO.size())
	__casewise_dLogLike_dParameters
	( *dLLSampCO                 // [params]
	, dSampAdjProbCO                // [nodes, params]
	, Probability->ptr(c)     // [nodes]
	, Data_Choice->values(c) // [elementals]
	, Xylem->n_elemental()    // number of elementals
	, dSampAdjProbCO.size2()        // number of parameters
	);





	} catch (ZeroProbWhenChosen) {
		std::string prob_row = Probability->printrow(c);
		std::string choice_row = Data_Choice->printcase(c);
		
//		FATAL(*msg)
//			<< "\n\nProbability\n"<< Probability.printrow(c)
//			<< "\n\nChoices\n"<< Data_Choice->printcase(c);
	
		OOPS_ZEROPROB( "zero probability found for a chosen alternative in case ", c, "/n",prob_row, "/n",choice_row);
		
	}
}

#define BIGGER(x,y) (x)>(y) ? (x) : (y)

elm::workshop_nl_gradient::workshop_nl_gradient
( const unsigned&   dF
 , const unsigned&   nNodes
 , elm::ca_co_packet UtilPK
 , elm::ca_co_packet SampPK
 , const paramArray& Params_LogSum
 , elm::darray_ptr     Data_Choice
 , elm::darray_ptr     Data_Weight
 , const etk::memarray* AdjProbability
 , const etk::memarray* Probability
 , const etk::memarray* Cond_Prob
 , const VAS_System* Xylem
 , etk::memarray* GCurrent
 , etk::symmetric_matrix* Bhhh
 , etk::logging_service* msgr
)
: dF         (dF)
, nNodes     (nNodes)
, nCA (UtilPK.Params_CA->length())
, nCO (UtilPK.Params_CO->length())
, nMU (Params_LogSum.length())
, nSA (SampPK.Params_CA->length())
, nSO (SampPK.Params_CO->length())
, nPar(nCA+nCO+nMU+nSA+nSO)
, UtilPacket (UtilPK)
, SampPacket (SampPK)
, dUtil      (nNodes,nPar)
, dProb      (nNodes,nPar)
, dSampWgt   (nNodes,nSA+nSO)
, dAdjProb   (nNodes,nPar)
, Workspace       (nPar)
, GradT_Fused     (nPar)
, CaseGrad        (dF)
, workshopBHHH    (dF,dF)
, workshopGCurrent(dF)
, Params_LogSum   (&Params_LogSum)
, Data_Choice     (Data_Choice)
, Data_Weight     (Data_Weight)
, _AdjProbability(AdjProbability )
, _Probability(Probability )
, _Cond_Prob  ( Cond_Prob)
, _Xylem     (Xylem)
, _GCurrent(GCurrent)
, _Bhhh(Bhhh)
, _lock(nullptr)
, msg_ (msgr)
{
}




elm::workshop_nl_gradient::~workshop_nl_gradient()
{
}


void elm::workshop_nl_gradient::workshop_nl_gradient_do(
  const unsigned& firstcase, const unsigned& numberofcases)
{
//	BUGGER_(msg_, "Beginning NL gradient calculation ["<<firstcase<<"]-["<<firstcase+numberofcases-1<<"]");

//	UtilPacket.Data_CA->load_values(firstcase,numberofcases);
//	UtilPacket.Data_CO->load_values(firstcase,numberofcases);
//	Data_Choice   ->load_values(firstcase,numberofcases);
//	if (Data_Weight) {
//		Data_Weight->load_values(firstcase,numberofcases);
//	}

	unsigned lastcase = firstcase+numberofcases;

	unsigned c;
	workshopBHHH.initialize();
	workshopGCurrent.initialize();
	
//	BUGGER_(msg_, "in NL gradient, sampling bias is "<< (SampPacket.relevant()? "" : "not ")<< "relevant");

	for (c=firstcase;c<lastcase;c++) {
		//BUGGER_(msg_, "NL grad ["<<c<<"]");
		
		
		CaseGrad.initialize();
		GradT_Fused.initialize();

		case_dUtility_dFusedParameters(c);
		case_dProbability_dFusedParameters(c);
		
		if (SampPacket.relevant()) {
			case_dSamplingFactor_dFusedParameters(c);
			case_dAdjProbability_dFusedParameters(c);
		}
		
		case_dLogLike_dFusedParameters(c);
		
		
		if (Data_Weight) {
			GradT_Fused.scale(Data_Weight->value(c, 0));
		}
				
		elm::push_to_freedoms2(*UtilPacket.Params_CA  , *GradT_Fused          , *CaseGrad);
		elm::push_to_freedoms2(*UtilPacket.Params_CO  , (*GradT_Fused)+nCA    , *CaseGrad);
		elm::push_to_freedoms2(*Params_LogSum         , (*GradT_Fused)+nCA+nCO, *CaseGrad);
		elm::push_to_freedoms2(*SampPacket.Params_CA  , (*GradT_Fused)+nCA+nCO+nMU    , *CaseGrad);
		elm::push_to_freedoms2(*SampPacket.Params_CO  , (*GradT_Fused)+nCA+nCO+nMU+nSA, *CaseGrad);

		
		// BHHH
		#ifdef SYMMETRIC_PACKED
		cblas_dspr(CblasRowMajor,CblasUpper, dF,1,*CaseGrad, 1, *workshopBHHH);
		#else
		cblas_dsyr(CblasRowMajor,CblasUpper, dF,1,*CaseGrad, 1, *workshopBHHH, workshopBHHH.size1());
		#endif
		
		// ACCUMULATE
		workshopGCurrent += CaseGrad;
		
	}
	//BUGGER_(msg_, "Finished NL gradient calculation ["<<firstcase<<"]-["<<firstcase+numberofcases-1<<"]");

}

void elm::workshop_nl_gradient::workshop_nl_gradient_send
()
{
	if (_lock) {
		std::lock_guard<std::mutex> lock_while_in_shope(*_lock);
		*_GCurrent += workshopGCurrent;
		*_Bhhh += workshopBHHH;
	} else {
		OOPS("No lock in elm::workshop_nl_gradient::workshop_nl_gradient_send");
	}
}




void elm::workshop_nl_gradient::work(size_t firstcase, size_t numberofcases, boosted::mutex* result_mutex)
{
	workshop_nl_gradient_do(firstcase,numberofcases);
	_lock = result_mutex;
	workshop_nl_gradient_send();
}

