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

#include "elm_workshop_ngev_gradient.h"

using namespace etk;
using namespace elm;
using namespace std;



void elm::workshop_ngev_gradient::case_dUtility_dFusedParameters( const unsigned& c )
{
//	if (c==0) BUGGER_(msg_, "case_dUtility_dFusedParameters()" )  ;

	const double*	   Pr = _Probability->ptr(c);
	const double*	   Qnt = _Quantity->size() ? _Quantity->ptr(c) : nullptr;
	const double*      CPr = _Cond_Prob->ptr(c);
	const double*	   Util = UtilPacket.Outcome->ptr(c);
	const double*	   Alloc = AllocPacket.relevant()? AllocPacket.Outcome->ptr(c) : nullptr;
	const VAS_System*  Xylem = _Xylem;

	dUtil.initialize(0.0);
	// for this case, this will be the derivative of utility at
	//  each node w.r.t. params [N_Nodes, N_Params]
	
	unsigned a,u,ou;
	size_t Offset_Phi = offset_alloc();
	
	for (a=0; a<Xylem->size()-1; a++) {
		// 'a' is iterated over all the relevant nodes in the network
		//  the last node is not relevant, as it is the root node and has no predecessors

		if (!Pr[a]) continue;
		
		auto xylem_a = (*Xylem)[a];
		
		// First, we calculate the effect of various parameters on the utility
		// of 'a' directly. For elemental alternatives, this means beta, gamma,
		// and theta parameters. For other nodes, only mu has a direct effect
		if (a<Xylem->n_elemental()) {
			// BETA for SELF (elemental alternatives)
			if (nCA) UtilPacket.Data_CA->ExportData(dUtil.ptr(a)    ,c,a,UtilPacket.nAlt());
			if (nCO) UtilPacket.Data_CO->ExportData(dUtil.ptr(a)+nCA,c,a,UtilPacket.nAlt());

			// GAMMA on SELF
			if (nQA) {
				double* gg = dUtil.ptr(a)+offset_quant();
				QuantPacket.Data_CA->ExportData(gg,c,a,QuantPacket.Data_CA->nAlts());
				simple_inplace_element_multiply(nQA, **QuantPacket.Coef_CA, 1, gg, 1);
				if (nQL>0) {
					OOPS("Theta not implemented");
				}
				//(nQL>0?OOPS("Theta not implemented"):1.0)
				cblas_dscal(nQA,(1.0)/Qnt[a],gg,1);
				
				if (nQL) {
					// THETA on SELF
					*(dUtil.ptr(a)+offset_quant()+nQA) = log(Qnt[a]);
				}
				
			}

		} else {
			// MU for SELF (adjust the kiddies contributions) /////HERE
			dUtil(a,a-(Xylem->n_elemental())+offset_mu()) += Util[a];
			dUtil(a,a-(Xylem->n_elemental())+offset_mu()) /= xylem_a->mu();
		}

		size_t xylem_a_upsize = xylem_a->upsize();

		// Then, we compute the carry-on effects from 'a' on other nodes in the
		// network. For utility, this can only include direct predecessor nodes.
		for (u=0; u< xylem_a_upsize; u++) {
		
			auto xylem_a_upcell_u = xylem_a->upcell(u);
			auto xylem_a_upedge_u = xylem_a->upedge(u);
		
			// define attributes of current edge
			size_t upcell_slot = xylem_a_upcell_u->slot();
			size_t slot_of_this_upedge = xylem_a_upedge_u->edge_slot();

//			if (c==0) BUGGER_(msg_, "gradx upslot "<<u<<", nodeslot "<<a<<", up_nodeslot "<<upcell_slot<<", mu_offset="<<xylem_a_upcell_u->mu_offset() )  ;
			
//			if (c==0) BUGGER_(msg_, "gradxx "<<bool(Alloc)<<","<<xylem_a_upedge_u->is_competitive() )  ;
			
			if (Alloc && (xylem_a_upsize>1)) {
				// When this edge is competitive
				
				
				
				// PHI for all competing edges
				for (ou=0; ou<xylem_a_upsize; ou++) {
//					if (c==0) BUGGER_(msg_, "dU+ "<<dUtil.printSize()<<"\n" << dUtil.printall())  ;
					
					unsigned slot_competing_edges = xylem_a->upedge(ou)->alloc_slot();
//					if (c==0) BUGGER_(msg_, "slot_competing_edges "<<slot_competing_edges<<" for number " << ou << ", slot_of_this_upedge="<<slot_of_this_upedge)  ;
					
					AllocPacket.Data_CO->OverlayData(dUtil.ptr(upcell_slot,Offset_Phi),
									  c,
									  slot_competing_edges,
									  -CPr[slot_of_this_upedge]*Alloc[slot_competing_edges], xylem_a_upsize);
//					if (c==0) BUGGER_(msg_, "dU!\n" << dUtil.printall())  ;
				}
				// PHI for this edge, adjustment
				AllocPacket.Data_CO->OverlayData(dUtil.ptr(upcell_slot,Offset_Phi),
								  c,
								  xylem_a_upedge_u->alloc_slot(),
								  CPr[slot_of_this_upedge], xylem_a_upsize);
				
				// MU for Parent (competitive edge)
				dUtil(upcell_slot,xylem_a_upcell_u->mu_offset()+offset_mu()) -=
				CPr[slot_of_this_upedge] * (Util[a] + log(Alloc[xylem_a_upedge_u->alloc_slot()]));
				
			} else {
				// When this edge is not competitive
				
				// MU for Parent (non-competitive edge)
				dUtil(upcell_slot,xylem_a_upcell_u->mu_offset()+offset_mu()) -=
				CPr[slot_of_this_upedge] * Util[a];
			}
			
			// Finally, roll up secondary effects on parents
			if (CPr[slot_of_this_upedge])
				cblas_daxpy(nPar,CPr[slot_of_this_upedge],dUtil.ptr(a),1,dUtil.ptr(upcell_slot),1);
		}

//		if (c==0) {
//		BUGGER_(msg_, "dU[at "<<a<<"]-> "<<dUtil.printSize()<<"\n" << dUtil.printall())  ;
//		}

	}
//	if (c==0) {
//	BUGGER_(msg_, "_Cond_Prob-> "<< _Cond_Prob->printrow(c))  ;
//	BUGGER_(msg_, "dU-> "<<dUtil.printSize()<<"\n" << dUtil.printall())  ;
//	}
}


void elm::workshop_ngev_gradient::case_dProbability_dFusedParameters( const unsigned& c )
{

	const double*	   Pr = _Probability->ptr(c);
	const double*      CPr = _Cond_Prob->ptr(c);
	const double*	   Util = UtilPacket.Outcome->ptr(c);
	const double*	   Alloc = AllocPacket.relevant()? AllocPacket.Outcome->ptr(c) : nullptr;
	const VAS_System*  Xylem = _Xylem;
//	datamatrix      Data_UtilityCA = UtilPacket.Data_CA;
//	datamatrix      Data_UtilityCO = UtilPacket.Data_CO;

	double*			   scratch = Workspace.ptr();
	const double*	   Cho = Data_Choice->values(c);
	
	dProb.initialize(0.0);
	size_t Offset_Phi = offset_alloc();

	unsigned i,u,ou;
	for (i=(*Xylem).size()-1; i!=0; ) {
		i--;
		size_t xylem_a_upsize = (*Xylem)[i]->upsize();
		for (u=0; u<xylem_a_upsize; u++) {
			
			auto xylem_i_upcell_u = (*Xylem)[i]->upcell(u);
			auto xylem_i_upcell_u_slot = xylem_i_upcell_u->slot();
			
			// if (c==0) std::cerr << "case_dProbability_dFusedParameters\n";
			// if (c==0) std::cerr << "i="<<i<<"\n";
			// if (c==0) std::cerr << "u="<<u<<"\n";
			// if (c==0) std::cerr << "xylem_i_upcell_u_slot="<<xylem_i_upcell_u_slot<<"\n";


			// scratch = dUtil[down] - dUtil[up]
			cblas_dcopy(nPar, dUtil.ptr(i), 1, scratch, 1);
			cblas_daxpy(nPar, -1, dUtil.ptr(xylem_i_upcell_u_slot), 1, scratch, 1);
			
			// for competitive edges, adjust phi
			// scratch += X_Phi()[edge] - sum over competes(Alloc[compete]*X_Phi[compete])
			if (Alloc && (*Xylem)[i]->upedge(u)->is_competitive()) {
				auto allocslot_upedge_u = (*Xylem)[i]->upedge(u)->alloc_slot();
				// if (c==0) std::cerr << "allocslot_upedge_u="<<allocslot_upedge_u<<"\n";

				// if (c==0) std::cerr << "WorkspaceX1 "<<Workspace.printSize()<<"\n" << Workspace.printall() ;
				
				AllocPacket.Data_CO->OverlayData(scratch+Offset_Phi, c, allocslot_upedge_u, 1.0, xylem_a_upsize);
				// if (c==0) std::cerr <<  "WorkspaceXa "<<Workspace.printSize()<<"\n" << Workspace.printall() ;
				for (ou=0; ou<xylem_a_upsize; ou++)  {
					auto slot = (*Xylem)[i]->upedge(ou)->alloc_slot();
					AllocPacket.Data_CO->OverlayData(scratch+Offset_Phi, c, slot, -Alloc[slot], xylem_a_upsize);
					// if (c==0) std::cerr <<  "WorkspaceXb "<<Workspace.printSize()<<"\n" << Workspace.printall()  ;
				}

				// if (c==0) std::cerr <<  "WorkspaceX2 "<<Workspace.printSize()<<"\n" << Workspace.printall() ;
				
				// adjust Mu for hierarchical structure, competitive
				scratch[xylem_i_upcell_u->mu_offset()+offset_mu()] += (Util[xylem_i_upcell_u_slot]
														   - Util[i] 
														   - log(Alloc[allocslot_upedge_u])
														   ) / xylem_i_upcell_u->mu();
				
			} else {
				
				// adjust Mu for hierarchical structure, noncompete
				scratch[xylem_i_upcell_u->mu_offset()+offset_mu()] += (Util[xylem_i_upcell_u_slot]
														   - Util[i] 
														   ) / xylem_i_upcell_u->mu();
			}
			
			// scratch *= Pr[up]/mu[up]
			cblas_dscal(nPar, Pr[xylem_i_upcell_u_slot]/xylem_i_upcell_u->mu(), scratch, 1);
			
			// scratch += dProb[up]
			cblas_daxpy(nPar, 1.0, dProb.ptr(xylem_i_upcell_u_slot), 1, scratch, 1);
			
			// dProb += scratch * CPr
			cblas_daxpy(nPar, CPr[(*Xylem)[i]->upedge(u)->edge_slot()], scratch, 1, dProb.ptr(i), 1);
		}
	}


//	if (c==0) BUGGER_(msg_, "dProb "<<dProb.printSize()<<"\n" << dProb.printall())  ;
	
}




void elm::workshop_ngev_gradient::case_dSamplingFactor_dFusedParameters( const unsigned& c )
{
	
	size_t nalt = _Xylem->n_elemental();
	const double*	   Pr = _Probability->ptr(c);
	elm::darray_ptr      Data_CA = SampPacket.Data_CA;
	elm::darray_ptr      Data_CO = SampPacket.Data_CO;

	for (int a=0; a<nalt; a++) {
		if (!Pr[a]) continue;
		
		if (nSA) Data_CA->ExportData(dSampWgt.ptr(a),c,a,nalt);
		if (nSO) Data_CO->ExportData(dSampWgt.ptr(a)+nSA,c,a,nalt);
				
	}

}

void elm::workshop_ngev_gradient::case_dAdjProbability_dFusedParameters( const unsigned& c )
{

	size_t nalt = _Xylem->n_elemental();

	const double* dTerm_dParam = dSampWgt.ptr();
	const double* OldP = _Probability->ptr(c);
	const double* NewP = _AdjProbability->ptr(c);

	dAdjProb.initialize();
	if (dSampWgt.size()) {
		for (int a=0; a<nalt; a++) {
			cblas_dcopy(dSampWgt.size2(), dSampWgt.ptr(a), 1, dAdjProb.ptr(a,offset_sampadj()), 1);
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





void elm::workshop_ngev_gradient::case_dLogLike_dFusedParameters( const unsigned& c )
{
	double*            dLL = GradT_Fused.ptr();    // [params]
	const double*      Pr  = _Probability->ptr(c);   // [nodes]
	const double*	   Cho = Data_Choice->values(c);
	const unsigned     nA  = _Xylem->n_elemental();    // number of elementals
	etk::memarray_raw* dPr = &dProb;

	if (SampPacket.relevant()) {
//		if (c==0) BUGGER_(msg_, "case_dLogLike_dFusedParameters->SampPacket.relevant")  ;
		Pr = _AdjProbability->ptr(c);
		dPr = &dAdjProb;
	} else {
//		if (c==0) BUGGER_(msg_, "case_dLogLike_dFusedParameters->SampPacket.NOTrelevant")  ;
	}

//	if (c==0) BUGGER_(msg_, "dPr-> "<<dPr->printSize()<<"\n" << dPr->printall())  ;

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
	
//	if (c==0) BUGGER_(msg_, " dProb:" << dProb.printall() << "\n dAdjProb"<<dAdjProb.printall()<<"\n GradT_Fused:\n"<<GradT_Fused.printall());
	
}










#define BIGGER(x,y) (x)>(y) ? (x) : (y)

elm::workshop_ngev_gradient::workshop_ngev_gradient
( const unsigned&   dF
 , const unsigned&   nNodes
 , elm::ca_co_packet UtilPK
 , elm::ca_co_packet AllocPK
 , elm::ca_co_packet SampPK
 , elm::ca_co_packet QuantPK
 , const paramArray& Params_LogSum
 , elm::darray_ptr     Data_Choice
 , elm::darray_ptr     Data_Weight
 , const etk::memarray* AdjProbability
 , const etk::memarray* Probability
 , const etk::memarray* Cond_Prob
 , const VAS_System* Xylem
 , etk::memarray* GCurrent
 , etk::ndarray*  GCurrentCasewise
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
, nAO (AllocPK.Params_CO->length())
, nQA (QuantPK.Params_CA->length())

, nQL (0)
, nPar(nCA+nCO+nMU+nSA+nSO+nAO+nQA+nQL)
, UtilPacket (UtilPK)
, AllocPacket(AllocPK)
, SampPacket (SampPK)
, QuantPacket (QuantPK)
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
, _Quantity   (QuantPK.Outcome  )
, _Cond_Prob  ( Cond_Prob)
, _Xylem     (Xylem)
, _GCurrent(GCurrent)
, _Bhhh(Bhhh)
, _GCurrentCasewise(GCurrentCasewise)
, _lock(nullptr)
, msg_ (msgr)
{
}




elm::workshop_ngev_gradient::~workshop_ngev_gradient()
{
}


void elm::workshop_ngev_gradient::workshop_ngev_gradient_do(
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
		
		
		CaseGrad.initialize();    // The CaseGrad holds the gradient for a single case on the freedoms
		GradT_Fused.initialize(); // GradT_Fused hold the gradient for this case on the 'parameters', which are
		                          // derived from the freedoms.

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
		elm::push_to_freedoms2(*Params_LogSum         , (*GradT_Fused)+offset_mu()         , *CaseGrad);
		elm::push_to_freedoms2(*SampPacket.Params_CA  , (*GradT_Fused)+offset_sampadj()    , *CaseGrad);
		elm::push_to_freedoms2(*SampPacket.Params_CO  , (*GradT_Fused)+offset_sampadj()+nSA, *CaseGrad);
		elm::push_to_freedoms2(*AllocPacket.Params_CO , (*GradT_Fused)+offset_alloc(), *CaseGrad);
		elm::push_to_freedoms2(*QuantPacket.Params_CA , (*GradT_Fused)+offset_quant(), *CaseGrad);

//		if (c==0) {
//			BUGGER_(msg_, "push list:\n"<<z);
//			BUGGER_(msg_, "AllocPacket.Params_CO.__str__():\n"<<AllocPacket.Params_CO->__str__());
//			BUGGER_(msg_, "CaseGrad [c==0]:\n"<<CaseGrad.printall());
//		}

		
		// BHHH
		#ifdef SYMMETRIC_PACKED
		cblas_dspr(CblasRowMajor,CblasUpper, dF,1,*CaseGrad, 1, *workshopBHHH);
		#else
		cblas_dsyr(CblasRowMajor,CblasUpper, dF,1,*CaseGrad, 1, *workshopBHHH, workshopBHHH.size1());
		#endif
		
		// ACCUMULATE
		workshopGCurrent += CaseGrad;
		
		if (_GCurrentCasewise) {
			cblas_dcopy(dF, *CaseGrad, 1, _GCurrentCasewise->ptr(c), 1);
		}
	}
	//BUGGER_(msg_, "Finished NL gradient calculation ["<<firstcase<<"]-["<<firstcase+numberofcases-1<<"]");

}

void elm::workshop_ngev_gradient::workshop_ngev_gradient_send
()
{
	if (_lock) {
		std::lock_guard<std::mutex> lock_while_in_shope(*_lock);
		*_GCurrent += workshopGCurrent;
		*_Bhhh += workshopBHHH;
	} else {
		OOPS("No lock in workshop_ngev_gradient_send");
	}
}



boosted::mutex workshop_ngev_display_mutex;




void elm::workshop_ngev_gradient::work(size_t firstcase, size_t numberofcases, boosted::mutex* result_mutex)
{
	workshop_ngev_gradient_do(firstcase,numberofcases);
	_lock = result_mutex;
	workshop_ngev_gradient_send();
}

