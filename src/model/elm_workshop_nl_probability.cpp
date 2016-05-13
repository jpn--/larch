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

#include "elm_workshop_nl_probability.h"

using namespace etk;
using namespace elm;
using namespace std;





void elm::__casewise_nl_utility 
( double* U		        // pointer to utility array [nN space]
, const VAS_System& Xy  // nesting structure
, double* Work	        // function workspace, [nN]
) 
{
	unsigned i; 
	double max = -INF;
	unsigned k;
	for (i=Xy.n_elemental(); i<Xy.size(); i++) {
		max = -INF;
		for (k=0; k<Xy[i]->dnsize(); k++) {
			Work[k] = U[Xy[i]->dncell(k)->slot()];
			if (Work[k] > max) max = Work[k];
		}			
		if (Xy[i]->mu() == 0) { // When Mu is zero, special calculation
			U[i] = max;
			//OOPS("error: zero value for Mu");
		} else { // Mu is not zero
			if (max == -INF) max = 0;
			max /= Xy[i]->mu();
			U[i] = 0;
			for (k=0; k<Xy[i]->dnsize(); k++) {
				if (Work[k] == -INF) continue;
				Work[k] /= Xy[i]->mu();
				Work[k] -= max;
				Work[k] = exp(Work[k]);
				U[i] += Work[k];
			}
			if (U[i]) {
				U[i] = log(U[i]);
				U[i] += max;
				U[i] *= Xy[i]->mu();
			} else {
				U[i] = -INF;
			}
		}
	}
}

void elm::__casewise_nl_probability
( double* U  			// pointer to utility array [nN space]
, double* CPr 		    // pointer to conditional probability
, double* Pr 		    // pointer to probability
, const VAS_System& Xy	// nesting structure
)
{
	unsigned i;
	unsigned u;
	unsigned nN = Xy.size();
	
	// Total Probability of the root
	Pr[nN-1] = 1.0;
	
	// Conditional and Total Probability of the Nodes
	for (i=nN-1; i!=0; ) {
		i--;
		Pr[i] = 0;
		u=0; 
		if (U[i]!=-INF) {
			if (Xy[i]->upcell(u)->mu() == 0) {
				if (U[i] == U[Xy[i]->upcell(u)->slot()]) {
					CPr[Xy[i]->upedge(u)->edge_slot()] = 1.0;
					// TODO: count number of other nodes with identical maximum utility (a pathological case)
				} else {
					CPr[Xy[i]->upedge(u)->edge_slot()] = 0.0;
				}
			} else {
				CPr[Xy[i]->upedge(u)->edge_slot()] = exp((U[i] - U[Xy[i]->upcell(u)->slot()]) / Xy[i]->upcell(u)->mu());
			}
		} else {
			CPr[Xy[i]->upedge(u)->edge_slot()] = 0.0;
		}
		Pr[i] += CPr[Xy[i]->upedge(u)->edge_slot()] * Pr[Xy[i]->upcell(u)->slot()];			
	}
}



void elm::workshop_nl_probability::case_logit_add_sampling(const unsigned& c)
{
	size_t n            = SampPacket.Outcome->size2();
	double* adjPr       = AdjProbability->ptr(c);
	const bool* av      = Data_Avail->boolvalues(c,1);
	
	double tot = 0.0;
	const double* origPr = Probability->ptr(c);
	const double* sampWt = SampPacket.Outcome->ptr(c);

	double* p = adjPr;
	for (size_t i=0; i<n; i++, origPr++, sampWt++, p++, av++) {
		if (*av) {
			*p = (*origPr) * ::exp(*sampWt);
			tot += *p;
		} else {
			*p = 0.0;
		}
	}
	
	p = adjPr;
	for (size_t i=0; i<n; i++, p++) {
		*p /= tot;
	}
}




elm::workshop_nl_probability::workshop_nl_probability
( const unsigned&   nNodes
, elm::ca_co_packet UtilPacket
, elm::ca_co_packet SampPacket
, const paramArray& Params_LogSum
, elm::darray_ptr     Data_Avail
,  ndarray* Probability
,  ndarray* Cond_Prob
,	etk::ndarray* AdjProbability
, const VAS_System* Xylem
, const bool& option_mute_nan_warnings
, etk::logging_service* msgr
)
: nNodes          (nNodes)
, UtilPacket      (UtilPacket)
, SampPacket      (SampPacket)
, Params_LogSum   (&Params_LogSum)
, Data_Avail      (Data_Avail)
, Probability     (Probability)
, Cond_Prob       (Cond_Prob)
, AdjProbability  (AdjProbability)
, Xylem           (Xylem)
, option_mute_nan_warnings (option_mute_nan_warnings)
, msg_            (msgr)
{
	Workspace.resize(nNodes);
}

elm::workshop_nl_probability::~workshop_nl_probability()
{
}


void elm::workshop_nl_probability::workshop_nl_probability_calc
( const unsigned&   firstcase
, const unsigned&   numberofcases
)
{
//	datamatrix Data_UtilityCA = UtilPacket.Data_CA;
//	datamatrix Data_UtilityCO = UtilPacket.Data_CO;
	etk::ndarray* Utility = UtilPacket.Outcome;
	const etk::ndarray* Coef_UtilityCA = UtilPacket.Coef_CA;
	const etk::ndarray* Coef_UtilityCO = UtilPacket.Coef_CO;

	elm::darray_ptr Data_SamplingCA = SampPacket.Data_CA;
	elm::darray_ptr Data_SamplingCO = SampPacket.Data_CO;
	etk::ndarray* SamplingWgt = SampPacket.Outcome;
	const etk::ndarray* Coef_SamplingCA = SampPacket.Coef_CA;
	const etk::ndarray* Coef_SamplingCO = SampPacket.Coef_CO;
	
	unsigned lastcase(firstcase+numberofcases);


	unsigned nElementals = Xylem->n_elemental();
	unsigned warningcount = 0;

	UtilPacket.logit_partial(firstcase, numberofcases);

	bool use_sampling = SampPacket.relevant();
	if (use_sampling) {
		SampPacket.logit_partial(firstcase, numberofcases);
	}
	
	for (unsigned c=firstcase;c<lastcase;c++) {
		// Unavailable alternatives become -INF
		for (unsigned a=0;a<nElementals;a++) {
			if (!Data_Avail->boolvalue(c,a)) {
				(*Utility)(c,a) = -INF;
			} 		
		}

		
		__casewise_nl_utility(Utility->ptr(c), *Xylem, *Workspace);
		__casewise_nl_probability(Utility->ptr(c), Cond_Prob->ptr(c), Probability->ptr(c), *Xylem);
		if (use_sampling) {
			case_logit_add_sampling(c);
		}
		
//		if (c==0) BUGGER_(msg_, "DEBUG REPORT: caserow "<<c << "\n"
//								<< "util: " << Utility->printrow(c)
//								<< "c|pr: " << Cond_Prob->printrow(c)
//								<< "prob: " << Probability->printrow(c)
//								<< "bias: " << SamplingWgt->printrow(c)
//								<< "adjp: " << AdjProbability->printrow(c)
//								<< "coef-bias-ca: " << SampPacket.Coef_CA->printall()
//								<< "coef-bias-co: " << SampPacket.Coef_CO->printall()
//		   );

		// NANCHECK
		if (!option_mute_nan_warnings) {
			bool found_nan = false;
			for (unsigned a=0; a<AdjProbability->size2();a++) {
				if (isNan((*AdjProbability)(c,a))) {
					found_nan = true;
					break;
				}
			}
			if (found_nan) {
				if (!warningcount) {
					WARN_(msg_, "WARNING: Probability is NAN for caserow "<<c
					  << "\n" << "W..util: " << Utility->printrow(c)
					  << "\n" << "W..c|pr: " << Cond_Prob->printrow(c)
					  << "\n" << "W..prob: " << Probability->printrow(c)
					  << "\n" << "W..adjp: " << AdjProbability->printrow(c)
					   );
				}
				warningcount++;
			}
		} // end if NANCHECK
		
	}

}

void elm::workshop_nl_probability::work(size_t firstcase, size_t numberofcases, boosted::mutex* result_mutex)
{
	workshop_nl_probability_calc(firstcase,numberofcases);
}


