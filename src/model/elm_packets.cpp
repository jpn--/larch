/*
 *  elm_packets.cpp
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


#include "elm_packets.h"
#include "elm_sql_scrape.h"
#include "elm_darray.h"


elm::ca_co_packet::ca_co_packet(
			 const paramArray*	Params_CA	,
			 const paramArray*	Params_CO	,
			 const etk::ndarray*	Coef_CA		,
			 const etk::ndarray*	Coef_CO		,
			 elm::darray_ptr		Data_CA		,
			 elm::darray_ptr		Data_CO		,
			 const darray_export_map* Data_CE ,
			 etk::ndarray*		Outcome		)
: Params_CA	(Params_CA)
, Params_CO	(Params_CO)
, Coef_CA	(Coef_CA)
, Coef_CO	(Coef_CO)
, Data_CA	(Data_CA)
, Data_CO	(Data_CO)
, Data_CE   (Data_CE)
, Outcome	(Outcome)
{
}

elm::ca_co_packet::~ca_co_packet()
{
}



void elm::ca_co_packet::logarithm_partial
( const unsigned&      firstcase
, const unsigned&      numberofcases
, const size_t&        dim2size
, const double&        scale
)
{
	for (unsigned c=firstcase; c<firstcase+numberofcases; c++)
	for (unsigned a=0;a<dim2size;a++) {
		Outcome->at(c,a) = ::log(Outcome->at(c,a)) * scale;
	}
}



void elm::ca_co_packet::logit_partial
( const unsigned&      firstcase
, const unsigned&      numberofcases
, const double&        U_premultiplier
)
{
	if (Data_CE) {


		if (U_premultiplier) {
			cblas_dscal(Outcome->size2()*Outcome->size3()*numberofcases, U_premultiplier, Outcome->ptr(firstcase), 1);
		} else {
			memset(Outcome->ptr(firstcase), 0, sizeof(double)*Outcome->size2()*Outcome->size3()*numberofcases);
		}

	
		size_t rowmarker = Data_CE->_casestarts->int64_at(firstcase);
		long long caseindex = Data_CE->_caseindexes->int64_at(rowmarker);
		while (caseindex < firstcase+numberofcases) {
			*(Outcome->ptr(caseindex, Data_CE->_altindexes->int64_at(rowmarker))) = cblas_ddot(Data_CE->nvars(), Coef_CA->ptr(), 1, Data_CE->_data_array->ptr(rowmarker), 1);
			rowmarker++;
			if (rowmarker<Data_CE->nrows()) {
				caseindex = Data_CE->_caseindexes->int64_at(rowmarker);
			} else {
				break;
			}
		}
	
	} else {


		if (Data_CA && Data_CA->nVars()>0) {
			// Fast Linear Algebra		
			if (Outcome->size2()==Data_CA->nAlts()) {
				cblas_dgemv(CblasRowMajor,CblasNoTrans, 
							numberofcases * Data_CA->nAlts(), Data_CA->nVars(), 
							1,
							Data_CA->values(firstcase,numberofcases),Data_CA->nVars(),
							Coef_CA->ptr(),1,
							U_premultiplier, Outcome->ptr(firstcase),1);
			} else {
				for (unsigned a=0;a<Data_CA->nAlts();a++) {
					cblas_dgemv(CblasRowMajor,CblasNoTrans,
								numberofcases,Data_CA->nVars(),
								1, 
								Data_CA->values(firstcase,numberofcases)+(a*Data_CA->nVars()), Data_CA->nAlts()*Data_CA->nVars(), 
								Coef_CA->ptr(),1, 
								U_premultiplier, Outcome->ptr(firstcase)+a, Outcome->size2() );
				}
			}		
		}
		if ((Data_CA && Data_CA->nVars()==0) || (!Data_CA)) {
			if (U_premultiplier) {
				cblas_dscal(Outcome->size2()*Outcome->size3()*numberofcases, U_premultiplier, Outcome->ptr(firstcase), 1);
			} else {
				memset(Outcome->ptr(firstcase), 0, sizeof(double)*Outcome->size2()*Outcome->size3()*numberofcases);
			}
		}
	}
	if (Data_CO && Data_CO->nVars()>0) {
		// Fast Linear Algebra
		
		if (Coef_CO->size2()>0) {
		
		cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
					numberofcases,Coef_CO->size2(), Data_CO->nVars(),
					1,
					Data_CO->values(firstcase,numberofcases), Data_CO->nVars(),
					Coef_CO->ptr(), Coef_CO->size2(),
					1,Outcome->ptr(firstcase),Outcome->size2());
			
		} else {
			OOPS("Coef_CO->size2()=",Coef_CO->size2()," while Data_CO->nVars()=",Data_CO->nVars());
		}
	}
}

bool elm::ca_co_packet::relevant()
{
	if (Params_CA && Params_CA->size1()*Params_CA->size2()*Params_CA->size3() > 0) {
		return true;
	}
	
	if (Params_CO && Params_CO->size1()*Params_CO->size2()*Params_CO->size3() > 0) {
		return true;
	}
	
	return false;
}


size_t elm::ca_co_packet::nAlt() const
{
	if (Data_CA) {
		return Data_CA->nAlts();
	}
	
	if (Data_CO) {
		return Coef_CO->size2();
	}
	
	// else (Data_CE) only...
	TODO;
	
	//return (Data_CA ? Data_CA->nAlts() : /*Data_CO->nAlts()*/ Coef_CO->size2());
}


void elm::ca_co_packet::logit_partial_deriv
( const unsigned&      c
, etk::memarray_raw*   dUtilCA
, etk::memarray_raw*   dUtilCO
)
{
	unsigned nAlts = nAlt();
	unsigned nParamCA = Params_CA->size1() * Params_CA->size2() * Params_CA->size3();
	unsigned nParamCO = Params_CO->size1() * Params_CO->size2() * Params_CO->size3();

	dUtilCA->initialize(0.0);
	dUtilCO->initialize(0.0);
	// for this case, this will be the derivative of utility at
	//  each node w.r.t. params [N_Nodes, N_Params]
	
	unsigned a,u;
	
	for (a=0; a<nAlts; a++) {
		// 'a' is iterated over all the relevant nodes in the network
		//  the last node is not relevant, as it is the root node and has no predecessors

		if (!Outcome->at(c,a)) continue;
		
		// First, we calculate the effect of various parameters on the utility
		// of 'a' directly.
		if (Data_CE && Data_CE->active()) {
			if (dUtilCA->size()) Data_CE->export_into(dUtilCA->ptr(a),c,a,Data_CE->nvars());
		} else {
			if (dUtilCA->size()) Data_CA->ExportData(dUtilCA->ptr(a),c,a,Data_CA->nAlts());
		}
		if (dUtilCO->size()) Data_CO->ExportData(dUtilCO->ptr(a),c,a,Data_CO->nAlts());
			
	}
}

