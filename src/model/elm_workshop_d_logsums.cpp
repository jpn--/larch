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
#include "etk.h"
#include <iostream>

#include "elm_workshop_d_logsums.h"




elm::d_logsums_w::d_logsums_w(  etk::ndarray* U
							, elm::ca_co_packet UtilPack
							, elm::ca_co_packet QuantPack
							, etk::logging_service* msgr
							, PyArrayObject** d_logsums_casewise
							, const paramArray* Params_QuantLogSum
							, const double* CoefQuantLogsum
							)
: Probability(U)
, msg_(msgr)
, UtilPacket(UtilPack)
, QuantPacket(QuantPack)
, d_logsums_casewise(d_logsums_casewise)
, Params_QuantLogSum (Params_QuantLogSum)
, CoefQuantLogsum (CoefQuantLogsum)
{
	//	BUGGER_(msg_, "CONSTRUCT elm::d_logsums_w::d_logsums_w()\n");
	
	// check that logsums out is at least the correct size
	if (d_logsums_casewise) {
		if (PyArray_DIM(*d_logsums_casewise, 0) < Probability->size1() ) {
			Py_CLEAR(*d_logsums_casewise);
		}
	}
	
	if ((!d_logsums_casewise) || (!*d_logsums_casewise)) {
		OOPS("you need to give a valid d_logsums_casewise array");
	}
	
	Py_XINCREF(*d_logsums_casewise);
}

elm::d_logsums_w::~d_logsums_w()
{
	Py_CLEAR(*d_logsums_casewise);
}


void elm::d_logsums_w::work(size_t firstcase, size_t numberofcases, boosted::mutex* result_mutex)
{

//	PYTHON VERSION, WORKS BUT SLOW
//	pr = m.work.probability
//	xca = m.data.utilityca
//	if xca is not None:
//		yca = (pr[:,:m.nAlts(),None] * xca).sum(1) # [cases, n_vars_utilca]
//	else:
//		yca = numpy.zeros([m.nCases(),0])
//	xco = m.data.utilityco
//	if xco is not None:
//		yco = (pr[:,:m.nAlts(),None] * xco[:,None,:])
//		yco = yco.reshape(yco.shape[0], yco.shape[1]*yco.shape[2])
//	else:
//		yco = numpy.zeros([m.nCases(),0])
//	yco_1 = correspond_utilityco(m, grand_m)
//	yca_1 = correspond_utilityca(m, grand_m)
//	q_1 = correspond_quantity(m, grand_m)
//	
//	z = m.data.quantity  # [cases, alts, datacolumns]
//	if z is not None:
//		egam =	m.Coef("QuantityCA").squeeze() #[ datacolumns ]
//		egz = m.work.quantity # [cases, alts]
//		q = (z * egam[None, None, :] * pr[:,:m.nAlts(),None] / egz[:,:,None] ).sum(1) #[ cases, datacolumns ]
//	else:
//		q = numpy.zeros([m.nCases(),0])
//
//	return (
//		+ (numpy.dot(yca,yca_1) if yca_1 is not None else 0)
//		+ (numpy.dot(yco,yco_1) if yco_1 is not None else 0)
//		+ (numpy.dot(q  ,q_1  ) if q_1   is not None else 0)
//	)




	unsigned nElementals = 0;
	if (UtilPacket.Data_CA && UtilPacket.Data_CA->nVars()>0) {
		nElementals = UtilPacket.Data_CA->nAlts();
	} else if (UtilPacket.Data_CE && UtilPacket.Data_CE->nalts()>0) {
		nElementals = UtilPacket.Data_CE->nalts();
	} else if (UtilPacket.Data_CO && UtilPacket.Coef_CO->size2()>0) {
		nElementals = UtilPacket.Coef_CO->size2();
	} else {
		return;
		OOPS("no useful data!");
	};

	unsigned nVarsUCA = 0;
	if (UtilPacket.Data_CA && UtilPacket.Data_CA->nVars()>0) {
		nVarsUCA = UtilPacket.Data_CA->nVars();
	} else if (UtilPacket.Data_CE && UtilPacket.Data_CE->nalts()>0) {
		nVarsUCA = UtilPacket.Data_CE->nvars();
		OOPS_NotImplemented('CE not implemented here');
	}
	
	unsigned nVarsUCO = 0;
	if (UtilPacket.Data_CO && UtilPacket.Coef_CO->size2()>0) {
		nVarsUCO = UtilPacket.Data_CO->nVars();
	}
	
	unsigned nVarsQCA = 0;
	if (QuantPacket.Data_CA && QuantPacket.Data_CA->nVars()>0) {
		nVarsQCA = QuantPacket.Data_CA->nVars();
	} else if (QuantPacket.Data_CE && QuantPacket.Data_CE->nalts()>0) {
		nVarsQCA = QuantPacket.Data_CE->nvars();
		OOPS_NotImplemented('CE not implemented here');
	}

	
	etk::memarray_raw WorkspaceCA (nVarsUCA);
	etk::memarray_raw WorkspaceCO (nVarsUCO, nElementals);
	etk::memarray_raw WorkspaceQA (nVarsQCA);
	double WorkspaceQT=0;

	for (unsigned c=firstcase; c<firstcase+numberofcases; c++) {
		WorkspaceCA.initialize();
		for (unsigned a=0;a<nElementals;a++) {
			for (unsigned v=0;v<nVarsUCA;v++) {
				WorkspaceCA(v) += Probability->at(c,a) * UtilPacket.Data_CA->value_double(c,a,v);
			}
		}
		
		WorkspaceCO.initialize();
		for (unsigned a=0;a<nElementals;a++) {
			for (unsigned v=0;v<nVarsUCO;v++) {
				WorkspaceCO(v,a) += Probability->at(c,a) * UtilPacket.Data_CO->value_double(c,v);
			}
		}
		
		if (nVarsQCA) {
			WorkspaceQA.initialize();
			WorkspaceQT=0;
			for (unsigned a=0;a<nElementals;a++) {
				for (unsigned v=0;v<nVarsQCA;v++) {
					if (QuantPacket.Outcome->at(c,a)) {
						WorkspaceQA(v) += Probability->at(c,a) * QuantPacket.Data_CA->value_double(c,a,v) * QuantPacket.Coef_CA->at(v) / QuantPacket.Outcome->at(c,a) ;
					}
				}
				
				WorkspaceQT += Probability->at(c,a) * log(QuantPacket.Outcome->at(c,a));
			}
			
			if (CoefQuantLogsum) {
				WorkspaceQA.scale(*CoefQuantLogsum);
			}
		}
		
		
		double* outpointer = (double*) PyArray_GETPTR1(*d_logsums_casewise,c);
		memset(outpointer, 0, PyArray_NBYTES(*d_logsums_casewise)/PyArray_DIM(*d_logsums_casewise,0));
		elm::push_to_freedoms2(*UtilPacket.Params_CA  , *WorkspaceCA  , outpointer);
		elm::push_to_freedoms2(*UtilPacket.Params_CO  , *WorkspaceCO  , outpointer);
		elm::push_to_freedoms2(*QuantPacket.Params_CA , *WorkspaceQA  , outpointer);
		elm::push_to_freedoms2(*Params_QuantLogSum ,    &WorkspaceQT  , outpointer);

	}
	
}

