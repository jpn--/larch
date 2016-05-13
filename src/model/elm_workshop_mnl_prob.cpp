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

#include "elm_workshop_mnl_prob.h"




elm::mnl_prob_w::mnl_prob_w(  etk::ndarray* U
							, etk::ndarray* CLL
							, elm::ca_co_packet UtilPack
							, elm::darray_ptr Data_AV
							, elm::darray_ptr Data_Ch
							, const double& U_premultiplier
							, etk::logging_service* msgr
							)
: Probability(U)
, CaseLogLike(CLL)
, Data_AV(Data_AV)
, Data_Ch(Data_Ch)
, U_premultiplier(U_premultiplier)
, msg_(msgr)
, UtilPacket(UtilPack)
{
	//	BUGGER_(msg_, "CONSTRUCT elm::mnl_prob_w::mnl_prob_w()\n");
}

elm::mnl_prob_w::~mnl_prob_w()
{
}


void elm::mnl_prob_w::work(size_t firstcase, size_t numberofcases, boosted::mutex* result_mutex)
{
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

	// UTILITY //

	UtilPacket.Outcome = Probability;
	UtilPacket.logit_partial(firstcase, numberofcases);

//	if (Data_CA && Data_CA->nVars()>0 /* && Data_CA->fully_loaded()*/) {
//		// Fast Linear Algebra
//		if (Probability->size2()==Data_CA->nAlts()) {
//			cblas_dgemv(CblasRowMajor,CblasNoTrans, 
//						numberofcases * Data_CA->nAlts(), Data_CA->nVars(), 
//						1,
//						Data_CA->values(firstcase,0),Data_CA->nVars(),
//						Coef_CA->ptr(),1,
//						U_premultiplier, Probability->ptr(firstcase),1);
//		} else {
//			for (unsigned a=0;a<Data_CA->nAlts();a++) {
//				if (Probability->size2()<=0){
//					OOPS("IncY Zero");
//				}
//				cblas_dgemv(CblasRowMajor,CblasNoTrans,
//							numberofcases,Data_CA->nVars(),
//							1, 
//							Data_CA->values(firstcase,0)+(a*Data_CA->nVars()), Data_CA->nAlts()*Data_CA->nVars(), 
//							Coef_CA->ptr(),1, 
//							U_premultiplier, Probability->ptr(firstcase)+a, Probability->size2() );
//			}
//		}		
//	} else if (Data_CA && Data_CA->nVars()>0) {
//		// Slow case-by-case
//		for (unsigned c=firstcase; c<firstcase+numberofcases; c++) {
//			if (Probability->size2()==Data_CA->nAlts()) {
//				cblas_dgemv(CblasRowMajor,CblasNoTrans, 
//							Data_CA->nAlts(), Data_CA->nVars(), 
//							1,
//							Data_CA->values(c,1),Data_CA->nVars(),
//							Coef_CA->ptr(),1,
//							U_premultiplier, Probability->ptr(c),1);
//			} else {
//				for (unsigned a=0;a<Data_CA->nAlts();a++) {
//					if (Probability->size2()<=0){
//						OOPS("IncY Zero");
//					}
//					cblas_dgemv(CblasRowMajor,CblasNoTrans,
//								1,Data_CA->nVars(),
//								1, 
//								Data_CA->values(c,1)+(a*Data_CA->nVars()), Data_CA->nAlts()*Data_CA->nVars(), 
//								Coef_CA->ptr(),1, 
//								U_premultiplier, Probability->ptr(c)+a, Probability->size2() );
//				}
//			}
//		}
//	}
//	if (!Data_CA || Data_CA->nVars()==0) {
//		if (U_premultiplier) {
//			cblas_dscal(Probability->size2()*Probability->size3()*numberofcases, U_premultiplier, Probability->ptr(firstcase), 1);
//		} else {
//			memset(Probability->ptr(firstcase), 0, sizeof(double)*Probability->size2()*Probability->size3()*numberofcases);
//		}
//	}
//	
//	if (Data_CO && Data_CO->nVars()>0 /*&& Data_CO->fully_loaded()*/) {
//		// Fast Linear Algebra		
//		cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
//					numberofcases,Coef_CO->size2(), Data_CO->nVars(),
//					1,
//					Data_CO->values_constptr(firstcase), Data_CO->nVars(),
//					Coef_CO->ptr(), Coef_CO->size2(),
//					1,Probability->ptr(firstcase),Probability->size2());
//	} else if (Data_CO && Data_CO->nVars()>0) {
//		// Slow case-by-case
//		for (unsigned c=firstcase; c<firstcase+numberofcases; c++) {
//			if (Data_CO->nVars())
//				cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
//							1,Coef_CO->size2(), Data_CO->nVars(),
//							1,
//							Data_CO->values(c,1), Data_CO->nVars(),
//							Coef_CO->ptr(), Coef_CO->size2(),
//							1,Probability->ptr(c),Probability->size2());
//		}
//	}

	// PROBABILITY //
	
//	if (firstcase==0) std::cerr <<"Util[0]="<<Probability->printrow(0) <<"\n";
	
	
	for (unsigned c=firstcase; c<firstcase+numberofcases; c++) {
		double sum_prob = 0.0;
		double sum_choice = 0.0;
		CaseLogLike->at(c) = 0.0;
		double shifter = 0.0;
		double min_av_utility = INF;
//		double min_ch_utility = INF;
//		double max_ch_utility = -INF;
		double max_av_utility = -INF;
		for (unsigned a=0;a<nElementals;a++) {
			if (Data_AV->boolvalue(c,a)) {
				double p = Probability->at(c,a);
				if (p > max_av_utility) max_av_utility = p;
				if (p < min_av_utility) min_av_utility = p;
//				if (Data_Ch->value(c,a)) {
//					if (p < min_ch_utility) min_ch_utility = p;
//					if (p > max_ch_utility) max_ch_utility = p;
//				}
			}
		}
		if (max_av_utility>700 || min_av_utility<-700) {
			shifter = 700-max_av_utility;
		}
		

		
//			std::cerr << "Shifter Breaks on "<<c<<"\n";
//			std::cerr << " max_av_utility= "<<max_av_utility<<"\n";
//			std::cerr << " min_av_utility= "<<min_av_utility<<"\n";
//			std::cerr << " shifter       = "<<shifter<<"\n";
		for (unsigned a=0;a<nElementals;a++) {
			if (!Data_AV->boolvalue(c,a)) {
				Probability->at(c,a) = 0.0;
			} else {
				double* p = Probability->ptr(c,a);
//					std::cerr << "   u["<<a<<"]= "<<*p<<"\n";
				*p += shifter;
				double data_ch_value_ca = Data_Ch->value(c,a);
				if (data_ch_value_ca) {
					CaseLogLike->at(c) += (*p) * data_ch_value_ca;
					sum_choice += data_ch_value_ca;
				}
				*p = exp(*p);
				sum_prob += *p;
//					std::cerr << "  Availability for case 0 alt "<<a<<" is YES, exp(Utility)=\t"<<*p<<"\n";
			}
		}


//			std::cerr << "  sum_prob="<<sum_prob<<"\n";
//			std::cerr << "  sum_choice="<<sum_choice<<"\n";


	
		
//		if (c==565) std::cerr << "Data_AV[565,0:3]="<<Data_AV->boolvalue(565,0)<<Data_AV->boolvalue(565,1)<<Data_AV->boolvalue(565,2)<<"\n";
//		if (c==565) std::cerr << "sum_prob="<<sum_prob<<"\n";
		if (sum_prob) {
			for (unsigned a=0;a<nElementals;a++) {
				Probability->at(c,a) /= sum_prob;
//				if (shifter) {
//					std::cerr << "  pr["<<a<<"]="<<Probability->at(c,a)<<"\n";
//				}
			}
			if (sum_choice) {
				CaseLogLike->at(c) -= log(sum_prob) * sum_choice;
//				if (shifter) {
//					std::cerr << "  CaseLogLike="<<CaseLogLike->at(c)<<"\n";
//				}
			}
		}
		
		
	}
	
//	if (firstcase==0) BUGGER_(msg_, "Prob[0]="<<Probability->printrow(0));

}

