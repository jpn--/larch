/*
 *  sherpa_mountaineer.cpp
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

#include "sherpa_mountaineer.h"
#define PERTURBATION_SIZE (1e-5)

using namespace std;

void mountaineer::finite_diff_gradient_(const etk::memarray& params, etk::memarray& fGrad)
{
	unsigned i;	
	double jiggle;
	if (fGrad.size() < dF()) OOPS("error(sherpa): not enough finite diff array space");
	etk::memarray FvalueCurrent (dF());
	FvalueCurrent = params;
	
	for (i=0;i<dF();i++) {
		jiggle = FvalueCurrent[i] * PERTURBATION_SIZE;
		if (!jiggle) jiggle = PERTURBATION_SIZE;
		
		FvalueCurrent[i] += jiggle;
		fGrad[i] = objective(FvalueCurrent); 
		FvalueCurrent[i] -= jiggle;
		
		FvalueCurrent[i] -= jiggle;
		fGrad[i] -= objective(FvalueCurrent); 
		FvalueCurrent[i] += jiggle;
		
		fGrad[i] /= -2*jiggle;
	}
}


void mountaineer::finite_diff_hessian (const etk::memarray& params, etk::triangle& fHESS) 
{
	// Formulation from Givens & Hoeting, "Computational Statistics"
	
	etk::memarray temp_fHess (dF(),dF());
	etk::memarray FvalueCurrent (dF());
	FvalueCurrent = params;

	const double* Gr (NULL);
	
	unsigned i,j;
	double jiggle;
	
	for (i=0;i<dF();i++) {
		jiggle = FvalueCurrent[i] * PERTURBATION_SIZE;
		if (!jiggle) jiggle = PERTURBATION_SIZE;
		
		FvalueCurrent[i] += jiggle;
		Gr = gradient(FvalueCurrent).ptr(); 
		cblas_dcopy(dF(), Gr,1, temp_fHess.ptr()+i,dF());    
		FvalueCurrent[i] -= jiggle;
		
		FvalueCurrent[i] -= jiggle;
		Gr = gradient(FvalueCurrent).ptr(); 
		cblas_daxpy(dF(), -1, Gr,1, temp_fHess.ptr()+i,dF());
		FvalueCurrent[i] += jiggle;
		
		if (jiggle) cblas_dscal(dF(), 1/(2*jiggle), temp_fHess.ptr()+i,dF());						
		//MONITOR(msg) << "xHESSIAN\n" << temp_fHess.printrows(0,dF()) ;
	}
	
	fHESS.initialize();
	for (i=0;i<dF();i++) for (j=0;j<dF();j++) {
		fHESS(i,j) += temp_fHess(i,j);
		if (i==j) fHESS(i,j) += temp_fHess(i,j);
		//MONITOR(msg) << "fHESSIAN\n" << fHESS.printSquare() ;
	}
	fHESS.scale(0.5);
}


double mountaineer::gradient_diagnostic (const etk::memarray& params) 
{
	unsigned i;
	INFO(msg)<< "Conducting Gradient Diagnostic..." ;
	etk::memarray GCurrent;
	GCurrent = gradient(params); 
	etk::memarray FiniteGrad (dF());
	finite_diff_gradient_(params,FiniteGrad);
	
	ostringstream buff;
//	buff << "Parameter           \t" ;
	buff << "ParamValue          \tAnalyticGrad        \tFiniteDiffGrad      \n" ;
	for ( i=0; i<dF(); i++) {
//		buff.width(20);
//		buff << FNames[i] << "\t";
		buff.width(20);
		buff << params[i] << "\t";
		buff.width(20);
		buff << GCurrent[i] << "\t";
		buff.width(20);
		buff << FiniteGrad[i] << "\n";
	}
	
	FiniteGrad -= GCurrent;
	double max_diff = 0;
	
	for (i=0; i<dF(); i++) {
		if ((fabs(FiniteGrad[i]) > max_diff) && FiniteGrad[i] && GCurrent[i])
			max_diff = fabs(FiniteGrad[i]);
	}
	
	double max_percent_diff = 0;
	bool found_nan =false;
	double max_diff_one_is_zero = 0;
	for (i=0; i<dF(); i++) {
		if ((fabs(FiniteGrad[i]/GCurrent[i]) > max_percent_diff) && FiniteGrad[i]!=0 && GCurrent[i]!=0) {
			max_percent_diff = fabs(FiniteGrad[i]/GCurrent[i]);
		}
		if (FiniteGrad[i]==0 || GCurrent[i]==0) {
			if (fabs(FiniteGrad[i]) + fabs(GCurrent[i]) > max_diff_one_is_zero) {
				max_diff_one_is_zero = fabs(FiniteGrad[i]) + fabs(GCurrent[i]);
			}
		}
			max_percent_diff = fabs(FiniteGrad[i]/GCurrent[i]);
		if (isNan(GCurrent[i])) found_nan = true;
	}
	
	buff << "Maximum Absolute Deviation:   " << max_diff << "\n";
	buff << "Maximum Percentage Deviation: " << max_percent_diff*100 << "%";
	
	INFO(msg)<< buff.str() ;
	if (max_percent_diff > 0.09 || found_nan || max_diff_one_is_zero > 0.001) {
		OOPS("error: gradient diagnostic disagreement\n",buff.str());
		//msg << "Press enter to continue...\n";
		//std::cin.get();
	}
	
	return max_diff;
}


double mountaineer::hessian_diagnostic (const etk::memarray& params) 
{
	unsigned i,j;
	INFO(msg)<< "Conducting Hessian Diagnostic..." ;
	etk::triangle Hess;
	Hess = calculate_hessian(params); 
	etk::triangle FiniteHess (dF());
	finite_diff_hessian(params,FiniteHess);
	
	ostringstream buff;
	buff << "Parameter           \t";
	buff << "Parameter           \tAnalyticHess        \tFiniteDiffHess(+/-)      \n" ;
	for ( i=0; i<dF(); i++) {
	}
	
	FiniteHess -= Hess;
	double max_diff = 0;
	
	for (i=0; i<dF(); i++) for (j=i; j<dF(); j++) {
		if ((fabs(FiniteHess(i,j)) > max_diff) && FiniteHess(i,j) && Hess(i,j))
			max_diff = fabs(FiniteHess(i,j));
	}
	
	double max_percent_diff = 0;
	bool found_nan =false;
	for (i=0; i<dF(); i++)  for (j=i; j<dF(); j++) {
		if ((fabs(FiniteHess(i,j)/Hess(i,j)) > max_percent_diff) && FiniteHess(i,j)!=0 && Hess(i,j)!=0) {
			max_percent_diff = fabs(FiniteHess(i,j)/Hess(i,j));
			if ((fabs(FiniteHess(i,j)/Hess(i,j)) > 0.001)) {
				buff.width(20);
				buff << /*FNames[i]*/i << "\t";
				buff.width(20);
				buff << /*FNames[i]*/j << "\t";
				buff.width(20);
				buff << Hess(i,j) << "\t";
				buff.width(20);
				buff << FiniteHess(i,j) << "\n";
			}
		}
		if (isNan(Hess(i,j))) found_nan = true;
	}
	
	buff << "Maximum Absolute Deviation:   " << max_diff << "\n";
	buff << "Maximum Percentage Deviation: " << max_percent_diff*100 << "%";
	
	INFO(msg)<< buff.str() ;
	if (max_percent_diff > 0.09 || found_nan) {
		OOPS("error: hessian diagnostic disagreement\n",buff.str());
		//msg << "Press enter to continue...\n";
		//std::cin.get();
	}
	
	return max_diff;
}


