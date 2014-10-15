/*
 *  sherpa_pack.h
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


#ifndef __SHERPA_PACK_H__
#define __SHERPA_PACK_H__


#ifdef SWIG
%feature("kwargs", 1) sherpa_pack::sherpa_pack;
%{
#include "sherpa_pack.h"
%}
%rename(OptimizationMethod) sherpa_pack;
%template(OptimizationMethodList) std::vector<sherpa_pack>;
#endif // def SWIG

#include <string>
#include <climits>
#include <vector>

class sherpa_pack { 
public:
	
	sherpa_pack(char algo='G', double thresh=0.0001, double ini=1, unsigned slow=0, 
				double min=1e-10, double max=4, double ext=2, double ret=.5,
				unsigned honey=3, double pat=1., unsigned maxiter=UINT_MAX, unsigned miniter=0);
	virtual ~sherpa_pack();
	
	double Min_Step;
	double Max_Step;
	double Step_Extend_Factor;
	double Step_Retract_Factor;
	double Initial_Step;
	
	unsigned Honeymoon;
	double Patience;
	unsigned Max_NumIter;
	unsigned Min_NumIter;

	char   Algorithm;
	bool   Fail;
	unsigned Slowness;
	
	virtual double get_step();
	virtual void tell_step(const double& step);
	
	virtual bool tell_turn(const double& val, const double& tol, std::string& explain_stop, const unsigned& iterationNumber);
	
	virtual std::string print_pack() const;
	
	std::string AlgorithmName() const;
	
	std::string __repr__() const;
	
private:
	double tolerance_threshold;
};


std::vector<sherpa_pack> default_optimization_recipe();
std::vector<sherpa_pack> bfgs_optimization_recipe();


#endif
