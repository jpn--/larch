/*
 *  sherpa_pack.cpp
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


#include "sherpa_pack.h"
#include <cmath>
#include <sstream>

std::string algorithm_name(const char& algo)
{
	switch (algo) {
		case 'A':
		case 'a':
			return "Newton";
		case 'B':	
		case 'b':	
			return "BFGS";
		case 'D':	
		case 'd':
			return "DFP";
		case 'J':	
		case 'j':	
			return "DFP(J)";
		case 'S':	
		case 's':	
			return "Steepest Ascent";
		case 'G':	
		case 'g':
			return "BHHH";
	}
	return "Incognito";
}


sherpa_pack::sherpa_pack(char algo, double thresh, double ini, unsigned slow, 
						 double min, double max, double ext, double ret,
						 unsigned honey, double pat, unsigned maxiter, unsigned miniter)
: Min_Step (min)
, Max_Step (max)
, Step_Extend_Factor (ext)
, Step_Retract_Factor (ret)
, Initial_Step (ini)
, Algorithm (algo)
, tolerance_threshold (thresh)
, Fail (false)
, Slowness (slow)
, Honeymoon (honey)
, Patience (pat)
, Max_NumIter(maxiter)
, Min_NumIter(miniter)
{
	if (algorithm_name(algo)=="Incognito") {
		Algorithm = 'G';
	}
}

sherpa_pack::~sherpa_pack()
{
	
}

double sherpa_pack::get_step()
{
	return Initial_Step;
}

void sherpa_pack::tell_step(const double& step)
{
	
}

bool sherpa_pack::tell_turn(const double& val, const double& tol, std::string& explain_stop, const unsigned& iterationNumber)
{
	// Return value should be "true" for keep going,
	// or "false" for stop: we are here.
	if (fabs(tol) > tolerance_threshold) {
		return true;
	}
	if (iterationNumber < Min_NumIter) {
		return true;
	}
	if (iterationNumber > Max_NumIter) {
		std::ostringstream x;
		x << "After "<<iterationNumber<<" iterations, this method is being abandoned";
		explain_stop = x.str();
		return false;
	}
	if (isnan(tol)) {
		explain_stop = "Tolerance is NaN";
		return false;
	}
	std::ostringstream x;
	x << "Tolerance is "<<tol<<" which is less than the threshold value of "<<tolerance_threshold;
	explain_stop = x.str();
	return false;
}

std::string sherpa_pack::print_pack() const
{
	std::ostringstream ret;
	ret << "minstep="<<Min_Step;
	ret << " maxstep="<< Max_Step;
	ret << " stepextend="<< Step_Extend_Factor;
	ret << " stepretract=" << Step_Retract_Factor;
	ret << " initialstep=" << Initial_Step;
	ret << " algorithm="<< Algorithm;
	ret << " tolthreshold=" << tolerance_threshold;
	return ret.str();
}




std::string sherpa_pack::AlgorithmName() const
{
	switch (Algorithm) {
			
			// Analytic or finite difference Hessian
		case 'A':
		case 'a':
			return "Newton(Analytic)";
			
			// BFGS (Broyden-Fletcher-Goldfarb-Shanno) Algorithm
		case 'B':
		case 'b':
			return "BFGS";
			
			// DFP (Davidson-Fletcher-Powell) Algorithm
		case 'D':
		case 'd':
			return "DFP";
			
			// Jeff's Undocumented DFP Variant
		case 'J':
		case 'j':
			return "DFP(J) (undocumented)" ;
			
			// Steepest Ascent
		case 'S':
		case 's':
			return "Steepest Ascent";
			
			// BHHH (Berndt-Hall-Hall-Hausman) Algorithm
		case 'G':
		case 'g':
		default:
			return "BHHH" ;
	}
	
}

std::string sherpa_pack::__repr__() const
{
	return std::string("<larch.core.OptimizationRecipe using ")+AlgorithmName()+">";
}


std::vector<sherpa_pack> default_optimization_recipe()
{
	std::vector<sherpa_pack> packs;
	//(char algo, double thresh,    double ini,  double min, double max, double ext, double ret,  unsigned honey, double pat)
	packs.push_back( sherpa_pack('G', 0.000001, 1, 0,   1e-10, 4, 2, .5,    1, 0.001) );
	//	packs.push_back( sherpa_pack('D', 0.000001) );
	packs.push_back( sherpa_pack('B', 0.000001) );
	//	packs.push_back( sherpa_pack('J', 0.000001) );
	packs.push_back( sherpa_pack('S', 0.000001, 1.e-6, 100,   1e-10, 4, 2, .5,    1, 100.) );
	
	return packs;
}

std::vector<sherpa_pack> bfgs_optimization_recipe()
{
	std::vector<sherpa_pack> packs;
	//(char algo, double thresh,    double ini,  double min, double max, double ext, double ret,  unsigned honey, double pat)
	packs.push_back( sherpa_pack('B', 0.000001) );
	packs.push_back( sherpa_pack('S', 0.000001, 1.e-6, 100,   1e-10, 4, 2, .5,    1, 100.) );
	
	return packs;
}


