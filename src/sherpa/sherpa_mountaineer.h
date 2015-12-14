/*
 *  sherpa_mountaineer.h
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


#ifndef __SHERPA_MOUNTAINEER_HEADER__
#define __SHERPA_MOUNTAINEER_HEADER__

#include "etk.h"

class mountaineer 
: public etk::object
{
	
public:
	
	virtual unsigned dF() const=0;

	virtual double objective(const etk::memarray& params)=0;
	virtual const etk::memarray& gradient(const etk::memarray& params)=0;
	virtual const etk::triangle& calculate_hessian(const etk::memarray& params)=0;
	
	void finite_diff_gradient_(const etk::memarray& params, etk::memarray& fGrad);
	void finite_diff_hessian (const etk::memarray& params, etk::triangle& fHESS);
	
	double gradient_diagnostic (const etk::memarray& params);
	double hessian_diagnostic (const etk::memarray& params) ;
	int flag_gradient_diagnostic;
	int flag_hessian_diagnostic;
	
};

#endif
