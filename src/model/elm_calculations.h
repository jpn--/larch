/*
 *  elm_calculations.h
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

#ifndef __ELM_CALCULATIONS_H__
#define __ELM_CALCULATIONS_H__


#include "etk_ndarray.h"
#include "elm_sql_scrape.h"
#include "elm_darray.h"


namespace elm {
		
	void __logit_utility
	( etk::memarray&       U
	, elm::darray_ptr        Data_CA
	, elm::darray_ptr        Data_CO
	, const etk::memarray& Coef_CA
	, const etk::memarray& Coef_CO
	, const double&        U_premultiplier
	);

	void __logit_utility_arrays
	( etk::memarray&  U
	 , const etk::ndarray*   Data_CA
	 , const etk::ndarray*   Data_CO
	 , const etk::memarray& Coef_CA
	 , const etk::memarray& Coef_CO
	 , const double&   U_premultiplier
	 );


	
}; // end namespace elm



#endif // __ELM_CALCULATIONS_H__

