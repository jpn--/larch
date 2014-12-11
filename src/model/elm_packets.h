/*
 *  elm_packets.h
 *
 *  Copyright 2013 Jeffrey Newman 
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

#ifndef __ELM_PACKETS_H__
#define __ELM_PACKETS_H__

#include "elm_sql_scrape.h"
#include "etk_ndarray.h"
#include "elm_parameter2.h"
#include "elm_darray.h"

namespace elm {

	struct ca_co_packet {

		const paramArray*	Params_CA	;
		const paramArray*	Params_CO	;
		const etk::ndarray*	Coef_CA		;
		const etk::ndarray*	Coef_CO		;
		elm::darray_ptr		Data_CA		;
		elm::darray_ptr		Data_CO		;
		etk::ndarray*		Outcome		;
		
		// Constructor
		ca_co_packet(const paramArray*	Params_CA	,
					 const paramArray*	Params_CO	,
					 const etk::ndarray*	Coef_CA		,
					 const etk::ndarray*	Coef_CO		,
					 elm::darray_ptr		Data_CA		,
					 elm::darray_ptr		Data_CO		,
					 etk::ndarray*		Outcome		);
		~ca_co_packet();
		
		// Compute
		void logit_partial
		( const unsigned&      firstcase
		, const unsigned&      numberofcases
		, const double&        U_premultiplier=0.0
		);
		void logit_partial_deriv
		( const unsigned&      c
		, etk::memarray_raw*   dUtilCA
		, etk::memarray_raw*   dUtilCO
		);
		
		bool relevant();
		size_t nAlt() const;
	};

}

#endif // __ELM_PACKETS_H__
