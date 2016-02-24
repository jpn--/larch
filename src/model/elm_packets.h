/*
 *  elm_packets.h
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

#ifndef __ELM_PACKETS_H__
#define __ELM_PACKETS_H__

#include "elm_sql_scrape.h"
#include "etk_ndarray.h"
#include "elm_parameter2.h"
#include "elm_darray.h"

namespace elm {

	struct ca_co_packet {

		const paramArray*	     Params_CA	;
		const paramArray*	     Params_CO	;
		const etk::ndarray*	     Coef_CA    ;
		const etk::ndarray*	     Coef_CO    ;
		elm::darray_ptr		     Data_CA    ;
		elm::darray_ptr		     Data_CO    ;
		const darray_export_map* Data_CE    ;
		etk::ndarray*		     Outcome    ;
		
		// Constructor
		ca_co_packet(const paramArray*	Params_CA	,
					 const paramArray*	Params_CO	,
					 const etk::ndarray*	 Coef_CA	,
					 const etk::ndarray*	 Coef_CO	,
					 elm::darray_ptr		 Data_CA	,
					 elm::darray_ptr		 Data_CO	,
					 const darray_export_map* Data_CE ,
					 etk::ndarray*		Outcome		);
		~ca_co_packet();
		
		// Compute
		void logarithm_partial
		( const unsigned&      firstcase
		, const unsigned&      numberofcases
		, const size_t&        dim2size
		);
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
