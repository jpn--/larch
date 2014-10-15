/*
 *  elm_model.cpp
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

#ifndef __ELM_WORKSHOP_NL_PROBABILITY_H__
#define __ELM_WORKSHOP_NL_PROBABILITY_H__


#include <cstring>
#include "elm_model2.h"
#include "elm_parameter2.h"
#include "elm_sql_scrape.h"
#include "elm_names.h"
#include "etk_workshop.h"
#include <iostream>


namespace elm {


void __casewise_nl_utility 
( double* U		        // pointer to utility array [nN space]
, const VAS_System& Xy  // nesting structure
, double* Work	        // function workspace, [nN]
) ;

void __casewise_nl_probability
( double* U  			// pointer to utility array [nN space]
, double* CPr 		    // pointer to conditional probability
, double* Pr 		    // pointer to probability
, const VAS_System& Xy	// nesting structure
);



class workshop_nl_probability 
: public etk::workshop
{

public:
	
	elm::ca_co_packet UtilPacket;
	elm::ca_co_packet SampPacket;
	
	const paramArray* Params_LogSum;
	
	datamatrix Data_Avail;

	etk::memarray_raw Workspace;
	unsigned nNodes;

	etk::ndarray* Probability;
	etk::ndarray* Cond_Prob;
	const VAS_System* Xylem;
	
	etk::ndarray* AdjProbability;
	
	etk::logging_service* msg_;

	workshop_nl_probability
	(  const unsigned&   nNodes
	 , elm::ca_co_packet UtilPacket
	 , elm::ca_co_packet SampPacket
	 , const paramArray& Params_LogSum
	 , datamatrix     Data_Avail
	 ,  etk::ndarray* Probability
	 ,  etk::ndarray* Cond_Prob
	 ,	etk::ndarray* AdjProbability
	 , const VAS_System* Xylem
	 , etk::logging_service* msgr=nullptr
	 );
	
	virtual ~workshop_nl_probability();

	void workshop_nl_probability_calc
	( const unsigned&   firstcase
	 , const unsigned&   numberofcases
	 );
	
	virtual void work(size_t firstcase, size_t numberofcases, boosted::mutex* result_mutex);	

	void case_logit_add_sampling(const unsigned& c);
	
	
};



}
#endif // __ELM_WORKSHOP_NL_PROBABILITY_H__

