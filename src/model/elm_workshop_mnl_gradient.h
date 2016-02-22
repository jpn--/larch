/*
 *  elm_model.cpp
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

#ifndef __ELM_WORKSHOP_MNL_GRADIENT_H__
#define __ELM_WORKSHOP_MNL_GRADIENT_H__


#include <cstring>
#include "elm_model2.h"
#include "elm_parameter2.h"
#include "elm_sql_scrape.h"
#include "elm_names.h"
#include <iostream>
#include "etk_workshop.h"
#include "elm_darray.h"

namespace elm {


	class workshop_mnl_gradient2
	: public etk::workshop
	{

	  public:
		boosted::mutex* _lock;

		// Fused Parameter Block Sizes
		size_t nCA;
		size_t nCO;
		size_t nQ;
		size_t nPar;

		
		unsigned dF;
		unsigned nElementals;
		
		// These are memory arrays bound to this workshop.
		// The workshop is free to read and write to these arrays at will.
		etk::memarray_raw Workspace;
		etk::memarray_raw CaseGrad;
		etk::memarray_raw workshopGCurrent;
		etk::symmetric_matrix workshopBHHH   ;
		etk::memarray_raw Grad_UtilityCA;
		etk::memarray_raw Grad_UtilityCO;
		etk::memarray_raw Grad_QuantityCA;

		
		// These are memory arrays that are the principle output accumulators of the workshop.
		//  The lock needs to be acquired before writing to these arrays.
		etk::memarray* _GCurrent;
		etk::symmetric_matrix* _Bhhh;
		

		
		// These are memory arrays that are shared among multiple places.
		// They are read-only for this workshop, and not expected to be written to
		//  by anyone else while this workshop is working.
		const etk::memarray* _Probability;
		const etk::bitarray* _multichoices;


		// This is a workshop packet. It contains members that are memory arrays
		//  that are shared among multiple places.
		elm::ca_co_packet UtilPacket;
		elm::ca_co_packet QuantPacket;


		
		// These are data arrays. If fully loaded, they should not need to be written to
		//  by this workshop. If not fully loaded, they will need to be updated for each
		//  call for a new case, but ScrapePtr's are designed (hopefully) to be thread safe
		elm::darray_ptr Data_Choice;
		elm::darray_ptr Data_Weight;
		
		etk::logging_service* msg_;

		workshop_mnl_gradient2
		(  const unsigned&   dF
		 , const unsigned&   nElementals
		 , elm::ca_co_packet UtilPK
		 , elm::ca_co_packet QuantPK
		 , elm::darray_ptr     Data_Choice
		 , elm::darray_ptr     Data_Weight
		 , const etk::memarray* Probability
		 , etk::memarray* GCurrent
		 , etk::symmetric_matrix* Bhhh
		 , etk::logging_service* msgr
		 , const etk::bitarray* Data_MultiChoice
		 );
		
		~workshop_mnl_gradient2();

		void workshop_mnl_gradient_do(const unsigned& firstcase, const unsigned& numberofcases);
		void workshop_mnl_gradient_send();
		
		virtual void work(size_t firstcase, size_t numberofcases, boosted::mutex* result_mutex);

		void case_gradient_mnl
		( const unsigned& c
		 , const etk::memarray& Probability
		 );
		
		void case_gradient_mnl_multichoice
		( const unsigned& c
		 , const etk::memarray& Probability
		 );
		
	};

	class workshop_mnl_gradient_full_casewise
	: public etk::workshop
	{

	  public:
		boosted::mutex* _lock;

		// Fused Parameter Block Sizes
		size_t nCA;
		size_t nCO;
		size_t nQ;
		size_t nPar;

		
		unsigned dF;
		unsigned nElementals;
		
		// These are memory arrays bound to this workshop.
		// The workshop is free to read and write to these arrays at will.
		etk::memarray_raw Workspace;
		etk::memarray_raw CaseGrad;
		etk::memarray_raw workshopGCurrent;
		etk::symmetric_matrix workshopBHHH   ;
		etk::memarray_raw Grad_UtilityCA;
		etk::memarray_raw Grad_UtilityCO;
		etk::memarray_raw Grad_QuantityCA;

		
		// These are memory arrays that are the principle output accumulators of the workshop.
		//  The lock needs to be acquired before writing to these arrays.
		etk::memarray* _GCurrent;
		etk::symmetric_matrix* _Bhhh;
		
		etk::ndarray* _GCurrentCasewise;

		
		// These are memory arrays that are shared among multiple places.
		// They are read-only for this workshop, and not expected to be written to
		//  by anyone else while this workshop is working.
		const etk::memarray* _Probability;
		const etk::bitarray* _multichoices;


		// This is a workshop packet. It contains members that are memory arrays
		//  that are shared among multiple places.
		elm::ca_co_packet UtilPacket;
		elm::ca_co_packet QuantPacket;


		
		// These are data arrays. If fully loaded, they should not need to be written to
		//  by this workshop. If not fully loaded, they will need to be updated for each
		//  call for a new case, but ScrapePtr's are designed (hopefully) to be thread safe
		elm::darray_ptr Data_Choice;
		elm::darray_ptr Data_Weight;
		
		etk::logging_service* msg_;

		workshop_mnl_gradient_full_casewise
		(  const unsigned&   dF
		 , const unsigned&   nElementals
		 , elm::ca_co_packet UtilPK
		 , elm::ca_co_packet QuantPK
		 , elm::darray_ptr     Data_Choice
		 , elm::darray_ptr     Data_Weight
		 , const etk::memarray* Probability
		 , etk::memarray* GCurrent
		 , etk::symmetric_matrix* Bhhh
		 , etk::logging_service* msgr
		 , const etk::bitarray* Data_MultiChoice
		 , etk::memarray* GCurrentCasewise
		 );
		
		~workshop_mnl_gradient_full_casewise();

		void workshop_mnl_gradient_do(const unsigned& firstcase, const unsigned& numberofcases);
		void workshop_mnl_gradient_send();
		
		virtual void work(size_t firstcase, size_t numberofcases, boosted::mutex* result_mutex);

		void case_gradient_mnl
		( const unsigned& c
		 , const etk::memarray& Probability
		 );
		
		void case_gradient_mnl_multichoice
		( const unsigned& c
		 , const etk::memarray& Probability
		 );
		
	};




class workshop_mnl_gradient 
{

public:

	unsigned dF;
	unsigned nElementals;
	unsigned firstcase;
	unsigned numberofcases;
	unsigned lastcase;

	etk::memarray_raw Workspace;
	etk::memarray_raw CaseGrad;
	
	etk::memarray Grad_UtilityCA;
	etk::memarray Grad_UtilityCO;
	etk::memarray Grad_QuantityCA;

	etk::memarray workshopGCurrent;
	etk::symmetric_matrix workshopBHHH   ;
	
	etk::bitarray multichoices;

	const paramArray* Params_UtilityCA;
	const paramArray* Params_UtilityCO;
	const paramArray* Params_QuantityCA;
	const paramArray* Params_LogSum;
	
	elm::darray_ptr Data_UtilityCA;
	elm::darray_ptr Data_UtilityCO;
	elm::darray_ptr Data_QuantityCA;
	elm::darray_ptr Data_Choice;
	elm::darray_ptr Data_Weight;



	workshop_mnl_gradient
	(  const unsigned&   dF
	 , const unsigned&   nNodes
	 , const paramArray& Params_UtilityCA
	 , const paramArray& Params_UtilityCO
	 , const paramArray& Params_QuantityCA
	 , const paramArray& Params_LogSum
	 , elm::darray_ptr     Data_UtilityCA
	 , elm::darray_ptr     Data_UtilityCO
	 , elm::darray_ptr     Data_QuantityCA
	 , elm::darray_ptr     Data_Choice
	 , elm::darray_ptr     Data_Weight
	 , const unsigned&   firstcase
	 , const unsigned&   numberofcases
	 );
	
	~workshop_mnl_gradient();


	void case_gradient_mnl
	( const unsigned& c
	, const etk::memarray& Probability
	);

	void case_gradient_mnl_multichoice
	( const unsigned& c
	, const etk::memarray& Probability
	);

	void workshop_mnl_gradient_do
	( const etk::memarray& Probability
	 );
	 
	void workshop_mnl_gradient_send
	(  etk::memarray& GCurrent
	 , etk::symmetric_matrix& Bhhh);
	
	
};



}
#endif // __ELM_WORKSHOP_MNL_GRADIENT_H__

