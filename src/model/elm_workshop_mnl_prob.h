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

#ifndef __ELM_WORKSHOP_MNL_PROBABILITY_H__
#define __ELM_WORKSHOP_MNL_PROBABILITY_H__

#include "etk.h"
#include "elm_sql_scrape.h"
#include "etk_workshop.h"
#include "elm_darray.h"
#include "elm_packets.h"

namespace elm {

	class mnl_prob_w
	: public etk::workshop
	{
		etk::ndarray* Probability;
		etk::ndarray* CaseLogLike;
		elm::darray_ptr Data_AV;
		elm::darray_ptr Data_Ch;
		double        U_premultiplier;


		elm::ca_co_packet UtilPacket;

		
		etk::logging_service* msg_;
		
	public:
		virtual void work(size_t firstcase, size_t numberofcases, boosted::mutex* result_mutex);
		mnl_prob_w(  etk::ndarray* U
				   , etk::ndarray* CLL
				   , elm::ca_co_packet UtilPack
				   , elm::darray_ptr Data_AV
				   , elm::darray_ptr Data_Ch
				   , const double& U_premultiplier
				   , etk::logging_service* msgr=nullptr
				   );
		~mnl_prob_w();
	}; 



}
#endif // __ELM_WORKSHOP_MNL_PROBABILITY_H__

