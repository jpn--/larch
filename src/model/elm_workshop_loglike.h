/*
 *  elm_workshop_loglike.h
 *
 *  Copyright 2007-2013 Jeffrey Newman.
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

#ifndef __ELM_WORKSHOP_LOGLIKE_H__
#define __ELM_WORKSHOP_LOGLIKE_H__

#include "etk.h"
#include "elm_sql_scrape.h"
#include "etk_workshop.h"

namespace elm {

	class loglike_w
	: public etk::workshop
	{
		etk::ndarray** Probability;
		size_t        nAlts;
		datamatrix Data_CH;
		datamatrix Data_WT;
		double*       LogL;
		
		etk::logging_service* msg_;
		
		bool mute_warnings;
		
	public:
		virtual void work(size_t firstcase, size_t numberofcases, boosted::mutex* result_mutex);
		loglike_w(   etk::ndarray** Pr
				   , const size_t& nAlts
				   , datamatrix Data_CH
				   , datamatrix Data_WT
				   , double*       LogL
				   , bool          mute_warnings
				   , etk::logging_service* msgr=nullptr
				   );
		~loglike_w();
	}; 



}
#endif // __ELM_WORKSHOP_LOGLIKE_H__

