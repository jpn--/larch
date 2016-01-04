/*
 *  elm_workshop_loglike.cpp
 *
 *  Copyright 2007-2015 Jeffrey Newman.
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


#include <cstring>
#include "etk.h"
#include <iostream>

#include "elm_workshop_loglike.h"

elm::loglike_w::loglike_w
(  etk::ndarray** Pr
 , const size_t& nAlts
 , elm::darray_ptr Data_CH
 , elm::darray_ptr Data_WT
 , double*       LogL
 , etk::ndarray* LogL_casewise
 , bool          mute_warnings
 , etk::logging_service* msgr
 )
: Probability(Pr)
, nAlts      (nAlts)
, Data_CH    (Data_CH)
, Data_WT    (Data_WT)
, LogL       (LogL)
, LogL_casewise(LogL_casewise)
, msg_       (msgr)
, mute_warnings(mute_warnings)
{
//	if (Data_CH) Data_CH->incref();
//	if (Data_WT) Data_WT->incref();
}

elm::loglike_w::~loglike_w()
{
//	REF_CLEAR(Data_CH);
//	REF_CLEAR(Data_WT);
}


void elm::loglike_w::work(size_t firstcase, size_t numberofcases, boosted::mutex* result_mutex)
{
	double LogL_local (0.0);
	double LogL_local_c (0.0);
	double choice_value;
	for (size_t c=firstcase; c<firstcase+numberofcases; c++) {
		LogL_local_c = 0.0;
		for (size_t a=0;a<nAlts;a++) {
			choice_value = Data_CH->value(c,a);
			if (choice_value) {
				if (Data_WT) {
					LogL_local_c += (log((*Probability)->at(c,a)) * choice_value * Data_WT->value(c,0));
				} else {
					LogL_local_c += (log((*Probability)->at(c,a)) * choice_value);
				}
			}			
		}
		
		if (LogL_casewise) {
			LogL_casewise->at(c) = LogL_local_c;
		}
		
		LogL_local += LogL_local_c;
		
		if (isNan(LogL_local) && !mute_warnings) {
			WARN_(msg_, "WARNING: Log Likelihood becomes NAN at caserow "<<c 
						<< "\nW..prob: "<<(*Probability)->printrows(c,c+1)
						<< "\nW..chos: "<<Data_CH->printcases(c,c+1) );
			break;
		}
		if (isInf(LogL_local) && !mute_warnings) {
			WARN_(msg_, "WARNING: Log Likelihood becomes INF at caserow "<<c 
						<< "\nW..prob: "<<(*Probability)->printrows(c,c+1)
						<< "\nW..chos: "<<Data_CH->printcases(c,c+1) );
			break;
		}
	}
	
	
	result_mutex->lock();
//	BUGGER_(msg_, "LogLike["<< firstcase <<"]->["<< firstcase+numberofcases-1 <<"] = "<<LogL_local);
	*LogL += LogL_local;
	result_mutex->unlock();

}

