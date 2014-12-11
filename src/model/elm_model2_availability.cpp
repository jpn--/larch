/*
 *  elm_model2_choice.cpp
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


#include <cstring>
#include "elm_model2.h"
#include "elm_sql_scrape.h"




//void elm::Model2::avail(const std::string& varname)
//{
//	std::ostringstream ret;
//	if (varname.empty() && !availability_ca_variable.empty()) {
//		OOPS("you can't delete the availability designator (this feature will come in the future)");
//	} else {
//		availability_ca_variable = varname;
//	}
//}




//PyObject*	elm::Model2::_get_avail()  const
//{
//	return etk::py_one_item_list(PyString_FromString(availability_ca_variable.c_str())); 
//
//}

//etk::ndarray* elm::Model2::tally_avail()
//{
//	if (!_Data) OOPS("A database must be linked to this model to do this.");
//	setUp();
//	etk::ndarray* tally = new etk::ndarray(_Data->nAlts());
//	for (size_t c=0; c<_Data->nCases(); c++) {
//		for (size_t a=0; a<_Data->nAlts(); a++) {
//			if (Data_Avail->boolvalue(c, a)) tally->at(a) += 1;
//		}
//	}
//	return tally;
//}



//void elm::Model2::_setUp_availability_data()
//{
//	if (!_Data) OOPS("A database must be linked to this model to do this.");
//
//	// By default, if a row exists in the idca table, it is available, otherwise not
//	// The use can override by naming a column in the idca table, which is 1 or 0
//	std::string av_col ("1");
//	if (availability_ca_variable != "") av_col = availability_ca_variable;
//	
////	if (Data_Avail) {
////		Data_Avail->decref();
////		Data_Avail = nullptr;
////	}
//	
//	Data_Avail = _Data->ask_avail();
//	
////	Data_Avail = _Data->get_scrape_aval();
////	try {
////		Data_Avail->load_values();
////	} SPOO {
////		WARN(msg) << "error in preloading availability of alternatives";
////		Data_Avail->load_values(0,1);
////	}
//}


