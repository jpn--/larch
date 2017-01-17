/*
 *  elm_model.cpp
 *
 *  Copyright 2007-2016 Jeffrey Newman
 *

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
#include "elm_model2.h"
#include "elm_sql_scrape.h"
#include "elm_names.h"
#include <iostream>
#include "elm_calculations.h"

#include "elm_workshop_mnl_gradient.h"
#include "elm_workshop_mnl_prob.h"
#include "elm_workshop_loglike.h"
#include "elm_workshop_nl_probability.h"

using namespace etk;
using namespace elm;
using namespace std;



void elm::Model2::_setUp_QMNL()
{
	INFO(msg)<< "Setting up QMNL model..." ;
	
	if (!_fountain()) OOPS("A data fountain must be linked to this model to do this.");
	
	// COUNTING
	nElementals = Xylem.n_elemental();
	nNests = Xylem.n_branches();
	nNodes = Xylem.size();
	
	if (_fountain()->nAlts()<=0) {
		OOPS("The number of alternatives given in the data is non-positive");
	}
		
	// Allocate Memory	
	Probability.resize(nCases,nElementals);
	CaseLogLike.resize(nCases);
	Quantity.resize(nCases,nElementals);
	
	Workspace.resize(nElementals);
	
		
	sherpa::allocate_memory();
	
	Data_MultiChoice.resize(nCases);
	Data_MultiChoice.initialize(true);
	
	INFO(msg)<< "Set up QMNL model complete." ;
	
}



void elm::Model2::pull_and_exp_from_freedoms(const paramArray& par,       double* ops, const double* fr, const bool& log)
{
	for (unsigned i=0; i<par.length(); i++) {
		if (par[i]){
			ops[i] = exp(par[i]->pullvalue(fr));
			if (log) {
				BUGGER(msg) << "PULL "<<par[i]->freedomName()<<" "<<ops[i]<<" into slot "<<i;
			}
		} else {
		}
	}
}










