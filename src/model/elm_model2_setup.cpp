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


#include <cstring>
#include "elm_model2.h"
#include "elm_sql_scrape.h"
#include "elm_names.h"
#include <iostream>

#include "elm_parameter2.h"

using namespace etk;
using namespace elm;
using namespace std;



etk::strvec __identify_needs(ComponentList& Input_List)
{
	etk::strvec u_ca;
	
	for (unsigned b=0; b<Input_List.size(); b++) {
		u_ca.push_back_if_unique(Input_List[b].apply_name);
	}
	
	return u_ca;
}


void __check_validity_of_needs(const etk::strvec& needs, Facet*	_Data, int k, etk::logging_service* msg)
{
	if (!_Data) return;

	if (k & IDCA) {
		for (auto i=needs.begin(); i!=needs.end(); i++) {
			BUGGER_(msg, "checking for validity of "<<*i<<" in idCA data");
			_Data->check_ca(*i);
		}
	} else if (k & IDCO) {
		for (auto i=needs.begin(); i!=needs.end(); i++) {
			BUGGER_(msg, "checking for validity of "<<*i<<" in idCO data");
			_Data->check_co(*i);
		}
	}
}


void _setUp_linear_data_and_params
(	ParameterList&			self
 ,	Facet*					_Data
 ,	VAS_System&				Xylem
 ,	ComponentList&			Input_UtilityCA
 ,	ComponentList&			Input_UtilityCO
 ,	datamatrix*			Data_UtilityCA_
 ,	datamatrix*			Data_UtilityCO_
 ,	paramArray&				Params_UtilityCA
 ,	paramArray&				Params_UtilityCO
 ,	etk::logging_service*	msg
)
{
	size_t slot, slot2;


	// utility.ca //
	// First, populate the data_port
	etk::strvec u_ca = __identify_needs(Input_UtilityCA);
	__check_validity_of_needs(u_ca, _Data, IDCA, msg);
	
	// Second, resize the paramArray
	BUGGER_(msg, "setting Params_?CA size to ("<<u_ca.size()<<")");
	Params_UtilityCA.resize(u_ca.size());
	
	// Third, populate the paramArray
	for (unsigned b=0; b<Input_UtilityCA.size(); b++) {
		slot = u_ca.push_back_if_unique(Input_UtilityCA[b].apply_name);
		Params_UtilityCA(slot) = self._generate_parameter(Input_UtilityCA[b].param_name,Input_UtilityCA[b].multiplier);
	}
	
	
	
	// utility.co //
	// First, populate the data_port
	etk::strvec u_co = __identify_needs(Input_UtilityCO);
	__check_validity_of_needs(u_co, _Data, IDCO, msg);

	// Second, resize the paramArray
	auto s = _Data ? _Data->nAlts() : Xylem.n_elemental();
	BUGGER_(msg, "setting Params_?CO size to ("<<u_co.size()<<","<< s <<")");
	Params_UtilityCO.resize(u_co.size(), s);
	
	// Third, populate the paramArray
	for (unsigned b=0; b<Input_UtilityCO.size(); b++) {
		slot = u_co.push_back_if_unique(Input_UtilityCO[b].apply_name);
		if (!Input_UtilityCO[b].altname.empty()) {
			slot2 = Xylem.slot_from_name(Input_UtilityCO[b].altname);
		} else {
			if (Input_UtilityCO[b].altcode==cellcode_empty) {
				OOPS("utilityco input does not specify an alternative.\n"
					 "Inputs in the utilityco space need to identify an alternative.");
			}
			slot2 = Xylem.slot_from_code(Input_UtilityCO[b].altcode);
		}
		Params_UtilityCO(slot,slot2) = self._generate_parameter(Input_UtilityCO[b].param_name, Input_UtilityCO[b].multiplier);
	}
	
	if (Data_UtilityCA_) {
		if (!_Data) OOPS("A database must be linked to this model to do this.");
		*Data_UtilityCA_ = _Data->ask_idca(u_ca);
	}
	
	if (Data_UtilityCO_) {
		if (!_Data) OOPS("A database must be linked to this model to do this.");
		*Data_UtilityCO_ = _Data->ask_idco(u_co);
	}
	
}




void elm::Model2::_setUp_utility_data_and_params(bool and_load_data)
{
	BUGGER(msg) << "--Params_Utility--\n";

	_setUp_linear_data_and_params
	(	*this
	 ,	_Data
	 ,	Xylem
	 ,	Input_Utility.ca
	 ,	Input_Utility.co
	 ,	and_load_data ? &Data_UtilityCA : nullptr
	 ,	and_load_data ? &Data_UtilityCO : nullptr
	 ,	Params_UtilityCA
	 ,	Params_UtilityCO
	 ,	&msg
	);

	BUGGER(msg) << "Params_UtilityCA \n" << Params_UtilityCA.__str__();
	BUGGER(msg) << "Params_UtilityCO \n" << Params_UtilityCO.__str__();

}

void elm::Model2::_setUp_samplefactor_data_and_params(bool and_load_data)
{
	BUGGER(msg) << "--Params_Sampling--\n";

	_setUp_linear_data_and_params
	(	*this
	 ,	_Data
	 ,	Xylem
	 ,	Input_Sampling.ca
	 ,	Input_Sampling.co
	 ,	and_load_data ? &Data_SamplingCA : nullptr
	 ,	and_load_data ? &Data_SamplingCO : nullptr
	 ,	Params_SamplingCA
	 ,	Params_SamplingCO
	 ,	&msg
	);

	BUGGER(msg) << "Params_SamplingCA \n" << Params_SamplingCA.__str__();
	BUGGER(msg) << "Params_SamplingCO \n" << Params_SamplingCO.__str__();

}




std::string __GetWeight(const std::string& varname, bool reweight, Facet* _Data) {
	std::string w;	
	if (reweight) {
		std::ostringstream sql1;
		sql1 << "SELECT SUM("<<varname<<") FROM " << _Data->tbl_idco();
		double average_weight = _Data->sql_statement(sql1)->execute()->getDouble(0); 
		
		average_weight /= _Data->nCases();
		ostringstream sql2;
		sql2 << "("<<varname<<")/"<<average_weight;
		w = (sql2.str());
	} else {
		w = (varname);
	}	
	return w;
}

void elm::Model2::_setUp_weight_data()
{
	if (!_Data) OOPS("A database must be linked to this model to do this.");

	if (_Data->unweighted()) {
		WARN(msg) << "No weights specified, defaulting to weight=1 for all cases.";
	} else {
		Data_Weight = _Data->ask_weight();
		weight_scale_factor = 1.0;
	}

}

void elm::Model2::auto_rescale_weights(const double& mean_weight)
{
	if (Data_Weight) {
		double total_weight = Data_Weight->_repository.sum();
		double factor = Data_Weight->_repository.scale_so_mean_is(mean_weight);
		weight_scale_factor *= factor;
		INFO(msg) << "automatically rescaled weights (total initial weight "<<total_weight
					<<" scaled by "<<weight_scale_factor<<" across "<<Data_Weight->nCases()
					<<" cases)";
	}
}

void elm::Model2::restore_scale_weights()
{
	if (Data_Weight && weight_scale_factor!=1.0) {
		Data_Weight->_repository.scale(1.0/weight_scale_factor);
		weight_scale_factor = 1.0;
	}
}



void elm::Model2::setUp(bool and_load_data)
{
	BUGGER(msg) << "Setting up the model...";
	if (_is_setUp>=2 || (_is_setUp>=1 && !and_load_data)) {
		BUGGER(msg) << "The model is already set up.";
		return;
	}

	BUGGER(msg) << "Rebuilding Xylem network...";
	Xylem.regrow( &Input_LogSum, &Input_Edges, _Data, &msg );
	
	if (Xylem.n_branches() > 0) {
		BUGGER(msg) << "Setting model features to include nesting.";
		features |= MODELFEATURES_NESTING;
	}
	
	BUGGER(msg) << "Setting up utility parameters...";
	_setUp_utility_data_and_params(and_load_data);
	if (features & MODELFEATURES_NESTING) {
		_setUp_NL();
	} else {
		_setUp_MNL();
	}
	if (and_load_data) {
		BUGGER(msg) << "Setting up weight data....";
		_setUp_weight_data();
		_setUp_choice_data();
		_setUp_availability_data();
		
		// multichoice
		if (Data_MultiChoice.size()<_Data->nCases()) {
			Data_MultiChoice.resize(_Data->nCases());
		}
		for (unsigned c=0;c<_Data->nCases();c++) {
			size_t m=c;
			Data_MultiChoice.input(false,m);
			int found=0;
			double sum = 0;
			for (size_t a=0;a<nElementals;a++) {
				if (Data_Choice->value(c,a,0)) {
					found++;
					sum += Data_Choice->value(c,a,0);
				}
			}
			if (found>1 || sum != 1.0) {
				Data_MultiChoice.input(true,m);
			}
		}
	}

	
	if (Input_Sampling.ca.size() || Input_Sampling.co.size()) {
		_setUp_samplefactor_data_and_params(and_load_data);
	}
	_setUp_coef_and_grad_arrays();
	
	if (features & MODELFEATURES_NESTING) {
		Xylem.repoint_parameters(*Coef_LogSum, NULL);
		Xylem.regrow( &Input_LogSum, &Input_Edges, _Data, &msg );
	}
	
	pull_coefficients_from_freedoms();
	
	BUGGER(msg) << "Params_UtilityCA \n" << Params_UtilityCA.__str__();
	BUGGER(msg) << "Params_UtilityCO \n" << Params_UtilityCO.__str__();
	
	_is_setUp = 1;
	if (and_load_data) _is_setUp = 2;
}


void elm::Model2::tearDown()
{
	_is_setUp = 0;
//	if (Data_UtilityCA  ) { Data_UtilityCA   ->decref(); Data_UtilityCA=nullptr;}
//	if (Data_UtilityCO  ) { Data_UtilityCO   ->decref(); Data_UtilityCO=nullptr;}
//	if (Data_QuantityCA ) { Data_QuantityCA  ->decref(); Data_QuantityCA=nullptr;}
//	if (Data_QuantLogSum) { Data_QuantLogSum ->decref(); Data_QuantLogSum=nullptr;}
//	if (Data_LogSum     ) { Data_LogSum      ->decref(); Data_LogSum=nullptr;}
//	if (Data_Avail      ) { Data_Avail       ->decref(); Data_Avail=nullptr;}
	
	probability_dispatcher.reset();
	gradient_dispatcher.reset();
	loglike_dispatcher.reset();
}

