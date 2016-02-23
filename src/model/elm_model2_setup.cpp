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


#include <cstring>
#include "elm_model2.h"
#include "elm_sql_scrape.h"
#include "elm_names.h"
#include <iostream>

#include "elm_parameter2.h"

using namespace etk;
using namespace elm;
using namespace std;



etk::strvec elm::__identify_needs(const ComponentList& Input_List)
{
	etk::strvec u_ca;
	
	for (unsigned b=0; b<Input_List.size(); b++) {
		u_ca.push_back_if_unique(Input_List[b].data_name);
	}
	
	return u_ca;
}

etk::strvec elm::__identify_needs(const LinearCOBundle_1& Input_ListMap)
{
	etk::strvec u_ca;
	
	for (auto b=Input_ListMap.begin(); b!=Input_ListMap.end(); b++) {
		for (auto i=b->second.begin(); i!=b->second.end(); i++) {
			u_ca.push_back_if_unique(i->data_name);
		}
	}
	
	return u_ca;
}


etk::strvec elm::__identify_needs(const LinearCOBundle_2& Input_EdgeMap)
{
	etk::strvec data_names;
	for (auto iter=Input_EdgeMap.begin(); iter!=Input_EdgeMap.end(); iter++) {
		__identify_additional_needs(iter->second, data_names);
	}
	return data_names;
}

void elm::__identify_additional_needs(const ComponentList& Input_List, etk::strvec& needs)
{
	for (unsigned b=0; b<Input_List.size(); b++) {
		needs.push_back_if_unique(Input_List[b].data_name);
	}
}


void __check_validity_of_needs(const etk::strvec& needs, Fountain*	_Data, int k, etk::logging_service* msg)
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
(	sherpa&			self
 ,	Fountain*				_fount
 ,	VAS_System&				Xylem
 ,	ComponentList*			Input_UtilityCA
 ,	LinearCOBundle_1*		Input_UtilityCO
 ,	paramArray*				Params_UtilityCA
 ,	paramArray*				Params_UtilityCO
 ,	etk::logging_service*	msg
)
{
	size_t slot, slot2;


	// utility.ca //
	if (Input_UtilityCA && Params_UtilityCA) {
		// First, populate the data_port
		etk::strvec u_ca = __identify_needs((*Input_UtilityCA));
		__check_validity_of_needs(u_ca, _fount, IDCA, msg);
		
		// Second, resize the paramArray
		BUGGER_(msg, "setting Params_?CA size to ("<<u_ca.size()<<")");
		(*Params_UtilityCA).resize(u_ca.size());
		
		// Third, populate the paramArray
		for (unsigned b=0; b<(*Input_UtilityCA).size(); b++) {
			slot = u_ca.push_back_if_unique((*Input_UtilityCA)[b].data_name);
			(*Params_UtilityCA)(slot) = self._generate_parameter((*Input_UtilityCA)[b].param_name,(*Input_UtilityCA)[b].multiplier);
		}
	}
	
	
	// utility.co //
	if (Input_UtilityCO && Params_UtilityCO) {
		// First, populate the data_port
		etk::strvec u_co = __identify_needs((*Input_UtilityCO));
		__check_validity_of_needs(u_co, _fount, IDCO, msg);

		// Second, resize the paramArray
		auto s = _fount ? _fount->nAlts() : Xylem.n_elemental();
		BUGGER_(msg, "setting Params_?CO size to ("<<u_co.size()<<","<< s <<")");
		(*Params_UtilityCO).resize(u_co.size(), s);



		// Third, populate the paramArray
		int count =0;
		for (auto top_i=(*Input_UtilityCO).begin(); top_i!=(*Input_UtilityCO).end(); top_i++) {
			for (auto i=top_i->second.begin(); i!=top_i->second.end(); i++) {

			BUGGER_(msg, "setting Params_?CO count="<<(++count));
			slot = u_co.push_back_if_unique(i->data_name);
			if (top_i->first==cellcode_empty) {
				OOPS("utilityco input does not specify an alternative.\n"
					 "Inputs in the utilityco space need to identify an alternative.");
			}
			slot2 = Xylem.slot_from_code(top_i->first);
			(*Params_UtilityCO)(slot,slot2) = self._generate_parameter(i->param_name, i->multiplier);

			}
		}
	}
	
	BUGGER_(msg, "_setUp_linear_data_and_params complete");
	
}



void _setUp_linear_data_and_params_edges
(	sherpa&			self
 ,	VAS_System&				Xylem
 ,	LinearCOBundle_2&		Input_Alloc
 ,	paramArray&				Params_Alloc
 ,	etk::logging_service*	msg
)
{
	size_t slot, slot2;

	
	
	// idco //
	etk::strvec data_names = __identify_needs(Input_Alloc);

	// resize the paramArray
	auto s = Xylem.n_compet_alloc();
	BUGGER_(msg, "setting Params_Alloc size to ("<<data_names.size()<<","<< s <<")");
	Params_Alloc.resize(data_names.size(), s);
	
	// Third, populate the paramArray
	for (auto iter=Input_Alloc.begin(); iter!=Input_Alloc.end(); iter++) {
		auto edge = Xylem.edge_from_codes(iter->first.up, iter->first.dn);
		if (!edge) {
			OOPS("allocation input up=",iter->first.up," dn=",iter->first.dn," does not specify a valid network link.");
		}
		
		try {
			slot2 = edge->alloc_slot();
		} SPOO {
			slot2 = UINT_MAX;
		}
		
		if (slot2 != UINT_MAX) {
			for (unsigned b=0; b<iter->second.size(); b++) {
				BUGGER_(msg, "setting Params_?CO b="<<b);
				slot = data_names.push_back_if_unique(iter->second[b].data_name);
				Params_Alloc(slot,slot2) = self._generate_parameter(iter->second[b].param_name, iter->second[b].multiplier);
			}
		}
		
		
	}
	
	
	
	
	BUGGER_(msg, "_setUp_linear_data_and_params_edges complete");
	
}




void elm::Model2::_setUp_utility_data_and_params()
{
	BUGGER(msg) << "--Params_Utility--\n";

	_setUp_linear_data_and_params
	(	*this
	 ,	_fountain()
	 ,	Xylem
	 ,	&Input_Utility.ca
	 ,	&Input_Utility.co
	 ,	&Params_UtilityCA
	 ,	&Params_UtilityCO
	 ,	&msg
	);

	BUGGER(msg) << "Params_UtilityCA \n" << Params_UtilityCA.__str__();
	BUGGER(msg) << "Params_UtilityCO \n" << Params_UtilityCO.__str__();

}

void elm::Model2::_setUp_quantity_data_and_params()
{
	BUGGER(msg) << "--Params_Quantity--\n";

	_setUp_linear_data_and_params
	(	*this
	 ,	_fountain()
	 ,	Xylem
	 ,	&Input_QuantityCA
	 ,	nullptr
	 ,	&Params_QuantityCA
	 ,	nullptr
	 ,	&msg
	);

	BUGGER(msg) << "Params_QuantityCA \n" << Params_QuantityCA.__str__();

}

void elm::Model2::_setUp_samplefactor_data_and_params()
{
	BUGGER(msg) << "--Params_Sampling--\n";

	_setUp_linear_data_and_params
	(	*this
	 ,	_fountain()
	 ,	Xylem
	 ,	&Input_Sampling.ca
	 ,	&Input_Sampling.co
	 ,	&Params_SamplingCA
	 ,	&Params_SamplingCO
	 ,	&msg
	);

	BUGGER(msg) << "Params_SamplingCA \n" << Params_SamplingCA.__str__();
	BUGGER(msg) << "Params_SamplingCO \n" << Params_SamplingCO.__str__();

}

void elm::Model2::_setUp_allocation_data_and_params()
{
	BUGGER(msg) << "--Params_Allocation--\n";

	_setUp_linear_data_and_params_edges(*this, Xylem, Input_Edges, Params_Edges, &msg);

	BUGGER(msg) << "Params_Allocation \n" << Params_Edges.__str__();

}



//std::string __GetWeight(const std::string& varname, bool reweight, Facet* _Data) {
//	std::string w;	
//	if (reweight) {
//		std::ostringstream sql1;
//		sql1 << "SELECT SUM("<<varname<<") FROM " << _Data->tbl_idco();
//		double average_weight = _Data->sql_statement(sql1)->execute()->getDouble(0); 
//		
//		average_weight /= _Data->nCases();
//		ostringstream sql2;
//		sql2 << "("<<varname<<")/"<<average_weight;
//		w = (sql2.str());
//	} else {
//		w = (varname);
//	}	
//	return w;
//}

std::string elm::Model2::auto_rescale_weights(const double& mean_weight)
{
	if (Data_Weight) {

		double current_total = Data_Weight->_repository.sum();
		double needed_scale_factor = (mean_weight*Data_Weight->_repository.size())/current_total;
		if ((needed_scale_factor > 1.0001) || (needed_scale_factor < 0.9999)) {
			Data_Weight_rescaled = boosted::make_shared<elm::darray>(*Data_Weight,needed_scale_factor);
			weight_scale_factor = needed_scale_factor;
			
			std::ostringstream s;
			s << "automatically rescaled weights (total initial weight "<<current_total
						<<" scaled by "<<weight_scale_factor<<" across "<<Data_Weight->nCases()
						<<" cases)";
			
			INFO(msg) << s.str();
			return s.str();
		} else {
			return "did not automatically rescale weights";
		}
	} else {
		return "no weights to automatically rescale";
	}
}

void elm::Model2::restore_scale_weights()
{
	Data_Weight_rescaled.reset();
	weight_scale_factor = 1.0;
	
	gradient_dispatcher.reset();
	probability_dispatcher.reset();
	loglike_dispatcher.reset();
}

double elm::Model2::get_weight_scale_factor() const
{
	return weight_scale_factor;
}


void elm::Model2::scan_for_multiple_choices()
{
	BUGGER(msg) << "Scanning choice data for instances of multiple or non-unit choice....";
	
	// multichoice
	if (Data_MultiChoice.size() != Data_Choice->nCases()) {
		Data_MultiChoice.resize(Data_Choice->nCases());
	}
	
	for (unsigned c=0;c<Data_Choice->nCases();c++) {
		size_t m=c;
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
		} else {
			Data_MultiChoice.input(false,m);
		}
	}
}


void elm::Model2::_pull_graph_from_db()
{
	if (option.suspend_xylem_rebuild) {
		OOPS("Xylem regrow is suspended");
	}
	BUGGER(msg) << "Rebuilding Xylem network...";
	elm::cellcode root = Xylem.root_cellcode();
	Xylem.touch();
	Xylem.regrow( &Input_LogSum, &Input_Edges, _fountain(), &root, &msg );
}


void elm::Model2::setUp(bool and_load_data)
{
	
	// MAYBE THIS IS NOT REALLY NEEDED?
//	if (is_provisioned()!=1) {
//		OOPS("data not provisioned");
//	}

//	BUGGER(msg) << "Setting up the model...";
	if (_is_setUp>=2 || (_is_setUp>=1 && !and_load_data)) {
		BUGGER(msg) << "The model is already set up.";
		return;
	}
	
	INFO(msg) << "Setting up the model...";

	if (!option.suspend_xylem_rebuild) _pull_graph_from_db();

	if (Data_UtilityCE.active()) {
		// Data_UtilityCE is only currently compatible with the full NGEV code.
		features |= MODELFEATURES_NESTING;
		features |= MODELFEATURES_ALLOCATION;
	}
	
	if (Xylem.n_branches() > 0) {
		BUGGER(msg) << "Setting model features to include nesting.";
		features |= MODELFEATURES_NESTING;
	}

	if (Xylem.n_compet_alloc() > 0) {
		BUGGER(msg) << "Setting model features to include nest allocation.";
		features |= MODELFEATURES_NESTING;
		features |= MODELFEATURES_ALLOCATION;
	}
	
	if (Input_QuantityCA.size()>0) {
		BUGGER(msg) << "Setting model features to include quantitative alternatives.";
		features |= MODELFEATURES_QUANTITATIVE;
	}
	
	BUGGER(msg) << "Setting up utility parameters...";
	_setUp_utility_data_and_params();
	if (features & MODELFEATURES_NESTING) {
		if (!option.suspend_xylem_rebuild) {
			elm::cellcode root = Xylem.root_cellcode();
			Xylem.touch();
			Xylem.regrow( &Input_LogSum, &Input_Edges, _fountain(), &root, &msg );
		}
		if (features & (MODELFEATURES_ALLOCATION|MODELFEATURES_QUANTITATIVE)) {
			_setUp_NGEV();
		} else {
			_setUp_NL();
		}
	} else if (features & MODELFEATURES_QUANTITATIVE) {
		_setUp_NGEV();
	} else {
		_setUp_MNL();
	}
	if (is_provisioned()>0) scan_for_multiple_choices();

	
	if (Input_Sampling.ca.size() || Input_Sampling.co.metasize()) {
		_setUp_samplefactor_data_and_params();
	}
	
	if (features & MODELFEATURES_ALLOCATION) {
		_setUp_allocation_data_and_params();
	}

	if (features & MODELFEATURES_QUANTITATIVE) {
		_setUp_quantity_data_and_params();
	}
	
	_setUp_coef_and_grad_arrays();
	
	if ((features & MODELFEATURES_NESTING) && (!option.suspend_xylem_rebuild)) {
		Xylem.repoint_parameters(*Coef_LogSum, NULL);
		elm::cellcode root = Xylem.root_cellcode();
		Xylem.regrow( &Input_LogSum, &Input_Edges, _fountain(), &root, &msg );
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
	
	clear_cache();
	
	CaseLogLike.destroy();
	Data_UtilityCE.clear();
}












std::string elm::Model2::_subprovision(const std::string& name, boosted::shared_ptr<const darray>& storage,
								const std::map< std::string, boosted::shared_ptr<const darray> >& input,
								const std::map<std::string, darray_req>& need,
								std::map<std::string, size_t>& ncases)
{

	auto i = input.find(name);
	auto n = need.find(name);
	if (i!=input.end()) {
		// This pointer is provisioned
		if (n->second.satisfied_by(&*i->second)<0) {
			// if it does not satisty the need, add to the exception
			storage = nullptr;
			return cat("\ndata for ",name," is provisioned by an array that does not satisfy the need");
		}
		storage = i->second;
		ncases[name] = storage->nCases();
	} else {
		// This pointer is not provisioned
		storage = nullptr;
		if (n != need.end()) {
			// if it is needed, add to the exception
			return cat("\ndata for ",name," is needed but not provisioned");
		}
	}
	return "";
}


void elm::Model2::provision()
{
	OOPS("Calling provision with no argument and no db set is not supported");
}

void elm::Model2::provision(const std::map< std::string, boosted::shared_ptr<const darray> >& input)
{
	BUGGER(msg) << "Provisioning model data...";
	
	std::string ret = "";
	
	std::map<std::string, darray_req> need = needs();
	std::map<std::string, size_t> ncases;
	
	ret += _subprovision("UtilityCA", Data_UtilityCA, input, need, ncases);
	ret += _subprovision("UtilityCO", Data_UtilityCO, input, need, ncases);
	ret += _subprovision("QuantityCA", Data_QuantityCA, input, need, ncases);
	ret += _subprovision("SamplingCA", Data_SamplingCA, input, need, ncases);
	ret += _subprovision("SamplingCO", Data_SamplingCO, input, need, ncases);
	ret += _subprovision("Allocation", Data_Allocation, input, need, ncases);

	ret += _subprovision("Avail",  Data_Avail , input, need, ncases);
	ret += _subprovision("Choice", Data_Choice, input, need, ncases);
	if (Data_Choice) scan_for_multiple_choices();
	ret += _subprovision("Weight", Data_Weight, input, need, ncases);

	
	auto caseiter = ncases.begin();
	
	if (caseiter==ncases.end()) {
		nCases = 0;
		_nCases_recall = nCases;
	} else {
		
		size_t nc = caseiter->second;
		for (; caseiter!=ncases.end(); caseiter++) {
			if (nc != caseiter->second) {
				OOPS_PROVISIONING("inconsistent numbers or cases");
			}
		}
		
		nCases = nc;
		_nCases_recall = nCases;
		
	}
	if (!ret.empty()) {
		OOPS_PROVISIONING(ret);
	}
}

std::map<std::string, darray_req> elm::Model2::needs() const
{
	std::map<std::string, darray_req> requires;
	
	etk::strvec u_ca = __identify_needs(Input_Utility.ca);
	if (u_ca.size()) {
		requires["UtilityCA"] = darray_req (3,NPY_DOUBLE,Xylem.n_elemental());
		requires["UtilityCA"].set_variables(u_ca);
	}
	
	etk::strvec u_co = __identify_needs(Input_Utility.co);
	if (u_co.size()) {
		requires["UtilityCO"] = darray_req (2,NPY_DOUBLE);
		requires["UtilityCO"].set_variables(u_co);
	}

	etk::strvec q_ca = __identify_needs(Input_QuantityCA);
	if (q_ca.size()) {
		requires["QuantityCA"] = darray_req (3,NPY_DOUBLE,Xylem.n_elemental());
		requires["QuantityCA"].set_variables(q_ca);
	}


	etk::strvec s_ca = __identify_needs(Input_Sampling.ca);
	if (s_ca.size()) {
		requires["SamplingCA"] = darray_req (3,NPY_DOUBLE,Xylem.n_elemental());
		requires["SamplingCA"].set_variables(s_ca);
	}
	
	etk::strvec s_co = __identify_needs(Input_Sampling.co);
	if (s_co.size()) {
		requires["SamplingCO"] = darray_req (2,NPY_DOUBLE);
		requires["SamplingCO"].set_variables(s_co);
	}
	
	etk::strvec allo = __identify_needs(Input_Edges);
	if (allo.size()) {
		requires["Allocation"] = darray_req (2,NPY_DOUBLE);
		requires["Allocation"].set_variables(allo);
	}
	
	requires["Avail"] = darray_req (3,NPY_BOOL);
	requires["Weight"] = darray_req (2,NPY_DOUBLE);
	requires["Choice"] = darray_req (3,NPY_DOUBLE);
	
	return requires;
}



#define MISSING_BUT_NEEDED 0x1
#define GIVEN_BUT_WRONG    0x2
#define NOT_NEEDED         0x0
#define GIVEN_CORRECTLY    0x0


int elm::Model2::_is_subprovisioned(const std::string& name, const elm::darray_ptr& arr, const std::map<std::string, darray_req>& requires, const bool& ex) const
{
	auto i = requires.find(name);
	if (i!=requires.end()) {
		if (!arr) {
			if (ex) {
				OOPS(name," is provisioned incorrectly, needs <",i->second.__str__(),"> but not provided");
			} else {
				return MISSING_BUT_NEEDED;
			}
		}
		if (i->second.satisfied_by(&*arr)==0) {
			return GIVEN_CORRECTLY;
		} else {
			if (ex) {
				OOPS(name," is provisioned incorrectly, needs <",i->second.__str__(),"> but provides <",arr->__str__(),">");
			} else {
				return GIVEN_BUT_WRONG;
			}
		}
	} else {
		return NOT_NEEDED;
	}
}

int elm::Model2::is_provisioned(bool ex) const
{
	std::map<std::string, darray_req> requires = needs();
	
	int i = 0;
	i |= _is_subprovisioned("UtilityCA", Data_UtilityCA, requires, ex);
	i |= _is_subprovisioned("UtilityCO", Data_UtilityCO, requires, ex);
	i |= _is_subprovisioned("QuantityCA", Data_QuantityCA, requires, ex);
	i |= _is_subprovisioned("SamplingCA", Data_SamplingCA, requires, ex);
	i |= _is_subprovisioned("SamplingCO", Data_SamplingCO, requires, ex);
	i |= _is_subprovisioned("Allocation", Data_Allocation, requires, ex);
	
	i |= _is_subprovisioned("Avail", Data_Avail, requires, ex);
	i |= _is_subprovisioned("Weight", Data_Weight, requires, ex);
	i |= _is_subprovisioned("Choice", Data_Choice, requires, ex);
	
	if (i & GIVEN_BUT_WRONG) {
		return -1;
	}
	if (i & MISSING_BUT_NEEDED) {
		return 0;
	}
	return 1;
}

const elm::darray* elm::Model2::Data(const std::string& label)
{
	if (label=="UtilityCA") return Data_UtilityCA ?   (&*Data_UtilityCA) : nullptr;
	if (label=="UtilityCO") return Data_UtilityCO ?   (&*Data_UtilityCO) : nullptr;
	if (label=="QuantityCA") return Data_QuantityCA ? (&*Data_QuantityCA) : nullptr;
	if (label=="SamplingCA") return Data_SamplingCA ? (&*Data_SamplingCA) : nullptr;
	if (label=="SamplingCO") return Data_SamplingCO ? (&*Data_SamplingCO) : nullptr;
	if (label=="Allocation") return Data_Allocation ? (&*Data_Allocation) : nullptr;

	if (label=="Avail" ) return Data_Avail ?  (&*Data_Avail ) : nullptr;
	if (label=="Choice") return Data_Choice ? (&*Data_Choice) : nullptr;
	if (label=="Weight") return Data_Weight ? (&*Data_Weight) : nullptr;

	OOPS(label, " is not a valid label for model data");
	
}

elm::darray* elm::Model2::DataEdit(const std::string& label)
{
	if (label=="UtilityCA") return Data_UtilityCA ?   const_cast<elm::darray*>(&*Data_UtilityCA) : nullptr;
	if (label=="UtilityCO") return Data_UtilityCO ?   const_cast<elm::darray*>(&*Data_UtilityCO) : nullptr;
	if (label=="QuantityCA") return Data_QuantityCA ? const_cast<elm::darray*>(&*Data_QuantityCA) : nullptr;
	if (label=="SamplingCA") return Data_SamplingCA ? const_cast<elm::darray*>(&*Data_SamplingCA) : nullptr;
	if (label=="SamplingCO") return Data_SamplingCO ? const_cast<elm::darray*>(&*Data_SamplingCO) : nullptr;
	if (label=="Allocation") return Data_Allocation ? const_cast<elm::darray*>(&*Data_Allocation) : nullptr;

	if (label=="Avail" ) return Data_Avail ?  const_cast<elm::darray*>(&*Data_Avail ) : nullptr;
	if (label=="Choice") return Data_Choice ? const_cast<elm::darray*>(&*Data_Choice) : nullptr;
	if (label=="Weight") return Data_Weight ? const_cast<elm::darray*>(&*Data_Weight) : nullptr;

	OOPS(label, " is not a valid label for model data");
	
}


const etk::ndarray* elm::Model2::Coef(const std::string& label)
{
	
	if (FCurrent.size1()!=dF()) {
		FCurrent.resize(dF());
	}

	for (unsigned i=0; i<dF(); i++) {
		freedom_info* f = &(FInfo[FNames[i]]);
		FCurrent[i] = f->value;
	}
	
	if (label=="UtilityCA") {
		if (Params_UtilityCA.length()==0) {
			_setUp_utility_data_and_params();
		}
		Coef_UtilityCA.resize_if_needed(Params_UtilityCA);
		pull_from_freedoms        (Params_UtilityCA  , *Coef_UtilityCA  , *ReadFCurrent());
		return &Coef_UtilityCA;
	}
	if (label=="UtilityCO") {
		if (Params_UtilityCO.length()==0) {
			_setUp_utility_data_and_params();
		}
		Coef_UtilityCO.resize_if_needed(Params_UtilityCO);
		pull_from_freedoms        (Params_UtilityCO  , *Coef_UtilityCO  , *ReadFCurrent());
		return &Coef_UtilityCO   ;
	}
	if (label=="SamplingCA") {
		if (Params_SamplingCA.length()==0) {
			_setUp_samplefactor_data_and_params();
		}
		Coef_SamplingCA.resize_if_needed(Params_SamplingCA);
		pull_from_freedoms        (Params_SamplingCA , *Coef_SamplingCA , *ReadFCurrent());
		return &Coef_SamplingCA ;
	}
	if (label=="SamplingCO") {
		if (Params_SamplingCO.length()==0) {
			_setUp_samplefactor_data_and_params();
		}
		Coef_SamplingCO.resize_if_needed(Params_SamplingCO);
		pull_from_freedoms        (Params_SamplingCO , *Coef_SamplingCO , *ReadFCurrent());
		return &Coef_SamplingCO ;
	}
	if (label=="Allocation") {
		if (Params_Edges.length()==0) {
			_setUp_allocation_data_and_params();
		}
		Coef_Edges.resize_if_needed(Params_Edges);
		pull_from_freedoms        (Params_Edges      , *Coef_Edges      , *ReadFCurrent());
		return &Coef_Edges ;
	}
	if (label=="QuantityCA") {
		if (Params_SamplingCA.length()==0) {
			_setUp_quantity_data_and_params();
		}
		Coef_QuantityCA.resize_if_needed(Params_QuantityCA);
		pull_and_exp_from_freedoms(Params_QuantityCA , *Coef_QuantityCA , *ReadFCurrent());
		return &Coef_QuantityCA ;
	}
	if (label=="LogSum") {
		if (Params_LogSum.length()==0) {
			OOPS("calling Coef('LogSum') currently requires the model is already setup");
		}
		Coef_LogSum.resize_if_needed(Params_LogSum);
		pull_from_freedoms        (Params_LogSum     , *Coef_LogSum     , *ReadFCurrent());
		return &Coef_LogSum ;
	}
	if (label=="QuantLogSum") {
		if (Params_QuantLogSum.length()==0) {
			OOPS("calling Coef('QuantLogSum') currently requires the model is already setup");
		}
		Coef_QuantLogSum.resize_if_needed(Params_QuantLogSum);
		pull_from_freedoms        (Params_QuantLogSum, *Coef_QuantLogSum, *ReadFCurrent());
		return &Coef_QuantLogSum ;
	}
	OOPS(label, " is not a valid label for model coefficients");
	
}



