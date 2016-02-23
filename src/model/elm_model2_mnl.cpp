/*
 *  elm_model.cpp
 *
 *  Copyright 2007-2015 Jeffrey Newman
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
#include "larch_modelparameter.h"

#include "elm_workshop_mnl_gradient.h"
#include "elm_workshop_mnl_prob.h"
#include "elm_workshop_loglike.h"
#include "elm_workshop_nl_probability.h"

using namespace etk;
using namespace elm;
using namespace std;



void elm::Model2::_setUp_MNL()
{
	INFO(msg)<< "Setting up MNL model..." ;
	
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
	
	Workspace.resize(nElementals);
	
		
	sherpa::allocate_memory();
	
	Data_MultiChoice.resize(nCases);
	Data_MultiChoice.initialize(true);
	
	INFO(msg)<< "Set up MNL model complete." ;
	
}










void elm::Model2::pull_from_freedoms(const paramArray& par,       double* ops, const double* fr, const bool& log)
{
	for (unsigned i=0; i<par.length(); i++) {
		if (par[i]){
			ops[i] = par[i]->pullvalue(fr);
//			if (log) {
//				BUGGER(msg) << "PULL "<<par[i]->freedomName()<<" "<<ops[i]<<" into slot "<<i;
//			}
		} else {
		}
	}
}
void elm::Model2::push_to_freedoms  ( paramArray& par, const double* ops,       double* fr)
{
	for (unsigned i=0; i<par.length(); i++) {
		if (par[i]){
			par[i]->pushvalue(fr,ops[i]);
		}
	}
}

void elm::Model2::_setUp_coef_and_grad_arrays()
{
	Coef_UtilityCA.resize(Params_UtilityCA.size1(),Params_UtilityCA.size2(),Params_UtilityCA.size3());
	Coef_UtilityCO.resize(Params_UtilityCO.size1(),Params_UtilityCO.size2(),Params_UtilityCO.size3());
	Coef_QuantityCA.resize(Params_QuantityCA.size1(),Params_QuantityCA.size2(),Params_QuantityCA.size3());
	Coef_QuantLogSum.resize(Params_QuantLogSum.size1(),Params_QuantLogSum.size2(),Params_QuantLogSum.size3());
	Coef_LogSum.resize(Params_LogSum.size1(),Params_LogSum.size2(),Params_LogSum.size3());
	Coef_Edges.resize(Params_Edges.size1(), Params_Edges.size2(), Params_Edges.size3());
	Coef_SamplingCA.resize(Params_SamplingCA.size1(),Params_SamplingCA.size2(),Params_SamplingCA.size3());
	Coef_SamplingCO.resize(Params_SamplingCO.size1(),Params_SamplingCO.size2(),Params_SamplingCO.size3());


	Grad_UtilityCA.resize(Params_UtilityCA.size1(),Params_UtilityCA.size2(),Params_UtilityCA.size3());
	Grad_UtilityCO.resize(Params_UtilityCO.size1(),Params_UtilityCO.size2(),Params_UtilityCO.size3());
	Grad_QuantityCA.resize(Params_QuantityCA.size1(),Params_QuantityCA.size2(),Params_QuantityCA.size3());
	Grad_QuantLogSum.resize(Params_QuantLogSum.size1(),Params_QuantLogSum.size2(),Params_QuantLogSum.size3());
	Grad_LogSum.resize(Params_LogSum.size1(),Params_LogSum.size2(),Params_LogSum.size3());
}

void elm::Model2::pull_coefficients_from_freedoms()
{
	pull_from_freedoms        (Params_UtilityCA  , *Coef_UtilityCA  , *ReadFCurrent());
	pull_from_freedoms        (Params_UtilityCO  , *Coef_UtilityCO  , *ReadFCurrent());
	pull_from_freedoms        (Params_SamplingCA , *Coef_SamplingCA , *ReadFCurrent());
	pull_from_freedoms        (Params_SamplingCO , *Coef_SamplingCO , *ReadFCurrent());
	pull_and_exp_from_freedoms(Params_QuantityCA , *Coef_QuantityCA , *ReadFCurrent());
	pull_from_freedoms        (Params_QuantLogSum, *Coef_QuantLogSum, *ReadFCurrent());
	pull_from_freedoms        (Params_LogSum     , *Coef_LogSum     , *ReadFCurrent(), true);
	pull_from_freedoms        (Params_Edges      , *Coef_Edges      , *ReadFCurrent());
}

void elm::Model2::freshen()
{
	setUp(false);
	allocate_memory();
	
	elm::cellcode root = Xylem.root_cellcode();
	if (!option.suspend_xylem_rebuild) Xylem.regrow( &Input_LogSum, &Input_Edges, _fountain(), &root, &msg );
	nElementals = Xylem.n_elemental();
	nNests = Xylem.n_branches();
	nNodes = Xylem.size();
	
	_setUp_utility_data_and_params();
	_setUp_samplefactor_data_and_params();
	_setUp_allocation_data_and_params();
	_setUp_quantity_data_and_params();

	Coef_UtilityCA.resize_if_needed(Params_UtilityCA);
	Coef_UtilityCO.resize_if_needed(Params_UtilityCO);
	Coef_QuantityCA.resize_if_needed(Params_QuantityCA);
	Coef_QuantLogSum.resize_if_needed(Params_QuantLogSum);
	Coef_LogSum.resize_if_needed(Params_LogSum);
	Coef_Edges.resize_if_needed(Params_Edges);
	Coef_SamplingCA.resize_if_needed(Params_SamplingCA);
	Coef_SamplingCO.resize_if_needed(Params_SamplingCO);

	pull_coefficients_from_freedoms();
}


void elm::Model2::calculate_probability()
{

	if ((features & MODELFEATURES_ALLOCATION)||(features & MODELFEATURES_QUANTITATIVE)) {
		ngev_probability();
	} else if ((features & MODELFEATURES_NESTING)) {
		nl_probability();
	} else {
		// MNL
		mnl_probability();
	}

}

void elm::Model2::calculate_utility_only()
{
	if ((features & MODELFEATURES_ALLOCATION)||(features & MODELFEATURES_QUANTITATIVE)) {
		ngev_probability();
	} else if ((features & MODELFEATURES_NESTING)) {
		nl_probability();
	} else {
		// MNL
		pull_coefficients_from_freedoms();
		
		Utility.resize(nCases,_fountain()->nAlts());
		
		__logit_utility(Utility, Data_UtilityCA, Data_UtilityCO, &Coef_UtilityCA, &Coef_UtilityCO, 0);
		
		unsigned c;
		unsigned a;
		
		// Unavailable alternatives become -INF
		for (c=0;c<nCases;c++) for (a=0;a<nElementals;a++)
			if (!Data_Avail->boolvalue(c,a)) Utility(c,a) = -INF;
		
		
	}
	
}


std::shared_ptr<etk::ndarray> elm::Model2::calc_utility() const
{
	return _calc_utility(Data_UtilityCO ? (&Data_UtilityCO->_repository) : nullptr,
						 Data_UtilityCA ? (&Data_UtilityCA->_repository) : nullptr,
						 Data_Avail ? &Data_Avail->_repository : nullptr);
}


std::shared_ptr<etk::ndarray> elm::Model2::calc_utility(const std::map< std::string, boosted::shared_ptr<const elm::darray> >& prov)
{
	provision(prov);
	return calc_utility();
}



//std::shared_ptr<ndarray> elm::Model2::calc_utility(datamatrix_t* dco, datamatrix_t* dca, datamatrix_t* av) const
//{
//	if (dco && dca) {
//		if (dco->dimty==case_alt_var && dca->dimty==case_var) {
//			std::swap(dco,dca);
//		}
//	} else if (dco && dco->dimty==case_alt_var) {
//		std::swap(dco,dca);
//	} else if (dca && dca->dimty==case_var) {
//		std::swap(dco,dca);
//	} else {
//		OOPS("no input data");
//	}
//	return calc_utility(dco ? &dco->_repository : nullptr, dca ? &dca->_repository : nullptr, av ? &av->_repository : nullptr);
//}

std::shared_ptr<ndarray> elm::Model2::calc_utility(ndarray* dco, ndarray* dca, ndarray* av) const
{
	return _calc_utility( dco, dca, av);
}

std::shared_ptr<ndarray> elm::Model2::_calc_utility(const ndarray* dco, const ndarray* dca, const ndarray* av) const
{
	if (nElementals==0) {
		OOPS("Model is not setUp (it has no alternatives defined)");
	}

	// swap arrays if they are backwards...
	if (dco && dca) {
		if (dco->ndim()==3 && dca->ndim()==2) {
			std::swap(dco,dca);
		}
	} else if (dco && dco->ndim()==3) {
		std::swap(dco,dca);
	} else if (dca && dca->ndim()==2) {
		std::swap(dco,dca);
	} else if (!dco && !dca) {
		OOPS("no input data");
	}
	if (dco && dco->ndim()==1) {
		OOPS("idco input array has only one dimension, did you mean to have only one case (if so, make 1st dimension size=1)");
	}
	if (dca && dca->ndim()==2) {
		OOPS("idca input array has only two dimensions, did you mean to have only one case (if so, make 1st dimension size=1)");
	}

	
	if (Input_Utility.ca.size() && !dca) OOPS("idca data needed but not given");
	if (Input_Utility.co.metasize() && !dco) OOPS("idco data needed but not given");

	
	size_t ncases = 0;
	if (dco && dca) {
		if (dco->size1() != dca->size1()) {
			OOPS("input data arrays do not have same number of cases");
		}
		ncases = dco->size1();
	} else if (dco) {
		ncases = dco->size1();
	} else if (dca) {
		ncases = dca->size1();
	} else {
		OOPS("no input data");
	}
 	
	
	size_t nalts = nElementals;
	if (dca) {
		if (dca->size2() != nElementals) {
			OOPS("input ca data array does not have correct number of alternatives, needs ",nElementals,", has ",dca->size2());
		}
	}
	
	std::shared_ptr<ndarray> U = std::make_shared<ndarray>(ncases,nalts);
		
	__logit_utility_arrays(*U, dca, dco, Coef_UtilityCA, Coef_UtilityCO, 0);

	if (av) {
		for (auto c=0;c<ncases;c++) for (auto a=0;a<nalts;a++)
			if (!av->bool_at(c,a)) (*U)(c,a) = -INF;
	}

	return U;
}

std::shared_ptr<ndarray> elm::Model2::calc_probability(ndarray* u) const
{
	if ((features & MODELFEATURES_ALLOCATION)) {
		OOPS("not implemented");  //TODO
	} else if ((features & MODELFEATURES_NESTING)) {
		OOPS("not implemented");  //TODO
	} else {
		
		if (!u) {
			OOPS("no utility given");
		}
		
		std::shared_ptr<ndarray> PR = std::make_shared<ndarray>(u->size1(), u->size2());
		*PR = *u;
		PR->exp();
		PR->prob_scale_2();
		
		return PR;
	}
}



std::shared_ptr<ndarray> elm::Model2::calc_logsums(ndarray* u) const
{
	if (!u) {
		OOPS("no utility given");
	}


	std::shared_ptr<ndarray> LogSum = std::make_shared<ndarray>(u->size1());
	if ((features & MODELFEATURES_ALLOCATION)) {
		TODO;
	} else if ((features & MODELFEATURES_NESTING)) {
				
		std::shared_ptr<ndarray> utility_workspace = std::make_shared<ndarray>(nNodes);
		std::shared_ptr<ndarray> utility_savespace = std::make_shared<ndarray>(nNodes);

		for (int c=0; c<u->size1(); c++) {
			cblas_dcopy(nElementals, u->ptr(c), 1, utility_savespace->ptr(), 1	);
			__casewise_nl_utility(utility_savespace->ptr(), Xylem, utility_workspace->ptr());
			LogSum->at(c) = *(utility_savespace->ptr(nNodes-1));
		}

	} else {
		
		
		std::shared_ptr<ndarray> PR = std::make_shared<ndarray>(u->size1(), u->size2());
		*PR = *u;
		PR->exp();
		PR->logsums_2(&*LogSum);
		
	}
	
	return LogSum;
}

//std::shared_ptr<ndarray> elm::Model2::calc_utility_probability(datamatrix_t* dco, datamatrix_t* dca, datamatrix_t* av) const
//{
//	return calc_probability( &*calc_utility(dco,dca,av) );
//}

std::shared_ptr<ndarray> elm::Model2::calc_utility_probability(ndarray* dco, ndarray* dca, ndarray* av) const
{
	return calc_probability( &*calc_utility(dco,dca,av) );
}

//std::shared_ptr<ndarray> elm::Model2::calc_utility_logsums(datamatrix_t* dco, datamatrix_t* dca, datamatrix_t* av) const
//{
//	return calc_logsums( &*calc_utility(dco,dca,av) );
//}

std::shared_ptr<ndarray> elm::Model2::calc_utility_logsums(ndarray* dco, ndarray* dca, ndarray* av) const
{
	return calc_logsums( &*calc_utility(dco,dca,av) );
}

#include "etk_workshop.h"

boosted::shared_ptr<workshop> elm::Model2::make_shared_workshop_mnl_probability ()
{
//	BUGGER(msg) << "CALL make_shared_workshop_mnl_probability()\n";
	return boosted::make_shared<elm::mnl_prob_w>(
			&Probability, &CaseLogLike, Data_UtilityCA, Data_UtilityCO, Data_Avail, Data_Choice,
			&Coef_UtilityCA, &Coef_UtilityCO, 0, &msg);
}


void elm::Model2::mnl_probability()
{
	Probability.resize(nCases,Xylem.n_elemental());
	Probability.initialize(0.0);
	CaseLogLike.resize(nCases);
	Workspace.resize(Xylem.n_elemental());
	pull_coefficients_from_freedoms();
	
//	BUGGER(msg) << "Coef_UtilityCA\n" << Coef_UtilityCA.printall();
//	BUGGER(msg) << "Coef_UtilityCO\n" << Coef_UtilityCO.printall();
//	if (Data_UtilityCA && Data_UtilityCA->_repository.size1()>1) {
//		BUGGER(msg) << "Data_UtilityCA (case 0)\n" << Data_UtilityCA->printcase(0);
//		BUGGER(msg) << "Data_UtilityCA (case 1)\n" << Data_UtilityCA->printcase(1);
//		BUGGER(msg) << "Data_UtilityCA (case "<<nCases-1<<")\n" << Data_UtilityCA->printcase(nCases-1);
//	} else {
//		BUGGER(msg) << "Data_UtilityCA is NULL\n";
//	}
//	if (Data_UtilityCO && Data_UtilityCO->_repository.size1()>1) {
//		BUGGER(msg) << "Data_UtilityCO (case 0)\n" << Data_UtilityCO->printcase(0);
//		BUGGER(msg) << "Data_UtilityCO (case 1)\n" << Data_UtilityCO->printcase(1);
//		BUGGER(msg) << "Data_UtilityCO (case "<<nCases-1<<")\n" << Data_UtilityCO->printcase(nCases-1);
//	} else {
//		BUGGER(msg) << "Data_UtilityCO is NULL\n";
//	}
//	if (Data_Choice && Data_Choice->_repository.size1()>1) {
//		BUGGER(msg) << "Data_Choice (case 0)\n" << Data_Choice->printcase(0);
//		BUGGER(msg) << "Data_Choice (case 1)\n" << Data_Choice->printcase(1);
//		BUGGER(msg) << "Data_Choice (case "<<nCases-1<<")\n" << Data_Choice->printcase(nCases-1);
//	} else {
//		BUGGER(msg) << "Data_Choice is NULL\n";
//	}
//	if (Data_Avail && Data_Avail->_repository.size1()>1) {
//		BUGGER(msg) << "Data_Avail (case 0)\n" << Data_Avail->printboolcase(0);
//		BUGGER(msg) << "Data_Avail (case 1)\n" << Data_Avail->printboolcase(1);
//		BUGGER(msg) << "Data_Avail (case "<<nCases-1<<")\n" << Data_Avail->printboolcase(nCases-1);
//	} else {
//		BUGGER(msg) << "Data_Avail is NULL\n";
//	}

	if (option.threads>=1 && _ELM_USE_THREADS_ && Input_QuantityCA.size()==0) {
//		BUGGER(msg) << "Using multithreading with "<<option.threads<<" threads in mnl_probability()\n";
		#ifndef __APPLE__
//		BUGGER(msg) << "Using non-APPLE compiled\n";
		openblas_set_num_threads(1);
		#endif
//		
//		BUGGER_(&msg, "Coef_CO->size2() =. \n");
//		BUGGER_(&msg, "Coef_CO->size2() = "<<Coef_UtilityCO.size2()<<"\n");
//		BUGGER_(&msg, "Data_CO->nVars() =. \n");
//		BUGGER_(&msg, "Data_UtilityCO->nVars() = "<<Data_UtilityCO->nVars()<<"\n");
//		BUGGER_(&msg, "Data_CO->values(0,0) =. \n");
//		BUGGER_(&msg, "Data_CO->values(0,0) = "<< ((const void*) Data_UtilityCO->values(0,0))<<"\n");
//		BUGGER_(&msg, "Coef_CO->ptr() =.\n");
//		BUGGER_(&msg, "Coef_UtilityCO.ptr() = "<<Coef_UtilityCO.ptr()<<"\n");
//		BUGGER_(&msg, "Probability->ptr(0) =.\n");
//		BUGGER_(&msg, "Probability->ptr(0) = "<<Probability.ptr(0)<<"\n");
//		BUGGER_(&msg, "Probability.size1() = "<<Probability.size1()<<"\n");
//		BUGGER_(&msg, "Probability.size2() = "<<Probability.size2()<<"\n");
//		BUGGER_(&msg, "Probability.size3() = "<<Probability.size3()<<"\n");
//		#ifndef __APPLE__
//		BUGGER_(&msg, "openblas_get_num_threads() = "<<openblas_get_num_threads()<<"\n");
//		#endif
//		BUGGER_(&msg, "mongo... \n");
		
		boosted::function<boosted::shared_ptr<workshop> ()> workshop_builder =
			boosted::bind(&elm::Model2::make_shared_workshop_mnl_probability, this);
		USE_DISPATCH(probability_dispatcher,option.threads, nCases, workshop_builder);
		
	} else {
		unsigned c;
		unsigned a;
	
		if (Input_QuantityCA.size()>0) {
			BUGGER(msg) << "Not using multithreading but using quantity\n";
			__logit_utility(Probability, Data_QuantityCA, nullptr, &Coef_QuantityCA, nullptr, 0);


			Probability.log();


//			if (Probability.size1()>0) {
//				FATAL(msg) << "ProbabilityQ (case 0)\n" << Probability.printrow(0) ;
//				FATAL(msg) << "ProbabilityQ (case n)\n" << Probability.printrow(nCases-1) ;
//			} else {
//				FATAL(msg) << "ProbabilityQ (size 0)\n" ;
//			}

			__logit_utility(Probability, Data_UtilityCA, Data_UtilityCO, &Coef_UtilityCA, &Coef_UtilityCO, 1);
			// TODO: scale by theta


//			if (Probability.size1()>0) {
//				FATAL(msg) << "ProbabilityX (case 0)\n" << Probability.printrow(0) ;
//				FATAL(msg) << "ProbabilityX (case n)\n" << Probability.printrow(nCases-1) ;
//			} else {
//				FATAL(msg) << "ProbabilityX (size 0)\n" ;
//			}
		} else {
			BUGGER(msg) << "Not using multithreading or quantity\n";
			__logit_utility(Probability, Data_UtilityCA, Data_UtilityCO, &Coef_UtilityCA, &Coef_UtilityCO, 0);
		}
	
	
	
		
		
		// Unavailable alternatives become -INF
		for (c=0;c<nCases;c++) for (a=0;a<nElementals;a++)
			if (!Data_Avail->boolvalue(c,a)) Probability(c,a) = -INF;
		
		// Take exp of utility
		Probability.exp();
		
		// Zero out non-alternative Utility
		if (Probability.size2() > nElementals)
			for (c=0;c<nCases;c++) for (a=nElementals; a<Probability.size2(); a++)
				Probability(c,a) = 0;
		
		// Scaling to Probabilities
		Probability.prob_scale_2();
		
		// NANCHECK
		if (!option.mute_nan_warnings) {
			MONITOR(msg) << "checking for NAN in Probability..." ;
			bool found_nan (false);
			unsigned warningcount (0);
			for (c=0;c<nCases;c++) {
				found_nan = false;
				for (a=0; a<Probability.size2();a++) {
					if (isNan(Probability(c,a))) {
						found_nan = true;
						break;
					}
				}
				if (found_nan) {
					if (!warningcount) {
						WARN(msg) << "WARNING: Probability is NAN for caserow "<<c ;
						WARN(msg) << "W..prob: " << Probability.printrow(c) ;
					}
					warningcount++;
				}
			}
			if (warningcount>1) {
				WARN(msg) << "W......: and for "<<warningcount-1<<" other cases" ;
			}
		} // end if NANCHECK
	}
	
//	if (Probability.size1()>0) {
//		BUGGER(msg) << "Probability (case 0)\n" << Probability.printrow(0) ;
//		BUGGER(msg) << "Probability (case n)\n" << Probability.printrow(nCases-1) ;
//	} else {
//		BUGGER(msg) << "Probability (size 0)\n" ;
//	}
	
}






//void elm::Model2::simulate_probability
//( const std::string& tablename
//, const std::string& columnnameprefix
//, const std::string& columnnamesuffix
//, bool use_alt_codes
//, bool overwrite
//)
//{
//	setUp();
//	sim_probability( tablename, columnnameprefix, columnnamesuffix, use_alt_codes, overwrite);
//	tearDown();
//}

//void elm::Model2::simulate_conditional_probability
//( const std::string& tablename
//, const std::string& columnnameprefix
//, const std::string& columnnamesuffix
//, bool use_alt_codes
//, bool overwrite
//)
//{
//	setUp();
//	sim_conditional_probability( tablename, columnnameprefix, columnnamesuffix, use_alt_codes, overwrite);
//	tearDown();
//}



//void elm::Model2::sim_conditional_probability
//( const std::string& tablename
//, const std::string& columnnameprefix
//, const std::string& columnnamesuffix
//, bool use_alt_codes
//, bool overwrite
//)
//{
//	if (!_Data) OOPS("A database must be linked to this model to do this.");
//
//	if (!(features & MODELFEATURES_NESTING)) {
//		OOPS("Conditional probability simulations are only meaningful for models with nesting.");
//	}
//	
//	BUGGER(msg) << "Simulating conditonal probability...";
//	std::ostringstream sql;
//	
//	_parameter_update();
//	
//	// Calculate the probabilities
//	calculate_probability();
//	
//	// Read the raw casenumbers from elm_case_ids
//	vector<elm::caseid_t> case_ids (_Data->caseids());
//	
//	// Drop the old table if it exists and is to be overwritten
//	if (overwrite) {
//		_Data->drop(tablename);
//		
////		_Data->sql_statement("DELETE FROM elm_tables WHERE tablename=?")->bind_text(1, tablename)-> execute_until_done();
////		_Data->sql_statement("DELETE FROM elm_tables_temp WHERE tablename=?")->bind_text(1, tablename)->execute_until_done();
//	} else {
//		etk::strvec alltablenames (_Data->all_table_names());
//		if (alltablenames.contains(tablename)) OOPS("The table ",tablename," already exists in the database.");
//	}
//	
//	// Create a new table to hold the probabilities
//	sql << "CREATE TABLE " << tablename << " (caseid INTEGER";
//	
//	std::vector< std::pair<cellcode, cellcode> >  X (Xylem.list_edges());
//	for (unsigned i=0; i<X.size(); i++) {
//		sql << ", " << columnnameprefix << X[i].first << "_" << X[i].second << columnnamesuffix << " DOUBLE";
//	}
//	
//	sql << ")";
//	_Data->sql_statement(sql)->execute_until_done();
//	BUGGER(msg) << sql.str();
//	sql.str(""); sql.clear();
//
//	// Write casenumbers and probabilities to file
//	sql << "INSERT INTO " << tablename << " VALUES (?";
//	for (unsigned i=0; i<Xylem.n_edges(); i++) {
//		sql << ",?";
//	}
//	sql << ");";	
//	SQLiteStmtPtr z = _Data->sql_statement(sql.str());
//	for (unsigned i=0; i<_Data->nCases(); i++) {
//		z->bind_int64(1, case_ids[i]); // tablename
//		for (unsigned a=0; a<Xylem.n_edges(); a++) {
//			z->bind_double(2+a, Cond_Prob(i,a)); // tablename
//		}
//		z->execute_until_done();
//		z->reset();
//	}	
//}

//void elm::Model2::sim_probability
//( const std::string& tablename
//, const std::string& columnnameprefix
//, const std::string& columnnamesuffix
//, bool use_alt_codes
//, bool overwrite
//)
//{
//	if (!_Data) OOPS("A database must be linked to this model to do this.");
//	BUGGER(msg) << "Simulating probability...";
//	std::ostringstream sql;
//	
//	_parameter_update();
//	
//	// Calculate the probabilities
//	calculate_probability();
//	
//	// Read the raw casenumbers from elm_case_ids
//	vector<elm::caseid_t> case_ids (_Data->caseids());
//	
//	// Drop the old table if it exists and is to be overwritten
//	if (overwrite) {
//		_Data->drop(tablename);
////		_Data->sql_statement("DELETE FROM elm_tables WHERE tablename=?")->bind_text(1, tablename)->execute_until_done();
////		_Data->sql_statement("DELETE FROM elm_tables_temp WHERE tablename=?")->bind_text(1, tablename)->execute_until_done();
//	} else {
//		etk::strvec alltablenames (_Data->all_table_names());
//		if (alltablenames.contains(tablename)) OOPS("The table ",tablename," already exists in the database.");
//	}
//	
//	// Create a new table to hold the probabilities
//	sql << "CREATE TABLE " << tablename << " (caseid INTEGER";
//	if (use_alt_codes) {
//		vector<cellcode> X (_Data->alternative_codes());
//		for (unsigned i=0; i<X.size(); i++) {
//			sql << ", " << columnnameprefix << X[i] << columnnamesuffix << " DOUBLE";
//		}
//	} else {
//		std::vector<std::string> X (_Data->alternative_names());
//		for (unsigned i=0; i<X.size(); i++) {
//			sql << ", " << columnnameprefix << X[i] << columnnamesuffix << " DOUBLE";
//		}
//	}
//	sql << ")";
//	_Data->sql_statement(sql)->execute_until_done();
//	BUGGER(msg) << sql.str();
//	sql.str(""); sql.clear();
//
//
//	// Write casenumbers and probabilities to file
//	sql << "INSERT INTO " << tablename << " VALUES (?";
//	for (unsigned i=0; i<_Data->nAlts(); i++) {
//		sql << ",?";
//	}
//	sql << ");";	
//
//	SQLiteStmtPtr z = _Data->sql_statement(sql.str());
//	for (unsigned i=0; i<_Data->nCases(); i++) {
//		z->bind_int64(1, case_ids[i]); // tablename
//		for (unsigned a=0; a<_Data->nAlts(); a++) {
//			z->bind_double(2+a, Probability(i,a)); // tablename
//		}
//		z->execute_until_done();
//		z->reset();
//	}	
//}

etk::ndarray* elm::Model2::probability(etk::ndarray* params)
{

	if (_is_setUp<2) setUp();

	if (!params) {
		_parameter_update();
	} else {
		if ( params->size()!=dF() ) OOPS("Incorrect number of parameters given, need ",dF()," but got ",params->size());
		for (unsigned i=0; i<dF(); i++) {
			FCurrent[i] = params->at(i);
		}
		freshen();
	}
	
	 
	_parameter_log();
	 
	// Calculate the probabilities
	calculate_probability();
	
	return &Probability;
}

etk::ndarray* elm::Model2::adjprobability(etk::ndarray* params)
{

	if (!params) {
		_parameter_update();
	} else {
		if ( params->size()!=dF() ) OOPS("Incorrect number of parameters given, need ",dF()," but got ",params->size());
		for (unsigned i=0; i<dF(); i++) {
			FCurrent[i] = params->at(i);
		}
		freshen();
	}
	 
	// Calculate the probabilities
	calculate_probability();
	
	return &AdjProbability;
}

etk::ndarray* elm::Model2::utility(etk::ndarray* params)
{

	if (!params) {
		_parameter_update();
	} else {
		if ( params->size()!=dF() ) OOPS("Incorrect number of parameters given, need ",dF()," but got ",params->size());
		for (unsigned i=0; i<dF(); i++) {
			FCurrent[i] = params->at(i);
		}
		freshen();
	}
	 
	// Calculate the utilities
	calculate_utility_only();
	
	return &Utility;
}

boosted::shared_ptr<etk::workshop> elm::Model2::make_shared_workshop_accumulate_loglike ()
{
	return boosted::make_shared<loglike_w>(&PrToAccum, Xylem.n_elemental(),
		Data_Choice, Data_Weight_active(), &accumulate_LogL, nullptr, option.mute_nan_warnings, &msg);
}

double elm::Model2::accumulate_log_likelihood() /*const*/
{
	
	accumulate_LogL = 0.0;

	if (CaseLogLike.size()) {
		if (Data_Weight_active()) {
			accumulate_LogL = cblas_ddot(nCases, *CaseLogLike, 1, Data_Weight_active()->values(0,0), 1);
			if (accumulate_LogL) {
				INFO(msg) << "LL(["<< ReadFCurrentAsString() <<"])->"<<accumulate_LogL<< "  (using weights)";
				return accumulate_LogL;
			}
		} else {
			accumulate_LogL = CaseLogLike.sum();
			if (accumulate_LogL) {
				INFO(msg) << "LL(["<< ReadFCurrentAsString() <<"])->"<<accumulate_LogL<< "  (using simple summation)";
				return accumulate_LogL;
			}
		}
	}

	PrToAccum = (sampling_packet().relevant() ? &AdjProbability : &Probability);
	
	#if _ELM_USE_THREADS_
	
	std::function<std::shared_ptr<workshop> ()> workshop_builder =
		[&](){return std::make_shared<loglike_w>(&PrToAccum, Xylem.n_elemental(),
		Data_Choice, Data_Weight_active(), &accumulate_LogL, nullptr, option.mute_nan_warnings, &msg);};

	USE_DISPATCH(loglike_dispatcher,option.threads, nCases, workshop_builder);

	INFO(msg) << "LL(["<< ReadFCurrentAsString() <<"])->"<<accumulate_LogL<< "  (using "<<option.threads<<" threads)";

	#else // not _ELM_USE_THREADS_

	double choice_value;
	for (unsigned c=0;c<nCases;c++) {
		for (unsigned a=0;a<nElementals;a++) {
			choice_value = Data_Choice->value(c,a);
			if (choice_value) {
				if (Data_Weight_active()) {
					accumulate_LogL += (log((*PrToAccum)(c,a)) * choice_value * Data_Weight_active()->value(c,0));
				} else {
					accumulate_LogL += (log((*PrToAccum)(c,a)) * choice_value);
				}
			}			
		}

		if (isNan(accumulate_LogL) && !option.mute_nan_warnings) {
			WARN(msg) << "WARNING: Log Likelihood becomes NAN at caserow "<<c ;
			WARN(msg) << "W..prob: "<<Probability.printrows(c,c+1) ;
			WARN(msg) << "W..chos: "<<Data_Choice->printcases(c,c+1) ;
			break;
		}
		if (isInf(accumulate_LogL) && !option.mute_nan_warnings) {
			WARN(msg) << "WARNING: Log Likelihood becomes INF at caserow "<<c ;
			WARN(msg) << "W..prob: "<<Probability.printrows(c,c+1) ;
			WARN(msg) << "W..chos: "<<Data_Choice->printcases(c,c+1) ;
			break;
		}
		
	} 	

	INFO(msg) << "LL(["<< ReadFCurrentAsString() <<"])->"<<accumulate_LogL<< "  (not using threads)";

	#endif // _ELM_USE_THREADS_




	return accumulate_LogL;
}






std::shared_ptr<etk::ndarray> elm::Model2::_mnl_gradient_full_casewise()
{
	periodic Sup (5);
	BUGGER(msg)<< "Beginning MNL Gradient (Full Casewise) Evaluation" ;
	GCurrent.initialize(0.0);
	Bhhh.initialize(0.0);
	
	std::shared_ptr<ndarray> gradient_casewise = make_shared<ndarray> (nCases, dF());

	BUGGER(msg)<< "Beginning MNL Gradient full casewise single-threaded Evaluation" ;
	
	boosted::mutex local_lock;
	
	workshop_mnl_gradient_full_casewise w
			(dF()
			 , nElementals
			 , utility_packet()
			 , quantity_packet()
			 , Data_Choice
			 , Data_Weight_active()
			 , &Probability
			 , &GCurrent
			 , &Bhhh
			 , &msg
			 , &Data_MultiChoice
			 , &*gradient_casewise
			 );

	w.work(0, nCases, &local_lock);


	BUGGER(msg)<< "End MNL Gradient v2 Evaluation" ;
	
	return gradient_casewise;
}





void elm::Model2::mnl_gradient_v2() 
{
	periodic Sup (5);
	BUGGER(msg)<< "Beginning MNL Gradient v2 Evaluation" ;
	GCurrent.initialize(0.0);
	if (Bhhh.size1() != dF()) {
		Bhhh.resize(dF());
	}
	Bhhh.initialize(0.0);

	if (nThreads >= 1 && _ELM_USE_THREADS_) {

//		std::cerr << "mnl_gradient_v2 threads\n";
		
		boosted::function<boosted::shared_ptr<workshop> ()> workshop_builder =
		[&](){
			return boosted::make_shared<workshop_mnl_gradient2>
			(dF()
			 , nElementals
			 , utility_packet()
			 , quantity_packet()
			 , Data_Choice
			 , Data_Weight_active()
			 , &Probability
			 , &GCurrent
			 , &Bhhh
			 , &msg
			 , &Data_MultiChoice
			 );
		};
		USE_DISPATCH(gradient_dispatcher,option.threads, nCases, workshop_builder);

		std::ostringstream ret;
		for (unsigned i=0; i<GCurrent.size(); i++) {
			ret << "," << GCurrent[i];
		}
		INFO(msg) << "MNL Grad->["<< ret.str().substr(1) <<"] (using "<<option.threads<<" threads)";

	} else {
		
		BUGGER(msg)<< "Beginning MNL Gradient single-threaded Evaluation" ;

//		std::cerr << "mnl_gradient_v2 no threads\n";
		
		
		workshop_mnl_gradient w
		(dF()
		 , nElementals
		 , Params_UtilityCA
		 , Params_UtilityCO
		 , Params_QuantityCA
		 , Params_LogSum
		 , Data_UtilityCA
		 , Data_UtilityCO
		 , Data_QuantityCA
		 , Data_Choice
		 , Data_Weight_active()
		 , 0
		 , nCases
		 );
		
		w.workshop_mnl_gradient_do
		(  Probability
		 );
	 
		w.workshop_mnl_gradient_send
		(  GCurrent
		 , Bhhh);
	}
	BUGGER(msg)<< "End MNL Gradient v2 Evaluation" ;
}


/*

void elm::Model2::hessian_mnl () {  // Inverse Hessian
	
	MONITOR(msg) << "Calculating Analytic Hessian..." ;
	triangle HessFull (nModelParams);
	
	memarray hess_part (nElementals, nModelParams);
	unsigned c,v,vc,i;
	unsigned a;
	Hess.initialize(0.0);	
	
	for (c=0; c<nCases; c++) { 
		hess_part.initialize(0.0);
		
		// idCA
		for (v=0; v<X_Beta().nVarsCA(); v++) {
			for (a=0; a<nElementals; a++) 
				hess_part(0,v) -= Probability(c,a) * X_Beta().valueCA(c,a,v);
			for (a=1; a<nElementals; a++)  
				hess_part(a,v) = hess_part(0,v);
		}
		
		// idCO
		v=X_Beta().nVarsCA();
		for (vc=0; vc<Data_UtilityCO->nVars(); vc++) {
			for (i=0; i<nElementals; i++,v++)
				for (a=0; a<nElementals; a++) 
					hess_part(a,v) -= Probability(c,i) * X_Beta().valueCO(c,vc);
		}
		
		// idCA
		for (a=0; a<nElementals; a++)
			if (X_Beta().nVarsCA()) cblas_daxpy(X_Beta().nVarsCA(),1,&(X_Beta().valueCA(c,a,0)),1,hess_part.ptr(a),1);
		
		// idCO
		v=X_Beta().nVarsCA();
		for (vc=0; vc<Data_UtilityCO->nVars(); vc++)
			for (a=0; a<nElementals; v++, a++ )
				if (X_Avail()(c,a))	hess_part(a,v) += X_Beta().valueCO(c,vc);
		
		// Rollup
		for (a=0; a<nElementals; a++) {
			if (X_Avail()(c,a)) {
				cblas_dspr(CblasRowMajor, CblasUpper, 
						   nModelParams, 
						   WeightCO()(c) * Probability(c,a), 
						   hess_part.ptr(a), 1, 
						   *HessFull);
			}
		}
		
	}
	
	hess_part.resize(0);

	
	// Translate HessFull down to Hess for Freedoms.

	memarray HessInter (HessFull);
	HessFull.resize(0);
	memarray HessInter2 (nModelParams,dF());
	for (i=0; i<nModelParams; i++) {
		push_to_freedoms(HessInter.ptr(i), HessInter2.ptr(i));
	}
	memarray HessInter3 (nModelParams);
	memarray HessInter4 (dF(),dF());
	for ( i=0; i<dF(); i++) {
		cblas_dcopy(nModelParams, HessInter2.ptr(0,i), dF(), *HessInter3, 1);
		push_to_freedoms(*HessInter3, HessInter4.ptr(i));
	}
	Hess.initialize();
	for ( i=0;i<dF();i++) for (unsigned j=0;j<dF();j++) {
		Hess(i,j) += HessInter4(i,j);
		if (i==j) Hess(i,j) += HessInter4(i,j);
	}
	Hess.scale(0.5);
	
	
}
*/

//#include <algorithm>






/*
freedom_info& elm::Model2::parameter(const std::string& param_name,
									 const double& value,
									 const double& null_value,
									 const double& initial_value,
									 const double& std_err,
									 const double& robust_std_err,
									 const double& max,
									 const double& min,
									 const int& holdfast,
									 PyObject* covariance,
									 PyObject* robust_covariance)
{
	return add_freedom2(param_name, value, null_value, initial_value,
						std_err, robust_std_err,
						max, min, holdfast, covariance, robust_covariance);
}

*/





void elm::Model2::utilityca
(const string& variable_name, 
 string        freedom_name, 
 const double& freedom_multiplier)
{
	if (freedom_name=="") freedom_name = variable_name;
	if (!parameter_exists(freedom_name)) {
		BUGGER(msg) << "automatically generating "<<freedom_name<<" parameter because it does not already exist";
		std::string fn = freedom_name;
		etk::uppercase(fn);
		if (fn!="CONSTANT") {
			parameter(freedom_name);
		}
	}
	LinearComponent x;
	x.data_name = variable_name;
	x.param_name = freedom_name;
	x.multiplier = freedom_multiplier;
	Input_Utility.ca.push_back( x );
	if (_fountain()) {
		BUGGER(msg) << "checking for validity of "<<variable_name<<" in idCA data";
		_fountain()->check_ca(variable_name);
	}
	MONITOR(msg) << "success: added "<<variable_name;
}



void elm::Model2::utilityco
(const string&    column_name, 
 const string&    alt_name, 
 string           freedom_name, 
 const double&    freedom_multiplier)
{
	 _add_utility_co(column_name, alt_name, cellcode_empty, freedom_name, freedom_multiplier);
}

void elm::Model2::utilityco
(const string&    column_name, 
 const long long& alt_code, 
 string           freedom_name, 
 const double&    freedom_multiplier)
{
	 _add_utility_co(column_name, "", alt_code, freedom_name, freedom_multiplier);
}


string elm::Model2::_add_utility_co
(const string&    column_name, 
 const string&    alt_name, 
 const long long& alt_code, 
 string           freedom_name, 
 const double&    freedom_multiplier)
{
	unsigned slot;
	std::string variable_name;
	if (!alt_name.empty()) {
		variable_name = cat(column_name, "@", alt_name);
	} else {
		variable_name = cat(column_name, "#", alt_code);
	}
	if (freedom_name=="") {
		freedom_name = variable_name;
	}
	if (!parameter_exists(freedom_name)) {
		BUGGER(msg) << "automatically generating "<<freedom_name<<" parameter because it does not already exist";
		std::string fn = freedom_name;
		etk::uppercase(fn);
		if (fn!="CONSTANT") {
			parameter(freedom_name);
		}
	}
	LinearComponent x;
	x.data_name = column_name;
	x.param_name = freedom_name;
	x.multiplier = freedom_multiplier;
//	x._altcode = alt_code;
//	x._altname = alt_name;
	Input_Utility.co[alt_code].push_back( x );
	
	if (_fountain()) {
		BUGGER(msg) << "checking for validity of "<<column_name<<" in idCO data";
		_fountain()->check_co(column_name);
		if (!alt_name.empty()) {
			slot = _fountain()->DataDNA()->slot_from_name(alt_name);
		} else {
			if (alt_code==cellcode_empty) {
				OOPS("utility.co input requires that you specify an alternative.");
			}
			slot = _fountain()->DataDNA()->slot_from_code(alt_code);
		}
	}

	MONITOR(msg) << "success: added "<<variable_name;
	return "success";
}


PyObject* __GetInputTupleUtilityCA(const elm::LinearComponent& i)
{
	if (i.multiplier==1.0) {
		return Py_BuildValue("(ss)", i.data_name.c_str(), i.param_name.c_str());
	}
	return Py_BuildValue("(ssd)", i.data_name.c_str(), i.param_name.c_str(), i.multiplier);
}

PyObject* elm::Model2::_get_utilityca() const
{
	PyObject* U = PyList_New(0);
	for (unsigned i=0; i<Input_Utility.ca.size(); i++) {
		PyObject* z = __GetInputTupleUtilityCA(Input_Utility.ca[i]);
		PyList_Append(U, z);
		Py_CLEAR(z);
	}
	return U;
}

PyObject* elm::Model2::_get_samplingbiasca() const
{
	PyObject* U = PyList_New(0);
	for (unsigned i=0; i<Input_Sampling.ca.size(); i++) {
		PyObject* z = __GetInputTupleUtilityCA(Input_Sampling.ca[i]);
		PyList_Append(U, z);
		Py_CLEAR(z);
	}
	return U;
}

PyObject* __GetInputTupleUtilityCO(const elm::cellcode& altcode, const elm::LinearComponent& i)
{
	if (altcode==cellcode_empty) {
		OOPS("co input does not specify an alternative.\n"
			 "Inputs in the co space need to identify an alternative.");
	}
	if (i.multiplier==1.0) {
		return Py_BuildValue("(Kss)", i._altcode, i.data_name.c_str(), i.param_name.c_str());
	}
	return Py_BuildValue("(Kssd)", i._altcode, i.data_name.c_str(), i.param_name.c_str(), i.multiplier);
}

PyObject* elm::Model2::_get_utilityco() const
{
	PyObject* U = PyList_New(0);
	for (auto a = Input_Utility.co.begin(); a!=Input_Utility.co.end(); a++) {
		for (unsigned i=0; i<a->second.size(); i++) {
			PyObject* z = __GetInputTupleUtilityCO(a->first, a->second[i]);
			PyList_Append(U, z);
			Py_CLEAR(z);
		}
	}
	return U;
}

PyObject* elm::Model2::_get_samplingbiasco() const
{
	PyObject* U = PyList_New(0);
	for (auto a = Input_Sampling.co.begin(); a!=Input_Sampling.co.end(); a++) {
		for (unsigned i=0; i<a->second.size(); i++) {
			PyObject* z = __GetInputTupleUtilityCO(a->first, a->second[i]);
			PyList_Append(U, z);
			Py_CLEAR(z);
		}
	}
	return U;
}



double elm::Model2::loglike_given_utility( )
{
	if (nCases==0) {
		return 0;
		OOPS("There are no cases in the given utility array.");
	}
	
	FatGCurrent.initialize(NAN); // tell gradient it needs to recalculate

	BUGGER(msg)<< "Calculating LL given utility" ;
	double LL_ = 0;
	
	pull_coefficients_from_freedoms();
	freshen(); // TODO : is this really needed here?
	
	ngev_probability_given_utility();
	
	LL_= accumulate_log_likelihood();
	
	BUGGER(msg)<< "Model Objective Eval = "<< LL_ ;
	
	if (_string_sender_ptr) {
		ostringstream update;
		update << "Model Objective Eval = "<< LL_;
		_string_sender_ptr->write(update.str());
	}
	
	_FCurrent_latest_objective = FCurrent;
	_FCurrent_latest_objective_value = LL_;

	if (isNan(LL_)) {
		LL_ = -INF;
	}

	return LL_;
}

std::shared_ptr<etk::ndarray> elm::Model2::negative_d_loglike_given_utility ()
{
	if (array_compare(_FCurrent_latest_objective.ptr(),_FCurrent_latest_objective.size())
		!=array_compare(FCurrent.ptr(), FCurrent.size()))
	{
		loglike_given_utility();
	}

	if (option.force_finite_diff_grad) {
		negative_finite_diff_gradient_(GCurrent);
	} else {
		ngev_gradient();
	}
	FatGCurrent = ReadFCurrent();

	std::shared_ptr<etk::ndarray> g = make_shared<etk::ndarray>(GCurrent, false);

	etk::symmetric_matrix bhhh (Bhhh, false);
	bhhh.copy_uppertriangle_to_lowertriangle();


	_cached_results.set_cached_grad(elm::array_compare(FCurrent.ptr(), FCurrent.size()), g);
	_cached_results.set_cached_bhhh(elm::array_compare(FCurrent.ptr(), FCurrent.size()), bhhh);

	return g;

}






double elm::Model2::objective () 
{
	if (nCases==0) {
		return 0;
		OOPS("There are no cases in the current data sample.");
	}
	
	FatGCurrent.initialize(NAN); // tell gradient it needs to recalculate

//	if (array_compare(_FCurrent_latest_objective.ptr(),_FCurrent_latest_objective.size())
//		==array_compare(FCurrent.ptr(), FCurrent.size()))
//	{
//		return _FCurrent_latest_objective_value;
//	}
	
	BUGGER(msg)<< "Calculating LL" ;
//	msg << printStatus(status_FNames | status_FCurrent) <<"\n"; //101110
	double LL_ = 0;
	
	pull_coefficients_from_freedoms();
//	freshen(); // TODO : is this really needed here?
	calculate_probability();
	
	LL_= accumulate_log_likelihood();
	
	BUGGER(msg)<< "Model Objective Eval = "<< LL_ ;
	
//	for (unsigned scar=0; scar<sidecars.size(); scar++) {
//		if (sidecars[scar]) {
//			LL_ += sidecars[scar]->objective(FNames, *FCurrent);
//		}
//	}
//	if (sidecars.size())
//		BUGGER(msg)<< "Model Objective Eval with sidecars= "<< LL_ ;
	
	if (_string_sender_ptr) {
		ostringstream update;
		update << "Model Objective Eval = "<< LL_;
		_string_sender_ptr->write(update.str());
	}
	
	_FCurrent_latest_objective = FCurrent;
	_FCurrent_latest_objective_value = LL_;
	return LL_;
}


const etk::memarray& elm::Model2::gradient (const bool& force_recalculate)
{
	if (FatGCurrent == ReadFCurrent() && !option.force_recalculate && !force_recalculate) {
		// do nothing, calculations already done
	} else {
		if (option.force_finite_diff_grad) {
			negative_finite_diff_gradient_(GCurrent);
		} else {
			objective();
			if ((features & MODELFEATURES_ALLOCATION)) {
				ngev_gradient();
			} else if (features & MODELFEATURES_QUANTITATIVE) {
				ngev_gradient();
			} else if (features & MODELFEATURES_NESTING) {
				nl_gradient();
			} else {
				mnl_gradient_v2();
			}
		}
		FatGCurrent = ReadFCurrent();
//		for (unsigned scar=0; scar<sidecars.size(); scar++) {
//			if (sidecars[scar]) {
//				sidecars[scar]->gradient(FNames, *FCurrent, *GCurrent, &Bhhh);
//			}
//		}
	}

	return GCurrent;
}

std::shared_ptr<etk::ndarray> elm::Model2::finite_diff_gradient ()
{
	std::shared_ptr<etk::ndarray> g = std::make_shared<etk::ndarray>(dF());
	finite_diff_gradient_(*g);
	return g;
}

std::shared_ptr<etk::ndarray> elm::Model2::finite_diff_gradient (std::vector<double> v)
{
	std::shared_ptr<etk::ndarray> g = std::make_shared<etk::ndarray>(dF());
	finite_diff_gradient_(*g);
	return g;
}










#include <iomanip>
#define LINER(ret,filler)  ret.fill(filler); ret.width(92); ret << "\n"; ret.fill(' ');


string elm::Model2::prints(const unsigned& precision, const unsigned& cell_width) const
{
	ostringstream ret;
	ret << std::setprecision(precision);
	
	unsigned max_length_freedom_name (cell_width);
	for (unsigned p=0; p<FNames.size(); p++) {
		if (FNames[p].size() > max_length_freedom_name) {
			max_length_freedom_name = FNames[p].size();
		}
	}
	if (max_length_freedom_name > 255) {
		max_length_freedom_name = 255;
	}
	
	LINER(ret,'=');
	ret << "Model Parameter Estimates\n";
	LINER(ret,'-');
	ret << std::setw(max_length_freedom_name) << std::left << "Parameter  " << std::right << "\t";
	ret << std::setw(cell_width) << "InitValue  " << "\t";
	ret << std::setw(cell_width) << "FinalValue " << "\t";
	ret << std::setw(cell_width) << "StdError   " << "\t";
	ret << std::setw(cell_width) << "t-Stat     " << "\t";
	ret << std::setw(cell_width) << "NullValue  " << "\n";
	for (unsigned p=0; p<FNames.size(); p++) {
		const freedom_info* fi = get_raw_info(FNames[p]);
		ret << std::setw(max_length_freedom_name) << std::left << FNames[p] << std::right << "\t";
		ret << std::setw(cell_width) << fi->initial_value << "\t";
		ret << std::setw(cell_width) << fi->value << "\t";
		ret << std::setw(cell_width) << fi->std_err << "\t";
		ret << std::setw(cell_width) << fi->t_stat() << "\t";
		ret << std::setw(cell_width) << fi->null_value << "\n";
	}
	for (auto al=AliasInfo.begin(); al !=AliasInfo.end(); al++) {
		const freedom_info* fi = get_raw_info(al->second.refers_to);
		ret << std::setw(max_length_freedom_name) << std::left << al->first << std::right << "\t";
		ret << std::setw(cell_width) << fi->initial_value*al->second.multiplier << "\t";
		ret << std::setw(cell_width) << fi->value*al->second.multiplier << "\t";
		ret << "= "<<al->second.refers_to<<" * "<<al->second.multiplier<< "\n";
	}
	
	if (Xylem.size()>Xylem.n_elemental()+1) {
	LINER(ret,'=');
	ret << "Model Network\n";
	LINER(ret,'-');
	ret << ComponentGraphDNA(&Input_LogSum,&Input_Edges,_fountain()).__repr__() << "\n";
//	ret << Xylem.display_phenotype();
	}
	
	LINER(ret,'=');
	ret << "Model Estimation Statistics\n";
	LINER(ret,'-');
	
	ios_base::fmtflags fmt (ret.flags());
	streamsize precio (ret.precision());
	ret << fixed;
	if (!isNan(ZBest) && !isInf(ZBest)) {
		ret << "Log Likelihood at Convergence     \t" << ZBest << "\n";
	}
	if (!isNan(_LL_null) && !isInf(_LL_null)) {
		ret << "Log Likelihood at Null Parameters \t" << _LL_null << "\n";
	}
	if (!isNan(_LL_nil) && !isInf(_LL_nil)) {
		ret << "Log Likelihood with No Model \t" << _LL_nil << "\n";
	}
	if (!isNan(_LL_constants) && !isInf(_LL_constants)) {
		ret << "Log Likelihood at Constants       \t" << _LL_constants << "\n";
	}
	ret.flags(fmt);
	ret.precision(precio);

	LINER(ret,'=');
	ret << "Latest Estimation Run Statistics\n";
	LINER(ret,'-');
	
	ret << "Number of Iterations: \t"<<_latest_run.iteration << "\n";
	ret << "Running Time: \t"<<_latest_run.runtime() << "\n";
	if (!_latest_run.notes().empty()) {
		ret << "Notes: \t"<<_latest_run.notes() << "\n";
	}
	if (!_latest_run.results.empty()) {
		ret << "Results: \t"<<_latest_run.results << "\n";
	}
	
	etk::ndarray* _TallyChosen = nullptr;
	etk::ndarray* _TallyAvail = nullptr;
	
	LINER(ret,'=');
	return ret.str();
}

string elm::Model2::full_report(const unsigned& precision, const unsigned& cell_width)
{
	setUp();
	std::string ret = prints(precision, cell_width);
	tearDown();
	return ret;
}

std::string elm::Model2::representation() const
{
	ostringstream m;
	m << "<larch.Model";
	
	if (dF()<0) {
		m << "(";
		if (dF()>0) {
			m << FNames[0];
		}
		for (unsigned i=1; i<dF(); i++) {
			m << ", " << FNames[i];
		}
		m << ")>";
	} else {
		m << " with "<<dF()<<" parameters>";
	}
//	
//	m << "\tparameters=(\n";
//	for (unsigned i=0; i<dF(); i++) {
//		const freedom_info* f = get_raw_info(FNames[i]);
//		m << "\t\t" << f->representation(false) << ",\n";
//	}
//	m << "\t)\n";
//	m << ")";
	return m.str();
}
