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
#include "elm_parameter2.h"
#include "elm_sql_scrape.h"
#include "elm_names.h"
#include "elm_workshop_nl_gradient.h"
#include "elm_workshop_nl_probability.h"
#include "elm_workshop_ngev_gradient.h"
#include "elm_workshop_ngev_probability.h"
#include <iostream>
#include "etk_thread.h"
#include "elm_calculations.h"
#include "larch_modelparameter.h"

using namespace etk;
using namespace elm;
using namespace std;




ComponentGraphDNA elm::Model2::Input_Graph()
{
	return ComponentGraphDNA(&Input_LogSum,&Input_Edges,_fountain());
}


void elm::Model2::_setUp_NL()
{
	INFO(msg)<< "Setting up NL model..." ;
	
	// COUNTING
	nElementals = Xylem.n_elemental();
	nNests = Xylem.n_branches();
	nNodes = Xylem.size();
	
	Params_LogSum.resize(nNests+1);

	nThreads = option.threads;
	if (nThreads < 1) nThreads = 1;
	if (nThreads > 1024) nThreads = 1024;
	
	if (!option.suspend_xylem_rebuild) Xylem.regrow(nullptr,nullptr,nullptr,nullptr,&msg);
	BUGGER(msg) << "_setUp_NL:Xylem:\n" << Xylem.display();
	
	unsigned slot;
	for (ComponentCellcodeMap::iterator m=Input_LogSum.begin(); m!=Input_LogSum.end(); m++) {
		slot = Xylem.slot_from_code(m->first);
	//	multiname itemname (m->second.data_name);
	//	if (is_cellcode_empty(itemname.node_code)) {
	//		slot = Xylem.slot_from_name(itemname.alternative);
	//	} else {
	//		slot = Xylem.slot_from_code(itemname.node_code);
	//	}
		if (slot < Xylem.n_elemental()) {
			OOPS("pointing to a negative slot");
		}
		slot -= Xylem.n_elemental();
		Params_LogSum[slot] = _generate_parameter(m->second.param_name, m->second.multiplier);
	}
	if (!Params_LogSum[nNests]) Params_LogSum[nNests] = boosted::make_shared<elm::parametex_constant>(1.0);

	
		
	// Allocate Memory	
	Cond_Prob.resize(nCases,Xylem.n_edges());
	Probability.resize(nCases,nNodes);
	Utility.resize(nCases,nNodes);
	
	if ( Input_Sampling.ca.size()>0 || Input_Sampling.co.metasize()>0 ) {
		AdjProbability.resize(nCases,nElementals);
		SamplingWeight.resize(nCases,nElementals);
	} else {
		AdjProbability.same_memory_as(Probability);
		SamplingWeight.resize(0);
	}
	
	Workspace.resize(nNodes);
		
	sherpa::allocate_memory();
	

	INFO(msg)<< "Set up NL model complete." ;
	
}



void elm::Model2::_setUp_NGEV()
{
//	if (!(is_provisioned())) {
//		OOPS("data not provisioned");
//	}
	INFO(msg)<< "Setting up NGEV model..." ;
	
	// COUNTING
//	nCases is set in provisioning
	nElementals = Xylem.n_elemental();
	nNests = Xylem.n_branches();
	nNodes = Xylem.size();
	
//	Params_UtilityCA and Params_UtilityCO are resized in _setUp_linear_data_and_params
	Params_LogSum.resize(nNests+1);

	nThreads = option.threads;
	if (nThreads < 1) nThreads = 1;
	if (nThreads > 1024) nThreads = 1024;
	
	if (!option.suspend_xylem_rebuild) Xylem.regrow(nullptr,nullptr,nullptr,nullptr,&msg);
	BUGGER(msg) << "_setUp_NL:Xylem:\n" << Xylem.display();
	
	unsigned slot;
	for (ComponentCellcodeMap::iterator m=Input_LogSum.begin(); m!=Input_LogSum.end(); m++) {
		slot = Xylem.slot_from_code(m->first);
		if (slot < Xylem.n_elemental()) {
			OOPS("pointing to a negative slot");
		}
		slot -= Xylem.n_elemental();
		Params_LogSum[slot] = _generate_parameter(m->second.param_name, m->second.multiplier);
	}
	if (!Params_LogSum[nNests]) Params_LogSum[nNests] = boosted::make_shared<elm::parametex_constant>(1.0);

	if (nCases) {
		
		// Allocate Memory
		Cond_Prob.resize(nCases,Xylem.n_edges());
		Probability.resize(nCases,nNodes);
		Utility.resize(nCases,nNodes);
		
		Allocation.resize(nCases,Xylem.n_compet_alloc());
		
		if (Input_QuantityCA.size()>0) Quantity.resize(nCases,nElementals);
		
		
		if ( Input_Sampling.ca.size()>0 || Input_Sampling.co.metasize()>0 ) {
			AdjProbability.resize(nCases,nElementals);
			SamplingWeight.resize(nCases,nElementals);
		} else {
			AdjProbability.same_memory_as(Probability);
			SamplingWeight.resize(0);
		}
		
	}
	
	Workspace.resize(nNodes);
		
	sherpa::allocate_memory();
	

	INFO(msg)<< "Set up NGEV model complete." ;
	
}



//#ifndef __APPLE__
boosted::shared_ptr<workshop> elm::Model2::make_shared_workshop_nl_probability ()
{return boosted::make_shared<workshop_nl_probability>(nNodes, utility_packet(), sampling_packet()
								 , Params_LogSum
								 , Data_Avail
								 , &Probability
								 , &Cond_Prob
								 , &AdjProbability
								 , &Xylem
								 , option.mute_nan_warnings
								 , &msg
								 );}

boosted::shared_ptr<workshop> elm::Model2::make_shared_workshop_nl_gradient ()
{return boosted::make_shared<workshop_nl_gradient>(  dF()
									 , nNodes
									 , utility_packet()
									 , sampling_packet()
									 , Params_LogSum
									 , Data_Choice
									 , Data_Weight_active()
									 , &AdjProbability
									 , &Probability
									 , &Cond_Prob
									 , &Xylem
									 , &GCurrent
									 , &Bhhh
									 , &msg
									 );}

boosted::shared_ptr<workshop> elm::Model2::make_shared_workshop_ngev_probability ()
{return boosted::make_shared<workshop_ngev_probability>(nNodes, utility_packet(), allocation_packet(), sampling_packet(), quantity_packet()
								 , Params_LogSum
								 , Data_Avail
								 , &Probability
								 , &Cond_Prob
								 , &AdjProbability
								 , &Xylem
								 , option.mute_nan_warnings
								 , &msg
								 );}

boosted::shared_ptr<workshop> elm::Model2::make_shared_workshop_ngev_gradient ()
{
	return boosted::make_shared<workshop_ngev_gradient>(  dF()
									 , nNodes
									 , utility_packet()
									 , &Data_UtilityCE_manual
									 , allocation_packet()
									 , sampling_packet()
									 , quantity_packet()
									 , Params_LogSum
									 , Data_Choice
									 , Data_Weight_active()
									 , &AdjProbability
									 , &Probability
									 , &Cond_Prob
									 , &Xylem
									 , &GCurrent
									 , nullptr
									 , &Bhhh
									 , &msg
									 , nullptr
									 , nullptr
									 );
}





std::shared_ptr<etk::ndarray> elm::Model2::_ngev_gradient_full_casewise()
{
	periodic Sup (5);
	BUGGER(msg)<< "Beginning NGEV Gradient (Full Casewise) Evaluation" ;
	GCurrent.initialize(0.0);
	Bhhh.initialize(0.0);
	
	std::shared_ptr<ndarray> gradient_casewise = make_shared<ndarray> (nCases, dF());

	BUGGER(msg)<< "Beginning NGEV Gradient full casewise single-threaded Evaluation" ;
	
	boosted::mutex local_lock;
	
	workshop_ngev_gradient w
			(  dF()
			 , nNodes
			 , utility_packet()
			 , &Data_UtilityCE_manual
			 , allocation_packet()
			 , sampling_packet()
			 , quantity_packet()
			 , Params_LogSum
			 , Data_Choice
			 , Data_Weight_active()
			 , &AdjProbability
			 , &Probability
			 , &Cond_Prob
			 , &Xylem
			 , &GCurrent
			 , &*gradient_casewise
			 , &Bhhh
			 , &msg
			 , nullptr
			 , &local_lock
			 );

	w.work(0, nCases, &local_lock);

	BUGGER(msg)<< "End NGEV Gradient Evaluation" ;
	
	return gradient_casewise;
}


std::shared_ptr<etk::ndarray> elm::Model2::_ngev_d_prob()
{
	periodic Sup (5);
	BUGGER(msg)<< "Beginning NGEV dProb Evaluation" ;
	_setUp_NGEV();
	freshen();
	GCurrent.initialize(0.0);
	Bhhh.initialize(0.0);
	
	std::shared_ptr<ndarray> dPr = make_shared<ndarray> (nCases, nNodes, dF());
	
	boosted::mutex local_lock;


	workshop_builder_t workshop_builder =
		[&](){return std::make_shared<workshop_ngev_gradient>(
		       dF()
			 , nNodes
			 , utility_packet()
			 , &Data_UtilityCE_manual
			 , allocation_packet()
			 , sampling_packet()
			 , quantity_packet()
			 , Params_LogSum
			 , Data_Choice
			 , Data_Weight_active()
			 , &AdjProbability
			 , &Probability
			 , &Cond_Prob
			 , &Xylem
			 , &GCurrent
			 , nullptr
			 , &Bhhh
			 , &msg
			 , &*dPr
			 , &local_lock
			 );};

	workshop_updater_t workshop_updater = [&](std::shared_ptr<workshop> w)
	{
		workshop_ngev_gradient* ww = dynamic_cast<workshop_ngev_gradient*>(&*w);
		ww->rebuild_local_data(
		       dF()
			 , nNodes
			 , utility_packet()
			 , &Data_UtilityCE_manual
			 , allocation_packet()
			 , sampling_packet()
			 , quantity_packet()
			 , Params_LogSum
			 , Data_Choice
			 , Data_Weight_active()
			 , &AdjProbability
			 , &Probability
			 , &Cond_Prob
			 , &Xylem
			 , &GCurrent
			 , nullptr
			 , &Bhhh
			 , &msg
			 , &*dPr
			 , &local_lock
			 );
	};

	
	UPDATE_AND_DISPATCH(gradient_dispatcher,option.threads, &workshop_updater, nCases, workshop_builder);


	
//	workshop_ngev_gradient w
//			(  dF()
//			 , nNodes
//			 , utility_packet()
//			 , &Data_UtilityCE_manual
//			 , allocation_packet()
//			 , sampling_packet()
//			 , quantity_packet()
//			 , Params_LogSum
//			 , Data_Choice
//			 , Data_Weight_active()
//			 , &AdjProbability
//			 , &Probability
//			 , &Cond_Prob
//			 , &Xylem
//			 , &GCurrent
//			 , nullptr
//			 , &Bhhh
//			 , &msg
//			 , &*dPr
//			 , &local_lock
//			 );
//
//	w.work(0, nCases, &local_lock);

	BUGGER(msg)<< "End NGEV dProb Evaluation" ;
	
	return dPr;
}



void elm::Model2::nl_probability()
{
	
	// COUNTING
	nElementals = Xylem.n_elemental();
	nNests = Xylem.n_branches();
	nNodes = Xylem.size();
	
	if (Params_LogSum.size1() != nNests+1) {
		
		Params_LogSum.resize(nNests+1);
		
		if (!option.suspend_xylem_rebuild) Xylem.regrow(nullptr,nullptr,nullptr,nullptr,&msg);
		
		unsigned slot;
		for (ComponentCellcodeMap::iterator m=Input_LogSum.begin(); m!=Input_LogSum.end(); m++) {
			slot = Xylem.slot_from_code(m->first);
			if (slot < Xylem.n_elemental()) {
				OOPS("pointing to a negative slot");
			}
			slot -= Xylem.n_elemental();
			Params_LogSum[slot] = _generate_parameter(m->second.param_name, m->second.multiplier);
		}
		if (!Params_LogSum[nNests]) Params_LogSum[nNests] = boosted::make_shared<elm::parametex_constant>(1.0);
		
	}
	
	// Allocate Memory	
	Cond_Prob.resize(nCases,Xylem.n_edges());
	Probability.resize(nCases,nNodes);
	Utility.resize(nCases,nNodes);
	
	if ( Input_Sampling.ca.size()>0 || Input_Sampling.co.metasize()>0 ) {
		AdjProbability.resize(nCases,nElementals);
		SamplingWeight.resize(nCases,nElementals);
	} else {
		AdjProbability.same_memory_as(Probability);
		SamplingWeight.resize(0);
	}
	
	Workspace.resize(nNodes);
		









	unsigned c,a;
	unsigned warningcount (0);
	unsigned zerosize_warncount (0);
	bool found_nan (false);
	pull_coefficients_from_freedoms();
	
	if (nThreads<=1) nThreads = 1;
	 
//	BUGGER(msg) << "Number of threads in nl_probability =" << nThreads;
	if (nThreads>=2 && _ELM_USE_THREADS_) {
		
		#ifdef __APPLE__
		boosted::function<boosted::shared_ptr<workshop> ()> workshop_builder =
			[&](){return boosted::make_shared<workshop_nl_probability>(nNodes, utility_packet(), sampling_packet()
								 , Params_LogSum
								 , Data_Avail
								 , &Probability
								 , &Cond_Prob
								 , &AdjProbability
								 , &Xylem
								 , option.mute_nan_warnings
								 , &msg
								 );};
		#else
		boosted::function<boosted::shared_ptr<workshop> ()> workshop_builder =
			boosted::bind(&elm::Model2::make_shared_workshop_nl_probability, this);
		#endif // def __APPLE__
		USE_DISPATCH(probability_dispatcher,option.threads, nCases, workshop_builder);
	
	} else {
	
	Utility.initialize(0.0);
	__logit_utility(Utility, Data_UtilityCA, Data_UtilityCO, &Coef_UtilityCA, &Coef_UtilityCO, 0);

	elm::ca_co_packet sampling_packet_ = sampling_packet();
	bool use_sampling = sampling_packet_.relevant();
	
	if (use_sampling) {
		sampling_packet_.logit_partial(0, nCases);
	}
	
	for (c=0;c<nCases;c++) {
		// Unavailable alternatives become -INF
		for (a=0;a<nElementals;a++) {
			if (!Data_Avail->boolvalue(c,a)) {
				Utility(c,a) = -INF;
			} 		
		}
		
		__casewise_nl_utility(Utility.ptr(c), Xylem, *Workspace);
		__casewise_nl_probability(Utility.ptr(c), Cond_Prob.ptr(c), Probability.ptr(c), Xylem);

		if (use_sampling) {
			// AdjProbability
			size_t n            = sampling_packet_.Outcome->size2();
			double* adjPr       = AdjProbability.ptr(c);
			const bool* av      = Data_Avail->boolvalues(c,1);
			
			double tot = 0.0;
			const double* origPr = Probability.ptr(c);
			const double* sampWt = sampling_packet_.Outcome->ptr(c);

			double* p = adjPr;
			for (size_t i=0; i<n; i++, origPr++, sampWt++, p++, av++) {
				if (*av) {
					*p = (*origPr) * ::exp(*sampWt);
					tot += *p;
				} else {
					*p = 0.0;
				}
			}
			
			p = adjPr;
			for (size_t i=0; i<n; i++, p++) {
				*p /= tot;
			}

		}




		
		// NANCHECK
		if (!option.mute_nan_warnings) {
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
					WARN(msg) << "W..util: " << Utility.printrow(c) ;
					WARN(msg) << "W..c|pr: " << Cond_Prob.printrow(c) ;
					WARN(msg) << "W..prob: " << Probability.printrow(c) ;
				}
				warningcount++;
			}
		} // end if NANCHECK
		
	}
	
	if (warningcount>1) {
		WARN(msg) << "W......: and for "<<warningcount-1<<" other cases" ;
	}
	}
	BUGGER(msg) << "Utility (case 0)\n" << Utility.printrow(0) ;
	BUGGER(msg) << "Probability (case 0)\n" << Probability.printrow(0) ;
	BUGGER(msg) << "Cond_Prob (case 0)\n" << Cond_Prob.printrow(0) ;
}


void elm::Model2::ngev_probability()
{





	// COUNTING
//	nCases is set in provisioning
	nElementals = Xylem.n_elemental();
	nNests = Xylem.n_branches();
	nNodes = Xylem.size();
	
	if (Params_LogSum.size1() != nNests+1) {
		Params_LogSum.resize(nNests+1);
		
		if (!option.suspend_xylem_rebuild) Xylem.regrow(nullptr,nullptr,nullptr,nullptr,&msg);
		BUGGER(msg) << "_setUp_NL:Xylem:\n" << Xylem.display();
		
		unsigned slot;
		for (ComponentCellcodeMap::iterator m=Input_LogSum.begin(); m!=Input_LogSum.end(); m++) {
			slot = Xylem.slot_from_code(m->first);
			if (slot < Xylem.n_elemental()) {
				OOPS("pointing to a negative slot");
			}
			slot -= Xylem.n_elemental();
			Params_LogSum[slot] = _generate_parameter(m->second.param_name, m->second.multiplier);
		}
		if (!Params_LogSum[nNests]) Params_LogSum[nNests] = boosted::make_shared<elm::parametex_constant>(1.0);
	}

	
	// Allocate Memory
	Cond_Prob.resize(nCases,Xylem.n_edges());
	Probability.resize(nCases,nNodes);
	Utility.resize(nCases,nNodes);
	
	Allocation.resize(nCases,Xylem.n_compet_alloc());
	
	if (Input_QuantityCA.size()>0) Quantity.resize(nCases,nElementals);
	
	if ( Input_Sampling.ca.size()>0 || Input_Sampling.co.metasize()>0 ) {
		AdjProbability.resize(nCases,nElementals);
		SamplingWeight.resize(nCases,nElementals);
	} else {
		AdjProbability.same_memory_as(Probability);
		SamplingWeight.resize(0);
	}
	
	Workspace.resize(nNodes);





	unsigned c,a;
	unsigned warningcount (0);
	unsigned zerosize_warncount (0);
	bool found_nan (false);
	pull_coefficients_from_freedoms();
	
	if (nThreads<=1) nThreads = 1;
	 
	BUGGER(msg) << "Number of threads in ngev_probability =" << nThreads;
	if (nThreads>=2 && _ELM_USE_THREADS_) {
		
		boosted::function<boosted::shared_ptr<workshop> ()> workshop_builder =
			boosted::bind(&elm::Model2::make_shared_workshop_ngev_probability, this);
		USE_DISPATCH(probability_dispatcher,option.threads, nCases, workshop_builder);
	
	} else {
	
		boosted::shared_ptr<workshop> local_prob_workshop;
		if (!local_prob_workshop) {
			local_prob_workshop = make_shared_workshop_ngev_probability ();
		}
		local_prob_workshop->work(0, nCases, nullptr);
	
	}
//	BUGGER(msg) << "Quantity (case 0)\n" << Quantity.printrow(0) ;
//	BUGGER(msg) << "Utility (case 0)\n" << Utility.printrow(0) ;
//	BUGGER(msg) << "Probability (case 0)\n" << Probability.printrow(0) ;
//	BUGGER(msg) << "Cond_Prob (case 0)\n" << Cond_Prob.printrow(0) ;
//
//	if (nCases>1) {
//	BUGGER(msg) << "Quantity (case 1)\n" << Quantity.printrow(1) ;
//	BUGGER(msg) << "Utility (case 1)\n" << Utility.printrow(1) ;
//	BUGGER(msg) << "Probability (case 1)\n" << Probability.printrow(1) ;
//	BUGGER(msg) << "Cond_Prob (case 1)\n" << Cond_Prob.printrow(1) ;
//	}
}


void elm::Model2::ngev_probability_given_utility( )
{

	pull_coefficients_from_freedoms();
	
	if (nThreads<=1) nThreads = 1;
	 
	BUGGER(msg) << "Number of threads in ngev_probability =" << nThreads;



	std::function<std::shared_ptr<workshop> ()> workshop_builder =
		[&](){return std::make_shared<workshop_ngev_probability_given_utility>(nNodes, utility_packet_without_data(), allocation_packet(), sampling_packet(), quantity_packet()
								 , Params_LogSum
								 , Data_Avail
								 , &Probability
								 , &Cond_Prob
								 , &AdjProbability
								 , &Xylem
								 , option.mute_nan_warnings
								 , &msg
								 );};

	USE_DISPATCH(probability_given_utility_dispatcher,option.threads, nCases, workshop_builder);
	
}



void elm::Model2::nl_gradient() 
{
	periodic Sup (5);
	BUGGER(msg)<< "Beginning NL Gradient Evaluation" ;
	
	if (Bhhh.size1() != dF()) {
		Bhhh.resize(dF());
	}
	
	GCurrent.initialize(0.0);
	Bhhh.initialize(0.0);
	
	
	
	if (nThreads <= 0) nThreads=1;
	//nThreads=1; /* there may be a bug in the threading */
	BUGGER(msg)<< "Using "<< nThreads <<" threads" ;
	
	#ifndef _ELM_USE_THREADS_
	nThreads = 1;
	#endif // ndef _ELM_USE_THREADS_
	
//	if (nThreads >= 1 && _ELM_USE_THREADS_) {

		boosted::function<boosted::shared_ptr<workshop> ()> workshop_builder =
			boosted::bind(&elm::Model2::make_shared_workshop_nl_gradient, this);
		USE_DISPATCH(gradient_dispatcher,option.threads, nCases, workshop_builder);
		
//	} else {
//
//		boosted::shared_ptr<workshop> local_grad_workshop;
//		if (!local_grad_workshop) {
//			local_grad_workshop = make_shared_workshop_nl_gradient ();
//		}
//		local_grad_workshop->work(0, nCases, nullptr);
//
//	}
	BUGGER(msg)<< "End NL Gradient Evaluation" ;
	
	std::ostringstream ret;
	for (unsigned i=0; i<GCurrent.size(); i++) {
		ret << "," << GCurrent[i];
	}
	INFO(msg) << "NL Grad->["<< ret.str().substr(1) <<"] (using "<<option.threads<<" threads)";
}

void elm::Model2::ngev_gradient()
{
	periodic Sup (5);
	BUGGER(msg)<< "Beginning NGEV Gradient Evaluation" ;
	GCurrent.initialize(0.0);
	if (Bhhh.size1() != dF()) {
		Bhhh.resize(dF());
	}
	Bhhh.initialize(0.0);
	if (nThreads <= 0) nThreads=1;
	//nThreads=1; /* there may be a bug in the threading */
	BUGGER(msg)<< "Using "<< nThreads <<" threads" ;
	
	boosted::function<boosted::shared_ptr<workshop> ()> workshop_builder =
		boosted::bind(&elm::Model2::make_shared_workshop_ngev_gradient, this);
	USE_DISPATCH(gradient_dispatcher,option.threads, nCases, workshop_builder);
	
	BUGGER(msg)<< "End NGEV Gradient Evaluation" ;

	std::ostringstream ret;
	for (unsigned i=0; i<GCurrent.size(); i++) {
		ret << "," << GCurrent[i];
	}
	INFO(msg) << "NGEV Grad->["<< ret.str().substr(1) <<"] (using "<<option.threads<<" threads)";
}


ELM_RESULTCODE elm::Model2::nest
( const std::string& nest_name
, const elm::cellcode& nest_code
, std::string freedom_name
, const double& freedom_multiplier
)
{
	ELM_RESULTCODE result = ELM_IGNORED;
	try {
		// If parameter not given, assign it the same as the nest name
		if (freedom_name=="") freedom_name = nest_name;
		// If parameter does not exist, create it using LOGSUM defaults
		if (!parameter_exists(freedom_name)) {
			MONITOR(msg) << "automatically generating "<<freedom_name<<" parameter because it does not already exist";
			std::string fn = freedom_name;
			etk::uppercase(fn);
			if (fn!="CONSTANT") {
				parameter(freedom_name,1,1,1,1,0);
			}
			result |= ELM_UPDATED;
		}
		// If node already exists in Input_LogSum, edit it instead of creating a new one 
		ComponentCellcodeMap::iterator i = Input_LogSum.find(nest_code);
		if (i!=Input_LogSum.end()) {
			if (i->second.param_name != freedom_name) {
				i->second.param_name = freedom_name;
				result |= ELM_UPDATED;
			}
			if (i->second.multiplier != freedom_multiplier) {
				i->second.multiplier = freedom_multiplier;
				result |= ELM_UPDATED;
			}
			if (i->second._altname != nest_name) {
				i->second._altname = nest_name;
				result |= ELM_UPDATED;
			}
			if (i->second._altcode != nest_code) {
				i->second._altcode = nest_code;
				result |= ELM_UPDATED;
			}
			MONITOR(msg) << "success: updated parameter on existing node "<<nest_name<<" (" <<nest_code<<")";
			return result;
		}

		result |= Xylem.add_cell(nest_code, nest_name, true);
		if (result & ELM_CREATED) {
			LinearComponent z;
			z.data_name = "";// n.fuse();
			z.param_name = freedom_name;
			z.multiplier = freedom_multiplier;
			z._altcode = nest_code;
			z._altname = nest_name;
			Input_LogSum[nest_code] =  z ;
		}
		INFO(msg) << "created "<<nest_name << "("<<nest_code<<")";
	} SPOO {
		OOPS( cat("error in adding nest: ",oops.what()));
	}
	if (result & ELM_CREATED) {
		if (!option.suspend_xylem_rebuild) {
			elm::cellcode root = Xylem.root_cellcode();
			Xylem.regrow( &Input_LogSum, &Input_Edges, _fountain(), &root, &msg );
			nElementals = Xylem.n_elemental();
			nNests = Xylem.n_branches();
			nNodes = Xylem.size();
		} else {
			// make a best guess for now, and regrow for real later...
			// nElementals = no change;
			nNests += 1;
			nNodes += 1;
		}
	}
	return result;
}


PyObject* __GetInputTupleNest(const elm::LinearComponent& i)
{
//	multiname x (i.data_name);
	if (i.multiplier==1.0) {
		return Py_BuildValue("(sKs)", i._altname.c_str(), i._altcode, i.param_name.c_str());
	}
	return Py_BuildValue("(sKsd)", i._altname.c_str(), i._altcode, i.param_name.c_str(), i.multiplier);
}

PyObject* elm::Model2::_get_nest() const
{
	PyObject* U = PyList_New(0);
	for (ComponentCellcodeMap::const_iterator i=Input_LogSum.begin(); i!=Input_LogSum.end(); i++) {
		PyObject* z = __GetInputTupleNest(i->second);
		PyList_Append(U, z);
		Py_CLEAR(z);
	}
	return U;
}



string elm::Model2::link (const long long& parentCode, const long long& childCode)
{
	return Xylem.add_edge((parentCode),(childCode));
}

PyObject* elm::Model2::_get_link() const
{
	PyObject* U = PyList_New(0);
	std::vector< std::pair<cellcode, cellcode> > List = Xylem.list_edges();
	for (unsigned i=0; i<List.size(); i++) {
		if (List[i].first) {
			PyObject* z = Py_BuildValue("(LL)", List[i].first, List[i].second);
			PyList_Append(U, z);
			Py_CLEAR(z);
		}
	}
	return U;
}


