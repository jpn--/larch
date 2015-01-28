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
#include <iostream>
#include "etk_thread.h"
#include "elm_calculations.h"

using namespace etk;
using namespace elm;
using namespace std;




ComponentGraphDNA elm::Model2::Input_Graph()
{
	return ComponentGraphDNA(&Input_LogSum,&Input_Edges,_Data);
}


void elm::Model2::_setUp_NL()
{
	if (!_Data) OOPS("A database must be linked to this model to do this.");
	INFO(msg)<< "Setting up NL model..." ;
	
	// COUNTING
//	nCases = _Data->nCases(); // set in provisioning
	nElementals = Xylem.n_elemental();
	nNests = Xylem.n_branches();
	nNodes = Xylem.size();
	
//	Params_UtilityCA.resize(Data_UtilityCA->nVars());
//	Params_UtilityCO.resize(Data_UtilityCO->nVars(),_Data->nAlts());
	Params_LogSum.resize(nNests+1);

	nThreads = option.threads;
	if (nThreads < 1) nThreads = 1;
	if (nThreads > 1024) nThreads = 1024;
	
	Xylem.regrow(nullptr,nullptr,nullptr,nullptr,&msg);
	BUGGER(msg) << "_setUp_NL:Xylem:\n" << Xylem.display();
	
	unsigned slot;
	for (ComponentCellcodeMap::iterator m=Input_LogSum.begin(); m!=Input_LogSum.end(); m++) {
		slot = Xylem.slot_from_code(m->first);
	//	multiname itemname (m->second.apply_name);
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
	
	if ( Input_Sampling.ca.size()>0 || Input_Sampling.co.size()>0 ) {
		AdjProbability.resize(nCases,_Data->nAlts());
		SamplingWeight.resize(nCases,_Data->nAlts());
	} else {
		AdjProbability.same_memory_as(Probability);
		SamplingWeight.resize(0);
	}
	
	Workspace.resize(nNodes);
		
	sherpa::allocate_memory();
	

	INFO(msg)<< "Set up NL model complete." ;
	
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

//#endif // ndef __APPLE__


void elm::Model2::nl_probability()
{
	BUGGER(msg) << "Calculate NL probability";
	BUGGER(msg) << "Coef_UtilityCA\n" << Coef_UtilityCA.printall();
	BUGGER(msg) << "Coef_UtilityCO\n" << Coef_UtilityCO.printall();
	BUGGER(msg) << "Coef_SamplingCA\n" << Coef_SamplingCA.printall();
	BUGGER(msg) << "Coef_SamplingCO\n" << Coef_SamplingCO.printall();
	BUGGER(msg) << "Coef_LogSum\n" << Coef_LogSum.printall();
//	if (!Data_UtilityCA) OOPS("data for utility calculation is not loaded, try setUp model first");
	if (Data_UtilityCA /*->is_loaded_in_range(0,1)*/) {
		BUGGER(msg) << "Data_UtilityCA\n" << Data_UtilityCA->printcase(0);
	}
	if (Data_UtilityCO /*->is_loaded_in_range(0, 1)*/) {
		BUGGER(msg) << "Data_UtilityCO\n" << Data_UtilityCO->printcase(0);
	}
	if (Data_Avail /*->is_loaded_in_range(0, 1)*/) {
		BUGGER(msg) << "Data_Avail\n" << Data_Avail->printboolcase(0);
	}

	unsigned c,a;
	unsigned warningcount (0);
	unsigned zerosize_warncount (0);
	bool found_nan (false);
	pull_coefficients_from_freedoms();
	
	if (nThreads<=1) nThreads = 1;
	 
	BUGGER(msg) << "Number of threads in nl_probability =" << nThreads;
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
								 , &msg
								 );};
		#else
		boosted::function<boosted::shared_ptr<workshop> ()> workshop_builder =
			boosted::bind(&elm::Model2::make_shared_workshop_nl_probability, this);
		#endif // def __APPLE__
		USE_DISPATCH(probability_dispatcher,option.threads, nCases, workshop_builder);
	
	} else {
	
	Utility.initialize(0.0);
	__logit_utility(Utility, Data_UtilityCA, Data_UtilityCO, Coef_UtilityCA, Coef_UtilityCO, 0);

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
		if (true) {
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

void elm::Model2::nl_gradient() 
{
	periodic Sup (5);
	BUGGER(msg)<< "Beginning NL Gradient Evaluation" ;
	GCurrent.initialize(0.0);
	Bhhh.initialize(0.0);
	if (nThreads <= 0) nThreads=1;
	//nThreads=1; /* there may be a bug in the threading */
	BUGGER(msg)<< "Using "<< nThreads <<" threads" ;
	
	#ifndef _ELM_USE_THREADS_
	nThreads = 1;
	#endif // ndef _ELM_USE_THREADS_
	
	if (nThreads >= 2 && _ELM_USE_THREADS_) {

//		#ifdef __APPLE__
		boosted::function<boosted::shared_ptr<workshop> ()> workshop_builder =
			boosted::bind(&elm::Model2::make_shared_workshop_nl_gradient, this);
//			[&](){return boosted::make_shared<workshop_nl_gradient>(  dF()
//									 , nNodes
//									 , utility_packet()
//									 , sampling_packet()
//									 , Params_LogSum
//									 , Data_Choice
//									 , Data_Weight_active()
//									 , &AdjProbability
//									 , &Probability
//									 , &Cond_Prob
//									 , &Xylem
//									 , &GCurrent
//									 , &Bhhh
//									 , &msg
//									 );};
//		#else
//		boosted::function<boosted::shared_ptr<workshop> ()> workshop_builder =
//			boosted::bind(&elm::Model2::make_shared_workshop_nl_gradient, this);
//		#endif // def __APPLE__
		USE_DISPATCH(gradient_dispatcher,option.threads, nCases, workshop_builder);
	} else {




		boosted::shared_ptr<workshop> local_grad_workshop;
		if (!local_grad_workshop) {
			local_grad_workshop = make_shared_workshop_nl_gradient ();
		}
		local_grad_workshop->work(0, nCases, nullptr);
		



//
//
//		
//		unsigned c;
//
//		memarray_raw dUtilCA    (nNodes,Params_UtilityCA.length());
//		memarray_raw dUtilCO    (nNodes,Params_UtilityCO.length());
//		memarray_raw dUtilMU    (nNodes,Params_LogSum.length());
//		memarray_raw dProbCA    (nNodes,Params_UtilityCA.length());
//		memarray_raw dProbCO    (nNodes,Params_UtilityCO.length());
//		memarray_raw dProbMU    (nNodes,Params_LogSum.length());
//		memarray_raw WorkspaceCA(Params_UtilityCA.length());
//		memarray_raw WorkspaceCO(Params_UtilityCO.length());
//		memarray_raw WorkspaceMU(Params_LogSum.length());
//		
//		memarray_raw CaseGrad(dF());
//			
//		for (c=0;c<nCases;c++) {
//			
//			CaseGrad.initialize();
//			__casewise_nl_gradient
//			( c
//			 , &Probability
//			 , &Cond_Prob
//			 , &Utility
//			 , &Xylem
//			 , Data_UtilityCA
//			 , Data_UtilityCO
//			 , Data_Choice
//			 , dUtilCA
//			 , dUtilCO
//			 , dUtilMU
//			 , dProbCA
//			 , dProbCO
//			 , dProbMU
//			 , WorkspaceCA
//			 , WorkspaceCO
//			 , WorkspaceMU
//			 , Grad_UtilityCA
//			 , Grad_UtilityCO
//			 , Grad_LogSum);		
//
//		if (Data_Weight_active()) {
//			Grad_UtilityCA.scale(Data_Weight_active()->value(c, 0));
//			Grad_UtilityCO.scale(Data_Weight_active()->value(c, 0));
//			Grad_LogSum.scale(Data_Weight_active()->value(c, 0));
//		}
//
//			push_to_freedoms(Params_UtilityCA  , *Grad_UtilityCA  , *CaseGrad);
//			push_to_freedoms(Params_UtilityCO  , *Grad_UtilityCO  , *CaseGrad);
//			push_to_freedoms(Params_LogSum     , *Grad_LogSum     , *CaseGrad);
//			
//			// BHHH
//			#ifdef SYMMETRIC_PACKED
//			cblas_dspr(CblasRowMajor,CblasUpper, dF(),1,*CaseGrad, 1, *Bhhh);
//			#else
//			cblas_dsyr(CblasRowMajor,CblasUpper, dF(),1,*CaseGrad, 1, *Bhhh, Bhhh.size1());
//			#endif
//			
//			// ACCUMULATE
//			// cblas_daxpy(dF(),1,*CaseGrad,1,*GCurrent,1);
//			GCurrent += CaseGrad;
//			
//			PERIODICALLY(Sup, INFO(msg)) << "  processing gradient for case "<<c<<", "<<100.0*double(c)/double(nCases)<<"% complete";
//			
//			
//		}
	}
	BUGGER(msg)<< "End NL Gradient Evaluation" ;
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
				parameter(freedom_name,1,1);
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
			if (i->second.altname != nest_name) {
				i->second.altname = nest_name;
				result |= ELM_UPDATED;
			}
			if (i->second.altcode != nest_code) {
				i->second.altcode = nest_code;
				result |= ELM_UPDATED;
			}
			INFO(msg) << "success: updated parameter on existing node "<<nest_name<<" (" <<nest_code<<")";
			return result;
		}

		result |= Xylem.add_cell(nest_code, nest_name, true);
		if (result & ELM_CREATED) {
			InputStorage z;
			z.apply_name = "";// n.fuse();
			z.param_name = freedom_name;
			z.multiplier = freedom_multiplier;
			z.altcode = nest_code;
			z.altname = nest_name;
			Input_LogSum[nest_code] =  z ;
		}
		INFO(msg) << "created "<<nest_name << "("<<nest_code<<")";
	} SPOO {
		OOPS( cat("error in adding nest: ",oops.what()));
	}
	if (result & ELM_CREATED) {
		elm::cellcode root = Xylem.root_cellcode();
		Xylem.regrow( &Input_LogSum, &Input_Edges, _Data, &root, &msg );
		nElementals = Xylem.n_elemental();
		nNests = Xylem.n_branches();
		nNodes = Xylem.size();
	}
	return result;
}


PyObject* __GetInputTupleNest(const InputStorage& i)
{
//	multiname x (i.apply_name);
	if (i.multiplier==1.0) {
		return Py_BuildValue("(sKs)", i.altname.c_str(), i.altcode, i.param_name.c_str());
	}
	return Py_BuildValue("(sKsd)", i.altname.c_str(), i.altcode, i.param_name.c_str(), i.multiplier);
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


