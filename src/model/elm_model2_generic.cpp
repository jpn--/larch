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
#include <iomanip>
#include "elm_workshop_loglike.h"


using namespace etk;
using namespace elm;
using namespace std;

elm::Model2::Model2()
: _Data (NULL)
, Data_UtilityCA  (nullptr)
, Data_UtilityCO  (nullptr)
, Data_SamplingCA (nullptr)
, Data_SamplingCO (nullptr)
, Data_Allocation (nullptr)
, Data_QuantityCA (nullptr)
//, Data_QuantLogSum(nullptr)
//, Data_LogSum     (nullptr)
, Data_Choice     (nullptr)
, Data_Weight     (nullptr)
, Data_Avail      (nullptr)
, _LL_null (NAN)
, _LL_nil (NAN)
, _LL_constants (NAN)
, weight_scale_factor (1.0)
, nCases (0)
, _nCases_recall (0)
, nElementals (0)
, nNests (0)
, nNodes (0)
, nThreads (1)
, availability_ca_variable ("")
, features (0)
, option()
, _is_setUp(0)
//, weight_autorescale (false)
, Input_Utility("utility",this)
, Input_QuantityCA(COMPONENTLIST_TYPE_UTILITYCA, this)
, Input_LogSum(COMPONENTLIST_TYPE_LOGSUM, this)
, Input_Edges(this)
, Input_Sampling("samplingbias",this)
, title("Untitled Model")
, _string_sender_ptr(nullptr)
, hessian_matrix(new etk::symmetric_matrix())
{
}


elm::Model2::Model2(elm::Facet& datafile)
: _Data (&datafile)
, Data_UtilityCA  (nullptr)
, Data_UtilityCO  (nullptr)
, Data_SamplingCA (nullptr)
, Data_SamplingCO (nullptr)
, Data_Allocation (nullptr)
, Data_QuantityCA (nullptr)
//, Data_QuantLogSum(nullptr)
//, Data_LogSum     (nullptr)
, Data_Choice     (nullptr)
, Data_Weight     (nullptr)
, Data_Avail      (nullptr)
, _LL_null (NAN)
, _LL_nil (NAN)
, _LL_constants (NAN)
, weight_scale_factor (1.0)
, nCases (0)
, _nCases_recall (0)
, nElementals (0)
, nNests (0)
, nNodes (0)
, nThreads (1)
, availability_ca_variable ("")
, features (0)
, option()
, _is_setUp(0)
//, weight_autorescale (false)
, Input_Utility("utility",this)
, Input_QuantityCA(COMPONENTLIST_TYPE_UTILITYCA, this)
, Input_LogSum(COMPONENTLIST_TYPE_LOGSUM, this)
, Input_Edges(this)
, Input_Sampling("samplingbias",this)
, title("Untitled Model")
, _string_sender_ptr(nullptr)
, hessian_matrix(new etk::symmetric_matrix())
{
	Py_INCREF(_Data->apsw_connection);
	//msg.change_logger_name(logfilename);
	Xylem.add_dna_sequence(_Data->alternatives_dna());
}

elm::ParameterList* elm::Model2::_self_as_ParameterListPtr()
{
	return this;
}




elm::ca_co_packet elm::Model2::utility_packet()
{
	if (true /*Data_UtilityCA->fully_loaded() && Data_UtilityCO->fully_loaded()*/) {
		BUGGER(msg) << "spawning utility packet fully loaded";
		return elm::ca_co_packet(&Params_UtilityCA	,
								 &Params_UtilityCO	,
								 &Coef_UtilityCA	,
								 &Coef_UtilityCO	,
								 Data_UtilityCA		,
								 Data_UtilityCO		,
								 &Utility			);
	} else {
	/*	BUGGER(msg) << "Data_UtilityCA->fully_loaded = " << Data_UtilityCA->fully_loaded();
		BUGGER(msg) << "Data_UtilityCO->fully_loaded = " << Data_UtilityCO->fully_loaded();
		WARN(msg) << "spawning utility packet when data is not fully loaded, things are gonna be slow...";
		ScrapePtr d_ca = Data_UtilityCA->copy();
		ScrapePtr d_co = Data_UtilityCO->copy();
		
		return elm::ca_co_packet(&Params_UtilityCA	,
								 &Params_UtilityCO	,
								 &Coef_UtilityCA	,
								 &Coef_UtilityCO	,
								 d_ca		        ,
								 d_co		        ,
								 Data_UtilityCA		,
								 Data_UtilityCO		,
								 &Utility			);*/
	}
	
}

elm::ca_co_packet elm::Model2::quantity_packet()
{
		return elm::ca_co_packet(&Params_QuantityCA	,
								 nullptr         	,
								 &Coef_QuantityCA	,
								 nullptr         	,
								 Data_QuantityCA	,
								 nullptr         	,
								 &Quantity			);
}


elm::ca_co_packet elm::Model2::sampling_packet()
{
		return elm::ca_co_packet(&Params_SamplingCA	,
								 &Params_SamplingCO	,
								 &Coef_SamplingCA	,
								 &Coef_SamplingCO	,
								 Data_SamplingCA	,
								 Data_SamplingCO	,
								 &SamplingWeight	);
}

elm::ca_co_packet elm::Model2::allocation_packet()
{
		return elm::ca_co_packet(nullptr	,
								 &Params_Edges	,
								 nullptr	,
								 &Coef_Edges	,
								 nullptr	,
								 Data_Allocation	,
								 &Allocation	);
}








void elm::Model2::change_data_pointer(elm::Facet& datafile)
{
	if (_Data) {
		Py_DECREF(_Data->apsw_connection);
	}
	_Data = &datafile;
	if (_Data) {
		Py_INCREF(_Data->apsw_connection);
	}
	elm::cellcode root = Xylem.root_cellcode();
	Xylem.clear();
	Xylem.add_dna_sequence(_Data->alternatives_dna());
	if (!option.suspend_xylem_rebuild) {
		Xylem.regrow( &Input_LogSum, &Input_Edges, _Data, &root, &msg );
	}
	
	nElementals = Xylem.n_elemental();
	nNests = Xylem.n_branches();
	nNodes = Xylem.size();

}

void elm::Model2::delete_data_pointer()
{
	if (_Data) {
		Py_DECREF(_Data->apsw_connection);
	}
	_Data = NULL;
	Xylem.clear();
}


PyObject* elm::Model2::logger (const std::string& logname)
{
	if (logname=="") {
//		msg.change_logger_name("");
	} else if (logname.substr(0,6)=="larch.") {
		msg.change_logger_name(logname);
	} else {
		msg.change_logger_name("larch."+logname);
	}
	
	return msg.get_logger();
}

PyObject* elm::Model2::logger (bool z)
{
	if (z) {
		msg.change_logger_name("larch.Model");
	} else {
		msg.change_logger_name("");
	}
	return msg.get_logger();
}

PyObject* elm::Model2::logger (int z)
{
	if (z>0) {
		msg.change_logger_name("larch.Model");
	} else {
		msg.change_logger_name("");
	}
	return msg.get_logger();
}

PyObject* elm::Model2::logger (PyObject* z)
{
	if (PyLong_Check(z)) {
//		std::cerr << "logger int "<<((int)PyLong_AS_LONG(z))<<"\n";
		return logger((int)PyLong_AS_LONG(z));
	}

	if (PyUnicode_Check(z)) {
//		std::cerr << "logger string "<<(PyString_ExtractCppString(z))<<"\n";
		return logger(PyString_ExtractCppString(z));
	}

	if (PyBool_Check(z)) {
//		std::cerr << "logger bool "<<(Py_True==z)<<"\n";
		return logger(Py_True==z);
	}

	if (z && z!=Py_None) {
//		if (z==Py_None) {std::cerr << "logger obj NONE \n";} else {std::cerr << "logger obj SOME \n";}
		return msg.set_logger(z);
	}
	
	return msg.get_logger();
}


PyObject* elm::Model2::_get_logger () const
{
	return py_one_item_list(PyString_FromString(msg.get_logger_name().c_str()));
}


elm::Model2::~Model2()
{ 
	tearDown();
	
}




elm::cellcode elm::Model2::_get_root_cellcode() const
{
	return Xylem.root_cellcode();
}


void elm::Model2::_set_root_cellcode(const elm::cellcode& r)
{
	Xylem.root_cellcode( r, &msg );
}




runstats elm::Model2::estimate_tight(double magnitude)
{
	vector<sherpa_pack> packs;
	// (char algorithm='G', double threshold=0.0001, double initial_step=1, unsigned slowness=0,
	//			double min_step=1e-10, double max_step=4, double step_extend_factor=2, double ste_retract_factor=.5,
	//			unsigned honeymoon=3, double patience=1.)
	packs.push_back( sherpa_pack('G', pow(10,-magnitude), 1.0,   0,    1e-20, 4, 2, .5,    1, 1.0) );
	packs.push_back( sherpa_pack('B', pow(10,-magnitude), 1.0,   0,    1e-20, 4, 2, .5,    3, 1.0) );
	packs.push_back( sherpa_pack('S', pow(10,-magnitude), 1.e-6, 100,  1e-20, 4, 2, .5,    1, 100.) );
	return estimate(&packs);
}

runstats elm::Model2::estimate(std::vector<sherpa_pack> opts)
{
	return estimate(&opts);
}


runstats elm::Model2::estimate()
{
	return estimate(NULL);
}


runstats elm::Model2::_maximize_bhhh()
{
	// this function is used in emulating the scipy.optimize interface

	BUGGER(msg) << "Estimating the model.";
	_latest_run.restart();
	_latest_run.number_cpu_cores = etk::number_of_cpu;
	_latest_run.number_threads = option.threads;

	try {
		ZBest = -INF;
		
		flag_gradient_diagnostic = option.gradient_diagnostic; //Option_integer("gradient_diagnostic");
		flag_hessian_diagnostic = option.hessian_diagnostic; // Option_integer("hessian_diagnostic");
		flag_log_turns = option.log_turns;
		BUGGER(msg) << "Diagnostic flags set.";
	
		if (flag_gradient_diagnostic) {
			WARN(msg) << "Gradient Diagnostic set to "<<flag_gradient_diagnostic;
		}
	
		
		if (option.weight_autorescale) {
			_latest_run.start_process("weight autorescale");
			write_runstats_note(auto_rescale_weights());
		}
		
		_latest_run.start_process("optimize:bhhh");

		vector<sherpa_pack> packs;
		packs.push_back( sherpa_pack('G', 0.000001, 1, 0,   1e-10, 4, 2, .5,    1, 0.001) );
		_latest_run.results += maximize(_latest_run.iteration,&packs);
		
		if (option.weight_autorescale) {
			_latest_run.start_process("weight unrescale");
			restore_scale_weights();
			MONITOR(msg) << "recalculating log likelihood and gradients at normal weights";
			ZBest = ZCurrent = objective();
			gradient();
		}
		

	} SPOO {
		if (oops.code()==-8) {
			_update_freedom_info_best();
			_latest_run.write_result( "User Interrupt" );
			PYTHON_INTERRUPT;
		}
		_latest_run.write_result( "error: ", oops.what() );
		MONITOR(msg) << "ERROR IN ESTIMATION: " << oops.what() ;
		_latest_run.end_process();
		throw;
	}
	
	
	BUGGER(msg) << "updating freedom info...";
	try {
		_update_freedom_info(&invHess, &robustCovariance);
	} SPOO {
		try {
			_update_freedom_info();
		} SPOO {
			_latest_run.write_result( "error: ", oops.what() );
			OOPS(oops.what());
		}
	}
	
	
	BUGGER(msg) << "record finish time...";
	
	tm * timeinfo;
	time_t rawtime;
	char timebuff[256] = {0};
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	strftime(timebuff, 255, "%A, %B %d %Y, %I:%M:%S %p", timeinfo);
	_latest_run.timestamp = string(timebuff);

	BUGGER(msg) << "estimation complete.";
	
	_latest_run.end_process();
	
	return _latest_run;
}


double elm::Model2::loglike_null()
{
	setUp();	
	
	std::vector< int > hold_save (dF());
	
	if (option.null_disregards_holdfast) {
		for (unsigned i=0; i<dF(); i++) {
			hold_save[i] = FInfo[FNames[i]].holdfast;
			FInfo[FNames[i]].holdfast = 0;
		}
	}
	
	for (unsigned i=0; i<dF(); i++) {
		FCurrent[i] = FInfo[FNames[i]].null_value;
	}
	freshen();
	_LL_null = objective();			
	for (unsigned i=0; i<dF(); i++) {
		FCurrent[i] = FInfo[FNames[i]].value;
	}
	if (option.null_disregards_holdfast) {
		for (unsigned i=0; i<dF(); i++) {
			FInfo[FNames[i]].holdfast = hold_save[i];
		}
	}
	freshen();
	
	return _LL_null;
}


runstats elm::Model2::estimate(std::vector<sherpa_pack>* opts)
{
	if (_string_sender_ptr) {
		ostringstream update;
		update << "Estimating the model now...";
		_string_sender_ptr->write(update.str());
	}


	BUGGER(msg) << "Estimating the model.";
	_latest_run.restart();
	_latest_run.number_cpu_cores = etk::number_of_cpu;
	_latest_run.number_threads = option.threads;

	try {
		_latest_run.start_process("setup");
		setUp();	
		BUGGER(msg) << "Setup of model complete.";
		ZBest = -INF;
		
		flag_gradient_diagnostic = option.gradient_diagnostic; //Option_integer("gradient_diagnostic");
		flag_hessian_diagnostic = option.hessian_diagnostic; // Option_integer("hessian_diagnostic");
		flag_log_turns = option.log_turns;
		BUGGER(msg) << "Diagnostic flags set.";
	
		if (flag_gradient_diagnostic) {
			WARN(msg) << "Gradient Diagnostic set to "<<flag_gradient_diagnostic;
		}
	
		if (option.calc_null_likelihood) {
			_latest_run.start_process("null_likelihood");
			
			std::vector< int > hold_save (dF());
			
			if (option.null_disregards_holdfast) {
				for (unsigned i=0; i<dF(); i++) {
					hold_save[i] = FInfo[FNames[i]].holdfast;
					FInfo[FNames[i]].holdfast = 0;
				}
			}
			
			for (unsigned i=0; i<dF(); i++) {
				FCurrent[i] = FInfo[FNames[i]].null_value;
			}
			freshen();
			_LL_null = objective();			
			for (unsigned i=0; i<dF(); i++) {
				FCurrent[i] = FInfo[FNames[i]].value;
			}
			if (option.null_disregards_holdfast) {
				for (unsigned i=0; i<dF(); i++) {
					FInfo[FNames[i]].holdfast = hold_save[i];
				}
			}
			freshen();
		}
		
		if (option.weight_autorescale) {
			_latest_run.start_process("weight autorescale");
			write_runstats_note(auto_rescale_weights());
		}
		
		_latest_run.start_process("maximize likelihood");
		_latest_run.results += maximize(_latest_run.iteration,opts);
		
		if (option.weight_autorescale) {
			_latest_run.start_process("weight unrescale");
			restore_scale_weights();
			MONITOR(msg) << "recalculating log likelihood and gradients at normal weights";
			ZBest = ZCurrent = objective();
			gradient();
		}
		
		if (option.calc_std_errors) {
			_latest_run.start_process("parameter covariance");
			//hessian();
			//MONITOR(msg) << "HESSIAN\n" << Hess.printSquare() ;
			//invHess = Hess;
			//invHess.inv(true);
			//MONITOR(msg) << "invHESSIAN\n" << invHess.printSquare() ;
			BUGGER(msg) << "calculate_parameter_covariance...";
			calculate_parameter_covariance(false);
			BUGGER(msg) << "calculate_parameter_covariance complete.";
		} else {
			Hess.initialize(NAN);
			invHess.initialize(NAN);
		}

	} SPOO {
		if (oops.code()==-8) {
			_update_freedom_info_best();
			_latest_run.write_result( "User Interrupt" );
			PYTHON_INTERRUPT;
		}
		_latest_run.write_result( "error: ", oops.what() );
		MONITOR(msg) << "ERROR IN ESTIMATION: " << oops.what() ;
		_latest_run.end_process();
		throw;
	}
	
	
	BUGGER(msg) << "updating freedom info...";
	try {
		_update_freedom_info(&invHess, &robustCovariance);
	} SPOO {
		try {
			_update_freedom_info();
		} SPOO {
			_latest_run.write_result( "error: ", oops.what() );
			OOPS(oops.what());
		}
	}
	
	
	if (option.teardown_after_estimate) {
		BUGGER(msg) << "teardown...";
		_latest_run.start_process("tearDown");
		tearDown();
	}
	
	
	BUGGER(msg) << "record finish time...";
	
	if (_latest_run.results.empty()) _latest_run.results = "ignored";

	tm * timeinfo;
	time_t rawtime;
	char timebuff[256] = {0};
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	strftime(timebuff, 255, "%A, %B %d %Y, %I:%M:%S %p", timeinfo);
	_latest_run.timestamp = string(timebuff);

	BUGGER(msg) << "estimation complete.";
	
	_latest_run.end_process();
	
	return _latest_run;
}



void elm::Model2::calculate_hessian_and_save()
{
	calculate_hessian();
	
	// Store hessian for later model user analysis if desired
	*hessian_matrix = Hess;
	hessian_matrix->copy_uppertriangle_to_lowertriangle();
}



//bool elm::Model2::any_holdfast()
//{
//	for (size_t hi=0; hi<dF(); hi++) {
//		if (FInfo[ FNames[hi] ].holdfast) {
//			return true;
//		}
//	}
//	return false;
//}
//
//size_t elm::Model2::count_holdfast()
//{
//	size_t n=0;
//	for (size_t hi=0; hi<dF(); hi++) {
//		if (FInfo[ FNames[hi] ].holdfast) {
//			n++;
//		}
//	}
//	return n;
//}


void elm::Model2::calculate_parameter_covariance(bool update_freedoms)
{
	MONITOR(msg) << "calculate_parameter_covariance()\n";

	if (any_holdfast()) {

		WARN(msg) << "Calculating inverse hessian without holdfast parameters";

		calculate_hessian_and_save();
		
		symmetric_matrix temp_free_hess (Hess.size1()-count_holdfast());
		
		// invert hessian
		MONITOR(msg) << "HESSIAN\n" << Hess.printSquare() ;
		hessfull_to_hessfree(&Hess, &temp_free_hess) ;
		MONITOR(msg) << "HESSIAN squeezed\n" << temp_free_hess.printSquare() ;
		temp_free_hess.inv();
		MONITOR(msg) << "invHESSIAN squeezed\n" << temp_free_hess.printSquare() ;
		hessfree_to_hessfull(&invHess, &temp_free_hess) ;
		MONITOR(msg) << "invHESSIAN\n" << invHess.printSquare() ;

		symmetric_matrix unpacked_bhhh;
		symmetric_matrix unpacked_invhess;
		MONITOR(msg) << "Bhhh\n" << Bhhh.printSquare() ;

		// unpack bhhh and inverse hessian for analysis
		unpacked_bhhh = Bhhh;
		unpacked_invhess = invHess;
		unsigned s = unpacked_bhhh.size1();
		unpacked_bhhh.copy_uppertriangle_to_lowertriangle();

		MONITOR(msg) << "unpacked_bhhh\n" << unpacked_bhhh.printall() ;
		MONITOR(msg) << "unpacked_invhess (calculate_parameter_covariance, yes holdfast)\n" << unpacked_invhess.printall() ;
		
		MONITOR(msg) << "s="<<s<<"\n";
		symmetric_matrix BtimeH (s,s);
		
		MONITOR(msg) << "cblas_dsymm...\n";
		
		cblas_dsymm(CblasRowMajor, CblasRight, CblasUpper, s, s,
					1, *unpacked_invhess, s, 
					*unpacked_bhhh, s, 
					0, *BtimeH, s); 
		MONITOR(msg) << "BtimeH\n" << BtimeH.printall() ;
		cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper, s, s, 
					1, *unpacked_invhess, s, 
					*BtimeH, s, 
					0, *unpacked_bhhh, s);
		
		MONITOR(msg) << "unpacked_bhhh.2\n" << unpacked_bhhh.printall() ;
		robustCovariance = unpacked_bhhh;
		MONITOR(msg) << "robustCovariance\n" << robustCovariance.printSquare() ;



	} else {

		calculate_hessian_and_save();
			
		// invert hessian
		MONITOR(msg) << "HESSIAN\n" << Hess.printSquare() ;
		invHess = Hess;
		invHess.inv();
		MONITOR(msg) << "invHESSIAN\n" << invHess.printSquare() ;

		symmetric_matrix unpacked_bhhh;
		symmetric_matrix unpacked_invhess;
		MONITOR(msg) << "Bhhh\n" << Bhhh.printSquare() ;

		// unpack bhhh and inverse hessian for analysis
		unpacked_bhhh = Bhhh;
		unpacked_invhess = invHess;
		unsigned s = unpacked_bhhh.size1();
		unpacked_bhhh.copy_uppertriangle_to_lowertriangle();

		MONITOR(msg) << "unpacked_bhhh\n" << unpacked_bhhh.printall() ;
		MONITOR(msg) << "unpacked_invhess (calculate_parameter_covariance, no holdfast)\n" << unpacked_invhess.printall() ;

		MONITOR(msg) << "s="<<s<<"\n";
		
		symmetric_matrix BtimeH (s,s);

		MONITOR(msg) << "cblas_dsymm...\n";
		
		cblas_dsymm(CblasRowMajor, CblasRight, CblasUpper, s, s,
					1, *unpacked_invhess, s, 
					*unpacked_bhhh, s, 
					0, *BtimeH, s); 
		MONITOR(msg) << "BtimeH\n" << BtimeH.printall() ;
		cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper, s, s, 
					1, *unpacked_invhess, s, 
					*BtimeH, s, 
					0, *unpacked_bhhh, s);
		
		MONITOR(msg) << "unpacked_bhhh.2\n" << unpacked_bhhh.printall() ;
		robustCovariance = unpacked_bhhh;
		MONITOR(msg) << "robustCovariance\n" << robustCovariance.printSquare() ;

	}

	if (update_freedoms) {
		BUGGER(msg) << "updating freedom info...";
		try {
			_update_freedom_info(&invHess, &robustCovariance);
		} SPOO {
			FATAL(msg) << "error in updating standard errors";
		}
	}

}


//std::string elm::Model2::weight(const std::string& varname, bool reweight) {
//	
//	weight_autorescale = reweight;
//	
//	ostringstream ret;
//	if (varname.empty()) {
//		weight_CO_variable.clear();
//		weight_autorescale = false;
//		ret << "success: weight is cleared";
//	} else {
//		weight_CO_variable = varname;
//		ret << "success: weight_CO is set to " << weight_CO_variable;
//		if (weight_autorescale) {
//			ret << " (auto-reweighted)";
//		}
//	}
//	return ret.str();
//	
//}
//
//PyObject*	elm::Model2::_get_weight() const
//{
//	return etk::py_one_item_list(
//		Py_BuildValue("(si)", weight_CO_variable.c_str(), int(weight_autorescale))
//	);
//}


void elm::Model2::_parameter_push(const std::vector<double>& v)
{
	if (v.size() != dF()) {
		OOPS("You must specify values for exactly the correct number of degrees of freedom (",dF(),"), you gave ",v.size(),".");
	}
	for (unsigned z=0; z<v.size(); z++) {
		if (FInfo[FNames[z]].holdfast) {
			if (v[z] != FInfo[FNames[z]].value) {
				WARN( msg ) << "WARNING: ignoring the given value of "<<v[z]<<" for " << FNames[z]
				<< ", it differs from the holdfast value of " <<FInfo[FNames[z]].value;
			}
		} else {
			FCurrent[z] = v[z];
			FInfo[FNames[z]].value = v[z];
		}
	}
	freshen();
}


void elm::Model2::_parameter_update()
{
	size_t FCurrent_size = FCurrent.size();
	size_t FMax_size = FMax.size();
	size_t FMin_size = FMin.size();
	size_t df_z = dF();

	for (unsigned i=0; i<dF(); i++) {
		freedom_info* f = &(FInfo[FNames[i]]);
		FCurrent[i] = f->value;
		FMax[i] = f->max_value;
		FMin[i] = f->min_value;
	}
	freshen();
}

void elm::Model2::_parameter_log()
{
	MONITOR(msg) << "-- Parameter Values --";
	for (unsigned i=0; i<dF(); i++) {
		freedom_info* f = &(FInfo[FNames[i]]);
		MONITOR(msg) << FNames[i] << ":" << f->value;
	}
	MONITOR(msg) << "----------------------";
}

std::string elm::Model2::_parameter_report() const
{
	std::ostringstream buff;
	int pname_width = 9;
	for ( size_t i=0; i<dF(); i++) {
		if (FNames[i].length() > pname_width) {
			pname_width = FNames[i].length();
		}
	}
	buff << "\n" ;
	buff.width(pname_width);
	buff << "Parameter" << "\t";
	buff.width(18);
	buff << "Value" << "\n" ;
	for ( size_t i=0; i<dF(); i++) {
		buff.width(pname_width);
		buff << FNames[i] << "\t";
		buff.width(18);
		buff << ReadFCurrent()[i] << "\n";
	}
	return buff.str();
}

std::vector< std::string > elm::Model2::parameter_names() const
{
	return FNames.strings();
}

std::vector< std::string > elm::Model2::alias_names() const
{
	std::vector< std::string > x;
	for (auto i=AliasInfo.begin(); i!=AliasInfo.end(); i++) {
		x.push_back(i->first);
	}
	return x;
}

PyObject* __GetParameterDict(const freedom_info& i)
{
	PyObject* P = PyDict_New();
	PyObject* item (nullptr);
	
	item = PyString_FromString(i.name.c_str());
	PyDict_SetItemString(P,"name",item);
	Py_CLEAR(item);

	item = PyFloat_FromDouble(i.initial_value);
	PyDict_SetItemString(P,"initial_value",item);
	Py_CLEAR(item);

	item = PyFloat_FromDouble(i.null_value);
	PyDict_SetItemString(P,"null_value",item);
	Py_CLEAR(item);

	item = PyFloat_FromDouble(i.value);
	PyDict_SetItemString(P,"value",item);
	Py_CLEAR(item);

	item = PyFloat_FromDouble(i.std_err);
	PyDict_SetItemString(P,"std_err",item);
	Py_CLEAR(item);

	item = PyFloat_FromDouble(i.robust_std_err);
	PyDict_SetItemString(P,"robust_std_err",item);
	Py_CLEAR(item);

	item = PyFloat_FromDouble(i.max_value);
	PyDict_SetItemString(P,"max_value",item);
	Py_CLEAR(item);

	item = PyFloat_FromDouble(i.min_value);
	PyDict_SetItemString(P,"min_value",item);
	Py_CLEAR(item);
	
	item = PyInt_FromLong(i.holdfast);
	PyDict_SetItemString(P,"holdfast",item);
	Py_CLEAR(item);
	
	item = i.getCovariance();
	if (item) PyDict_SetItemString(P,"covariance",item);
	Py_CLEAR(item);

	item = i.getRobustCovariance();
	if (item) PyDict_SetItemString(P,"robust_covariance",item);
	Py_CLEAR(item);

	return P;
}


PyObject* elm::Model2::_get_parameter() const
{
	PyObject* U = PyList_New(0);
	for (unsigned i=0; i<FNames.size(); i++) {
		std::map<std::string,freedom_info>::const_iterator FInfoIter = FInfo.find(FNames[i]);
		if (FInfoIter==FInfo.end()) continue;
		PyObject* z =  __GetParameterDict(FInfoIter->second);
		PyList_Append(U,z);
		Py_CLEAR(z);
	}
	return U;
}

std::vector<double> elm::Model2::parameter_values() const {
	std::vector<double> ret (dF());
	for (unsigned i=0; i<dF(); i++) {
		ret[i] = FInfo.find(FNames[i])->second.value;
	}
	return ret;
}


PyObject* elm::Model2::_get_estimation_statistics () const
{
	PyObject* P = PyDict_New();
	etk::py_add_to_dict(P, "log_like", ZCurrent);
	etk::py_add_to_dict(P, "log_like_null", _LL_null);
	etk::py_add_to_dict(P, "log_like_nil", _LL_nil);
	etk::py_add_to_dict(P, "log_like_constants", _LL_constants);
	etk::py_add_to_dict(P, "log_like_best", ZBest);
	
	return py_one_item_list(P);
}

PyObject* elm::Model2::_get_estimation_run_statistics () const
{
	return etk::py_one_item_list(_latest_run.dictionary());
}

void elm::Model2::_set_estimation_statistics
(	const double& log_like
,	const double& log_like_null
,	const double& log_like_nil
,	const double& log_like_constants
,	const double& log_like_best
)
{

	if (!isNan(log_like)) {
		ZCurrent = log_like;
	}
	if (!isNan(log_like_null)) {
		_LL_null = log_like_null;
	}
	if (!isNan(log_like_nil)) {
		_LL_nil = log_like_nil;
	}
	if (!isNan(log_like_constants)) {
		_LL_constants = log_like_constants;
	}
	if (!isNan(log_like_best)) {
		ZBest = log_like_best;
	}
}


void elm::Model2::_set_estimation_run_statistics_pickle
(	PyObject* other )
{
	if (other) {
		Py_INCREF(other);
		_latest_run.read_from_dictionary(other);
		Py_DECREF(other);
	}
}

runstats& elm::Model2::RunStatistics()
{
	return _latest_run;
}


//#define AsPyFloat(x) "float.fromhex('" << std::hexfloat << (x) << std::defaultfloat << "')"

std::string AsPyFloat(const double& x)
{
	ostringstream s;

	if(isNan(x)) {
		return "nan";
	}

	if(isInf(x)) {
		if (x<0) {
			return "-inf";
		} else {
			return "inf";
		}
	}

	double integer_part;
	if (modf(x, &integer_part)==0) {
		// the fractional part is exactly 0 (i.e. the value x is an integer)
		s << x;
		return s.str();
	}
	
	s << std::scientific << std::setprecision(17) << x;
	
//	auto f = s.str().find("0000000");
//	if (f!=std::string::npos) {
//		// when the full precision value has seven zeros, it is almost exactly the value before that point
//		std::string z = s.str();
//		return z.substr(0,f) + z.substr(z.find("e"));
//	}

	// if not an integer, use hex to preserve the exact value at full precision
	
	ostringstream h;
	h << "float.fromhex('" << std::hexfloat << std::setprecision(13) << (x) << std::defaultfloat << "')";
	return h.str();
}


std::string _repr(const std::string& x)
{
	bool has_endline = false;
	bool has_singlequote = false;

	if (x.find("\n")!=std::string::npos) {
		has_endline = true;
	}

	if (x.find("'")!=std::string::npos) {
		has_singlequote = true;
	}
	
	if (!has_singlequote) {
		if (has_endline) {
			return "'''" + x + "'''";
		}
		return "'" + x + "'";
	} else {
		if (has_endline) {
			return "\"\"\"" + x + "\"\"\"";
		}
		return "\"" + x + "\"";
	}
}

std::string __base64encode_wrap(const std::string& in)
{

	PyObject* b64string = PyObject_CallMethod(etk::base64_module, "standard_b64encode", "y", in.c_str());
	std::string out = PyBytes_AsString(b64string);
	Py_CLEAR(b64string);
	return "_Str('"+out+"')";
}

std::string elm::Model2::save_buffer() const
{
	ostringstream sv;

//	this stuff is defined in the load method now...
//	sv << "import numpy\n" << "inf=numpy.inf\n" << "nan=numpy.nan\n";
//	sv << "def _Str(s): import base64; return (base64.standard_b64decode(s)).decode()\n\n";

	
	
	etk::logging_service msg (this->msg);
	
	sv << "self.title = "<<__base64encode_wrap(title)<<"\n\n";
	
	// save parameter
//	BUGGER( msg ) << "save parameter";
	for (auto p=FNames.strings().begin(); p!=FNames.strings().end(); p++) {
//		BUGGER( msg ) << "save parameter "<<*p;
		auto i = FInfo.find(*p);
		if (i==FInfo.end()) {
    		BUGGER( msg ) << "error in save parameter "<<*p<<", not found in FInfo";
		} else {
//    		BUGGER( msg ) << "save parameter "<<*p<<" found in FInfo";
			auto j = i->second;
//			BUGGER( msg ) << "j.value=" << j.value;
			
			sv << "self.parameter("<<__base64encode_wrap(*p);
			sv << ","<<AsPyFloat(j.value);
//			BUGGER( msg ) << "j.null_value=" << j.null_value;
			sv << ","<<AsPyFloat(j.null_value);
//			BUGGER( msg ) << "j.initial_value=" << j.initial_value;
			sv << ","<<AsPyFloat(j.initial_value);
//			BUGGER( msg ) << "j.max_value=" << j.max_value;
			sv << ","<<AsPyFloat(j.max_value);
//			BUGGER( msg ) << "j.min_value=" << j.min_value;
			sv << ","<<AsPyFloat(j.min_value);
//			BUGGER( msg ) << "j.std_err=" << j.std_err;
			sv << ","<<AsPyFloat(j.std_err);
//			BUGGER( msg ) << "j.robust_std_err=" << j.robust_std_err;
			sv << ","<<AsPyFloat(j.robust_std_err);
//			BUGGER( msg ) << "j.holdfast=" << j.holdfast;
			sv << ","<<j.holdfast;
			
			if (j._covar) {
//				BUGGER( msg ) << "j._covar is not null";
//				int check = PyMapping_Check(j._covar);
//				if (check) {
//				BUGGER( msg ) << "j._covar check "<<check;
//				BUGGER( msg ) << "j._covar length "<< (int)PyMapping_Size(j._covar);
//				}
				PyObject* c = PyObject_Str(j._covar);
//				BUGGER( msg ) << "next line";
				if (c) {
//					BUGGER( msg ) << "j.covariance=";
//					BUGGER( msg ) << PyString_ExtractCppString(c);
					sv << ",covariance="<<PyString_ExtractCppString(c);
				}
				Py_CLEAR(c);
			}
			
			if (j._robust_covar) {
//				BUGGER( msg ) << "j._robust_covar is not null";
//				int check = PyMapping_Check(j._robust_covar);
//				BUGGER( msg ) << "j._robust_covar check "<<check;
				PyObject* c = PyObject_Str(j._robust_covar);
//				BUGGER( msg ) << "next line 2";
				if (c) {
//					BUGGER( msg ) << "j.robust_covariance=";
//					BUGGER( msg ) << PyString_ExtractCppString(c);
					sv << ",robust_covariance="<<PyString_ExtractCppString(c);
				}
				Py_CLEAR(c);
			}
			
			sv << ")\n";
		}
	}
	sv << "\n";
	
	// save alias
	BUGGER( msg ) << "save alias";
	for (auto p=AliasInfo.begin(); p!=AliasInfo.end(); p++) {
		BUGGER( msg ) << "save alias "<<p->first;
		auto j = p->second;
		
		sv << "self.alias("<<__base64encode_wrap(j.name);
		sv << ","<<__base64encode_wrap(j.refers_to);
		sv << ","<<AsPyFloat(j.multiplier);
		sv << ",True)\n";
	}
	sv << "\n";
	
	
	// save utility
	BUGGER( msg ) << "save utility";
	for (auto u=Input_Utility.ca.begin(); u!=Input_Utility.ca.end(); u++) {
		sv << "self.utility.ca("<<__base64encode_wrap(u->data_name)<<","<<__base64encode_wrap(u->param_name)<<","<<AsPyFloat(u->multiplier)<<")\n";
	}
	for (auto u=Input_Utility.co.begin(); u!=Input_Utility.co.end(); u++) {
		for (auto k=u->second.begin(); k!=u->second.end(); k++) {
			sv << "self.utility.co("<<u->first<<","<<__base64encode_wrap(k->data_name)<<","<<__base64encode_wrap(k->param_name)<<","<<AsPyFloat(k->multiplier)<<")\n";
		}
	}
	sv << "\n";

	// save quantity
	BUGGER( msg ) << "save quantity";
	for (auto u=Input_QuantityCA.begin(); u!=Input_QuantityCA.end(); u++) {
		sv << "self.quantity("<<__base64encode_wrap(u->data_name)<<","<<__base64encode_wrap(u->param_name)<<","<<AsPyFloat(u->multiplier)<<")\n";
	}
	sv << "\n";

	// save root_id
	BUGGER( msg ) << "save root_id";
	sv << "self.root_id = "<< _get_root_cellcode() <<"\n";

	// save nest
	BUGGER( msg ) << "save nest";
	for (auto n=Input_LogSum.begin(); n!=Input_LogSum.end(); n++) {
		sv << "self.nest("<<__base64encode_wrap(n->second._altname)<<","<<n->second._altcode<<","<<__base64encode_wrap(n->second.param_name);
		if (n->second.multiplier!=1.0) {
			sv << ","<<AsPyFloat(n->second.multiplier);
		}
		sv << ")\n";
	}
	
	// save link
	BUGGER( msg ) << "save link";
	for (auto n=Input_Edges.begin(); n!=Input_Edges.end(); n++) {
		if (n->second.size()) {
			// This edge has a component list
			//OOPS("not yet implemented for edges with components");
			sv << "self.link("<<n->first.up<<","<<n->first.dn<<")\n";
			
			for (auto edgecompo=n->second.begin(); edgecompo!=n->second.end(); edgecompo++) {
				sv << "self.link["<<n->first.up<<","<<n->first.dn<<"](data="<<__base64encode_wrap(edgecompo->data_name)<<",param="<<__base64encode_wrap(edgecompo->param_name);
				if (edgecompo->multiplier != 1.0) {
					sv << ", multiplier="<<edgecompo->multiplier;
				}
				sv << ")\n";
			}
//			sv << "import warnings\n";
//			sv << "warnings.warn('an edge had components that were not saved; this feature is not implented yet, sorry.')\n";
		} else {
			// This edge has no component list
			sv << "self.link("<<n->first.up<<","<<n->first.dn<<")\n";
		}
	}
	
	// save samplingbias
	BUGGER( msg ) << "save samplingbias";
	for (auto u=Input_Sampling.ca.begin(); u!=Input_Sampling.ca.end(); u++) {
		sv << "self.samplingbias.ca("<<__base64encode_wrap(u->data_name)<<","<<__base64encode_wrap(u->param_name)<<","<<AsPyFloat(u->multiplier)<<")\n";
	}
	for (auto u=Input_Sampling.co.begin(); u!=Input_Sampling.co.end(); u++) {
		for (auto k=u->second.begin(); k!=u->second.end(); k++) {
			sv << "self.samplingbias.co("<<u->first<<","<<__base64encode_wrap(k->data_name)<<","<<__base64encode_wrap(k->param_name)<<","<<AsPyFloat(k->multiplier)<<")\n";
		}
	}
	sv << "\n";
	
	BUGGER( msg ) << "save statistics";
	sv << "self._set_estimation_statistics("<<
		AsPyFloat(ZCurrent)<<","<<
		AsPyFloat(_LL_null) <<","<<
		AsPyFloat(_LL_nil) <<","<<
		AsPyFloat(_LL_constants) <<","<<
		AsPyFloat(ZBest) <<
		")\n";
	sv << "self._set_estimation_run_statistics_pickle('"<< _latest_run.pickled_dictionary() <<"')\n";


	sv << "\n";

	sv << option._save_buffer() << "\n";

	
	sv << "self.recall(nCases="<< _nCases_recall <<")\n";
	
	
	return sv.str();
}



void elm::Model2::parameter_values(std::vector<double> v) {
	
	_parameter_update();
	if (v.size() != dF()) {
		OOPS("You must specify values for exactly the correct number of degrees of freedom (",dF(),"), you gave ",v.size(),".");
	}
	for (unsigned z=0; z<v.size(); z++) {
		if (FInfo[FNames[z]].holdfast) {
			if (v[z] != FInfo[FNames[z]].value) {
				WARN( msg ) << "WARNING: ignoring the given value of "<<v[z]<<" for " << FNames[z]
				<< ", it differs from the holdfast value of " <<FInfo[FNames[z]].value;
			}
		} else {
			FCurrent[z] = v[z];
			FInfo[FNames[z]].value = v[z];
		}
	}
	freshen();

}



std::shared_ptr<etk::ndarray> elm::Model2::negative_d_loglike_cached() {
	std::shared_ptr<etk::ndarray> cached_grad;
	setUp();
	_parameter_update();
	if (FCurrent.size()<=0) {
		OOPS_CACHE("error in recovering cached value for d_loglike at the current parameters (init fail)");
	}

	if (_cached_results.read_cached_grad(elm::array_compare(FCurrent.ptr(), FCurrent.size()), cached_grad) && cached_grad) {
		return cached_grad;
	} else {
		OOPS_CACHE("there is no cached value for d_loglike at the current parameters");
	}
}

std::shared_ptr<etk::ndarray> elm::Model2::negative_d_loglike_cached(const std::vector<double>& v) {
	std::shared_ptr<etk::ndarray> cached_grad;
	if (_cached_results.read_cached_grad(elm::array_compare(v), cached_grad) && cached_grad) {
		return cached_grad;
	} else {
		OOPS_CACHE("there is no cached value for d_loglike at the given parameters");
	}
}


std::shared_ptr<etk::ndarray> elm::Model2::negative_d_loglike_nocache() {
	setUp();

	_parameter_update();
	std::shared_ptr<etk::ndarray> g = make_shared<etk::ndarray>(gradient(), false);
	bool z = true;
	for (auto i=0; i!=g->size(); i++) {
		if (g->at(i) != 0.0) {
			z = false;
			break;
		}
	}
	if (z) {
		auto fr = option.force_recalculate;
		option.force_recalculate = true;
		*g = gradient();
		option.force_recalculate = fr;
	}

	etk::symmetric_matrix bhhh (Bhhh, false);
	bhhh.copy_uppertriangle_to_lowertriangle();

	_cached_results.set_cached_grad(elm::array_compare(FCurrent.ptr(), FCurrent.size()), g);
	_cached_results.set_cached_bhhh(elm::array_compare(FCurrent.ptr(), FCurrent.size()), bhhh);
	return g;
}


std::shared_ptr<etk::ndarray> elm::Model2::negative_d_loglike() {
	try {
		return negative_d_loglike_cached();
	} catch (LarchCacheError) {
		return negative_d_loglike_nocache();
	}
}


std::shared_ptr<etk::ndarray> elm::Model2::negative_d_loglike_nocache(const std::vector<double>& v) {
		
	setUp();
	//if (!$self->_is_setUp) OOPS("Model is not setup, try calling setUp() first.");
	_parameter_update();
	_parameter_push(v);
	
	std::shared_ptr<etk::ndarray> g;
	try {
		g = std::make_shared<etk::ndarray>(gradient(), false);
	} catch (ZeroProbWhenChosen) {
		g = std::make_shared<etk::ndarray>(dF());
		for (size_t i=0; i!=g->size(); i++) {
			(*g)[i] = NAN;
		}
		return g;
	}
	
	
	bool z = true;
	for (size_t i=0; i!=g->size(); i++) {
		if ((*g)[i] != 0.0) {
			z = false;
			break;
		}
	}
	if (z) {
		auto fr = this->option.force_recalculate;
		this->option.force_recalculate = true;
		*g = this->gradient();
		this->option.force_recalculate = fr;
	}
	
	_cached_results.set_cached_grad(elm::array_compare(v), g);

	etk::symmetric_matrix bhhh (Bhhh, false);
	bhhh.copy_uppertriangle_to_lowertriangle();
	_cached_results.set_cached_bhhh(elm::array_compare(v), bhhh);

	return g;
	
	
}

std::shared_ptr<etk::ndarray> elm::Model2::negative_d_loglike(const std::vector<double>& v) {
	try {
		return negative_d_loglike_cached(v);
	} catch (LarchCacheError) {
		return negative_d_loglike_nocache(v);
	}
}

std::shared_ptr<etk::ndarray> elm::Model2::_gradient_casewise() {

	if ((features & MODELFEATURES_ALLOCATION)) {
		return _ngev_gradient_full_casewise();
	} else if (features & MODELFEATURES_QUANTITATIVE) {
		return _ngev_gradient_full_casewise();
	} else if (features & MODELFEATURES_NESTING) {
		return _ngev_gradient_full_casewise();
	} else {
		return _mnl_gradient_full_casewise();
	}
	
}

std::shared_ptr<etk::ndarray> elm::Model2::_gradient_casewise(std::vector<double> v) {

	loglike_nocache(v);

	if ((features & MODELFEATURES_ALLOCATION)) {
		return _ngev_gradient_full_casewise();
	} else if (features & MODELFEATURES_QUANTITATIVE) {
		return _ngev_gradient_full_casewise();
	} else if (features & MODELFEATURES_NESTING) {
		return _ngev_gradient_full_casewise();
	} else {
		return _mnl_gradient_full_casewise();
	}
	
}



std::shared_ptr<etk::symmetric_matrix> elm::Model2::bhhh_nocache() {
	setUp();
	_parameter_update();
	
	std::shared_ptr<etk::ndarray> g = std::make_shared<etk::ndarray>(gradient(), false);
	bool z = true;
	for (size_t i=0; i!=g->size(); i++) {
		if ((*g)[i] != 0.0) {
			z = false;
			break;
		}
	}
	if (z) {
		auto fr = this->option.force_recalculate;
		this->option.force_recalculate = true;
		*g = this->gradient();
		this->option.force_recalculate = fr;
	}

	std::shared_ptr<etk::symmetric_matrix> bhhh = std::make_shared<etk::symmetric_matrix>(Bhhh, false);
	
	bhhh->copy_uppertriangle_to_lowertriangle();
	_cached_results.set_cached_grad(elm::array_compare(FCurrent.ptr(), FCurrent.size()), g);
	_cached_results.set_cached_bhhh(elm::array_compare(FCurrent.ptr(), FCurrent.size()), *bhhh);
	
	return bhhh;
	
	
}

std::shared_ptr<etk::symmetric_matrix> elm::Model2::bhhh_cached() {

	std::shared_ptr<etk::symmetric_matrix> cached_bhhh;
	setUp();
	_parameter_update();
	if (FCurrent.size()<=0) {
		OOPS_CACHE("error in recovering cached value for bhhh at the current parameters (init fail)");
	}

	if (_cached_results.read_cached_bhhh(elm::array_compare(FCurrent.ptr(), FCurrent.size()), cached_bhhh) && cached_bhhh) {
		return cached_bhhh;
	} else {
		OOPS_CACHE("there is no cached value for bhhh at the current parameters");
	}
}

std::shared_ptr<etk::symmetric_matrix> elm::Model2::bhhh() {
	try {
		return bhhh_cached();
	} catch (LarchCacheError) {
		return bhhh_nocache();
	}
	
}

std::shared_ptr<etk::symmetric_matrix> elm::Model2::bhhh_nocache(const std::vector<double>& v) {
	setUp();
	_parameter_update();
	_parameter_push(v);
	
	std::shared_ptr<etk::ndarray> g = std::make_shared<etk::ndarray>(gradient(), false);

	bool z = true;
	for (size_t i=0; i!=g->size(); i++) {
		if ((*g)[i] != 0.0) {
			z = false;
			break;
		}
	}
	if (z) {
		auto fr = this->option.force_recalculate;
		this->option.force_recalculate = true;
		*g = this->gradient();
		this->option.force_recalculate = fr;
	}
	
	std::shared_ptr<etk::symmetric_matrix> bhhh = std::make_shared<etk::symmetric_matrix>(Bhhh, false);
	
	bhhh->copy_uppertriangle_to_lowertriangle();
	_cached_results.set_cached_grad(elm::array_compare(v), g);
	_cached_results.set_cached_bhhh(elm::array_compare(v), *bhhh);
	return bhhh;
	
	
}


std::shared_ptr<etk::symmetric_matrix> elm::Model2::bhhh_cached(const std::vector<double>& v) {
	std::shared_ptr<etk::symmetric_matrix> cached_bhhh;
	if (_cached_results.read_cached_bhhh(elm::array_compare(v), cached_bhhh) && cached_bhhh) {
		return cached_bhhh;
	} else {
		OOPS_CACHE("there is no cached value for bhhh at the given parameters");
	}
}


std::shared_ptr<etk::symmetric_matrix> elm::Model2::bhhh(const std::vector<double>& v) {
	try {
		return bhhh_cached(v);
	} catch (LarchCacheError) {
		return bhhh_nocache(v);
	}
}

std::shared_ptr< etk::ndarray > elm::Model2::bhhh_direction()
{
	return bhhh_direction( parameter_values() );
}

std::shared_ptr< etk::ndarray > elm::Model2::bhhh_direction(const std::vector<double>& v)
{
	
	std::shared_ptr<etk::symmetric_matrix> use_bhhh = bhhh(v);

	int status = _bhhh_update(&*use_bhhh);
	
	if (PyErr_Occurred()) {
		OOPS("error in finding bhhh_direction");
	}
	
	if (status<0) {
		OOPS("BHHH direction error");
	}

	std::shared_ptr< etk::ndarray > direction = std::make_shared<etk::ndarray>(dF());
	
	direction->initialize(0.0);
	
	#ifdef SYMMETRIC_PACKED
	cblas_dspmv(CblasRowMajor,CblasUpper,dF(), -1, 
				*invHessTemp, 
				*GCurrent,1, 
				0,**direction,1);
	#else
	cblas_dsymv(CblasRowMajor,CblasUpper,dF(), -1,
				*invHessTemp, invHessTemp.size1(),
				*GCurrent,1, 
				0,**direction,1);
	#endif
	
	double tolerance = -((*direction)*GCurrent);
	_cached_results.set_cached_bhhh_tol(v, tolerance);
	
	return direction;
}

double elm::Model2::bhhh_tolerance()
{
	return bhhh_tolerance( parameter_values() );
}


double elm::Model2::bhhh_tolerance(const std::vector<double>& v)
{
	double cached_bhhh_tol = NAN;
	if (_cached_results.read_cached_bhhh_tol(elm::array_compare(v), cached_bhhh_tol)) {
		return cached_bhhh_tol;
	} else {
		return bhhh_tolerance_nocache(v);
	}
}

double elm::Model2::bhhh_tolerance_nocache()
{
	return bhhh_tolerance_nocache( parameter_values() );
}

double elm::Model2::bhhh_tolerance_nocache(const std::vector<double>& v)
{
	double cached_bhhh_tol = NAN;
	try {
		bhhh_direction(v);
	} SPOO {
		if (PyErr_Occurred()) PyErr_Clear();
		return cached_bhhh_tol;
	}
	
	for (auto i=v.begin(); i!=v.end(); i++) {
		if (isNan(*i)) return cached_bhhh_tol;
	}
	
	if (_cached_results.read_cached_bhhh_tol(elm::array_compare(v), cached_bhhh_tol)) {
		return cached_bhhh_tol;
	} else {
		OOPS_CACHE("error in finding bhhh_tolerance at the given parameters");
	}
}


double elm::Model2::loglike_cached() {
	setUp();
	//if (!$self->_is_setUp) OOPS("Model is not setup, try calling setUp() first.");
	_parameter_update();

	double cached_ll = -INF;
	const double* FCurrent_ptr = FCurrent.ptr(0);
	size_t FCurrent_size = FCurrent.size();
	if (FCurrent_size && _cached_results.read_cached_loglike(elm::array_compare(FCurrent_ptr,FCurrent_size), cached_ll)) {
		return cached_ll;
	} else {
		OOPS_CACHE("there is no cached value for loglike at the current parameters");
	}
	
}


double elm::Model2::loglike_nocache() {
	
	double x (-INF);
	const double* FCurrent_ptr = FCurrent.ptr();
	size_t FCurrent_size = FCurrent.size();
	x = objective();
	if (isNan(x)) {
		x = -INF;
	}
	_cached_results.set_cached_loglike(elm::array_compare(FCurrent_ptr,FCurrent_size), x);
	
	return x;
	
}

double elm::Model2::loglike() {
	
	try {
		return loglike_cached();
	} catch (etk::LarchCacheError) {
		return loglike_nocache();
	}
	
}

double elm::Model2::loglike_cached(std::vector<double> v) {
	double cached_ll = -INF;
	if (_cached_results.read_cached_loglike(elm::array_compare(v), cached_ll)) {
		return cached_ll;
	} else {
		OOPS_CACHE("there is no cached value for loglike at the given parameters");
	}
}

double elm::Model2::loglike_nocache(std::vector<double> v) {
	setUp();
	//if (!$self->_is_setUp) OOPS("Model is not setup, try calling setUp() first.");
	_parameter_update();
	_parameter_push(v);
	double x = objective();
	if (isNan(x)) {
		x = -INF;
	}
	_cached_results.set_cached_loglike(elm::array_compare(v), x);
	return x;
}

double elm::Model2::loglike(std::vector<double> v) {
	
	double cached_ll = NAN;
	if (_cached_results.read_cached_loglike(elm::array_compare(v), cached_ll)) {
		return cached_ll;
	} else {

	}
	
	setUp();
	//if (!$self->_is_setUp) OOPS("Model is not setup, try calling setUp() first.");
	_parameter_update();
	_parameter_push(v);
	double x = objective();
	if (isNan(x)) {
		x = -INF;
	}
	_cached_results.set_cached_loglike(elm::array_compare(v), x);
	return x;
}

std::shared_ptr<etk::ndarray> elm::Model2::loglike_casewise()
{

	setUp();
	_parameter_update();
	objective();
	
	std::shared_ptr<ndarray> ll_casewise = make_shared<ndarray> (nCases);
	PrToAccum = (sampling_packet().relevant() ? &AdjProbability : &Probability);

	loglike_w w (&PrToAccum, Xylem.n_elemental(),
		Data_Choice, Data_Weight_active(), &accumulate_LogL, &*ll_casewise, option.mute_nan_warnings, &msg);

	
	boosted::mutex local_lock;
	w.work(0, nCases, &local_lock);
	
	return ll_casewise;
}

std::shared_ptr<etk::ndarray> elm::Model2::loglike_casewise(std::vector<double> v)
{

	loglike(v);
	
	std::shared_ptr<ndarray> ll_casewise = make_shared<ndarray> (nCases);
	PrToAccum = (sampling_packet().relevant() ? &AdjProbability : &Probability);

	loglike_w w (&PrToAccum, Xylem.n_elemental(),
		Data_Choice, Data_Weight_active(), &accumulate_LogL, &*ll_casewise, option.mute_nan_warnings, &msg);

	
	boosted::mutex local_lock;
	w.work(0, nCases, &local_lock);
	
	return ll_casewise;
}

void elm::Model2::clear_cache()
{
	_cached_results.clear();
	_FCurrent_latest_objective.initialize(NAN);
	
}



void elm::Model2::start_timing(const std::string& name)
{
	_latest_run.start_process(name);
}

void elm::Model2::finish_timing()
{
	_latest_run.end_process();
}

void elm::Model2::write_runstats_note(const std::string& comment)
{
	_latest_run.write(comment);
}

std::string elm::Model2::read_runstats_notes() const
{
	return _latest_run.notes();
}

