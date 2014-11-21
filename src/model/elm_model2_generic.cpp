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
#include <iomanip>


using namespace etk;
using namespace elm;
using namespace std;

elm::Model2::Model2()
: _Data (NULL)
, Data_UtilityCA  (nullptr)
, Data_UtilityCO  (nullptr)
, Data_SamplingCA (nullptr)
, Data_SamplingCO (nullptr)
, Data_QuantityCA (nullptr)
, Data_QuantLogSum(nullptr)
, Data_LogSum     (nullptr)
, Data_Choice     (nullptr)
, Data_Weight     (nullptr)
, Data_Avail      (nullptr)
, _LL_null (NAN)
, _LL_nil (NAN)
, _LL_constants (NAN)
, weight_scale_factor (1.0)
, nCases (0)
, nElementals (0)
, nNests (0)
, nNodes (0)
, nThreads (1)
, availability_ca_variable ("")
, features (0)
, option()
, _is_setUp(0)
, weight_autorescale (false)
, Input_Utility(COMPONENTLIST_TYPE_UTILITYCA,COMPONENTLIST_TYPE_UTILITYCO,"utility",this)
, Input_LogSum(COMPONENTLIST_TYPE_LOGSUM, this)
, Input_Edges(this)
, Input_Sampling(COMPONENTLIST_TYPE_UTILITYCA,COMPONENTLIST_TYPE_UTILITYCO,"samplingbias",this)
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
, Data_QuantityCA (nullptr)
, Data_QuantLogSum(nullptr)
, Data_LogSum     (nullptr)
, Data_Choice     (nullptr)
, Data_Weight     (nullptr)
, Data_Avail      (nullptr)
, _LL_null (NAN)
, _LL_nil (NAN)
, _LL_constants (NAN)
, weight_scale_factor (1.0)
, nCases (0)
, nElementals (0)
, nNests (0)
, nNodes (0)
, nThreads (1)
, availability_ca_variable ("")
, features (0)
, option()
, _is_setUp(0)
, weight_autorescale (false)
, Input_Utility(COMPONENTLIST_TYPE_UTILITYCA,COMPONENTLIST_TYPE_UTILITYCO,"utility",this)
, Input_LogSum(COMPONENTLIST_TYPE_LOGSUM, this)
, Input_Edges(this)
, Input_Sampling(COMPONENTLIST_TYPE_UTILITYCA,COMPONENTLIST_TYPE_UTILITYCO,"samplingbias",this)
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

elm::ca_co_packet elm::Model2::sampling_packet()
{
//	if (Data_SamplingCA->fully_loaded() && Data_SamplingCO->fully_loaded()) {
//		BUGGER(msg) << "spawning sampling packet fully loaded";
		return elm::ca_co_packet(&Params_SamplingCA	,
								 &Params_SamplingCO	,
								 &Coef_SamplingCA	,
								 &Coef_SamplingCO	,
								 Data_SamplingCA	,
								 Data_SamplingCO	,
								 &SamplingWeight	);
//	} else {
//		BUGGER(msg) << "Data_SamplingCA->fully_loaded = " << Data_SamplingCA->fully_loaded();
//		BUGGER(msg) << "Data_SamplingCO->fully_loaded = " << Data_SamplingCO->fully_loaded();
//		WARN(msg) << "spawning sampling packet when data is not fully loaded, things are gonna be slow...";
//		ScrapePtr d_ca = Data_SamplingCA->copy();
//		ScrapePtr d_co = Data_SamplingCO->copy();
//
//		return elm::ca_co_packet(&Params_SamplingCA	,
//								 &Params_SamplingCO	,
//								 &Coef_SamplingCA	,
//								 &Coef_SamplingCO	,
//								 Data_SamplingCA	,
//								 Data_SamplingCO	,
//								 Data_SamplingCA	,
//								 Data_SamplingCO	,
//								 &SamplingWeight	);
//	}
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
	Xylem.clear();
	Xylem.add_dna_sequence(_Data->alternatives_dna());
	Xylem.regrow( &Input_LogSum, &Input_Edges, _Data, &msg );
	
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


void elm::Model2::logger (const std::string& logname)
{
	if (logname=="") {
		msg.change_logger_name("");
	} else if (logname.substr(0,6)=="larch.") {
		msg.change_logger_name(logname);
	} else {
		msg.change_logger_name("larch."+logname);
	}
}

void elm::Model2::logger (bool z)
{
	if (z) {
		msg.change_logger_name("larch.Model");
	} else {
		msg.change_logger_name("");
	}
}

void elm::Model2::logger (int z)
{
	if (z>0) {
		msg.change_logger_name("larch.Model");
	} else {
		msg.change_logger_name("");
	}
}


PyObject* elm::Model2::_get_logger () const
{
	return py_one_item_list(PyString_FromString(msg.get_logger_name().c_str()));
}


elm::Model2::~Model2()
{ 
	tearDown();
	
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

runstats elm::Model2::estimate(std::vector<sherpa_pack>* opts)
{
	if (_string_sender_ptr) {
		ostringstream update;
		update << "Estimating the model now...";
		_string_sender_ptr->write(update.str());
	}


	BUGGER(msg) << "Estimating the model.";
	_latest_run.restart();

	try {
		setUp();	
		BUGGER(msg) << "Setup of model complete.";
		ZBest = -INF;
		
		flag_gradient_diagnostic = option.gradient_diagnostic; //Option_integer("gradient_diagnostic");
		flag_hessian_diagnostic = option.hessian_diagnostic; // Option_integer("hessian_diagnostic");
		BUGGER(msg) << "Diagnostic flags set.";
	
		if (flag_gradient_diagnostic) {
			WARN(msg) << "Gradient Diagnostic set to "<<flag_gradient_diagnostic;
		}
	
		if (option.calc_null_likelihood) {
			for (unsigned i=0; i<dF(); i++) {
				FCurrent[i] = FInfo[FNames[i]].null_value;
			}
			freshen();
			_LL_null = objective();			
			for (unsigned i=0; i<dF(); i++) {
				FCurrent[i] = FInfo[FNames[i]].value;
			}
			freshen();
		}
		
		if (option.weight_autorescale) {
			auto_rescale_weights();
			ostringstream oss;
			oss << "autoscaled weights " << weight_scale_factor << "\n";
			_latest_run._notes += oss.str();
		}
		
		_latest_run.results += maximize(_latest_run.iteration,opts);
		
		if (option.weight_autorescale) {
			restore_scale_weights();
			MONITOR(msg) << "recalculating log likelihood and gradients at normal weights";
			ZBest = ZCurrent = objective();
			gradient();
		}
		
		if (option.calc_std_errors) {
			//hessian();
			//MONITOR(msg) << "HESSIAN\n" << Hess.printSquare() ;
			//invHess = Hess;
			//invHess.inv(true);
			//MONITOR(msg) << "invHESSIAN\n" << invHess.printSquare() ;
			calculate_parameter_covariance();
		} else {
			Hess.initialize(NAN);
			invHess.initialize(NAN);
		}

	} SPOO {
		if (oops.code()==-8) {
			_update_freedom_info();
			_latest_run.write_result( "User Interrupt" );
			_latest_run.finish();
			PYTHON_INTERRUPT;
		}
		_latest_run.write_result( "error: ", oops.what() );
		MONITOR(msg) << "ERROR IN ESTIMATION: " << oops.what() ;
		throw;
	}
	
	
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
	
	
	if (option.teardown_after_estimate) tearDown();
	
	
	
	_latest_run.finish();
	if (_latest_run.results.empty()) _latest_run.results = "ignored";

	tm * timeinfo;
	time_t rawtime;
	char timebuff[256] = {0};
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	strftime(timebuff, 255, "%A, %B %d %Y, %I:%M:%S %p", timeinfo);
	_latest_run.timestamp = string(timebuff);
	
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


void elm::Model2::calculate_parameter_covariance()
{

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

		memarray_symmetric unpacked_bhhh;
		memarray_symmetric unpacked_invhess;
		MONITOR(msg) << "Bhhh\n" << Bhhh.printSquare() ;

		// unpack bhhh and inverse hessian for analysis
		unpacked_bhhh = Bhhh;
		unpacked_invhess = invHess;
		unsigned s = unpacked_bhhh.size1();
		unpacked_bhhh.copy_uppertriangle_to_lowertriangle();

		MONITOR(msg) << "unpacked_bhhh\n" << unpacked_bhhh.printall() ;
		MONITOR(msg) << "unpacked_invhess\n" << unpacked_invhess.printall() ;
		
		memarray_symmetric BtimeH (s,s);
		
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

		memarray_symmetric unpacked_bhhh;
		memarray_symmetric unpacked_invhess;
		MONITOR(msg) << "Bhhh\n" << Bhhh.printSquare() ;

		// unpack bhhh and inverse hessian for analysis
		unpacked_bhhh = Bhhh;
		unpacked_invhess = invHess;
		unsigned s = unpacked_bhhh.size1();
		unpacked_bhhh.copy_uppertriangle_to_lowertriangle();

		MONITOR(msg) << "unpacked_bhhh\n" << unpacked_bhhh.printall() ;
		MONITOR(msg) << "unpacked_invhess\n" << unpacked_invhess.printall() ;
		
		memarray_symmetric BtimeH (s,s);
		
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
}


std::string elm::Model2::weight(const std::string& varname, bool reweight) {
	
	weight_autorescale = reweight;
	
	ostringstream ret;
	if (varname.empty()) {
		weight_CO_variable.clear();
		weight_autorescale = false;
		ret << "success: weight is cleared";
	} else {
		weight_CO_variable = varname;
		ret << "success: weight_CO is set to " << weight_CO_variable;
		if (weight_autorescale) {
			ret << " (auto-reweighted)";
		}
	}
	return ret.str();
	
}

PyObject*	elm::Model2::_get_weight() const
{
	return etk::py_one_item_list(
		Py_BuildValue("(si)", weight_CO_variable.c_str(), int(weight_autorescale))
	);
}


void elm::Model2::_parameter_update()
{
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


std::vector< std::string > elm::Model2::parameter_names() const
{
	return FNames.strings();
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
	return _latest_run.dictionary();
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

void elm::Model2::_set_estimation_run_statistics
(	const long& startTimeSec
,	const long& startTimeUSec
,	const long& endTimeSec
,	const long& endTimeUSec
,	const unsigned& iteration
,	const std::string& results
,	const std::string& notes
)
{
#ifdef __APPLE__
	_latest_run.startTime.tv_sec = startTimeSec;
	_latest_run.startTime.tv_usec = startTimeUSec;
	_latest_run.endTime.tv_sec = endTimeSec;
	_latest_run.endTime.tv_usec = endTimeUSec;
#endif // __APPLE
	_latest_run.iteration = iteration;
	_latest_run.results = results;
	_latest_run._notes = notes;	
}

runstats& elm::Model2::RunStatistics()
{
	return _latest_run;
}


//#define AsPyFloat(x) "float.fromhex('" << std::hexfloat << (x) << std::defaultfloat << "')"

std::string AsPyFloat(const double& x)
{
	ostringstream s;

	double integer_part;
	if (modf(x, &integer_part)==0) {
		s << x;
		return s.str();
	}
	
	s << std::scientific << std::setprecision(17) << x;
	
	auto f = s.str().find("0000000");
	if (f!=std::string::npos) {
		std::string z = s.str();
		return z.substr(0,f) + z.substr(z.find("e"));
	}
	
	ostringstream h;
	h << "float.fromhex('" << std::hexfloat << (x) << std::defaultfloat << "')";
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


std::string elm::Model2::save_buffer() const
{
	ostringstream sv;
	
	//sv << "import numpy\n" << "inf=numpy.inf\n" << "nan=numpy.nan\n\n";
	
	sv << "self.title = '''"<<title<<"'''\n\n";
	
	// save parameter
	for (auto p=FNames.strings().begin(); p!=FNames.strings().end(); p++) {
		auto i = FInfo.find(*p);
		auto j = i->second;
		
		sv << "self.parameter('"<<*p<<"'";
		sv << ","<<AsPyFloat(j.value);
		sv << ","<<AsPyFloat(j.null_value);
		sv << ","<<AsPyFloat(j.initial_value);
		sv << ","<<AsPyFloat(j.max_value);
		sv << ","<<AsPyFloat(j.min_value);
		sv << ","<<AsPyFloat(j.std_err);
		sv << ","<<AsPyFloat(j.robust_std_err);
		sv << ","<<j.holdfast;
		
		PyObject* c = PyObject_Str(j._covar);
		if (c) {
			sv << ",covariance="<<PyString_ExtractCppString(c);
		}
		Py_CLEAR(c);

		c = PyObject_Str(j._robust_covar);
		if (c) {
			sv << ",robust_covariance="<<PyString_ExtractCppString(c);
		}
		Py_CLEAR(c);

		sv << ")\n";
	}
	sv << "\n";
	
	// save utility
	for (auto u=Input_Utility.ca.begin(); u!=Input_Utility.ca.end(); u++) {
		sv << "self.utility.ca('"<<u->apply_name<<"','"<<u->param_name<<"',"<<AsPyFloat(u->multiplier)<<")\n";
	}
	for (auto u=Input_Utility.co.begin(); u!=Input_Utility.co.end(); u++) {
		if (u->altcode) {
			sv << "self.utility.co('"<<u->apply_name<<"',"<<u->altcode<<",'"<<u->param_name<<"',"<<AsPyFloat(u->multiplier)<<")\n";
		} else {
			sv << "self.utility.co('"<<u->apply_name<<"','"<<u->altname<<"','"<<u->param_name<<"',"<<AsPyFloat(u->multiplier)<<")\n";
		}
	}
	sv << "\n";

	// save nest
	for (auto n=Input_LogSum.begin(); n!=Input_LogSum.end(); n++) {
		sv << "self.nest('"<<n->second.altname<<"',"<<n->second.altcode<<",'"<<n->second.param_name<<"'";
		if (n->second.multiplier!=1.0) {
			sv << ","<<AsPyFloat(n->second.multiplier);
		}
		sv << ")\n";
	}
	
	// save link
	for (auto n=Input_Edges.begin(); n!=Input_Edges.end(); n++) {
		if (n->second.size()) {
			// This edge has a component list
			OOPS("not yet implemented for edges with components");
		} else {
			// This edge has no component list
			sv << "self.link("<<n->first.up<<","<<n->first.dn<<")\n";
		}
	}
	
	// save samplingbias
	for (auto u=Input_Sampling.ca.begin(); u!=Input_Sampling.ca.end(); u++) {
		sv << "self.samplingbias.ca('"<<u->apply_name<<"','"<<u->param_name<<"',"<<AsPyFloat(u->multiplier)<<")\n";
	}
	for (auto u=Input_Sampling.co.begin(); u!=Input_Sampling.co.end(); u++) {
		if (u->altcode) {
			sv << "self.samplingbias.co('"<<u->apply_name<<"',"<<u->altcode<<",'"<<u->param_name<<"',"<<AsPyFloat(u->multiplier)<<")\n";
		} else {
			sv << "self.samplingbias.co('"<<u->apply_name<<"','"<<u->altname<<"','"<<u->param_name<<"',"<<AsPyFloat(u->multiplier)<<")\n";
		}
	}
	sv << "\n";
	
	sv << "self._set_estimation_statistics("<<
		AsPyFloat(ZCurrent)<<","<<
		AsPyFloat(_LL_null) <<","<<
		AsPyFloat(_LL_nil) <<","<<
		AsPyFloat(_LL_constants) <<","<<
		AsPyFloat(ZBest) <<
		")\n";
	sv << "self._set_estimation_run_statistics("<<
	_latest_run.startTime.tv_sec  <<","<<
	_latest_run.startTime.tv_usec <<","<<
	_latest_run.endTime.tv_sec    <<","<<
	_latest_run.endTime.tv_usec   <<","<<
	_latest_run.iteration   <<",'''"<<
	_latest_run.results   <<"''','''"<<
	_latest_run._notes   <<
		"''')\n";



	sv << "\n";

	sv << option._save_buffer() << "\n";

	
	
	
	return sv.str();
}




PyObject* elm::Model2::d_loglike(std::vector<double> v) {
		
	this->setUp();
	//if (!this->_is_setUp) OOPS("Model is not setup, try calling setUp() first.");
	//this->_parameter_update();
	if (v.size() != this->dF()) {
		OOPS("You must specify values for exactly the correct number of degrees of freedom (",this->dF(),"), you gave ",v.size(),".");
	}
	for (unsigned z=0; z<v.size(); z++) {
		this->FCurrent[z] = v[z];
	}
	this->freshen();
	this->setUp();
	//if (!this->_is_setUp) OOPS("Model is not setup, try calling setUp() first.");
	this->_parameter_update();
	etk::ndarray g = this->gradient();
	bool z = true;
	for (size_t i=0; i!=g.size(); i++) {
		if (g[i] != 0.0) {
			z = false;
			break;
		}
	}
	if (z) {
		auto fr = this->option.force_recalculate;
		this->option.force_recalculate = true;
		g = this->gradient();
		this->option.force_recalculate = fr;
	}
	
	Py_INCREF(g.get_object());
	return g.get_object();
	
	
}





PyObject* elm::Model2::negative_d_loglike(std::vector<double> v) {
	
	this->setUp();
	//if (!this->_is_setUp) OOPS("Model is not setup, try calling setUp() first.");
	//this->_parameter_update();
	if (v.size() != this->dF()) {
		OOPS("You must specify values for exactly the correct number of degrees of freedom (",this->dF(),"), you gave ",v.size(),".");
	}
	for (unsigned z=0; z<v.size(); z++) {
		this->FCurrent[z] = v[z];
	}
	this->freshen();
	this->setUp();
	//if (!this->_is_setUp) OOPS("Model is not setup, try calling setUp() first.");
	this->_parameter_update();
	etk::ndarray g = this->gradient();
	bool z = true;
	for (size_t i=0; i!=g.size(); i++) {
		if (g[i] != 0.0) {
			z = false;
			break;
		}
	}
	if (z) {
		auto fr = this->option.force_recalculate;
		this->option.force_recalculate = true;
		g = this->gradient();
		this->option.force_recalculate = fr;
	}
	
	g.neg();
	Py_INCREF(g.get_object());
	return g.get_object();
	
	
}

