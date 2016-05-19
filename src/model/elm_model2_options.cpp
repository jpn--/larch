/*
 *  elm_model_options.cpp
 *
 *  Copyright 2007-2016 Jeffrey Newman
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


#include "elm_model2.h"
#include <set>
#include "etk_python.h"

#include "elm_model2_options.h"

#include <iostream>
using namespace std;
using namespace etk;


elm::model_options_t::model_options_t(
			int threads,
			bool calc_null_likelihood,
			bool null_disregards_holdfast,
			bool calc_std_errors,
			int gradient_diagnostic,
			int hessian_diagnostic,
			bool mute_nan_warnings,
			bool force_finite_diff_grad,
			bool save_db_hash,
			bool force_recalculate,
			std::string author,
			bool teardown_after_estimate,
			bool weight_autorescale,
			bool weight_choice_rebalance,
			bool suspend_xylem_rebuild,
			bool log_turns,
			bool enforce_bounds,
			bool enforce_constraints,
			double idca_avail_ratio_floor,
			bool autocreate_parameters
		)
: gradient_diagnostic   (gradient_diagnostic)
, hessian_diagnostic    (hessian_diagnostic)
, threads               (threads)
, calc_null_likelihood  (calc_null_likelihood)
, null_disregards_holdfast(null_disregards_holdfast)
, calc_std_errors       (calc_std_errors)
, mute_nan_warnings     (mute_nan_warnings)
, force_finite_diff_grad(force_finite_diff_grad)
, save_db_hash          (save_db_hash)
, force_recalculate     (force_recalculate)
, author                (author)
, teardown_after_estimate(teardown_after_estimate)
, weight_autorescale    (weight_autorescale)
, weight_choice_rebalance(weight_choice_rebalance)
, suspend_xylem_rebuild (suspend_xylem_rebuild)
, log_turns             (log_turns)
, enforce_bounds        (enforce_bounds)
, enforce_constraints   (enforce_constraints)
, idca_avail_ratio_floor(idca_avail_ratio_floor)
, autocreate_parameters (autocreate_parameters)
{
	boosted::lock_guard<boosted::mutex> LOCK(etk::python_global_mutex);
//#ifdef __APPLE__
	if (this->threads<=0) {
//		PyObject* multiprocessing_module = PyImport_ImportModule("multiprocessing");
//		PyObject* cpu_count = PyObject_GetAttrString(multiprocessing_module, "cpu_count");
//		PyObject* n_cpu = PyObject_CallFunction(cpu_count, "()");
//		int t = PyInt_AsLong(n_cpu);
//		if (!PyErr_Occurred()) {
//			this->threads = t;
//		} else {
//			PyErr_Clear();
//		}
//		Py_CLEAR(n_cpu);
//		Py_CLEAR(cpu_count);
//		Py_CLEAR(multiprocessing_module);
		if ((number_of_cpu>0)&&(number_of_cpu<100000)) {
			this->threads = number_of_cpu;
		} else {
			this->threads = 1;
		}
	}
//#else
//	this->threads = 1;
//#endif
}


void elm::model_options_t::__call__(
			int threads,
			int calc_null_likelihood,
			int null_disregards_holdfast,
			int calc_std_errors,
			int gradient_diagnostic,
			int hessian_diagnostic,
			int mute_nan_warnings,
			int force_finite_diff_grad,
			int save_db_hash,
			int force_recalculate,
			std::string author,
			int teardown_after_estimate,
			int weight_autorescale,
			int weight_choice_rebalance,
			int suspend_xylem_rebuild,
			int log_turns,
			int enforce_bounds,
			int enforce_constraints,
			double idca_avail_ratio_floor,
			int autocreate_parameters
		)
{
	if (gradient_diagnostic     != -9 ) (this->gradient_diagnostic     = gradient_diagnostic     );
	if (hessian_diagnostic      != -9 ) (this->hessian_diagnostic      = hessian_diagnostic      );
	if (threads                 != -9 ) (this->threads                 = threads                 );
	if (calc_null_likelihood    != -9 ) (this->calc_null_likelihood    = calc_null_likelihood    );
	if (null_disregards_holdfast!= -9 ) (this->null_disregards_holdfast= null_disregards_holdfast);
	if (calc_std_errors         != -9 ) (this->calc_std_errors         = calc_std_errors         );
	if (mute_nan_warnings       != -9 ) (this->mute_nan_warnings       = mute_nan_warnings       );
	if (force_finite_diff_grad  != -9 ) (this->force_finite_diff_grad  = force_finite_diff_grad  );
	if (save_db_hash            != -9 ) (this->save_db_hash            = save_db_hash            );
	if (force_recalculate       != -9 ) (this->force_recalculate       = force_recalculate       );
	if (author                  !="-9") (this->author                  = author                  );
	if (teardown_after_estimate != -9 ) (this->teardown_after_estimate = teardown_after_estimate );
	if (weight_autorescale      != -9 ) (this->weight_autorescale      = weight_autorescale      );
	if (weight_choice_rebalance != -9 ) (this->weight_choice_rebalance = weight_choice_rebalance );
	if (suspend_xylem_rebuild   != -9 ) (this->suspend_xylem_rebuild   = suspend_xylem_rebuild   );
	if (log_turns               != -9 ) (this->log_turns               = log_turns               );
	if (enforce_bounds          != -9 ) (this->enforce_bounds          = enforce_bounds          );
	if (enforce_constraints     != -9 ) (this->enforce_constraints     = enforce_constraints     );
	if (idca_avail_ratio_floor  != -9 ) (this->idca_avail_ratio_floor  = idca_avail_ratio_floor  );
	if (autocreate_parameters   != -9 ) (this->autocreate_parameters   = autocreate_parameters   );
	
}

void elm::model_options_t::copy(const model_options_t& other)
{
	this->gradient_diagnostic     = other.gradient_diagnostic     ;
	this->hessian_diagnostic      = other.hessian_diagnostic      ;
	this->threads                 = other.threads                 ;
	this->calc_null_likelihood    = other.calc_null_likelihood    ;
	this->null_disregards_holdfast= other.null_disregards_holdfast;
	this->calc_std_errors         = other.calc_std_errors         ;
	this->mute_nan_warnings       = other.mute_nan_warnings       ;
	this->force_finite_diff_grad  = other.force_finite_diff_grad  ;
	this->save_db_hash            = other.save_db_hash            ;
	this->force_recalculate       = other.force_recalculate       ;
	this->teardown_after_estimate = other.teardown_after_estimate ;
	this->weight_autorescale      = other.weight_autorescale      ;
	this->weight_choice_rebalance = other.weight_choice_rebalance ;
	this->suspend_xylem_rebuild   = other.suspend_xylem_rebuild   ;
	this->log_turns               = other.log_turns               ;
	this->enforce_bounds          = other.enforce_bounds          ;
	this->enforce_constraints     = other.enforce_constraints     ;
	this->idca_avail_ratio_floor  = other.idca_avail_ratio_floor  ;
	this->autocreate_parameters   = other.autocreate_parameters   ;
}


std::string elm::model_options_t::__repr__() const
{
	std::ostringstream x;
	x << "larch.core.model_options_t(\n";
	x << "                 threads= "<<threads                 <<",\n";
	x << "    calc_null_likelihood= "<<calc_null_likelihood    <<",\n";
	x << "null_disregards_holdfast= "<<null_disregards_holdfast<<",\n";
	x << "         calc_std_errors= "<<calc_std_errors         <<",\n";
	x << "     gradient_diagnostic= "<<gradient_diagnostic     <<",\n";
	x << "      hessian_diagnostic= "<<hessian_diagnostic      <<",\n";
	x << "       mute_nan_warnings= "<<mute_nan_warnings       <<",\n";
	x << "  force_finite_diff_grad= "<<force_finite_diff_grad  <<",\n";
	x << "       force_recalculate= "<<force_recalculate       <<",\n";
	x << "            save_db_hash= "<<save_db_hash            <<",\n";
	x << "                  author= "<<author                  <<",\n";
	x << " teardown_after_estimate= "<<teardown_after_estimate <<",\n";
	x << "      weight_autorescale= "<<weight_autorescale      <<",\n";
	x << " weight_choice_rebalance= "<<weight_choice_rebalance <<",\n";
	x << "   suspend_xylem_rebuild= "<<suspend_xylem_rebuild   <<",\n";
	x << "               log_turns= "<<log_turns               <<",\n";
	x << "          enforce_bounds= "<<enforce_bounds          <<",\n";
	x << "     enforce_constraints= "<<enforce_constraints     <<",\n";
	x << "  idca_avail_ratio_floor= "<<idca_avail_ratio_floor  <<",\n";
	x << "   autocreate_parameters= "<<autocreate_parameters   <<",\n";
	x << ")";
	return x.str();
}

std::string elm::model_options_t::_save_buffer() const
{
	std::ostringstream x;
	x << "self.option.threads= "                << threads                                 <<"\n";
	x << "self.option.calc_null_likelihood= "   <<(calc_null_likelihood    ?"True":"False")<<"\n";
	x << "self.option.null_disregards_holdfast="<<(null_disregards_holdfast?"True":"False")<<"\n";
	x << "self.option.calc_std_errors= "        <<(calc_std_errors         ?"True":"False")<<"\n";
	x << "self.option.gradient_diagnostic= "    << gradient_diagnostic                     <<"\n";
	x << "self.option.hessian_diagnostic= "     << hessian_diagnostic                      <<"\n";
	x << "self.option.mute_nan_warnings= "      <<(mute_nan_warnings       ?"True":"False")<<"\n";
	x << "self.option.force_finite_diff_grad= " <<(force_finite_diff_grad  ?"True":"False")<<"\n";
	x << "self.option.force_recalculate= "      <<(force_recalculate       ?"True":"False")<<"\n";
	x << "self.option.save_db_hash= "           <<(save_db_hash            ?"True":"False")<<"\n";
	x << "self.option.author= '"                << author                                  <<"'\n";
	x << "self.option.teardown_after_estimate= "<<(teardown_after_estimate ?"True":"False")<<"\n";
	x << "self.option.weight_autorescale= "     <<(weight_autorescale      ?"True":"False")<<"\n";
	x << "self.option.weight_choice_rebalance= "<<(weight_choice_rebalance ?"True":"False")<<"\n";
	x << "self.option.suspend_xylem_rebuild= "  <<(suspend_xylem_rebuild   ?"True":"False")<<"\n";
	x << "self.option.log_turns= "              <<(log_turns               ?"True":"False")<<"\n";
	x << "self.option.enforce_bounds= "         <<(enforce_bounds          ?"True":"False")<<"\n";
	x << "self.option.enforce_constraints= "    <<(enforce_constraints     ?"True":"False")<<"\n";
	x << "self.option.idca_avail_ratio_floor= " << idca_avail_ratio_floor                  <<"\n";
	x << "self.option.autocreate_parameters= "  <<(autocreate_parameters   ?"True":"False")<<"\n";
	return x.str();
}

std::string elm::model_options_t::__str__() const
{
	std::ostringstream x;
	x << "                 threads: "<< threads               <<"\n";
	x << "    calc_null_likelihood: "<<(calc_null_likelihood  ?"True":"False")<<"\n";
	x << "null_disregards_holdfast: "<<(null_disregards_holdfast  ?"True":"False")<<"\n";
	x << "         calc_std_errors: "<<(calc_std_errors       ?"True":"False")<<"\n";
	x << "     gradient_diagnostic: "<< gradient_diagnostic   <<"\n";
	x << "      hessian_diagnostic: "<< hessian_diagnostic    <<"\n";
	x << "       mute_nan_warnings: "<<(mute_nan_warnings     ?"True":"False")<<"\n";
	x << "  force_finite_diff_grad: "<<(force_finite_diff_grad?"True":"False")<<"\n";
	x << "       force_recalculate: "<<(force_recalculate     ?"True":"False")<<"\n";
	x << "            save_db_hash: "<<(save_db_hash          ?"True":"False")<<"\n";
	x << "                  author: "<< author                <<"\n";
	x << " teardown_after_estimate: "<< teardown_after_estimate<<"\n";
	x << "      weight_autorescale: "<<(weight_autorescale    ?"True":"False")<<"\n";
	x << " weight_choice_rebalance: "<<(weight_choice_rebalance?"True":"False")<<"\n";
	x << "   suspend_xylem_rebuild: "<<(suspend_xylem_rebuild ?"True":"False")<<"\n";
	x << "               log_turns: "<<(log_turns             ?"True":"False")<<"\n";
	x << "          enforce_bounds: "<<(enforce_bounds        ?"True":"False")<<"\n";
	x << "     enforce_constraints: "<<(enforce_constraints   ?"True":"False")<<"\n";
	x << "  idca_avail_ratio_floor: "<<idca_avail_ratio_floor<<"\n";
	x << "   autocreate_parameters: "<<(autocreate_parameters ?"True":"False")<<"\n";
	return x.str();
}



std::set<std::string> elm::Model2::valid_options()
{
	std::set<std::string> valid_options_init;
	valid_options_init.insert("gradient diagnostic");
	valid_options_init.insert("hessian diagnostic");
	valid_options_init.insert("log");
	valid_options_init.insert("finite_diff");  // BOOL: Skip the analytic gradient, use finte differences only
	valid_options_init.insert("skip_ll_null");  // BOOL: Skip calculateing the LL at Null values
	valid_options_init.insert("calculate std err"); // BOOL: calculate std errors after maximization
	valid_options_init.insert("allavail"); // BOOL: Force calculations to assume that all alternatives are available for every decision maker
	valid_options_init.insert("autoscale choices");
	valid_options_init.insert("multichoice");
	valid_options_init.insert("mute_nan_warnings");
	valid_options_init.insert("teardown_after_estimate");
	valid_options_init.insert("weight_autorescale");
	valid_options_init.insert("weight_choice_rebalance");
	valid_options_init.insert("suspend_xylem_rebuild");
	valid_options_init.insert("log_turns");
	valid_options_init.insert("enforce_bounds");
	valid_options_init.insert("enforce_constraints");
	valid_options_init.insert("idca_avail_ratio_floor");
	valid_options_init.insert("autocreate_parameters");
	return valid_options_init;
}

bool elm::Model2::is_valid_option(const string& optname)
{
	static set<std::string> valid_options_list = valid_options();
	set<std::string>::const_iterator i = valid_options_list.find(optname);
	if (i == valid_options_list.end()) return false;
	return true;
}
/*
bool elm::Model2::is_option(const string& optname) const
{
	ss_map::const_iterator i = _options.find(optname);
	if (i == _options.end()) return false;
	return true;
}

bool elm::Model2::bool_option(const string& optname) const
{
	ss_map::const_iterator i = _options.find(optname);
	if (i == _options.end()) return false;
	if (i->second == "no") return false;
	if (i->second == "0") return false;
	if (i->second == "false") return false;
	if (i->second == "off") return false;
	if (i->second == "f") return false;
	if (i->second == "n") return false;
	return true;
}

unsigned elm::Model2::unsigned_option(const string& optname) const
{
	ss_map::const_iterator i = _options.find(optname);
	if (i == _options.end()) return 0;
	char** a;
	return strtoul( i->second.c_str(), a, 10 );
}

int elm::Model2::int_option(const string& optname) const
{
	ss_map::const_iterator i = _options.find(optname);
	if (i == _options.end()) return 0;
	return atoi( i->second.c_str() );
}


double elm::Model2::double_option(const string& optname) const
{
	ss_map::const_iterator i = _options.find(optname);
	if (i == _options.end()) return 0;
	char** a;
	return strtod( i->second.c_str(),a );
}

string elm::Model2::string_option(const string& optname) const
{
	ss_map::const_iterator i = _options.find(optname);
	if (i == _options.end()) return "";
	return ( i->second );
}

void elm::Model2::setOption(const string& optname, const string& optvalue)
{
	_options[optname] = optvalue;
	process_options();
}

string elm::Model2::getOption(const string& optname) const
{
	return string_option(optname);
}
*/
void elm::Model2::process_options()
{
	flag_gradient_diagnostic = option.gradient_diagnostic;
	flag_hessian_diagnostic = option.hessian_diagnostic;
	flag_log_turns = option.log_turns;
}
/*
etk::strvec valid_options;
valid_options.push_back("log");
valid_options.push_back("gradient diagnostic");
valid_options.push_back("hessian diagnostic");
valid_options.push_back("skip_std_err");

*/

/*
void elm::Model2::_Option(PyObject* x) 
{
	if (Option)	Py_CLEAR(Option);
	Option = x; 
	Py_INCREF(Option);
}

*/




/*
double elm::Model2::Option_double(const string& optname) const
{
	double x (0.0);
	int interrupt ( PyErr_CheckSignals() );
	bool error_in_options (0);
	PyObject* ptype (NULL);
	PyObject* pvalue (NULL); 
	PyObject* ptraceback (NULL);
	PyErr_Fetch(&ptype, &pvalue, &ptraceback);
	try {
		char* opt = const_cast<char*>(optname.c_str()) ;
		PyObject* obj (PyMapping_GetItemString(Option , opt)); 
		if (obj) {
			x = PyFloat_AsDouble(obj);
			Py_CLEAR(obj);
			if (x==-1.0) {
				if (PyErr_Occurred()) x=0;
			}
		} else {
			x=0;
		}
	} catch (...) {
		error_in_options = true;
	}
	PyErr_Restore( ptype,  pvalue,  ptraceback);
	if (interrupt) PyErr_SetInterrupt();
	return x;
}

long elm::Model2::Option_integer(const string& optname) const
{
	long x (0);
	int interrupt ( PyErr_CheckSignals() );
	bool error_in_options (0);
	PyObject* ptype (NULL);
	PyObject* pvalue (NULL); 
	PyObject* ptraceback (NULL);
	PyErr_Fetch(&ptype, &pvalue, &ptraceback);
	try {
		char* opt = const_cast<char*>(optname.c_str()) ;
		PyObject* obj (PyMapping_GetItemString(Option , opt)); 
		if (obj) {
			x = PyInt_AsLong(obj);
			Py_CLEAR(obj);
			if (x==-1) {
				if (PyErr_Occurred()) x= 0;
			}
		} else {
			x= 0;
		}
	} catch (...) {
		error_in_options = true;
	}
	PyErr_Restore( ptype,  pvalue,  ptraceback);
	if (interrupt) PyErr_SetInterrupt();
	return x;
}

bool elm::Model2::Option_bool(const string& optname) const
{
	bool ret (false);
	int interrupt ( PyErr_CheckSignals() );
	bool error_in_options (0);
	PyObject* ptype (NULL);
	PyObject* pvalue (NULL); 
	PyObject* ptraceback (NULL);
	PyErr_Fetch(&ptype, &pvalue, &ptraceback);
	try {
		char* opt = const_cast<char*>(optname.c_str()) ;
		PyObject* obj (PyMapping_GetItemString(Option , opt)); 
		if (obj) {
			ret = PyObject_IsTrue(obj);
			Py_CLEAR(obj);
		} else {
			ret=false;
		}
	} catch (...) {
		error_in_options = true;
	}
	PyErr_Restore( ptype,  pvalue,  ptraceback);
	if (interrupt) PyErr_SetInterrupt();
	return ret;
}
*/

