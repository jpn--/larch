//
//  elm_parameterlist.cpp
//  Yggdrasil
//
//  Created by Jeffrey Newman on 12/21/12.
//
//

#include "elm_parameterlist.h"
#include "elm_parameter2.h"


elm::ParameterList::ParameterList()
: FNames ()
, FInfo ()
{
	
}
elm::ParameterList::ParameterList(const elm::ParameterList& dupe)
: FNames (dupe.FNames)
, FInfo (dupe.FInfo)
{
	
}



elm::parametexr elm::ParameterList::_generate_parameter(const std::string& freedom_name,
										   const double& freedom_multiplier)
{
	std::string fn = freedom_name;
	etk::uppercase(fn);
	if (fn == "CONSTANT") {
		return boosted::make_shared<parametex_constant> (freedom_multiplier);
	} else if (freedom_multiplier==0) {
		OOPS("multiplier cannot be zero");
	}
	if (!FNames.has_key(freedom_name) ) {
		throw(etk::ParameterNameError(etk::cat("parameter name '",freedom_name,"' not found")));
	}
	if (FInfo[freedom_name].holdfast) {
		return boosted::make_shared<parametex_constant>(freedom_multiplier*FInfo[freedom_name].value);
	}
	if (freedom_multiplier==1.0) {
		return boosted::make_shared<elm::parametex_equal>(freedom_name,this);
	} 
	return boosted::make_shared<elm::parametex_scale>(freedom_name,this,freedom_multiplier);
}



freedom_info& elm::ParameterList::parameter(const std::string& param_name,
								   const double& value,
								   const double& null_value,
								   const double& initial_value,
								   const double& max,
								   const double& min,
								   const double& std_err,
								   const double& robust_std_err,
								   const int& holdfast,
								   PyObject* covariance,
								   PyObject* robust_covariance)
{
	if (param_name=="") {
		throw(etk::ParameterNameError("Cannot name a parameter with an empty string."));
	}
	FNames[param_name];
	FInfo[param_name].name = param_name;
	if (!isNan(value)) {
		FInfo[param_name].initial_value = value;
		FInfo[param_name].value = value;
	}
	if (!isNan(null_value)) {
		FInfo[param_name].null_value = null_value;
	}
	if (!isNan(initial_value)) {
		FInfo[param_name].initial_value = initial_value;
	}
	if (!isNan(std_err)) {
		FInfo[param_name].std_err = std_err;
	}
	if (!isNan(robust_std_err)) {
		FInfo[param_name].robust_std_err = robust_std_err;
	}
	if (!isNan(min)) {
		FInfo[param_name].min_value = min;
	}
	if (!isNan(max)) {
		FInfo[param_name].max_value = max;
	}
	if (holdfast>=0) {
		FInfo[param_name].holdfast = holdfast;
	}
	if (covariance) {
		FInfo[param_name].setCovariance(covariance);
	}
	if (robust_covariance) {
		FInfo[param_name].setRobustCovariance(robust_covariance);
	}
	return FInfo[param_name];
}

freedom_info& elm::ParameterList::__getitem__(const std::string& param_name)
{
	return parameter(param_name);
}

freedom_info& elm::ParameterList::__getitem__(const int& param_num)
{
	if (param_num > FNames.size()-1) OOPS("Parameter number ",param_num," out of range (there are only ",FNames.size()," parameters)");
	if (param_num < 0 && param_num >= -int(FNames.size())) {
		return parameter(FNames[FNames.size()+param_num]);
	}
	return parameter(FNames[param_num]);
}

void elm::ParameterList::__setitem__(const std::string& param_name, freedom_info& value)
{
	parameter(param_name) = value;
}

void elm::ParameterList::__delitem__(const std::string& param_name)
{
	bool s = FNames.drop(param_name);
	FInfo.erase(param_name);
//	if (!s) OOPS("No parameter named ",param_name," exists.");
}

bool elm::ParameterList::__contains__(const std::string& param_name) const
{
	return FNames.has_key(param_name);
}

PyObject* elm::ParameterList::values() const
{
	size_t s = FNames.size();
	etk::ndarray ret (s);
	for (unsigned i=0; i<s; i++) {
		ret[i] = FInfo.find(FNames[i])->second.value;
	}
	return ret.get_object();

}

std::string elm::ParameterList::values_string()
{
	std::ostringstream ret;
	size_t s = FNames.size();
	for (unsigned i=0; i<s; i++) {
		ret << "," << FInfo.find(FNames[i])->second.value;
	}
	return ret.str().substr(1);
}


void elm::ParameterList::values(PyObject* obj)
{
	Py_XINCREF(obj);
	if (!PySequence_Check(obj)) OOPS("Setting values requires a sequence");
	if (PySequence_Length(obj)<FNames.size()) OOPS("Sequence too short for setting values");
	
	for (unsigned i=0; i<FNames.size(); i++) {
		PyObject* item = PySequence_GetItem(obj, i);
		FInfo.find(FNames[i])->second.value = PyFloat_AsDouble(item);
		Py_CLEAR(item);
	}
	Py_XDECREF(obj);
}


PyObject* elm::ParameterList::zeros() const
{
	size_t s = FNames.size();
	etk::ndarray ret (s);
	return ret.get_object();

}

size_t elm::ParameterList::_len() const
{
	return FNames.size();
}

PyObject* elm::ParameterList::constraints() const
{
	Py_RETURN_NONE;
}

//void elm::ParameterList::covariance(etk::symmetric_matrix* obj)
//{
//	if (obj->size1()!=FNames.size()) OOPS("Input covariance must be a square with side equal to number of parameters");
//	for (unsigned i=0; i<FNames.size(); i++) {
//		FInfo.find(FNames[i])->second.std_err = sqrt((*obj)(i,i));
//	}
//}
//void elm::ParameterList::robustcovariance(etk::symmetric_matrix* obj)
//{
//	if (obj->size1()!=FNames.size()) OOPS("Input covariance must be a square with side equal to number of parameters");
//	for (unsigned i=0; i<FNames.size(); i++) {
//		FInfo.find(FNames[i])->second.robust_std_err = sqrt((*obj)(i,i));
//	}
//}
//


void elm::ParameterList::freshen()
{
	
}



