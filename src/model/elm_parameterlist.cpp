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
														double freedom_multiplier)
{
	std::string fn = freedom_name;
	etk::uppercase(fn);
	if (fn == "CONSTANT") {
		return boosted::make_shared<parametex_constant> (freedom_multiplier);
	} else if (freedom_multiplier==0) {
		OOPS("multiplier cannot be zero");
	}
	
	auto iter = AliasInfo.find(freedom_name);
	if (iter!=AliasInfo.end()) {
		freedom_alias* x = &iter->second;
		fn = x->refers_to;
		freedom_multiplier *= x->multiplier;
		if (!FNames.has_key(fn) &&  AliasInfo.find(fn)==AliasInfo.end()) {
			throw(etk::ParameterNameError(etk::cat("parameter name '",fn,"' referred to by alias '",freedom_name,"' not found")));
		}
		return _generate_parameter(fn,freedom_multiplier);
	} else {
		fn = freedom_name;
	}
	
	if (!FNames.has_key(fn) ) {
		throw(etk::ParameterNameError(etk::cat("parameter name '",fn,"' not found")));
	}
	if (FInfo[fn].holdfast) {
		return boosted::make_shared<parametex_constant>(freedom_multiplier*FInfo[fn].value);
	}
	if (freedom_multiplier==1.0) {
		return boosted::make_shared<elm::parametex_equal>(fn,this);
	} 
	return boosted::make_shared<elm::parametex_scale>(fn,this,freedom_multiplier);
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

freedom_alias& elm::ParameterList::alias(const std::string& alias_name, const std::string& refers_to, const double& multiplier, const bool& force)
{
	if (alias_name=="") {
		throw(etk::ParameterNameError("Cannot name an alias with an empty string."));
	}
	if (refers_to=="") {
		throw(etk::ParameterNameError("Cannot refer to a parameter with an empty string."));
	}
	
	if (!force) {
		auto iter = FInfo.find(refers_to);
		if ((iter == FInfo.end()) && (AliasInfo.find(refers_to)==AliasInfo.end())) {
			throw(etk::ParameterNameError(etk::cat("Cannot refer to parameter '",refers_to,"' that has not been previously defined.")));
		}

		if (alias_name==refers_to) {
			throw(etk::ParameterNameError(etk::cat("Cannot create an alias '",refers_to,"' that refers to a parameter with the same name.")));
		}
	}

	if (AliasInfo.find(alias_name)==AliasInfo.end()) {
		AliasInfo.emplace(alias_name,freedom_alias(alias_name, refers_to, multiplier));
	} else {
		AliasInfo.at(alias_name) = freedom_alias(alias_name, refers_to, multiplier);
	}
	
	// If the alias_name is a parameter and not self-referential, delete the parameter
	auto iter2 = FInfo.find(alias_name);
	if ((alias_name!=refers_to) && (iter2 != FInfo.end())) {
		__delitem__(alias_name);
	}
	
	return AliasInfo.at(alias_name);
}

freedom_alias& elm::ParameterList::alias(const std::string& alias_name)
{
	if (alias_name=="") {
		throw(etk::ParameterNameError("Cannot reference an alias with an empty string."));
	}

	if (AliasInfo.find(alias_name)==AliasInfo.end()) {
		throw(etk::ParameterNameError(etk::cat("Cannot find an alias named '",alias_name,"'.")));
	} else {
		return AliasInfo.at(alias_name);
	}
	
}

void elm::ParameterList::del_alias(const std::string& alias_name)
{
	if (alias_name=="") {
		throw(etk::ParameterNameError("Cannot delete an alias named <empty string>."));
	}
	AliasInfo.erase(alias_name);
}

void elm::ParameterList::unlink_alias(const std::string& alias_name)
{
	if (alias_name=="") {
		throw(etk::ParameterNameError("Cannot unlink an alias named <empty string>."));
	}
	
	if (AliasInfo.find(alias_name)==AliasInfo.end()) {
		throw(etk::ParameterNameError("Cannot unlink an alias that does not exist."));
	}
	std::string refers_to = AliasInfo.at(alias_name).refers_to;
	double multiplier = AliasInfo.at(alias_name).multiplier;
	
	parameter(alias_name,
			  FInfo[refers_to].value*multiplier,
			  FInfo[refers_to].null_value*multiplier,
			  FInfo[refers_to].initial_value*multiplier,
			  FInfo[refers_to].max_value*multiplier,
			  FInfo[refers_to].min_value*multiplier,
			  NAN,
			  NAN,
			  FInfo[refers_to].holdfast);
	
	AliasInfo.erase(alias_name);
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
	bool t = FNames.has_key(param_name);
	if (t) return true;
	auto iter = AliasInfo.find(param_name);
	if (iter != AliasInfo.end()) return true;
	return false;
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



