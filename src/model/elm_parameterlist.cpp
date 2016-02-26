//
//  elm_parameterlist.cpp
//  Yggdrasil
//
//  Created by Jeffrey Newman on 12/21/12.
//
//

#include "elm_parameterlist.h"
#include "elm_parameter2.h"
#include "sherpa.h"

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



elm::parametexr sherpa::_generate_parameter(const std::string& freedom_name,
														double freedom_multiplier)
{
	std::string fn = freedom_name;
	etk::uppercase(fn);
	if (fn == "CONSTANT") {
		return boosted::make_shared<elm::parametex_constant> (freedom_multiplier);
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
	
	size_t freedom_number = FNames.index_from_string(fn);
	
	if (FHoldfast.int8_at(freedom_number)) {
		return boosted::make_shared<elm::parametex_constant>(freedom_multiplier*(FCurrent.at(freedom_number)));
	}
	if (freedom_multiplier==1.0) {
		return boosted::make_shared<elm::parametex_equal>(fn,this);
	} 
	return boosted::make_shared<elm::parametex_scale>(fn,this,freedom_multiplier);
}

size_t elm::ParameterList::parameter_index(const std::string& param_name) const
{
	return FNames[param_name];
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

void elm::ParameterList::tearDown()
{

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



