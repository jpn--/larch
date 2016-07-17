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
{
	
}
elm::ParameterList::ParameterList(const elm::ParameterList& dupe)
: FNames (dupe.FNames)
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

//PyObject* elm::ParameterList::constraints() const
//{
//	Py_RETURN_NONE;
//}

void elm::ParameterList::tearDown()
{

}



void elm::ParameterList::freshen()
{
	
}



