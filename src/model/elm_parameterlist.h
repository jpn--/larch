//
//  elm_parameterlist.h
//  Yggdrasil
//
//  Created by Jeffrey Newman on 12/21/12.
//
//

#ifndef __Yggdrasil__elm_parameterlist__
#define __Yggdrasil__elm_parameterlist__

#ifndef SWIG
#include "sherpa_freedom.h"
#include "etk.h"
#endif // not SWIG

#ifdef SWIG
%{
#include "elm_parameterlist.h"
%}
%feature("kwargs", 1) elm::ParameterList::parameter;

#endif // SWIG

namespace elm {

	#ifndef SWIG
	class parametex;
	typedef boosted::shared_ptr<parametex> parametexr;
	#endif // not SWIG

	class ParameterList
	#ifndef SWIG
	: public etk::object
	#endif // not SWIG
	{


	public:
		#ifdef SWIG
		%rename(_parameter_name_index) FNames;
		#endif // SWIG
		etk::autoindex_string FNames;


	#ifndef SWIG
	public:
		
		std::map<std::string,freedom_alias> AliasInfo;
		
	#endif // not SWIG

public:

	ParameterList();
	ParameterList(const ParameterList& dupe);

	bool __contains__(const std::string& param_name) const;
	size_t _len() const;
	
	size_t parameter_index(const std::string& param_name) const;

//	PyObject* values() const;
//	void values(PyObject*);
	PyObject* zeros() const;

//	PyObject* constraints() const;
	virtual void tearDown();

	virtual void freshen();
	
	#ifndef SWIG
//	std::string values_string();
	
	#endif
};



}; // end namespace elm

#ifdef SWIG
%pythoncode %{
ParameterList.__len__ = lambda self: int(self._len())
%}
#endif // SWIG


#endif /* defined(__Yggdrasil__elm_parameterlist__) */
