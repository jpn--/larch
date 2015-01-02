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

	#ifndef SWIG
	public:
		etk::autoindex_string FNames;
		std::map<std::string,freedom_info> FInfo;
		parametexr _generate_parameter(const std::string& freedom_name,
									   const double& freedom_multiplier);
	#endif // not SWIG

public:

	ParameterList();
	ParameterList(const ParameterList& dupe);

	freedom_info& parameter(const std::string& param_name,
							   const double& value=NAN,
							   const double& null_value=NAN,
							   const double& initial_value=NAN,
							   const double& max=NAN,
							   const double& min=NAN,
							   const double& std_err=NAN,
							   const double& robust_std_err=NAN,
							   const int& holdfast=-1,
							   PyObject* covariance=NULL,
							   PyObject* robust_covariance=NULL); 

	freedom_info& __getitem__(const std::string& param_name);
	freedom_info& __getitem__(const int& param_num);
	void __setitem__(const std::string& param_name, freedom_info& value);
	void __delitem__(const std::string& param_name);


	bool __contains__(const std::string& param_name) const;
	size_t _len() const;

	PyObject* values() const;
	void values(PyObject*);
	PyObject* zeros() const;

	PyObject* constraints() const;

	virtual void freshen();
	
//	void covariance(etk::symmetric_matrix*);
//	void robustcovariance(etk::symmetric_matrix*);

	#ifndef SWIG
	std::string values_string();
	
	#endif
};



}; // end namespace elm

#ifdef SWIG
%pythoncode %{
ParameterList.__len__ = lambda self: int(self._len())
%}
#endif // SWIG


#endif /* defined(__Yggdrasil__elm_parameterlist__) */
