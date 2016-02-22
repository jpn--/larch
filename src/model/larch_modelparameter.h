/*
 *  larch_modelparameter.h
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

#ifndef __LARCH_MODELPARAMETER_H__
#define __LARCH_MODELPARAMETER_H__

#include "etk.h"

namespace elm {

	class Model2;
	
	class ModelParameter
	{
	protected:
		Model2* model;
		size_t slot;
		
	public:
	
		ModelParameter(Model2* model, const size_t& slot);
		~ModelParameter();
		
		double _get_value() const;
		void _set_value(const double& value);

		double _get_min() const;
		void _set_min(const double& value);
		void _del_min();

		double _get_max() const;
		void _set_max(const double& value);
		void _del_max();


		double _get_std_err() const;
		double _get_robust_std_err() const;

		std::string _get_name() const;
		
		bool _get_holdfast() const;
		void _set_holdfast(const bool& value);
		void _del_holdfast();
		
		double _get_nullvalue() const;
		void _set_nullvalue(const double& value);

		double _get_initvalue() const;
		void _set_initvalue(const double& value);


		#ifdef SWIG
		%pythoncode %{
		value = property(_get_value, _set_value)
		null_value = property(_get_nullvalue, _set_nullvalue)
		initial_value = property(_get_initvalue, _set_initvalue)
		minimum = property(_get_min, _set_min, _del_min)
		maximum = property(_get_max, _set_max, _del_max)
		holdfast = property(_get_holdfast, _set_holdfast, _del_holdfast)
		std_err = property(_get_std_err, None, None, "the standard error of the estimator")
		robust_std_err = property(_get_robust_std_err, None, None, "the robust standard error of the estimator via bhhh sandwich")
		name = property(_get_name, None, None, "the parameter name")
		def __repr__(self):
			return "ModelParameter('{}', value={})".format(self.name, self.value)
		%}
		#endif // def SWIG



	};
	
}




#endif // ndef __LARCH_MODELPARAMETER_H__


