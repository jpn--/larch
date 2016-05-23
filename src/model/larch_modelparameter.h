/*
 *  larch_modelparameter.h
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

#ifndef __LARCH_MODELPARAMETER_H__
#define __LARCH_MODELPARAMETER_H__

#include "etk.h"

class sherpa;

namespace elm {

	class Model2;
	
	class ModelParameter
	{
	protected:
		sherpa* model;
		size_t slot;
	
	private:
		PyObject* model_as_pyobject;
		
	public:
	
		ModelParameter(sherpa* model, const size_t& slot);
		ModelParameter(const elm::ModelParameter& original);
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
		double _get_t_stat() const;
		double _get_robust_std_err() const;

		std::string _get_name() const;
		
		signed char _get_holdfast() const;
		void _set_holdfast(const bool& value);
		void _set_holdfast(const signed char& value);
		void _del_holdfast();
		
		double _get_nullvalue() const;
		void _set_nullvalue(const double& value);

		double _get_initvalue() const;
		void _set_initvalue(const double& value);

		size_t _get_index() const;

		etk::symmetric_matrix* _get_complete_covariance_matrix() const;
		
		PyObject* _get_model();

		#ifdef SWIG
		%pythoncode %{
		value = property(_get_value, _set_value, None, "the current value for the parameter")
		null_value = property(_get_nullvalue, _set_nullvalue, None, "the null value for the parameter (used for null models and t-stats)")
		initial_value = property(_get_initvalue, _set_initvalue, None, "the initial value of the parameter")
		minimum = property(_get_min, _set_min, _del_min, "the min bound for the parameter during estimation")
		min_value = minimum
		maximum = property(_get_max, _set_max, _del_max, "the max bound for the parameter during estimation")
		max_value = maximum
		holdfast = property(_get_holdfast, _set_holdfast, _del_holdfast, "a flag indicating if the parameter value should be held fast (constrained to keep its value) during estimation")
		std_err = property(_get_std_err, None, None, "the standard error of the estimator (read-only)")
		robust_std_err = property(_get_robust_std_err, None, None, "the robust standard error of the estimator via bhhh sandwich (read-only)")
		name = property(_get_name, None, None, "the parameter name (read-only)")
		index = property(_get_index, None, None, "the parameter index within the model (read-only)")
		t_stat = property(_get_t_stat, None, None, "the t-statistic for the estimator (read-only)")
		def __repr__(self):
			return "ModelParameter('{}', value={})".format(self.name, self.value)
		@property
		def covariance(self):
			"the covariance of the estimator (read-only)"
			slot = self.index
			cov = self._get_complete_covariance_matrix()
			model = self._get_model()
			ret = {}
			for name, val in zip(model.parameter_names(), cov[:,slot]):
				ret[name] = val
			return ret
		@property
		def robust_covariance(self):
			"the robust covariance of the estimator via bhhh sandwich (read-only)"
			slot = self.index
			model = self._get_model()
			cov = model.robust_covariance_matrix
			ret = {}
			for name, val in zip(model.parameter_names(), cov[:,slot]):
				ret[name] = val
			return ret
		%}
		#endif // def SWIG



	};
	
}




#endif // ndef __LARCH_MODELPARAMETER_H__


