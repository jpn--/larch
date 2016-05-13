/*
 *  sherpa_freedom.h
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


%rename(ParameterAlias) freedom_alias;
%rename(Parameter) freedom_info;
%feature("kwargs", 1) elm::freedom_info::update;

%feature("kwargs", 1) freedom_info;
//%feature("compactdefaultargs") freedom_info;



%feature("autodoc", "1") freedom_info;
%feature("docstring") freedom_info
"	This object represents a discrete choice model parameter.

	Parameters
	----------
	name : str
		The name of the parameter. This name is used both in commands
		that refer to this parameter, as well as in reports. Generally
		it is best to choose a short but descriptive name that does
		not include any special characters, although any unicode string
		should be acceptable.
	value : float
		This value represents the current value of the parameter.
	null_value : float
		This value represents the default value of the parameter, which
		would be assumed if no information is available. It is generally
		zero, although for some parameters -- notably the logsum parameters
		in a nest logit model, but also certain others -- the default value
		might be one, or some other value.
	holdfast : bool
		Sets the holdfast attribute. When True, the value of this parameter
		is held constant during parameter estimation.

	Other Parameters
	----------------
	initial_value : float
		The initial value of the parameter. This is where the
		search algorithm began.
	std_err, robust_std_err : float
		This is the standard error of the estimate of this parameter. The
		standard error is derived from the curvature of the log likelihood
		function at its maximum. The robust standard error of the estimate
		is derived from the sandwich estimator.
	covariance, robust_covariance : dict
		These are dictionary with parameter names as keys and floats
		as values, representing the (robust) covariance between this estimator
		and the other estimators in the model.

	Notes
	-----
	It is not usually necessary to define the 'other parameters' explicitly.
	The values are normally derived as an outcome of the model estimation
	process, and the ability to set them here is provided to allow
	the save and load methods to accurately
	recreate a model with all attributes intact.";







%feature("docstring") freedom_info::t_stat
"Calculates the t statistic against the null value of the parameter. 

The t statistic is calculated as value - null_value)/std_err.";



