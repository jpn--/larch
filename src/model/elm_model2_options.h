/*
 *  elm_model2_options.h
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

#ifndef __ELM_MODEL2_OPTIONS_H__
#define __ELM_MODEL2_OPTIONS_H__

#ifdef SWIG
%feature("kwargs", 1) elm::model_options_t::model_options_t;
%feature("kwargs", 1) elm::model_options_t::__call__;

%feature("docstring") elm::model_options_t::gradient_diagnostic
"Setting the gradient diagnostic to a positive integer N will cause the estimation \
algorithm to evaluate both analytic and finite difference gradients on the first N \
iterations and raise an Exception if they differ significantly.";

%feature("docstring") elm::model_options_t::hessian_diagnostic
"Setting the hessian diagnostic to a positive integer N will cause the estimation \
algorithm to evaluate both analytic and finite difference hessians (if available) on the first N \
iterations and raise an Exception if they differ significantly.";

%feature("docstring") elm::model_options_t::author
"This option currently does nothing.";

%feature("docstring") elm::model_options_t::calc_null_likelihood
"Calculate the null model log likelihood in conjunction with an estimation.";

%feature("docstring") elm::model_options_t::threads
"For certain easy to parallel-ize calculations, ELM will create this many worker \
threads. The default value is the number of processor cores on your computer, and \
it is recommended that you not change this value.";

%feature("docstring") elm::model_options_t::force_recalculate
"Force the recalculation of gradient every time it is requested, instead of using \
stored values. Primarily for debugging, it is recommended to leave this option off.";

%feature("docstring") elm::model_options_t::mute_nan_warnings
"Disable logging warnings of not-a-number error messages, which can occur sometimes in \
likelihood maximization.";

%feature("docstring") elm::model_options_t::calc_std_errors
"Calculate the standard errors of the parameter estimates in conjunction with an \
estimation. These values can sometimes take a long time to generate, so if you \
don't need the standard errors (ex. during early model testing) you can save some \
time by disabling this option.";

#endif // SWIG

namespace elm {
  
	struct model_options_t
	{
		
		int gradient_diagnostic;
		int hessian_diagnostic;
		
		int threads;
		
		bool calc_null_likelihood;
		bool null_disregards_holdfast;
		bool calc_std_errors;
		bool mute_nan_warnings;
		bool force_finite_diff_grad;
		bool save_db_hash;
		bool force_recalculate;
		bool teardown_after_estimate;
		bool weight_autorescale;
		bool weight_choice_rebalance;
		bool suspend_xylem_rebuild;
		bool log_turns;
		bool enforce_bounds;
		bool enforce_constraints;
		
		std::string author;
		
		// Constructor
		model_options_t(
			int threads=-9,
			bool calc_null_likelihood=true,
			bool null_disregards_holdfast=true,
			bool calc_std_errors=true,
			int gradient_diagnostic=0,
			int hessian_diagnostic=0,
			bool mute_nan_warnings=true,
			bool force_finite_diff_grad=false,
			bool save_db_hash=false,
			bool force_recalculate=false,
			std::string author="Chuck Finley",
			bool teardown_after_estimate=true,
			bool weight_autorescale=true,
			bool weight_choice_rebalance=true,
			bool suspend_xylem_rebuild=false,
			bool log_turns=false,
			bool enforce_bounds=true,
			bool enforce_contraints=false
		);
	
		// Re-constructor
		void __call__(
			int threads=-9,
			int calc_null_likelihood=-9,
			int null_disregards_holdfast=-9,
			int calc_std_errors=-9,
			int gradient_diagnostic=-9,
			int hessian_diagnostic=-9,
			int mute_nan_warnings=-9,
			int force_finite_diff_grad=-9,
			int save_db_hash=-9,
			int force_recalculate=-9,
			std::string author="-9",
			int teardown_after_estimate=-9,
			int weight_autorescale=-9,
			int weight_choice_rebalance=-9,
			int suspend_xylem_rebuild=-9,
			int log_turns=-9,
			int enforce_bounds=-9,
			int enforce_contraints=-9
		);

		void copy(const model_options_t& other);
	
		// Repr
		std::string __repr__() const;
		std::string __str__() const;
		std::string _save_buffer() const;
		

		#ifdef SWIG
		%extend {
			%pythoncode %{
			def __getitem__(self, k):
				return getattr(self, k)
			def __setitem__(self, k, v):
				return setattr(self, k, v)
			def _as_dict(self):
				keys = dir(self)
				dct = {}
				for k in keys:
					if k[0:2]!="__" and k not in ['copy', 'this', 'thisown', '_as_dict']:
						dct[k] = getattr(self, k)
				return dct
			def __setattr__(self, key, value):
				if key not in dir(self) and key not in ['copy', 'this', 'thisown', '_as_dict']:
					raise TypeError( "cannot create the new attribute '%s' for %s" % (str(key),str(type(self))) )
				super(model_options_t, self).__setattr__(key, value)
			%}
		}
		#endif // SWIG

			
	};
  
};




#endif // __ELM_MODEL2_OPTIONS_H__
