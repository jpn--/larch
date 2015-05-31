/*
 *  sherpa_freedom.h
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


#ifndef __SHERPA_FREEDOM__
#define __SHERPA_FREEDOM__

#ifndef SWIG
#include <string>
#include <map>
#include "etk.h"
#endif

#ifdef SWIG 
%include "sherpa_freedom.i"
#endif // SWIG


struct freedom_alias {
	std::string name;
	std::string refers_to;
	double multiplier;
	
	freedom_alias(const std::string& name, const std::string& refers_to, const double& multiplier)
	: name(name)
	, refers_to(refers_to)
	, multiplier(multiplier)
	{ }
	
};



class freedom_info {
public:
	std::string name;
	double value;
	double null_value;
	double initial_value;
	double std_err;
	double robust_std_err;
	double max_value;
	double min_value;
	int holdfast;
	PyObject* _covar;
	PyObject* _robust_covar;
	
	PyObject* getCovariance() const;
	void setCovariance(PyObject* covariance);
	PyObject* getRobustCovariance() const;
	void setRobustCovariance(PyObject* covariance);
	
	double t_stat() const;
	
	std::string representation(bool pretty=true) const;
	
	freedom_info(const std::string& name="",
						   const double& value=0,
						   const double& null_value=0,
						   const int& holdfast=0,
						   const double& initial_value=NAN,
						   const double& std_err=NAN,
						   const double& robust_std_err=NAN,
						   const double& min_value=-INF,
						   const double& max_value=INF,
						   PyObject* covariance=nullptr,
						   PyObject* robust_covariance=nullptr);
	~freedom_info();
	
	#ifndef SWIG
	freedom_info(const freedom_info& that);
	#endif
	
	void update(const double& value=NAN,
						   const double& null_value=NAN,
						   const int& holdfast=-1,
						   const double& initial_value=NAN,
						   const double& std_err=NAN,
						   const double& robust_std_err=NAN,
						   const double& min_value=NAN,
						   const double& max_value=NAN,
						   PyObject* covariance=nullptr,
						   PyObject* robust_covariance=nullptr);
};




#ifdef SWIG 
%extend freedom_info {
	std::string __str__(void* z=nullptr) const {
		return $self->representation(true);			
	}
	std::string __repr__(void* z=nullptr) const {
		return $self->representation(false);			
	}
	%pythoncode %{
		covariance = _swig_property(getCovariance, setCovariance)
		robust_covariance = _swig_property(getRobustCovariance, setRobustCovariance)
		def __getitem__(self, *arg):
			return self.__getattribute__(*arg)
		
		def t_stat_signif(self, df=None):
			'''Calulates the significance level of the t-test.
			
			When df is not given, the reported value is calculated as
			:math:`2(1-\Phi(t))`, with :math:`\Phi(t)` as the CDF of a the standard
			normal distribution evaluated at :math:`t`. When df is given, the
			t distribution with the indicated number of degrees of freedom is
			used in place of the normal diatribution. In most discrete choice modeling
			scenarios, the number of degrees of freedom is large enough that the
			resulting values are indistinguishable.
			'''
			import scipy.stats
			t = self.t_stat()
			if df is None:
				return 2.0*scipy.stats.norm.sf(abs(t), loc=0, scale=1)
			return 2.0*scipy.stats.t.sf(abs(t), df, loc=0, scale=1)
	%}
};
#endif // SWIG


#endif // __SHERPA_FREEDOM__

