/*
 *  sherpa.h
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


#ifndef __SHERPA_HEADER__
#define __SHERPA_HEADER__

#ifndef SWIG

#include "etk.h"
#include "elm_parameterlist.h"
#include "sherpa_pack.h"
#include "sherpa_freedom.h"

#define SHERPA_SUCCESS    2
#define SHERPA_IMPROVED   1
#define SHERPA_SLOW       0
#define SHERPA_IMP_STUCK -1
#define SHERPA_IMP_ERROR -2
#define SHERPA_FAIL	     -3
#define SHERPA_BAD       -4
#define SHERPA_FATAL     -5
#define SHERPA_MAXITER   -6
#define SHERPA_NAN_TOL   -7

#define LINE_SEARCH_SUCCESS_BIG		( 2)
#define LINE_SEARCH_SUCCESS_SMALL	( 1)
#define LINE_SEARCH_NO_IMPROVEMENT	( 0)
#define LINE_SEARCH_FAIL         	(-1)
#define LINE_SEARCH_ERROR_NAN   	(-2)

#endif // ndef SWIG

std::string algorithm_name(const char& algo);


struct sherpa_result {
	double starting_obj_value;
	double best_obj_value;
	int result;
	std::string explain_stop;
};


class sherpa
: public elm::ParameterList
{


#ifndef SWIG
	
public:
	virtual double objective();
	virtual const etk::memarray& gradient();
	virtual void calculate_hessian();
	
	void finite_diff_gradient(etk::memarray& fGrad);
	void finite_diff_hessian (etk::triangle& fHESS);
	
	double gradient_diagnostic (bool shout=false);
	double hessian_diagnostic () ;
	int flag_gradient_diagnostic;
	int flag_hessian_diagnostic;
	int flag_log_turns;

protected:
	void allocate_memory();
	void free_memory();
	void _initialize();

public:
	inline size_t dF() const { return FNames.size(); }
	
private:
	double _check_for_improvement();
	void _reset_to_best_known_value();
	double _line_search_evaluation(double& step);
	int _line_search (sherpa_pack& method);
	//	line_search() is called to search along a direction for some improvement
	//		Return Values:
	//		 2	Found improvement at initial step
	//		 1	Found improvement at smaller than initial step
	//		-1	Did not find an improvement at minimum step value
	//		-2	A function evaluation returned NaN even at minimum step value
	
private:
	// Direction finding schemes, return 0 for good values, <0 if there is a problem
	int _bfgs_update();
	int _dfp_update();
	int _dfpj_update();
	int _find_ascent_direction (char& Method);
	void _basecamp();
	
public:
	void full_evaluation(int with_derivs=0);
private:
	sherpa_result _maximize_pack(sherpa_pack& Norgay, unsigned& iteration_number);
public:
	std::string maximize(unsigned& iteration_number, std::vector<sherpa_pack>* opts=NULL);
	
	std::string calculate_errors();

private:
	void _initial_evaluation();

public:
	std::string add_freedom(const std::string& param_name, const double& value=NAN, const double& nullvalue=NAN,
					   const double& max=NAN, const double& min=NAN);
	std::string add_freedom(const freedom_info& fInfo);
	std::string remove_freedom(const std::string& param_name);
	
	
	///// RESULTS /////
	
protected:
	const freedom_info* get_raw_info (const std::string& freedom_name) const;
public:
	bool   parameter_exists(const std::string& freedom_name) const;
	double parameter_value (const std::string& freedom_name) const;
	double parameter_stderr(const std::string& freedom_name) const;
	freedom_info& get_freedom_info (const std::string& freedom_name);
	
	///// MEMORY /////
	
public:
//	etk::autoindex_string FNames;
//	std::map<std::string,freedom_info> FInfo;
protected:
	void _update_freedom_info(const etk::triangle* ihess=NULL, const etk::triangle* robust_covar=NULL);
	void _update_freedom_info_best();
	
public:
	void reset_to_initial_value();
	void refresh_initial_value();

public:
#define status_FNames            0x01
#define status_FCurrent          0x02
#define status_FLastTurn         0x04
#define status_FDirection        0x08
#define status_GCurrent          0x10
#define status_FDirectionLarge   0x20
	std::string printStatus(int which=0x3, double total_tol=0) const;
	
protected:
	double ZCurrent;
	double ZBest;
	double ZLastTurn;
	
public:
	etk::memarray FCurrent;
	
public:
	const etk::memarray& ReadFCurrent() const {return FCurrent;};
	const etk::memarray& ReadFBest() const {return FBest;};
	std::string ReadFCurrentAsString() const;
	
protected:
	etk::memarray FBest;
	etk::memarray FLastTurn;
	etk::memarray FPrevTurn;
	etk::memarray FMotion;
	etk::memarray FDirection;
	
	etk::memarray FMax;
	etk::memarray FMin;
	
protected:
	etk::memarray GCurrent;
private:
	etk::memarray GLastTurn;
	etk::memarray GPrevTurn;
	etk::memarray GMotion;
protected:
	etk::memarray FatGCurrent;
	
protected:	
	etk::memarray_symmetric Bhhh;
	etk::triangle Hess;
	etk::triangle invHess;
	etk::triangle invHessTemp;
	etk::triangle robustCovariance;
	
#endif // ndef SWIG

public:
	double LL() const; 

	unsigned max_iterations;

	etk::symmetric_matrix* covariance_matrix();
	etk::symmetric_matrix* robust_covariance_matrix();
	
	///// CONSTRUCTOR /////
public:
	sherpa();
	sherpa(const sherpa& dupe);
	
	bool any_holdfast();
	size_t count_holdfast();
	void hessfull_to_hessfree(const etk::symmetric_matrix* full_matrix, etk::symmetric_matrix* free_matrix) ;
	void hessfree_to_hessfull(etk::symmetric_matrix* full_matrix, const etk::symmetric_matrix* free_matrix) ;

};



#endif // __SHERPA_HEADER__

