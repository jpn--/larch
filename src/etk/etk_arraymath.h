//
//  elm_arraymath.h
//  Yggdrasil
//
//  Created by Jeffrey Newman on 12/20/12.
//
//

#ifndef __Yggdrasil__elm_arraymath__
#define __Yggdrasil__elm_arraymath__

#ifndef SWIG
#include "etk.h"
namespace elm { class paramArray; }
#endif // ndef SWIG

#ifdef SWIG
%feature("kwargs", 1) elm::logit_simple;
%feature("kwargs", 1) elm::log_likelihood_simple;
#endif // def SWIG



namespace elm {


extern PyObject* elm_linalg_module;

void set_linalg(PyObject* mod);


void case_logit_add_term
(size_t n,
 const double* old_p, // array size [n]
 const double* term2, // array size [n]
 double* new_p,       // array size [n]
 const bool* avail    // array size [n]
 );

void case_logit_add_term_deriv
(size_t nalt,
 size_t npar,
 const double* dOldP_dParam,  //[ nAlts, nParams ]
 const double* dTerm_dParam,  //[ nAlts, nParams ]
 double* dNewP_dParam,        //[ nAlts, nParams ]
 double* workspace,           //[ nParams ]
 const double* OldP,          //[ nAlts ]
 const double* NewP           //[ nAlts ]
 );


void logit_simple (etk::ndarray* utility, etk::ndarray* probability=nullptr, const etk::ndarray_bool* avail=nullptr);
void logit_simple_deriv (etk::ndarray* dLLdprobability, etk::ndarray* dLLdutility,
								   const etk::ndarray* probability, bool initialize);

void logit_gev (etk::ndarray* utility, etk::ndarray* probability, etk::ndarray* Gi, const etk::ndarray_bool* avail=nullptr);
void logit_gev_deriv (etk::ndarray* dLLdprobability, etk::ndarray* dLLdutility,
								   const etk::ndarray* probability,
								   const etk::ndarray* Yi,
								   const etk::ndarray* Gi,
								   const etk::ndarray* dGij,
								   bool initialize);




				
double log_likelihood_simple(const etk::ndarray* probability, etk::ndarray* choices, etk::ndarray* weights=nullptr);
void log_likelihood_simple_deriv(const etk::ndarray* probability, etk::ndarray* choices, etk::ndarray* dLLdProb, etk::ndarray* weights=nullptr);

void stdthread_go();

void workshoptester();


/*
void Gi_nested_logit(const etk::ndarray* utility, const etk::ndarray_bool* avail,
					 elm::VAS_System* xylem, 
					 etk::ndarray* Gi, etk::ndarray* Gij);
*/

};

#endif /* defined(__Yggdrasil__elm_arraymath__) */
