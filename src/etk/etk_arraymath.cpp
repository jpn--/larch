//
//  elm_arraymath.cpp
//  Yggdrasil
//
//  Created by Jeffrey Newman on 12/20/12.
//
//

#include "etk.h"
#include "etk_arraymath.h"
#include "elm_parameter2.h"
#include <iostream>

#include "etk_thread.h"




PyObject* elm::elm_linalg_module = nullptr;


void elm::set_linalg(PyObject* mod)
{
	Py_CLEAR(elm::elm_linalg_module);
	elm::elm_linalg_module = mod;
	Py_XINCREF(elm::elm_linalg_module);
}



void* heavy(void* input)
{
	double* val = (double*) input;
	double hold = *val;

	srand ( time(NULL) );
	

	double ex = exp(*val);
	for (unsigned i=0; i<100000000; i++) {
		for (unsigned j=0; j<8; j++) {
			ex += double((rand() % 10))/double(1000);
			ex -= double((rand() % 10))/double(1000);
		}
	}
	
	*val = ex;
	
	//std::cout <<"[" << *val << "]\n";
	return (void*) val;
}

void elm::stdthread_go()
{
	std::cout << "stdthread go!\n";


	std::vector< boosted::thread* > threads (8,NULL);
	std::vector< double > z (8,1.0);
	std::vector< double > data (14,2.0);

	for (unsigned i=0; i<14; i++) {
		data[i] = i*i;
	}
	
	
	for (unsigned i=0; i<4; i++) {
		threads[i] = new boosted::thread(heavy, &data[i]);
	}
	for (unsigned i=0; i<4; i++) {
		threads[i]->join();
		std::cout << i <<"[" << data[i] << "]\n";
		delete threads[i];
		threads[i] = NULL;
	}
	std::cout << "stdthread end!\n";
}

#include "etk_workshop.h"

void elm::workshoptester()
{
//	etk::dispatch(4, 5000, []{return boosted::make_shared< etk::workshop >();});
}














void elm::logit_simple (etk::ndarray* utility, etk::ndarray* probability, const etk::ndarray_bool* avail)
{
	PyArrayObject* avail_temp = nullptr;
	if (avail) {
		avail_temp = (PyArrayObject*) PyArray_Squeeze(avail->pool);
		Py_XINCREF(avail_temp);
		if (!utility->pool || !avail->pool || !PyArray_SAMESHAPE(utility->pool, avail_temp)) {
			OOPS("Availability array must match shape of utility array");
		}
	}
	if (!probability) { probability = utility; }
	if (!PyArray_SAMESHAPE(utility->pool, probability->pool)) {
		probability->resize(*utility);
	}
	double* u;
	double* p;
	double* startu=(double*)( PyArray_DATA(utility->pool) );
	double* startp=(double*)( PyArray_DATA(probability->pool) );
	if (avail_temp) {
		const bool* av = static_cast<const bool*>( PyArray_DATA((PyArrayObject*)avail_temp) );
		for (u=startu, p=startp; u!=startu+PyArray_SIZE(utility->pool); u++, p++, av++) {
			if (*av) {
				*p = ::exp(*u);
			} else {
				*p = 0;
			}
		}
	} else {
		for (u=startu, p=startp; u!=startu+PyArray_SIZE(utility->pool); u++, p++) {
			*p = ::exp(*u);
		}
	}
	probability->prob_scale_2();
	Py_CLEAR(avail_temp);
}


void elm::logit_simple_deriv (etk::ndarray* dLLdprobability, etk::ndarray* dLLdutility,
								   const etk::ndarray* probability, bool initialize)
{
	double initializer = 0.0;
	if (!initialize) initializer = 1.0;
	
	if (!dLLdutility) { dLLdutility = dLLdprobability; }
	if (!PyArray_SAMESHAPE(dLLdprobability->pool, dLLdutility->pool)) {
		dLLdutility->resize(*dLLdprobability);
	}
	double* u;
	double* p;
	double* startu=(double*)( PyArray_DATA(dLLdutility->pool) );
	double* startp=(double*)( PyArray_DATA(dLLdprobability->pool) );

	size_t nC = dLLdprobability->size1();
	size_t nA = dLLdprobability->size2();
	etk::ndarray dPdU (nA,nA);
	double Pi;

	for (size_t c=0; c<nC; c++) {
		dPdU.initialize();
		for (size_t i=0; i<nA; i++) {
			Pi = (*probability)(c,i);
			for (size_t j=i; j<nA; j++) {
				if (i==j) {
					dPdU(i,j) = Pi*(1.0-Pi);
				} else {
					dPdU(j,i) = dPdU(i,j) = -Pi*(*probability)(c,j);
				}
			}
		}
		cblas_dgemv(CblasRowMajor, CblasNoTrans, nA, nA, 1, dPdU.ptr(), nA, dLLdprobability->ptr(c), 1, initializer, dLLdutility->ptr(c), 1);
	}
}




void elm::logit_gev (etk::ndarray* utility, etk::ndarray* probability, etk::ndarray* Gi, const etk::ndarray_bool* avail)
{
	PyArrayObject* avail_temp = nullptr;
	if (avail) {
		avail_temp = (PyArrayObject*)PyArray_Squeeze(avail->pool);
		Py_XINCREF(avail_temp);
		if (!utility->pool || !avail->pool || !PyArray_SAMESHAPE(utility->pool, avail_temp)) {
			OOPS("Availability array must match shape of utility array");
		}
	}

	if (!PyArray_SAMESHAPE(utility->pool, Gi->pool)) {
			OOPS("Gi array must match shape of utility array");
	}
	
	
	if (!probability) { probability = utility; }
	if (!PyArray_SAMESHAPE(utility->pool, probability->pool)) {
		probability->resize(*utility);
	}
	double* u;
	double* p;
	double* startu=(double*)( PyArray_DATA(utility->pool) );
	double* startp=(double*)( PyArray_DATA(probability->pool) );
	double* g = (double*)( PyArray_DATA(Gi->pool) );
	if (avail_temp) {
		const bool* av = static_cast<const bool*>( PyArray_DATA((PyArrayObject*)avail_temp) );
		for (u=startu, p=startp; u!=startu+PyArray_SIZE(utility->pool); u++, p++, av++, g++) {
			if (*av) {
				*p = ::exp(*u) * (*g);
			} else {
				*p = 0;
			}
		}
	} else {
		for (u=startu, p=startp; u!=startu+PyArray_SIZE(utility->pool); u++, p++, g++) {
			*p = ::exp(*u) * (*g);
		}
	}
	probability->prob_scale_2();
	Py_CLEAR(avail_temp);
}

/*
logit_gev_deriv

etk::ndarray* dLLdprobability    [usedcases,alts]
etk::ndarray* dLLdutility        [usedcases,alts]
const etk::ndarray* probability  [usedcases,alts]
const etk::ndarray* Gi           [usedcases,alts]
const etk::ndarray* dGi_dVj      [usedcases,alts,alts]
bool initialize
*/
void elm::logit_gev_deriv (etk::ndarray* dLLdprobability, etk::ndarray* dLLdutility,
								   const etk::ndarray* probability,
								   const etk::ndarray* Yi,
								   const etk::ndarray* Gi,
								   const etk::ndarray* dGij,
								   bool initialize)
{



	double initializer = 0.0;
	if (!initialize) initializer = 1.0;
	
	if (!dLLdutility) { dLLdutility = dLLdprobability; }
	if (!PyArray_SAMESHAPE(dLLdprobability->pool, dLLdutility->pool)) {
		dLLdutility->resize(*dLLdprobability);
	}
	double* u;
	double* p;
	double* startu=(double*)( PyArray_DATA(dLLdutility->pool) );
	double* startp=(double*)( PyArray_DATA(dLLdprobability->pool) );

	size_t nC = dLLdprobability->size1();
	size_t nA = dLLdprobability->size2();
	etk::ndarray dPdU (nA,nA);
	etk::ndarray sum_term (nA);
	double Pi;

	for (size_t c=0; c<nC; c++) {
		dPdU.initialize();
		sum_term.initialize();
		// calculate sum_term
		for (size_t k=0; k<nA; k++) {
			cblas_daxpy(nA, (*probability)(c,k)/(*Gi)(c,k), dGij->ptr(c,k), 1, sum_term.ptr(), 1);
		}
		
		for (size_t i=0; i<nA; i++) {
			Pi = (*probability)(c,i);
			for (size_t j=0; j<nA; j++) {
				if (i==j) {
					dPdU(i,j) = Pi*( 1.0-Pi+((*dGij)(c,i,i)*Yi->at(c,i)/(*Gi)(c,i))-sum_term(i) );
				} else {
					dPdU(i,j) = Pi*( -(*probability)(c,j)+((*dGij)(c,i,j)*Yi->at(c,j)/(*Gi)(c,i))-sum_term(j) );
				}
			}
		}
		cblas_dgemv(CblasRowMajor, CblasNoTrans, nA, nA, 1, dPdU.ptr(), nA, dLLdprobability->ptr(c), 1, initializer, dLLdutility->ptr(c), 1);
	}
}









double elm::log_likelihood_simple(const etk::ndarray* probability, etk::ndarray* choices, etk::ndarray* weights)
{
	double LL=0;
	if (!probability->pool) OOPS("Probability array must be given");
	if (!choices->pool) OOPS("Choices array must be given");
	
	PyObject* choices_temp = nullptr;
	choices_temp = PyArray_Squeeze(choices->pool);
	Py_XINCREF(choices_temp);
	if (!PyArray_SAMESHAPE(probability->pool, (PyArrayObject*)choices_temp)) {
		Py_CLEAR(choices_temp);
		OOPS("Choices array must match shape of probability array");
	}
	
	if (weights) {
		if (weights->size1()!=probability->size1()) OOPS("Weights array must have same number of cases as probability array");
		if (weights->ndim()==2 && weights->size2()>1) OOPS("Weights array must have only one column");
		if (weights->ndim()>2 ) OOPS("Weights array has too many dimensions");
	}
	
	
	double* c;
	double* p;
	double* startc=(double*)( PyArray_DATA((PyArrayObject*)choices_temp) );
	double* startp=(double*)( PyArray_DATA(probability->pool) );
	
	
	if (weights) {
		double* c;
		for (unsigned cas=0; cas<probability->size1(); cas++) {
			for (unsigned alt=0; alt<probability->size2(); alt++) {
				c = (double*)PyArray_GETPTR2((PyArrayObject*)choices_temp, cas, alt);
				if (*c) {
					LL += (*c) * (::log((*probability)(cas,alt))) * (*weights)(cas);
				}
			}
		}
	}
	
	/*
	int a=0;
	int as = probability->size2();
	if (weights) {
		double* w = (double*)( PyArray_DATA(weights->pool) );
		for (c=startc, p=startp; p!=startp+PyArray_SIZE(probability->pool); c++, p++) {
			if (*c) {
				LL += (*c) * (::log(*p)) * (*w);
			}
			a++;
			if (a>=as) {
				a=0;
				w++;
			}
		}
	}*/
	
	else {
		for (c=startc, p=startp; p!=startp+PyArray_SIZE(probability->pool); c++, p++) {
			if (*c) {
				LL += (*c) * (::log(*p));
			}
		}
	}
	Py_CLEAR(choices_temp);
	return LL;
}

void elm::log_likelihood_simple_deriv(const etk::ndarray* probability, etk::ndarray* choices, etk::ndarray* dLLdProb, etk::ndarray* weights)
{
	if (!probability->pool) OOPS("Probability array must be given");
	if (!choices->pool) OOPS("Choices array must be given");
	if (!dLLdProb->pool) OOPS("dLLdProb array must be given");
	
	PyObject* choices_temp = nullptr;
	choices_temp = PyArray_Squeeze(choices->pool);
	Py_XINCREF(choices_temp);
	if (!PyArray_SAMESHAPE(probability->pool, (PyArrayObject*)choices_temp)) {
		Py_CLEAR(choices_temp);
		OOPS("Choices array must match shape of probability array");
	}
	
	if (!PyArray_SAMESHAPE(probability->pool, dLLdProb->pool)) {
		OOPS("dLLdProb array must match shape of probability array");
	}
	
	if (weights) {
		if (weights->size1()!=probability->size1()) OOPS("Weights array must have same number of cases as probability array");
		if (weights->ndim()==2 && weights->size2()>1) OOPS("Weights array must have only one column");
		if (weights->ndim()>2 ) OOPS("Weights array has too many dimensions");
	}
	
	double* c;
	double* p;
	double* dp;
	double* startc=(double*)( PyArray_DATA((PyArrayObject*)choices_temp) );
	double* startp=(double*)( PyArray_DATA(probability->pool) );
	double* startdp=(double*)( PyArray_DATA(dLLdProb->pool) );
	if (weights) {
		int a=0;
		int as = probability->size2();
		double* w = (double*)( PyArray_DATA(weights->pool) );
		for (c=startc, p=startp, dp=startdp; p!=startp+PyArray_SIZE(probability->pool); c++, p++, dp++) {
			if (*c) {
				*dp = (*c)*(*w) / (*p);
			} else {
				*dp = 0.0;
			}
			a++;
			if (a>=as) {
				a=0;
				w++;
			}

		}
	} else {
		for (c=startc, p=startp, dp=startdp; p!=startp+PyArray_SIZE(probability->pool); c++, p++, dp++) {
			if (*c) {
				*dp = (*c) / (*p);
			} else {
				*dp = 0.0;
			}
		}
	}
	Py_CLEAR(choices_temp);
	
}


/*

void elm::Gi_nested_logit(const etk::ndarray* utility, const etk::ndarray_bool* avail,
					 elm::VAS_System* xylem, 
					 etk::ndarray* Gi, etk::ndarray* Gij)
{
	if (avail) {
		if (!utility->pool || !avail->pool || !PyArray_SAMESHAPE(utility->pool, PyArray_Squeeze(avail->pool))) {
			OOPS("Availability array must match shape of utility array");
		}
	}
	
	etk::ndarray Y (xylem->size());
	
	size_t nC = utility->size1();
	
	unsigned child, parent;
	double mu;
	
	for (size_t c=0; c<nC; c++) {
		Y.initialize();
		size_t a=0;
		for (; a<xylem->n_elemental(); a++) {
			if ((*avail)(c,a)) {
				Y[a] = exp((*utility)(c,a));
				Y[xylem->at(a)->upcell(0)->slot()] += pow(Y[a],xylem->at(a)->upcell(0)->mu());
			} else {
				Y[a] = 0;
			}
		}
		for (; a<xylem->size()-1; a++) {
			if (Y[a]) {
				Y[a] = pow(Y[a], 1.0/xylem->at(a)->mu());
				Y[xylem->at(a)->upcell(0)->slot()] += pow(Y[a],xylem->at(a)->upcell(0)->mu());
			}
		}
		
		// Gi
//		// better for wide and shallow trees //
//		for (a=0; a<xylem->n_elemental(); a++) {
//			unsigned child, parent;
//			double* gi;
//			child = a;
//			parent = xylem->at(a)->upcell(0)->slot();
//			gi = &(*Gi)(c,a);
//			*gi = pow(Y[a],(xylem->at(parent)->mu()-1.0));
//			while (xylem->at(parent)->code()!=0) {
//				child = parent;
//				parent = xylem->at(child)->upcell(0)->slot();
//				*gi *= pow(Y[child],(xylem->at(parent)->mu()-1.0));
//			}
//		} 
		
		// Gi
		// better for deep trees //
		a=xylem->size()-1;
		(*Gi)(c,a) = 1;
		for (; a!=0; ) {
			a--;
			mu = xylem->at(a)->upcell(0)->mu();
			(*Gi)(c,a) = pow(Y[a],(1-mu)/mu)*pow(Y[xylem->at(a)->upcell(0)->slot()],(mu-1)/mu);
		}
		
		
		
	}
	
	
}
*/








void elm::case_logit_add_term
(size_t n,
 const double* old_p, // array size [n]
 const double* term2, // array size [n]
 double* new_p,       // array size [n]
 const bool* avail    // array size [n]
 )
{
	double tot = 0.0;
	const double* t1 = old_p;
	const double* t2 = term2;
	double* p = new_p;
	if (avail) {
		const bool* av = avail;
		for (size_t i=0; i<n; i++, t1++, t2++, p++, av++) {
			if (*av) {
				*p = (*t1) * ::exp(*t2);
				tot += *p;
			} else {
				*p = 0.0;
			}
		}
		p = new_p;
		for (size_t i=0; i<n; i++) {
			*p /= tot;
		}
	} else {
		for (size_t i=0; i<n; i++, t1++, t2++, p++) {
			*p = (*t1) * ::exp(*t2);
			tot += *p;
		}
		p = new_p;
		for (size_t i=0; i<n; i++) {
			*p /= tot;
		}
	}
}

void elm::case_logit_add_term_deriv
(size_t nalt,
 size_t npar,
 const double* dOldP_dParam,  //[ nAlts, nParams ]
 const double* dTerm_dParam,  //[ nAlts, nParams ]
 double* dNewP_dParam,        //[ nAlts, nParams ]
 double* workspace,           //[ nParams ]
 const double* OldP,          //[ nAlts ]
 const double* NewP           //[ nAlts ]
 )
{
	if (dTerm_dParam) {
	cblas_dcopy(nalt*npar, dTerm_dParam, 1, dNewP_dParam, 1);
	} else {
		memset(dNewP_dParam, 0, sizeof(double)*nalt*npar);
	}
	if (dOldP_dParam) {
		for (int a=0; a<nalt; a++) {
			cblas_daxpy(npar, 1/OldP[a], dOldP_dParam+(a*npar), 1, dNewP_dParam+(a*npar), 1);
		}
	}
	cblas_dgemv(CblasRowMajor, CblasNoTrans, nalt, npar, 1, dNewP_dParam, npar, NewP, 1, 0, workspace, 1);
	for (int a=0; a<nalt; a++) {
		cblas_daxpy(npar, -1, workspace, 1, dNewP_dParam+(a*npar), 1);
		cblas_dscal(npar, NewP[a], dNewP_dParam+(a*npar), 1);
	}	
}






