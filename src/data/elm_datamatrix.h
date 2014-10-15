/*
 *  elm_data.h
 *
 *  Copyright 2007-2013 Jeffrey Newman
 *
 *  This file is part of ELM.
 *  
 *  ELM is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  ELM is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with ELM.  If not, see <http://www.gnu.org/licenses/>.
 *  
 */


#ifndef __ELM2_DATAMATRIX_H__
#define __ELM2_DATAMATRIX_H__





#ifdef SWIG

//%feature("ref")   datamatrix_t "$this->establish();"
//%feature("unref") datamatrix_t "$this->release();"




%{

	// In SWIG, these headers are available to the c++ wrapper,
	// but are not themselves wrapped

	#include "elm_sql_facet.h"
	#include "etk_thread.h"
	#include "elm_sql_scrape.h"
	#include "elm_caseindex.h"
	#include "elm_datamatrix.h"

%}
#endif // SWIG

#ifndef SWIG
#include "etk.h"
#include "elm_caseindex.h"
#endif // ndef SWIG

namespace elm {


	#ifndef SWIG
	class Facet;
	class datamatrix_t;
	typedef boosted::shared_ptr<datamatrix_t> datamatrix;
	typedef boosted::weak_ptr<datamatrix_t>   datamatrix_;
	#endif // ndef SWIG

	enum dimensionality {
		case_var = 0x1,
		case_alt_var = 0x2,
	};

	enum matrix_dtype {
		mtrx_bool,
		mtrx_double,
		mtrx_int64,
	};

	enum matrix_purpose {
		purp_vars,
		purp_choice,
		purp_avail,
		purp_weight,
	};

	class datamatrix_t
	{

	#ifndef SWIG
	public:
		matrix_dtype    dtype;
		dimensionality  dimty;
		matrix_purpose  dpurp;

	protected:
		datamatrix_ myself;

		// The datamatrix_t provides the data to the core program. 
		// There can be more than one for a single model,
		//  as well as more than one to a single data_set.
		
	protected:
		etk::strvec   _VarNames;

	public:
		datamatrix_t(dimensionality dim, matrix_dtype tp, matrix_purpose purp);
		// This constructor initializes the datamatrix_t object.
		//  The constructor should only be called from the create method
		//  if created independently, will not have smart pointer functionality.
		
		datamatrix_t(const datamatrix_t&);
		//copy constructor is public
		
		datamatrix_t();
		//default constructor is public, but raises an exception
		
	public:
		virtual ~datamatrix_t();

	#endif // ndef SWIG

		

	public:
		datamatrix           pointer();
		static datamatrix    create(dimensionality dim, matrix_dtype tp, matrix_purpose purp=purp_vars);
		datamatrix           copy() const;


		long refcount();

	public:
		void set_variables( const std::vector<std::string>& varnames );
		// Takes a vector of strings naming the variables to use in providing the
		//  This replaces any previously provided
		//  vector of names. If any item from the input vector does not match an
		//  item in the set of available variables, throw an elmdata_exception.

		const std::vector<std::string>& get_variables() const;
		// Returns a vector of strings, indicating the names of variables currently
		//  being used from this port.
		//  This provides a check of what was input through the use_variables 
		//  functions.
				
		
		//// MARK: PROVIDING DATA //////////////////////////////////////////////////
		//
		//  Data can be provided in one of two frameworks: either the entirety of
		//  the data is loaded into memory simultaneously (Contiguous Data
		//  Framework), or the data for each case is loaded dynamically as called
		//  for (Dynamic Data Framework), either from disk or an in-memory
		//  database. This may be used when there is not enough memory available
		//  to do it all in one chunk.  At least one of these two frameworks must
		//  be available for any dataset, but it is not necessary to have both.
	public:		
		void tearDown(bool force=false);
		// This function is where memory is freed.
		//  If there is a data problem, throw an elmdata_exception (unlikely here).
			
	#ifndef SWIG
	
		etk::ptr_lockout<const double> values(const unsigned& firstcasenum=0, const size_t& numberofcases=0);
		etk::ptr_lockout<const double> values(const unsigned& firstcasenum=0, const size_t& numberofcases=0) const;
		etk::ptr_lockout<const bool> boolvalues(const unsigned& firstcasenum=0, const size_t& numberofcases=0);
		etk::ptr_lockout<const bool> boolvalues(const unsigned& firstcasenum=0, const size_t& numberofcases=0) const;
		// Returns a pointer to memory where the entire block of data is stored 
		//  in contiguous memory. The data is in (case,alt,var) three dimensional
		//  matrix in row major format. This memory space should remain constant
		//  and available to the program until tearDown is called.
		//  firstcasenum determines which case to begin with.
		//  numberofcases determines how many cases to return; 0 means all.
		//  For idca data, the total size of this array should be 
		//   [nCases * nAlts * nVars] or [numberofcases * nAlts * nVars].
		//  For idco data, the total size of this array should be 
		//   [nCases * nVars] or [numberofcases * nVars].
			
		std::string printcase(const unsigned& r) const;
		std::string printcases(unsigned rstart, const unsigned& rfinish) const;
		std::string printboolcase(const unsigned& r) const;
		std::string printboolcases(unsigned rstart, const unsigned& rfinish) const;
		// This method prints a human-readable string output of the case data.
		
		//// MARK: Applying Data ///////////////////////////////////////////////////

	public:
		etk::memarray _repository;
	private:
		elm::caseindex_t* _casedex;

	public:
		etk::readlock _repo_lock;

//	public:
//		static datamatrix create_idca(elm::Facet* db, const std::vector<std::string>& varnames);
//		static datamatrix create_idco(elm::Facet* db, const std::vector<std::string>& varnames);
//		static datamatrix create_choo(elm::Facet* db);
//		static datamatrix create_wght(elm::Facet* db);
//		static datamatrix create_aval(elm::Facet* db);

	#endif // ndef SWIG

	public:
		static datamatrix read_idca(elm::Facet* db, const std::vector<std::string>& varnames, long long* caseid=nullptr);
		static datamatrix read_idco(elm::Facet* db, const std::vector<std::string>& varnames, long long* caseid=nullptr);
		static datamatrix read_choo(elm::Facet* db, long long* caseid=nullptr);
		static datamatrix read_wght(elm::Facet* db, long long* caseid=nullptr);
		static datamatrix read_aval(elm::Facet* db, long long* caseid=nullptr);
		
	protected:
		void read_from_facet(elm::Facet* db, int style, const std::vector<std::string>* varnames=nullptr, long long* caseid=nullptr);
		
	public:
		double value    (const unsigned& c, const unsigned& a, const unsigned& v) const;
		double value    (const unsigned& c, const unsigned& v) const;
		bool   boolvalue(const unsigned& c, const unsigned& a, const unsigned& v) const;
		bool   boolvalue(const unsigned& c, const unsigned& v) const;

	#ifndef SWIG

	public:
		void ExportData (double* ExportTo, const unsigned& c, const unsigned& a, const unsigned& numberOfAlts) const;
		void ExportData	(double* ExportTo, const unsigned& c, const unsigned& a, const double& scale, const unsigned& numberOfAlts) const;
		void OverlayData(double* ExportTo, const unsigned& c, const unsigned& a, const double& scale, const unsigned& numberOfAlts=0) const;

	#endif // ndef SWIG
		
		size_t nCases() const;
		size_t nAlts() const;
		size_t nVars() const;
		
		PyObject* getArray();
		
		std::string __str__() const;
		std::string __repr__() const;
	};











	
} // end namespace elm
#endif // __ELM2_SQL_SCRAPE_H__

