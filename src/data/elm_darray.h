/*
 *  elm_data.h
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


#ifndef __ELM2_DARRAY_H__
#define __ELM2_DARRAY_H__





#ifdef SWIG
%{

	// In SWIG, these headers are available to the c++ wrapper,
	// but are not themselves wrapped

	#include "elm_sql_facet.h"
	#include "etk_thread.h"
	#include "elm_darray.h"

%}
#endif // SWIG

#ifndef SWIG
#include "etk.h"
#endif // ndef SWIG

#ifdef SWIG
%{
namespace elm {
	class darray;
};
%}
#endif // SWIG

namespace elm {

	#ifndef SWIG
	class darray;
	#endif // ndef SWIG

	class darray_req
	{
		
	public:
		int       dtype;
		int       dimty;
		int	      n_alts;
		bool      contig;
		
	protected:
		etk::strvec variables;
		std::map< std::string, size_t > variables_map;

	public:
		darray_req(int dim, int dtype, int nalts=0);
		// This constructor initializes the datamatrix_t object.
		//  The constructor should only be called from the create method
		//  if created independently, will not have smart pointer functionality.
		
		darray_req(const darray_req&);
		//copy constructor is public
		
		darray_req();
		//default constructor is public, but raises an exception
		
	public:
		virtual ~darray_req();



	public:
		virtual size_t      nVars()    const;
		virtual size_t      nAlts()    const;

		const std::vector<std::string>& get_variables() const;
		void                            set_variables(const std::vector<std::string>& v);
		
		virtual std::string __str__()  const;
		virtual std::string __repr__() const;
		
		int satisfied_by(const elm::darray* x) const;
	};



	#ifndef SWIG

	class darray:
	public darray_req
	{


	public:
		darray(PyObject* source_arr);
		// This constructor initializes the ldarray object.
		//  The constructor should only be called from the create method
		//  if created independently, will not have smart pointer functionality.
		
		darray(const darray&);
		//copy constructor is public

		darray(const darray&, double scale);
		//copy constructor with rescaling is public
		
		darray();
		//default constructor is public, but raises an exception
		
	public:
		virtual ~darray();

		
		//// MARK: PROVIDING DATA //////////////////////////////////////////////////
		//
		//  Data can be provided in one of two frameworks: either the entirety of
		//  the data is loaded into memory simultaneously (Contiguous Data
		//  Framework), or the data for each case is loaded dynamically as called
		//  for (Dynamic Data Framework), either from disk or an in-memory
		//  database. This may be used when there is not enough memory available
		//  to do it all in one chunk.  At least one of these two frameworks must
		//  be available for any dataset, but it is not necessary to have both.
		
	
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

	public:
		etk::readlock _repo_lock;


		
	public:
		double value    (const unsigned& c, const unsigned& a, const unsigned& v) const;
		double value    (const unsigned& c, const unsigned& v) const;
		bool   boolvalue(const unsigned& c, const unsigned& a, const unsigned& v) const;
		bool   boolvalue(const unsigned& c, const unsigned& v) const;

		double&    value_double   (const size_t& c, const size_t& a, const size_t& v);
		double&    value_double   (const size_t& c, const size_t& v);
		long long& value_int64    (const size_t& c, const size_t& a, const size_t& v);
		long long& value_int64    (const size_t& c, const size_t& v);
		bool&      value_bool     (const size_t& c, const size_t& a, const size_t& v);
		bool&      value_bool     (const size_t& c, const size_t& v);

	public:
		void ExportData (double* ExportTo, const unsigned& c, const unsigned& a, const unsigned& numberOfAlts) const;
		void ExportData	(double* ExportTo, const unsigned& c, const unsigned& a, const double& scale, const unsigned& numberOfAlts) const;
		void OverlayData(double* ExportTo, const unsigned& c, const unsigned& a, const double& scale, const unsigned& numberOfAlts=0) const;

		
		size_t nCases() const;
		virtual size_t      nVars()    const;
		
		PyObject* get_array();
		
		virtual std::string __str__() const;
		virtual std::string __repr__() const;
	};



	
	typedef boosted::shared_ptr<const elm::darray> darray_ptr;


	#endif // ndef SWIG









	std::string check_darray(const elm::darray* x);



	
} // end namespace elm
#endif // __ELM2_SQL_SCRAPE_H__

