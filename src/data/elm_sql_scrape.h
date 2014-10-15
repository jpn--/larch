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


#ifndef __ELM2_SQL_SCRAPE_H__
#define __ELM2_SQL_SCRAPE_H__


#define IDCO   0x01
#define IDCA   0x02
#define CHOO   0x10
#define WGHT   0x20
#define AVAL   0x40

#ifdef SWIG
%{

	// In SWIG, these headers are available to the c++ wrapper,
	// but are not themselves wrapped

	#include "elm_sql_facet.h"
	#include "etk_thread.h"
	#include "elm_sql_scrape.h"

%}
#endif // SWIG

#ifndef SWIG
#include "elm_sql_facet.h"
#endif // ndef SWIG

namespace elm {

	class Scrape;
	typedef boosted::shared_ptr<Scrape> ScrapePtr;
	typedef boosted::weak_ptr<Scrape>   ScrapePtr_weak;

	class Scrape {

	#ifndef SWIG
		friend class Facet;

	protected:
		ScrapePtr_weak   myself;
	

		// The Scrape provides the data to the core program. 
		// There can be more than one for a single model,
		//  as well as more than one to a single data_set.
	public:
		Facet* parent;
		// This provides a pointer back to the Facet that this connection will
		//  draw from.
		
	protected:
		Scrape(Facet* parent, int style);
		// This constructor links the Scrape to the Facet.
		//  The constructor should only be called from the create method
		//  if created independently, will not have smart pointer functionality.

	public:
		~Scrape();
		
	protected:
		etk::strvec   _VarNames;
		size_t		  _nVars_;
		int           _style;
		SQLiteStmtPtr _stmt;

	public:
		ScrapePtr        pointer();
		static ScrapePtr create(Facet* parent, int style);
		ScrapePtr        copy() const;

	private:
		bool _as_bool() const;
	#endif // ndef SWIG

	
	
	public:
		Scrape(const Scrape&); //copy constructor is public
		Scrape(); //default constructor is public, but raises an exception

	public:
		static Scrape create_idca(Facet* parent);
		static Scrape create_idco(Facet* parent);
		static Scrape create_choo(Facet* parent);
		static Scrape create_wght(Facet* parent);
		static Scrape create_aval(Facet* parent);
		// These static functions create Scrapes that are managed by python,
		// and not by c++ smart pointers.
		
	public:
		int style() const { return _style; }
		// This is an indicator of the data style. 
				

	public:
		void use_variables( const std::vector<std::string>& varnames );
		// Takes a vector of strings naming the variables to use in providing the
		//  This replaces any previously provided
		//  vector of names. If any item from the input vector does not match an
		//  item in the set of available variables, throw an elmdata_exception.

		size_t add_var( const std::string& varname );
		// Takes a string naming a variable to use in providing the 
		//  data. This adds to any previously provided vector of names, if the 
		//  variable was not already in the list. If the input
		//  does not match an item in the set of available variables, throw an
		//  elmdata_exception.
		//  Returns the position of the newly added variable name in the used
		//  variables list.
		
		std::vector<std::string> using_variables() const;
		// Returns a vector of strings, indicating the names of variables currently
		//  being used from this port.
		//  This provides a check of what was input through the use_variables 
		//  functions.
		
		size_t nVars() const;
		// Returns the number of variables currently in use.
		
		
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
		etk::ndarray_bool _bools;
	private:
		size_t _firstcasenum;
		size_t _numberofcases;

	public:
		etk::readlock _repo_lock;
		etk::readlock _bool_lock;

	#endif // ndef SWIG

	public:
		void load_values(const size_t& firstcasenum=0, const size_t& numberofcases=0);
		bool fully_loaded(bool boolean=false) const;
		bool is_loaded_in_range(const size_t& firstcasenum, const size_t& numberofcases) const;
		std::string describe_loaded_range() const;

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
		
		const unsigned& nCases() const;
		// Simple pass-through
		
		const unsigned& nAlts(const long long& c=0) const;
		// Passes through the nAlts, 

		PyObject* getArray();
		PyObject* getBoolArray();
	};











	
} // end namespace elm
#endif // __ELM2_SQL_SCRAPE_H__

