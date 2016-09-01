/*
 *  elm_Fountain.h
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

#ifndef __Hangman__elm_fountain__
#define __Hangman__elm_fountain__

#include <vector>

#include "elm_datamatrix.h"
#include "elm_vascular.h"

namespace elm {

#ifndef SWIG
	class caseindex_t;
	class datamatrix_t;
	class VAS_System;
#endif // ndef SWIG
	
	class Fountain
	{
	
	  public:
		virtual elm::VAS_dna   ask_dna(const long long& c=0);
		virtual const elm::VAS_dna   ask_dna(const long long& c=0) const;

		#ifdef SWIG
		%feature("docstring") nCases "The number of cases currently active in this Fountain."
		%feature("docstring") nAlts "The number of alternatives currently active in this Fountain."
		#endif // def SWIG

		virtual unsigned nCases() const ;
		virtual unsigned nAlts() const  ;

#ifndef SWIG

	  protected:
		mutable boosted::weak_ptr< const std::vector<std::string> >  _alternative_names_cached;
		mutable boosted::weak_ptr< const std::vector<long long>   >  _alternative_codes_cached;

	  public:
		boosted::shared_ptr< const std::vector<std::string> > cache_alternative_names() const ;
		boosted::shared_ptr< const std::vector<long long>   > cache_alternative_codes() const ;

		size_t alternative_slot_from_name(const std::string&) const;
		size_t alternative_slot_from_code(const long long&) const;
		
		int alternative_code_from_name(const std::string&, long long&) const;
		int alternative_name_from_code(const long long&, std::string&) const;

#endif // ndef SWIG

	  public:
		
		#ifdef SWIG
		%feature("docstring") alternative_names "A vector of the alternative names used by this Fountain."
		%feature("docstring") alternative_codes "A vector of the alternative codes (64 bit integers) used by this Fountain."
		%feature("docstring") alternative_name "Given an alternative code, return the name."
		%feature("docstring") alternative_code "Given an alternative name, return the code."
		#endif // def SWIG
		
		virtual std::vector<std::string>    alternative_names() const =0;
		virtual std::vector<long long>      alternative_codes() const =0;
		virtual std::string    alternative_name(long long) const      =0;
		virtual long long      alternative_code(std::string) const    =0;

		void uncache_alternatives();
		void cache_alternatives();

		virtual bool check_ca(const std::string& column) const =0;
		virtual bool check_co(const std::string& column) const =0;

		virtual std::vector<std::string> variables_ca() const =0;
		virtual std::vector<std::string> variables_co() const =0;
	
		Fountain();
		virtual ~Fountain();

		std::string source_filename;

	  protected:
		elm::VAS_System  _Data_DNA;
	  public:
		elm::VAS_System* DataDNA(const long long& c=0);
		elm::VAS_dna alternatives_dna() const;
		
		void _refresh_dna(const std::vector<std::string>& a_names, const std::vector<long long>& a_codes);

	};

	#ifdef SWIG
	%include "elm_fountain.i"
	#endif // def SWIG
		
	
};




#endif /* defined(__Hangman__elm_fountain__) */
