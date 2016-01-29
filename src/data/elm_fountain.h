/*
 *  elm_Fountain.h
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

		virtual unsigned nCases() const ;
		virtual unsigned nAlts() const  ;

#ifndef SWIG

	  protected:
		boosted::weak_ptr< std::vector<std::string> >  _alternative_names;
		boosted::weak_ptr< std::vector<long long>   >  _alternative_codes;

	  public:
		boosted::shared_ptr< std::vector<std::string> > cache_alternative_names();
		boosted::shared_ptr< std::vector<long long>   > cache_alternative_codes();

#endif // ndef SWIG

	  public:
		virtual std::vector<std::string>    alternative_names() const =0;
		virtual std::vector<long long>      alternative_codes() const =0;
		virtual std::string    alternative_name(long long) const      =0;
		virtual long long      alternative_code(std::string) const    =0;

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
		
		std::vector<long long>      _echo_alternative_codes() const;
		void _director_test();
	};
	
};




#endif /* defined(__Hangman__elm_fountain__) */
