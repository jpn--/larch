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
//		virtual elm::caseindex ask_caseids();
		virtual elm::VAS_dna   ask_dna(const long long& c=0);
//		virtual elm::datamatrix ask_idco(std::vector<std::string> variables);
//		virtual elm::datamatrix ask_idca(std::vector<std::string> variables);
//		virtual elm::datamatrix ask_choice();
//		virtual elm::datamatrix ask_weight();
//		virtual elm::datamatrix ask_avail();

//		virtual const elm::caseindex ask_caseids() const;
		virtual const elm::VAS_dna   ask_dna(const long long& c=0) const;
//		virtual const elm::datamatrix ask_idco(std::vector<std::string> variables) const;
//		virtual const elm::datamatrix ask_idca(std::vector<std::string> variables) const;
//		virtual const elm::datamatrix ask_choice() const;
//		virtual const elm::datamatrix ask_weight() const;
//		virtual const elm::datamatrix ask_avail() const;

		virtual const unsigned& nCases() const;
		virtual const unsigned& nAlts() const;
	
		Fountain();
		virtual ~Fountain();
	};
	
};




#endif /* defined(__Hangman__elm_fountain__) */
