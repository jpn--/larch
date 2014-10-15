//
//  elm_fountain.h
//  Hangman
//
//  Created by Jeffrey Newman on 4/16/14.
//  Copyright (c) 2014 Jeffrey Newman. All rights reserved.
//

#ifndef __Hangman__elm_fountain__
#define __Hangman__elm_fountain__

#include <vector>

#include "elm_caseindex.h"
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
		virtual elm::caseindex ask_caseids();
		virtual elm::VAS_dna   ask_dna(const long long& c=0);
		virtual elm::datamatrix ask_idco(std::vector<std::string> variables);
		virtual elm::datamatrix ask_idca(std::vector<std::string> variables);
		virtual elm::datamatrix ask_choice();
		virtual elm::datamatrix ask_weight();
		virtual elm::datamatrix ask_avail();

		virtual const elm::caseindex ask_caseids() const;
		virtual const elm::VAS_dna   ask_dna(const long long& c=0) const;
		virtual const elm::datamatrix ask_idco(std::vector<std::string> variables) const;
		virtual const elm::datamatrix ask_idca(std::vector<std::string> variables) const;
		virtual const elm::datamatrix ask_choice() const;
		virtual const elm::datamatrix ask_weight() const;
		virtual const elm::datamatrix ask_avail() const;

		virtual const unsigned& nCases() const;
		virtual const unsigned& nAlts() const;
	
		Fountain();
		virtual ~Fountain();
	};
	
};




#endif /* defined(__Hangman__elm_fountain__) */
