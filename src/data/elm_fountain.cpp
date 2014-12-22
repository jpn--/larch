//
//  elm_Fountain.cpp
//  Hangman
//
//  Created by Jeffrey Newman on 4/16/14.
//  Copyright (c) 2014 Jeffrey Newman. All rights reserved.
//

#include "elm_caseindex.h"
#include "elm_datamatrix.h"
#include "elm_fountain.h"

#include "etk.h"

elm::Fountain::Fountain()
{

}


elm::Fountain::~Fountain()
{
	// curiously, an abstract base class must have a real destructor.
}







//elm::caseindex elm::Fountain::ask_caseids()
//{
//	OOPS("Fountain is an abstract base class");
//}

elm::VAS_dna  elm::Fountain::ask_dna(const long long& c)
{
	OOPS("Fountain is an abstract base class");
}

/*
elm::datamatrix elm::Fountain::ask_idco(std::vector<std::string> variables)
{
	OOPS("Fountain is an abstract base class");
}

elm::datamatrix elm::Fountain::ask_idca(std::vector<std::string> variables)
{
	OOPS("Fountain is an abstract base class");
}

elm::datamatrix elm::Fountain::ask_choice()
{
	OOPS("Fountain is an abstract base class");
}

elm::datamatrix elm::Fountain::ask_weight()
{
	OOPS("Fountain is an abstract base class");
}

elm::datamatrix elm::Fountain::ask_avail()
{
	OOPS("Fountain is an abstract base class");
}
*/


const unsigned& elm::Fountain::nCases() const
{
	OOPS("Fountain is an abstract base class");
}

const unsigned& elm::Fountain::nAlts() const
{
	OOPS("Fountain is an abstract base class");
}







//const elm::caseindex elm::Fountain::ask_caseids() const
//{
//	return const_cast<elm::Fountain*>(this)->ask_caseids();
//}

const elm::VAS_dna  elm::Fountain::ask_dna(const long long& c) const
{
	return const_cast<elm::Fountain*>(this)->ask_dna(c);
}

/*
const elm::datamatrix elm::Fountain::ask_idco(std::vector<std::string> variables) const
{
	return const_cast<elm::Fountain*>(this)->ask_idco(variables);
}

const elm::datamatrix elm::Fountain::ask_idca(std::vector<std::string> variables) const
{
	return const_cast<elm::Fountain*>(this)->ask_idca(variables);
}

const elm::datamatrix elm::Fountain::ask_choice() const
{
	return const_cast<elm::Fountain*>(this)->ask_choice();
}

const elm::datamatrix elm::Fountain::ask_weight() const
{
	return const_cast<elm::Fountain*>(this)->ask_weight();
}

const elm::datamatrix elm::Fountain::ask_avail() const
{
	return const_cast<elm::Fountain*>(this)->ask_avail();
}
*/

