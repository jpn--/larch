/*
 *  elm_Fountain.cpp
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

