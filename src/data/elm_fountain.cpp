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
: _alternative_names()
, _alternative_codes()
, source_filename()
{

}


elm::Fountain::~Fountain()
{
	// curiously, an abstract base class must have a real destructor.
}






unsigned elm::Fountain::nCases() const
{
	OOPS("fountain is an abstract base class, use a derived class instead");
}

unsigned elm::Fountain::nAlts() const
{
	OOPS("fountain is an abstract base class, use a derived class instead");
}



elm::VAS_dna  elm::Fountain::ask_dna(const long long& c)
{
	OOPS("Fountain is an abstract base class");
}








const elm::VAS_dna  elm::Fountain::ask_dna(const long long& c) const
{
	return const_cast<elm::Fountain*>(this)->ask_dna(c);
}




boosted::shared_ptr< std::vector<std::string> > elm::Fountain::cache_alternative_names()
{
	elm::Fountain* modthis = const_cast<elm::Fountain*>(this);
	
	if (modthis->_alternative_names.expired()) {
    	boosted::shared_ptr< std::vector<std::string> > x = boosted::make_shared< std::vector<std::string> >( alternative_names() );
		modthis->_alternative_names = x;
		return x;
	} else {
		return modthis->_alternative_names.lock();
	}
}


boosted::shared_ptr< std::vector<long long> > elm::Fountain::cache_alternative_codes()
{
	elm::Fountain* modthis = const_cast<elm::Fountain*>(this);
	
	if (modthis->_alternative_codes.expired()) {
    	boosted::shared_ptr< std::vector<long long> > x = boosted::make_shared< std::vector<long long> >( alternative_codes() );
		modthis->_alternative_codes = x;
		return x;
	} else {
		return modthis->_alternative_codes.lock();
	}
}


elm::VAS_System* elm::Fountain::DataDNA(const long long& c)
{
	// In the future, the basic data structure may be allowed to vary based on the case
	return &_Data_DNA;
}


elm::VAS_dna elm::Fountain::alternatives_dna() const
{
	std::vector<std::string> the_names (alternative_names());
	std::vector<elm::cellcode> the_codes (alternative_codes());
	if (the_names.size() != the_codes.size()) OOPS("vector sizes do not match");
	VAS_dna output;
	for (unsigned i=0; i<the_names.size(); i++) {
		output[the_codes[i]] = VAS_dna_info(the_names[i]);
	}
	return output;
}



std::vector<long long> elm::Fountain::_echo_alternative_codes() const
{
	return this->alternative_codes();
}

void elm::Fountain::_director_test()
{
	std::cerr << "elm::Fountain::_director_test():\n";
	
	std::cerr << " nCases="<<nCases()<<"\n";
	std::cerr << " nAlts="<<nAlts()<<"\n";

	std::cerr << "end test.\n";
	
}


