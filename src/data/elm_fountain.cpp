/*
 *  elm_Fountain.cpp
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

#include "elm_datamatrix.h"
#include "elm_fountain.h"

#include "etk.h"

elm::Fountain::Fountain()
: _alternative_names_cached()
, _alternative_codes_cached()
, source_filename()
{

}


elm::Fountain::~Fountain()
{
	// curiously, an abstract base class must have a real destructor.
}



void elm::Fountain::_refresh_dna(const std::vector<std::string>& a_names, const std::vector<long long>& a_codes)
{

	size_t n_a = a_codes.size();
	
	_Data_DNA.clear();
	for (size_t i=0; i<n_a; i++) {
		_Data_DNA.add_cell(a_codes[i],a_names[i]);
	}
	_Data_DNA.regrow();
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


void elm::Fountain::uncache_alternatives()
{
	_alternative_names_cached.reset();
	_alternative_codes_cached.reset();
}



boosted::shared_ptr< const std::vector<std::string> > elm::Fountain::cache_alternative_names() const
{
	
	if (_alternative_names_cached.expired()) {
    	boosted::shared_ptr< const std::vector<std::string> > x = boosted::make_shared< const std::vector<std::string> >( alternative_names() );
		_alternative_names_cached = x;
		return x;
	} else {
		return _alternative_names_cached.lock();
	}
}


boosted::shared_ptr< const std::vector<long long> > elm::Fountain::cache_alternative_codes() const
{
	
	if (_alternative_codes_cached.expired()) {
    	boosted::shared_ptr< const std::vector<long long> > x = boosted::make_shared< const std::vector<long long> >( alternative_codes() );
		_alternative_codes_cached = x;
		return x;
	} else {
		return _alternative_codes_cached.lock();
	}
}



size_t elm::Fountain::alternative_slot_from_name(const std::string& node_name) const
{
	elm::Fountain* modthis = const_cast<elm::Fountain*>(this);
	return etk::find_first(node_name, *modthis->cache_alternative_names());
}

size_t elm::Fountain::alternative_slot_from_code(const long long& node_code) const
{
	elm::Fountain* modthis = const_cast<elm::Fountain*>(this);
	return etk::find_first(node_code, *modthis->cache_alternative_codes());
}

int elm::Fountain::alternative_code_from_name(const std::string& node_name, long long& node_code) const
{
	elm::Fountain* modthis = const_cast<elm::Fountain*>(this);
	size_t k = etk::find_first(node_name, *modthis->cache_alternative_names());
	if (k != SIZE_T_MAX) {
		node_code = (*modthis->cache_alternative_codes())[k];
		return 1;
	}
	return 0;
}

int elm::Fountain::alternative_name_from_code(const long long& node_code, std::string& output) const
{
	elm::Fountain* modthis = const_cast<elm::Fountain*>(this);
	size_t k =  etk::find_first(node_code, *modthis->cache_alternative_codes());
	if (k != SIZE_T_MAX) {
		output = (*modthis->cache_alternative_names())[k];
		return 1;
	}
	return 0;
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




