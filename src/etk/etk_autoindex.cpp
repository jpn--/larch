/*
 *  etk_autoindex.cpp
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


#include "etk_autoindex.h"
#include "etk_random.h"
#include <climits>

using namespace etk;

// INT

unsigned& autoindex_int::operator[] (const int& codex) {
	std::map<int,unsigned>::iterator i;
	i = _index.find(codex);	
	if (_strap) {
		if (_strap->at(i->second) >= _knife) OOPS("autoindex: case is knifed");
	} 
	if (i!=_index.end()) return (i->second);		
	size_t j = _index.size();
	if (_strap) {
		_codex.at(_strap->at(j)) = codex;
		if (_strap->at(j) >= _knife) OOPS("autoindex: case is knifed");
		return (_index.insert( std::pair<int,unsigned>(codex, _strap->at(j)) )).first->second;
	} 
	_codex.push_back(codex);
	return (_index.insert( std::pair<int,size_t> (codex, j) )).first->second;
}

const unsigned int& autoindex_int::operator[] (const int& codex) const {
	std::map<int,unsigned>::const_iterator i = _index.find(codex);		
	if (i==_index.end()) {
		OOPS("autoindex: codex not found in const autoindex");
	}
	if (_strap) {
		if (_strap->at(i->second) >= _knife) OOPS("autoindex: case is knifed");
	} 
	return (i->second);		
}

const int& autoindex_int::at_index (const unsigned& index) const {
	if (index >= _codex.size()) OOPS("autoindex: index out of range");
	return _codex[index];
}

size_t autoindex_int::size() const	
{ 
	return _codex.size(); 
}

void autoindex_int::clear()			
{ 
	_index.clear(); 
	_codex.clear();
	if (_strap) {
		_strap->clear();
		delete _strap;
		_strap =0;
	}
	_knife = UINT_MAX;
}

void autoindex_int::make_strap (const unsigned& n_cases)
{
	clear();
	_strap = new std::vector<unsigned> (n_cases);
	for (unsigned i=0; i<n_cases; i++) (*_strap)[i]=i;
	etk::randomizer R;
	shuffle(*_strap,&R);
}

void autoindex_int::set_knife (const unsigned& n_cases)
{
	_knife = n_cases;
}

void autoindex_int::set_knife (const double& fraction_cases)
{
	if (_strap) _knife = unsigned((double)_strap->size() * fraction_cases);
}

autoindex_int::autoindex_int()		
:	_strap	(0)
,	_knife	(UINT_MAX)
{ }

autoindex_int::~autoindex_int()	
{ 
	clear();
}



// STRING

size_t& autoindex_string::operator[] (const std::string& codex) {
	std::map<std::string,size_t>::iterator i;
	i = _index.find(codex);	
	if (i!=_index.end()) return (i->second);		
	size_t j = _index.size();
	_codex.push_back(codex);
	return (_index.insert( std::pair<std::string,size_t> (codex, j) )).first->second;
}

const size_t& autoindex_string::operator[] (const std::string& codex) const {
	std::map<std::string,size_t>::const_iterator i = _index.find(codex);
	if (i==_index.end()) OOPS("autoindex: codex %s not found in const autoindex",codex);
	return (i->second);		
}

const std::string& autoindex_string::operator[] (const size_t& index) const {
	if (index >= _codex.size()) OOPS("autoindex: index out of range");
	return _codex[index];
}




size_t autoindex_string::index_from_string(const std::string& codex)
{
	std::map<std::string,size_t>::iterator i;
	i = _index.find(codex);	
	if (i!=_index.end()) return (i->second);		
	size_t j = _index.size();
	_codex.push_back(codex);
	return (_index.insert( std::pair<std::string,size_t> (codex, j) )).first->second;
}

std::string autoindex_string::string_from_index(const size_t& index) const
{
	if (index >= _codex.size()) OOPS_IndexError("autoindex: index out of range");
	return _codex[index];
}








const std::string& autoindex_string::at_index (const size_t& index) const {
	if (index >= _codex.size()) OOPS_IndexError("autoindex: index out of range");
	return _codex[index];
}

size_t autoindex_string::size() const	
{ 
	return _codex.size(); 
}

void autoindex_string::clear()			
{ 
	_index.clear(); 
	_codex.clear();
}



size_t autoindex_string::drop (const std::string& codex)
{
	std::map<std::string,size_t>::iterator i;
	i = _index.find(codex);	
	if (i==_index.end()) OOPS_KeyError("key '",codex,"' not found"); // no drop required
	
	size_t dropped = i->second;
	
	strvec::iterator j = _codex.begin() + i->second;
	_codex.erase(j);	
	_index.clear();
	for (size_t k=0; k<_codex.size(); k++) {
		_index[_codex[k]] = k;
	}
	return dropped;
}

bool autoindex_string::has_key(const std::string& codex) const
{
	std::map<std::string,size_t>::const_iterator i;
	i = _index.find(codex);	
	if (i==_index.end()) return false; 
	return true;
}



autoindex_string::autoindex_string()		
{ }

void autoindex_string::extend(const std::vector<std::string>& init)
{
	size_t k = _codex.size();
	_codex.insert(_codex.end(), init.begin(), init.end());
	for (; k<_codex.size(); k++) {
		_index[_codex[k]] = k;
	}
}


autoindex_string::autoindex_string(const std::vector<std::string>& init)
: _codex(init)
{
	for (size_t k=0; k<_codex.size(); k++) {
		_index[_codex[k]] = k;
	}
}


autoindex_string::~autoindex_string()	
{ 
	clear();
}



void autoindex_string::reorder(const std::vector<std::string>& replacement_list)
{
	if (replacement_list.size() != _codex.size()) OOPS_IndexError("can only reorder with same length list");
	clear();
	extend(replacement_list);
}


