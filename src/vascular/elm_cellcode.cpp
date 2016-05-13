/*
 *  elm_cellcode.cpp
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

#include "etk.h"
#include "elm_cellcode.h"
#include <vector>
#include <iostream>
using namespace std;

elm::cellcodeset::cellcodeset (const cellcodevec& iv)
{
	for (unsigned i=0; i<iv.size(); i++) {
		_codes->insert(iv[i]);
	}
}

elm::cellcodeset::cellcodeset (const elm::cellcode& i)
{
	_codes->insert(i);
}

elm::cellcodeset::cellcodeset ()
: _codes(boosted::make_shared< std::set<elm::cellcode> >())
{

}

elm::cellcodeset::cellcodeset (const elm::cellcodeset& other)
: _codes(boosted::make_shared<std::set<cellcode> >())
{
	_codes->insert(other._codes->begin(),other._codes->end());
}

void elm::cellcodeset::append (const elm::cellcode& i)
{
	_codes->insert(i);
}

bool elm::cellcodeset::contains (const elm::cellcode& j) const
{
	std::set<cellcode>::iterator k = _codes->find(j);
	if (k==_codes->end()) return false;
	return true;
}

void elm::cellcodeset::insert_set (const elm::cellcodeset& i)
{
	_codes->insert(i._codes->begin(),i._codes->end());
}

bool elm::cellcodeset::remove(const elm::cellcode& i)
{
	std::set<cellcode>::iterator k = _codes->find(i);
	if (k==_codes->end()) return false;
	_codes->erase(k);
	return true;
}


elm::cellcodevec elm::cellcodevec_from_uintvec (const vector<unsigned int>& x)
{
	cellcodevec c (x.size());
	for (unsigned i=0; i<x.size(); i++) {
		c[i] = x[i];
	}
	return c;
}

etk::strvec splice_string (string block, const char sep)
{
	etk::strvec temp;
	string::size_type loc = block.find_first_of(sep);
	while (loc != string::npos) {
		temp.push_back(block.substr(0,loc));
		block = block.substr(++loc);
		loc = block.find_first_of(sep);
	}
	if (!block.empty())
		temp.push_back(block);
	return temp;
}


#define iswhitespace(q) ((q==' ')||(q=='\t')||(q=='\n'))

elm::cellcodevec elm::read_cellcodevec (const string& s)
{
    cellcodevec result;
	std::string temp;
	etk::strvec temps;
    istringstream is (s);
    while (is) {
        while (is && iswhitespace(is.peek())) is.ignore();
        if (is) {
			is >> temp;
			temps = splice_string(temp,',');
			for (unsigned j=0; j<temps.size(); j++) result.push_back(cellcode_from_string(temps[j]));
        }
    }
    return result;
}

long long elm::longlong_from_string(const std::string& s)
{
    istringstream is (s);
	unsigned long long temp (-1);
	is >> temp;
	return temp;
}

elm::cellcode elm::max_cellcode()
{
	return LLONG_MAX;
}


elm::cellcodeset_iterator::cellcodeset_iterator(elm::cellcodeset* p)
: it(p->_codes->begin())
, ender(p->_codes->end())
, parent(p->_codes)
{
}


elm::cellcodeset_iterator elm::cellcodeset_iterator::__iter__()
{
	return *this;
}

elm::cellcode elm::cellcodeset_iterator::next()
{
	if (it==ender) { PYTHON_STOP_ITERATION; }
	elm::cellcode i = *it;
	it++;
	return i;
}

elm::cellcode elm::cellcodeset_iterator::__next__()
{
	return next();
}

elm::cellcodeset_iterator elm::cellcodeset::__iter__()
{
	return elm::cellcodeset_iterator(this);
}

int elm::cellcodeset::__len__() const
{
	return _codes->size();
}

std::string elm::cellcodeset::__repr__() const
{
	std::ostringstream x;
	x << "cellcodeset(";
	std::set<cellcode>::iterator i=_codes->begin();
	if (i!=_codes->end()) {
		x << *i;
		i++;
	}
	while (i!=_codes->end()) {
		x << "," << *i++;
	}
	x << ")";
	return x.str();
}


elm::cellcodeset& elm::cellcodeset::operator+=(const elm::cellcodeset& i)
{
	insert_set(i);
	return *this;
}
elm::cellcodeset& elm::cellcodeset::operator-=(const elm::cellcodeset& i)
{
	for (elm::cellcode_set_citer j=i.begin(); j!=i.end(); j++) erase(*j);
	return *this;
}

elm::cellcodeset& elm::cellcodeset::operator+=(const elm::cellcode& i)
{
	insert(i);
	return *this;
}

elm::cellcodeset& elm::cellcodeset::operator-=(const elm::cellcode& i)
{
	erase(i);
	return *this;
}


