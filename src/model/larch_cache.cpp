/*
 *  larch_cache.cpp
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

#include "larch_cache.h"
#include <iostream>

using namespace elm;

array_compare::array_compare(const double* ptr, const size_t& len, bool make_copy)
: holder ()
, firstpointer (ptr)
, length (len)
{
	if (make_copy) {
		holder.resize(len);
		for (size_t i=0; i<length; i++) {
			holder[i] = ptr[i];
		}
		firstpointer = &(holder[0]);
	}
}

array_compare::array_compare(const std::vector<double>& array)
: holder ()
, firstpointer (&(array[0]))
, length (array.size())
{

}

array_compare::array_compare(const std::vector<double>& array, bool make_copy)
: holder (array)
, firstpointer (&(holder[0]))
, length (holder.size())
{

}

array_compare::array_compare(const array_compare& other)
: holder (other.length)
, firstpointer (&(holder[0]))
, length (other.length)
{
	for (size_t i=0; i<length; i++) {
		holder[i] = other.firstpointer[i];
	}
}



bool array_compare::operator==(const array_compare& other) const
{
	if (length!=other.length) return false;
	for (size_t i=0; i<length; i++) {
		if (firstpointer[i] != other.firstpointer[i]) return false;
	}
	return true;
}

bool array_compare::operator!=(const array_compare& other) const
{
	return ~operator==(other);
}

bool array_compare::operator< (const array_compare& other) const
{
	if (length<other.length) return true;
	if (length>other.length) return false;
	// then lengths the same...
	for (size_t i=0; i<length; i++) {
		if (firstpointer[i] < other.firstpointer[i]) return true;
		if (firstpointer[i] > other.firstpointer[i]) return false;
	}
	// then equal...
	return false;
}


bool array_compare::operator<=(const array_compare& other) const
{
	if (length<other.length) return true;
	if (length>other.length) return false;
	// then lengths the same...
	for (size_t i=0; i<length; i++) {
		if (firstpointer[i] < other.firstpointer[i]) return true;
		if (firstpointer[i] > other.firstpointer[i]) return false;
	}
	// then equal...
	return true;
}


bool array_compare::operator> (const array_compare& other) const
{
	if (length<other.length) return false;
	if (length>other.length) return true;
	// then lengths the same...
	for (size_t i=0; i<length; i++) {
		if (firstpointer[i] < other.firstpointer[i]) return false;
		if (firstpointer[i] > other.firstpointer[i]) return true;
	}
	// then equal...
	return false;
}


bool array_compare::operator>=(const array_compare& other) const
{
	if (length<other.length) return false;
	if (length>other.length) return true;
	// then lengths the same...
	for (size_t i=0; i<length; i++) {
		if (firstpointer[i] < other.firstpointer[i]) return false;
		if (firstpointer[i] > other.firstpointer[i]) return true;
	}
	// then equal...
	return true;
}


void array_compare::print_to_cerr() const
{
	std::cerr <<"array_compare(";
	std::cerr << length << ",";
	for (size_t i=0; i<length; i++) {
		std::cerr << std::hexfloat <<firstpointer[i] << " | ";
	}
	std::cerr <<")\n";
}






result_cache::result_cache()
: my_ll(nullptr)
, my_grad(nullptr)
, my_bhhh(nullptr)
, my_hess(nullptr)
, _ll (NAN)
, _grad()
, _bhhh()
, _hess()
{ }









const result_cache* cache_set::_get_results(const array_compare& key) const
{
	auto iter = _saved_results.find(key);
	if (iter == _saved_results.end()) {
		return nullptr;
	}
	return &iter->second;
}

result_cache* cache_set::_get_results(const array_compare& key)
{
	auto iter = _saved_results.find(key);
	if (iter == _saved_results.end()) {
		return nullptr;
	}
	return &iter->second;
}



bool cache_set::read_cached_loglike(const array_compare& key, double& ll) const
{
	auto x = _get_results(key);
	if (!x) {
		return false;
	}
	
	ll = *x->my_ll;
	return true;
}

bool cache_set::read_cached_grad   (const array_compare& key, etk::ndarray*& grad)
{
	auto x = _get_results(key);
	if (!x) {
		grad = nullptr;
		return false;
	}
	
	grad = x->my_grad;
	return true;
}


bool cache_set::read_cached_bhhh   (const array_compare& key, etk::ndarray*& bhhh)
{
	auto x = _get_results(key);
	if (!x) {
		bhhh = nullptr;
		return false;
	}
	
	bhhh = x->my_bhhh;
	return true;
}


bool cache_set::read_cached_hess   (const array_compare& key, etk::ndarray*& hess)
{
	auto x = _get_results(key);
	if (!x) {
		hess = nullptr;
		return false;
	}
	
	hess = x->my_hess;
	return true;
}



// These functions save the thing
void cache_set::set_cached_loglike(const array_compare& key, const double& ll)
{
	result_cache* x = &_saved_results[key];
	x->_ll = ll;
	x->my_ll = &x->_ll;
}


void cache_set::set_cached_grad   (const array_compare& key, const etk::ndarray& grad)
{
	result_cache* x = &_saved_results[key];
	x->_grad = grad;
	x->my_grad = &x->_grad;
}


void cache_set::set_cached_bhhh   (const array_compare& key, const etk::ndarray& bhhh)
{
	result_cache* x = &_saved_results[key];
	x->_bhhh = bhhh;
	x->my_bhhh = &x->_bhhh;
}


void cache_set::set_cached_hess   (const array_compare& key, const etk::ndarray& hess)
{
	result_cache* x = &_saved_results[key];
	x->_hess = hess;
	x->my_hess = &x->_hess;
}

void cache_set::clear()
{
	_saved_results.clear();
}

