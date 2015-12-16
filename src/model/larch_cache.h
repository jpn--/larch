/*
 *  larch_cache.h
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

#ifndef __LARCH_CACHE_H__
#define __LARCH_CACHE_H__

#include <vector>
#include <map>
#include "etk_ndarray.h"

namespace elm {

	class array_compare {
		std::vector<double> holder;
		const double*       firstpointer;
		size_t              length;
		
	public:
		array_compare(const double* ptr, const size_t& len, bool make_copy=false);
		array_compare(const std::vector<double>& array);
		array_compare(const std::vector<double>& array, bool make_copy);
		array_compare(const array_compare& other);
	
		bool operator==(const array_compare& other) const;
		bool operator!=(const array_compare& other) const;
		bool operator< (const array_compare& other) const;
		bool operator<=(const array_compare& other) const;
		bool operator> (const array_compare& other) const;
		bool operator>=(const array_compare& other) const;
		
		void print_to_cerr() const;
	};


	class result_cache {
	public:
		double*       my_ll;
		etk::ndarray* my_grad;
		etk::ndarray* my_bhhh;
		etk::ndarray* my_hess;

		double        _ll;
		etk::ndarray  _grad;
		etk::ndarray  _bhhh;
		etk::ndarray  _hess;
		
		result_cache();
		
		
	};


	class cache_set
	{
		
		std::map< array_compare, result_cache > _saved_results;
		
		const result_cache* _get_results(const array_compare& key) const;
		result_cache* _get_results(const array_compare& key);
		
	public:
		// these functions return true if the thing is saved,
		// and put a pointer to the thing into the 2nd argument.
		bool read_cached_loglike(const array_compare& key, double& ll) const;
		bool read_cached_grad   (const array_compare& key, etk::ndarray*& grad) ;
		bool read_cached_bhhh   (const array_compare& key, etk::ndarray*& bhhh) ;
		bool read_cached_hess   (const array_compare& key, etk::ndarray*& hess) ;
		
		// These functions save the thing
		void set_cached_loglike(const array_compare& key, const double& ll);
		void set_cached_grad   (const array_compare& key, const etk::ndarray& grad);
		void set_cached_bhhh   (const array_compare& key, const etk::ndarray& bhhh);
		void set_cached_hess   (const array_compare& key, const etk::ndarray& hess);
		
		void clear();
	};




	
};




#endif // __LARCH_CACHE_H__