/*
 *  etk_basic.h
 *
 *  Copyright 2007-2013 Jeffrey Newman
 *
 *  This file is part of ELM.
 *  
 *  ELM is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  ELM is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with ELM.  If not, see <http://www.gnu.org/licenses/>.
 *  
 */


#ifndef __ETK_PYDICT__
#define __ETK_PYDICT__

#include "etk_python.h"
#include <map>


namespace etk {
	
	class dictionary_sd {
		std::string	_key;
		double		_value;
		double		_value_orig;
		PyObject*	_pyo;
		
	public:
		dictionary_sd(PyObject* dict);
		~dictionary_sd();
		double& operator[](const std::string& key);
		double& key(const std::string& key);
	};
		
	
	
} // end namespace etk


#endif // __ETK_PYDICT__

