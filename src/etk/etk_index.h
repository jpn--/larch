/*
 *  toolbox_index.h
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


#ifndef __TOOLBOX_INDEX__
#define __TOOLBOX_INDEX__

#include <map>
#include <string>

namespace etk {
	
	template <class T>
	class index: public std::map<std::string,T*> {
	public:
		T* operator()(const std::string& lookup) const {
			if (this->find(lookup)==this->end()) return NULL;
			return this->find(lookup)->second;
			// TODO: single instance of "find" instead of two
		}
	};
	
}

#endif // __TOOLBOX_INDEX__
