/*
 *  etk_autoindex.h
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


#ifndef ___ETK_AUTOINDEX__
#define ___ETK_AUTOINDEX__

#include <map>
#include <vector>
#include "etk.h"

namespace etk {
		
	class autoindex_int {
		
		std::map<int,unsigned>	_index;
		std::vector<int>        _codex;
		std::vector<unsigned>*	_strap;
		unsigned				_knife;
		
	public:
		unsigned&       operator[] (const int& codex) ;
		const unsigned& operator[] (const int& codex) const ;	
		const int&      at_index (const unsigned& index) const ;	
		size_t          size() const	;
		void			clear()			;
		void            make_strap (const unsigned& n_cases);
		void            set_knife (const unsigned& n_cases);
		void            set_knife (const double& fraction_cases);	
		autoindex_int()	;
		~autoindex_int();
	};
	

	class autoindex_string {
		
		std::map<std::string,unsigned>	_index;
		std::vector<std::string>        _codex;
		std::vector<unsigned>*          _strap;
		unsigned				        _knife;
		
	public:
		unsigned&       operator[] (const std::string& codex) ;
		const unsigned& operator[] (const std::string& codex) const ;	
		const std::string& operator[] (const unsigned& index) const ;	
		const std::string& at_index (const unsigned& index) const ;	
		size_t          size() const	;
		void			clear()			;
		void            make_strap (const unsigned& n_cases);
		void            set_knife (const unsigned& n_cases);
		void            set_knife (const double& fraction_cases);
		const std::vector<std::string>& strings() const { return _codex; } 
		bool            drop (const std::string& codex);
		bool            has_key(const std::string& codex) const;
		autoindex_string()	;
		~autoindex_string();
	};
	
	


} // end namespace etk




#endif // ___ETK_AUTOINDEX__

