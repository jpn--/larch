/*
 *  etk_autoindex.h
 *
 *  Copyright 2007-2017 Jeffrey Newman
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


#ifndef ___ETK_AUTOINDEX__
#define ___ETK_AUTOINDEX__

#include <map>
#include <vector>
#include "etk.h"

namespace etk {

#ifndef SWIG
	
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
	
#endif // ndef SWIG

	class autoindex_string {
		
		std::map<std::string,size_t>	_index;
		std::vector<std::string>        _codex;
		
	public:

		#ifdef SWIG
		%rename(__len__) size;
		%rename(__contains__) has_key;
		#endif // def SWIG


		#ifndef SWIG
		size_t&       operator[] (const std::string& codex) ;
		const size_t& operator[] (const std::string& codex) const ;
		const std::string& operator[] (const size_t& index) const ;
		#endif // ndef SWIG
		
		const std::string& at_index (const size_t& index) const ;
		size_t          size() const	;
		void			clear()			;
		const std::vector<std::string>& strings() const { return _codex; } 
		size_t          drop (const std::string& codex);
		bool            has_key(const std::string& codex) const;
		autoindex_string()	;
		autoindex_string(const std::vector<std::string>& init)	;
		~autoindex_string();

		void extend(const std::vector<std::string>& init)	;
		
		size_t index_from_string(const std::string& codex) ;
		std::string string_from_index(const size_t& index) const ;

		void reorder(const std::vector<std::string>& replacement_list);
		
		#ifdef SWIG
		%pythoncode %{
		def __getitem__(self, key):
			if isinstance(key, int):
				if key<len(self):
					if key>=0:
						return key
					else:
						key = len(self)+key
						if key<0:
							raise IndexError()
						return key
				else:
					raise IndexError()
			elif isinstance(key, str):
				return self.index_from_string(key)
			elif isinstance(key, bytes):
				return self.index_from_string(key.decode('utf8'))
		def __repr__(self):
			return "larch.core.autoindex_string(" + repr(self.strings()) + ")"
		%}
		#endif // def SWIG
	};
	
	


} // end namespace etk




#endif // ___ETK_AUTOINDEX__

