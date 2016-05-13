/*
 *  etk_basic.h
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


#ifndef __et_basic__
#define __et_basic__

#ifndef NO_PYTHON_LOGGING
// et_logger.h includes Python.h, which must be first if it is included.
//#  include "et_basic_logger.h"
#endif

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#include <string>
#include <sstream>
#include <vector>
#include <set>
#include <map>



#include "etk_memory.h"
#include "etk_math.h"

#include "etk_exception.h"
#include "etk_vectors.h"
#include "etk_cat.h"

// CONSTANTS //
#define INF (-log(0.0))



namespace etk {
	
	typedef std::vector<int>          intvec;
	typedef std::vector<intvec>       intvecs;
	
	typedef std::vector<unsigned int> uintvec;
	typedef std::vector<uintvec>      uintvecs;
	typedef std::set<unsigned>        uintset;
	
	typedef std::vector<strvec>       strvecs;
	std::string fuse_strvec (const strvec& elements, const std::string sep);
	strvec alias(const strvec& elements, const std::string& replaceMe, const std::string& withMe);
		
//	typedef	std::vector<double>       dblvec; // see etk_vector.h
	typedef std::vector<dblvec>       dblvecs;
		
	typedef std::map<int,int>                 ii_map;
	typedef std::map<std::string,std::string> ss_map;
	
	typedef std::pair<int,int>                 ii_pair;
	typedef std::pair<std::string,std::string> ss_pair;
	

	// CONCATENATION //



	// STR Print 

//	void str_print (char * format, char* buffer, ...);


	// A vector of pointers, to store dynamic objects

	template <class T> class pvec {
		
		std::vector<T*> V;	
		
	public:
		void push_back(const T& val) {
			V.push_back(NULL);
			*(V.back()) = val;
		}
		
		void clear() {
			for (unsigned i=0; i<V.size(); i++) {
				delete V[i];
			}
			V.clear();
		}
		
		void resize(const unsigned& i){
			while (i<V.size()) {
				delete V.back();
				V.resize(V.size()-1);
			}
			while (i>V.size()) {
				V.push_back(new T);
			}
		}
		
		// Square Brackets returns the value
		T& operator[](const unsigned& i) { return (*V[i]); }
		const T& operator[](const unsigned& i) const { return (*V[i]); }
		
		// Round Paren returns the pointer
		T* operator()(const unsigned& i) { return (V[i]); }
		const T *const operator()(const unsigned& i) const { return (V[i]); }
		
	};

	
	
	bool XOR ( bool p, bool q ) ;
	 
	
	
	std::string compiled_filepath();
	
	std::string& uppercase(std::string& s);
	std::string  to_uppercase(const std::string& s);
	std::string& lowercase(std::string& s);
	
	std::string& trim_right_inplace(
		std::string&       s,
		const std::string& delimiters = " \f\n\r\t\v\'\"" );

	std::string& trim_left_inplace(
		std::string&       s,
		const std::string& delimiters = " \f\n\r\t\v\'\"" );

	std::string& trim(
		std::string&       s,
		const std::string& delimiters = " \f\n\r\t\v\'\"" );

	std::map<std::string,std::string> parse_option_string (std::string& input, const std::string& default_tag="label");
	std::string trim_to_brackets (std::string& input, 
		const std::string& open_bracket="[", const std::string& close_bracket="]");


	double flux(const double& should_be, const double& actually_is);
	double maxflux(const size_t& n, const double* should_be, const double* actually_is);
	
} // end namespace et_basic


#endif

