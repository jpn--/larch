/*
 *  etk_basic.cpp
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


#include "etk.h"

using namespace std;

/*
void str_print (char * format, char* bufferz, char* buffer, ...)
{
	va_list args;
	va_start (args, buffer);
	vsprintf (buffer,format, args);
	va_end (args);
	va_start (args, buffer);
	vsprintf (bufferz,format, args);
	va_end (args);
}
*/

string etk::fuse_strvec (const strvec& elements, const string sep)
{
	string ret = "";
	if ( elements.empty() ) return ret;
	
	ret = elements[0];
	
	for(unsigned i=1; i< elements.size(); ++i)
	{
		ret.append( sep );
		ret.append( elements[i] );
	}
	return ret;	
}

etk::strvec etk::alias(const strvec& elements, const string& replaceMe, const string& withMe)
{
	strvec ret (elements);
	for(unsigned i=0; i< elements.size(); ++i) {
		if (ret[i]==replaceMe) ret[i]=withMe;
	}
	return ret;
}

string etk::compiled_filepath() {
	string full (__FILE__);
	size_t i = full.find_last_of("/\\");
	// char delim = full[i];
	string part = full.substr(0,++i);
	return part;
}

bool etk::XOR ( bool p, bool q ) {
	return ( (p || q) && !(p && q) );
}

std::string& etk::uppercase(std::string& s)
{
	size_t len (s.length());
	for (unsigned i=0; i<len; i++) {
		s[i] = toupper(s[i]);
	}
	return s;
}

std::string etk::to_uppercase(const std::string& s)
{
	std::string s_out = s;
	size_t len (s_out.length());
	for (unsigned i=0; i<len; i++) {
		s_out[i] = toupper(s_out[i]);
	}
	return s_out;
}

std::string& etk::lowercase(std::string& s)
{
	size_t len (s.length());
	for (unsigned i=0; i<len; i++) {
		s[i] = tolower(s[i]);
	}
	return s;
}


std::string& etk::trim_right_inplace(
  std::string&       s,
  const std::string& delimiters)
{
  return s.erase( s.find_last_not_of( delimiters ) + 1 );
}

std::string& etk::trim_left_inplace(
  std::string&       s,
  const std::string& delimiters)
{
  return s.erase( 0, s.find_first_not_of( delimiters ) );
}

std::string& etk::trim(
  std::string&       s,
  const std::string& delimiters )
{
  return trim_left_inplace( trim_right_inplace( s, delimiters ), delimiters );
}

std::string etk::trim_to_brackets (std::string& input, 
		const std::string& open_bracket, const std::string& close_bracket)
{
	std::string s = input;
	s.erase( s.find_last_of( close_bracket ) + 1 );
	s.erase( 0, s.find_first_of( open_bracket ) );
	s.erase( s.find_last_not_of( close_bracket ) + 1 );
	s.erase( 0, s.find_first_not_of( open_bracket ) );
	return s;
}


std::map<std::string,std::string> etk::parse_option_string (std::string& input, const std::string& default_tag)
{

    std::stringstream input_process(input);
    std::string item;
	std::map<std::string,std::string> parsed;
    while(std::getline(input_process, item, ',')) {
        
		size_t eq = item.find("=");
		if (eq != std::string::npos) {
			std::string a = item.substr(0,eq);		 
			std::string b = item.substr(eq+1);	
			trim(a);
			trim(b);	 
			parsed[a] = b;
		} else {
			parsed[default_tag] = trim(item);
		}
		 
    }
	return parsed;
}

double etk::flux(const double& should_be, const double& actually_is)
{
	double difference = abs(should_be - actually_is);
	double magnitude  = (abs(should_be)+abs(actually_is)) / 2 ;
	if (magnitude) return difference/magnitude;
	return 0.0;
}

double etk::maxflux(const size_t& n, const double* should_be, const double* actually_is)
{
	double flux = 0;
	for (size_t i=0; i<n; i++) {
		double thisflux = etk::flux(should_be[i], actually_is[i]);
		if (thisflux > flux) flux = thisflux;
	}
	return flux;
}

