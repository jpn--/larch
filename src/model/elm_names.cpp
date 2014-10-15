/*
 *  elm_names.cpp
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


#include "elm_names.h"
#include "elm_cellcode.h"
using namespace etk;
using namespace elm;
using namespace std;

#define RESERVED_SYMBOLS "@#$%^&ß"

elm::multiname::multiname(const string& namepack)
: node_code (cellcode_empty)
, pred_code (cellcode_empty)
{
	string::size_type i = namepack.find_first_of(RESERVED_SYMBOLS);
	variable = namepack.substr(0,i);
	string::size_type j;
	char k;
	while (i != string::npos) {
		k = namepack.at(i++);
		j = namepack.find_first_of(RESERVED_SYMBOLS,i);
		switch (k) {
			case '@': alternative = namepack.substr(i,j-i);  break;
			case '#': node_code = cellcode_from_string(namepack.substr(i,j-i));  break;
			case '^': pred_code = cellcode_from_string(namepack.substr(i,j-i));  break;
			case '$': submodel = namepack.substr(i,j-i); break;
	//		case '~': var_name = "~" + namepack.substr(i,j-i) + var_name; break;
	//		case '%': relmodel_name = namepack.substr(i,j-i); break;
	//		case '^': 
	//		case '&': 
	//		case '*': var_name += namepack.substr(i,j-i); break;
			default: break;
		}
		i=j;
	}
}

string elm::multiname::fuse() const
{
	ostringstream ret;
	ret << variable;
	if (!alternative.empty()) ret << "@" << alternative;
	if (!is_cellcode_empty(node_code)) ret << "#" << node_code;
	if (!is_cellcode_empty(pred_code)) ret << "^" << pred_code;
	if (!submodel.empty()) ret << "$" << submodel;
	return ret.str();
}



bool elm::isText_constant(const string& text)
{
	std::string s (text);
	std::string::iterator i = s.begin();
	std::string::iterator end = s.end();
	while (i != end) {
		*i = std::toupper((unsigned char)*i);
		++i;
	}
	if (s=="CONSTANT") {
		return true;
	}
	return false;
}

string elm::constant_to_one(const string& x)
{
	if (isText_constant(x)) {
		//OOPS("The use of CONSTANT is currently not functioning correctly.");
		return "1";
	} 
	return x;
}

string elm::fuse_constant_to_one(const strvec& elements, const string& sep)
{
	std::string ret = "";
	if ( elements.empty() ) return ret;
	
	ret = constant_to_one(elements[0]);
	
	for(unsigned i=1; i< elements.size(); ++i)
	{
		ret.append( sep );
		ret.append( constant_to_one(elements[i]) );
	}
	return ret;	
	
}
