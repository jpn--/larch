/*
 *  elm_names.h
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


#ifndef __ELM2_NAMES_H__
#define __ELM2_NAMES_H__

#include "etk.h"
#include "elm_cellcode.h"

namespace elm {
	
	class multiname {
		
	public:
		std::string variable;
		std::string alternative;
		cellcode node_code;
		cellcode pred_code;
		std::string submodel;
		
		multiname(const std::string& s="");
		std::string fuse() const;
	};
	
	bool isText_constant(const std::string& text);
	std::string constant_to_one(const std::string& x);
	std::string fuse_constant_to_one(const etk::strvec& v, const std::string& sep=", ");
	
}

#endif

