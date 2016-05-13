/*
 *  toolbox_object.cpp
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


#include "etk_object.h"

using namespace etk;

etk::object::object(logging_service* m)
: msg (m)
{ 
	//MONITOR(msg)<< "Creating object:"<<pointer_as_string(this) ;
}

etk::object::object(logging_service& ms)
: msg (ms)
{ 
	//MONITOR(msg)<< "Creating object:"<<pointer_as_string(this) ;
}

etk::object::~object()
{

}



