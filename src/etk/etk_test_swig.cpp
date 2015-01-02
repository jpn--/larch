/*
 *  etk_basic.cpp
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

#include "etk.h"
#include "etk_pydict.h"
#include "etk_test_swig.h"



PyObject* etk::_swigtest_empty_dict()
{
	PyObject* z = PyDict_New();
	return z;
}

PyObject* etk::_swigtest_alpha_dict()
{
	PyObject* z = PyDict_New();
	etk::dictionary_sd(z).key("a") = 1;
	etk::dictionary_sd(z).key("b") = 2;
	etk::dictionary_sd(z).key("c") = 3;
	return z;
}







namespace etk {
  

	
	
}

etk::ostream_c::ostream_c(int i)
: _receiver(i<0 ? &std::cerr : &std::cout)
{ }

etk::ostream_c::ostream_c(std::ostream* direction)
: _receiver(direction)
{ }

etk::ostream_c::~ostream_c()
{
	if (_receiver) _receiver->flush();
}


std::string etk::ostream_c::mode() const
{
	return "w";
}

std::string etk::ostream_c::__repr__() const
{
	return "<etk::ostream_c>";
}


void etk::ostream_c::flush()
{
	if (_receiver) _receiver->flush();
}

int etk::ostream_c::write(std::string x)
{
	if (_receiver) {
		(*_receiver) << x;
		return x.length();
	}
	return 0;
}




