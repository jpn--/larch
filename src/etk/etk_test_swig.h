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


#ifndef __ETK_TEST_SWIG__
#define __ETK_TEST_SWIG__

#include "etk_python.h"
#include <iostream>



namespace etk {
	
	PyObject* _swigtest_empty_dict();
	PyObject* _swigtest_alpha_dict();

	class ostream_c
	{
		std::ostream* _receiver;
	public:
		ostream_c(int i=0);
		ostream_c(std::ostream* direction);
		~ostream_c();
		void flush();
		int write(std::string x);
		std::string mode() const;
		std::string __repr__() const;
	};

	class string_sender
	{
	public:
		virtual void write(std::string x)=0;
	};
	
} // end namespace etk


#endif // __ETK_TEST_SWIG__

