/*
 *  etk_cat.h
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

#ifndef __ETK_CAT__
#define __ETK_CAT__

#include <string>
#include <sstream>


namespace etk {

/*
	template <typename T>
	void mash_into(std::ostringstream& output, const T& value)
	{
		output << value;
	}

	template <typename U, typename... T>
	void mash_into(std::ostringstream& output, const U& head, const T&... tail)
	{
		output << head;
		mash_into(output, tail...);
	}
	
	template <typename... T>
	std::string mash(const T&... tail)
	{
		std::ostringstream output;
		mash_into(output, tail...);
		return output.str();
	}
*/		

	// CONCATENATION //


	template <class T1>
	std::string cat ( T1 a )
	{
		std::ostringstream output;
		output << a;
		return output.str();
	}

	template <class T1, class T2>
	std::string cat ( T1 a, T2 b )
	{
		std::ostringstream output;
		output << a << b;
		return output.str();
	}

	template <class T1, class T2, class T3>
	std::string cat ( T1 a, T2 b, T3 c )
	{
		std::ostringstream output;
		output << a << b << c;
		return output.str();
	}

	template <class T1, class T2, class T3, class T4>
	std::string cat ( T1 a, T2 b, T3 c, T4 d )
	{
		std::ostringstream output;
		output << a << b << c << d;
		return output.str();
	}

	template <class T1, class T2, class T3, class T4, class T5>
	std::string cat ( T1 a, T2 b, T3 c, T4 d, T5 e )
	{
		std::ostringstream output;
		output << a << b << c << d << e;
		return output.str();
	}

	template <class T1, class T2, class T3, class T4, class T5, class T6>
	std::string cat ( T1 a, T2 b, T3 c, T4 d, T5 e, T6 f )
	{
		std::ostringstream output;
		output << a << b << c << d << e << f;
		return output.str();
	}
	
	template <class T1, class T2, class T3, class T4, class T5, class T6, class T7>
	std::string cat ( T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g )
	{
		std::ostringstream output;
		output << a << b << c << d << e << f << g;
		return output.str();
	}
	
	template <class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8>
	std::string cat ( T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h )
	{
		std::ostringstream output;
		output << a << b << c << d << e << f << g << h;
		return output.str();
	}
	
	template <class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9>
	std::string cat ( T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 i )
	{
		std::ostringstream output;
		output << a << b << c << d << e << f << g << h << i;
		return output.str();
	}
	
	template <class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10>
	std::string cat ( T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 i, T10 j )
	{
		std::ostringstream output;
		output << a << b << c << d << e << f << g << h << i << j;
		return output.str();
	}

	template <class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11>
	std::string cat ( T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 i, T10 j, T11 k )
	{
		std::ostringstream output;
		output << a << b << c << d << e << f << g << h << i << j << k;
		return output.str();
	}

	template <class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11, class T12>
	std::string cat ( T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 i, T10 j, T11 k, T12 l )
	{
		std::ostringstream output;
		output << a << b << c << d << e << f << g << h << i << j << k << l;
		return output.str();
	}
	 
	template <class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11, class T12, class T13>
	std::string cat ( T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 i, T10 j, T11 k, T12 l, T13 m )
	{
		std::ostringstream output;
		output << a << b << c << d << e << f << g << h << i << j << k << l << m;
		return output.str();
	}
	 

	
} // end namespace etk

#endif // __ETK_CAT__

