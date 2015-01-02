/*
 *  etk_refcount.cpp
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

#include "etk_refcount.h"
#include <iostream>

etk::refcounted::refcounted()
: _refcount(1)
{
//	std::cerr << "construct_ref "<<this<<"\n";
}

int etk::refcounted::increase_ref()
{
	_refcount += 1;
//	std::cerr << "increase_ref "<<this<<"\n";
	return _refcount;
}

int etk::refcounted::decrease_ref()
{
	_refcount -= 1;
//	std::cerr << "decrease_ref "<<this<<"\n";
	return _refcount;
}

int etk::refcounted::ref_count() const
{
	return _refcount;
}



int etk::refcounted::incref() const
{
	return const_cast<refcounted*>(this)->increase_ref();
}

int etk::refcounted::decref() const   {
	if (const_cast<refcounted*>(this)->ref_count() == 0 || const_cast<refcounted*>(this)->decrease_ref() == 0 ) {
//		std::cerr << "delete_ref "<<this<<"\n";
		delete this;
		return 0;
	}
	return const_cast<refcounted*>(this)->ref_count();
}

void etk::refcounted::lifeboat() const
{
	if ( ref_count() <= 0 ) {
		const_cast<refcounted*>(this)->_refcount = 1;
	}
}
