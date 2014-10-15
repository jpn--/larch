//
//  etk_refcount.cpp
//  Hangman
//
//  Created by Jeffrey Newman on 4/15/14.
//  Copyright (c) 2014 Jeffrey Newman. All rights reserved.
//

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
