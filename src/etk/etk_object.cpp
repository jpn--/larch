/*
 *  toolbox_object.cpp
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
	//MONITOR(msg)<< "Destroying object:"<<pointer_as_string(this) ;
	if (_subjects.size()) {		
		//MONITOR(msg)<<"Deleting "<<int(_subjects.size())<<" subjects..." ;
		while (_subjects.size()>0) {
			delete (*( _subjects.begin() ));
		}
	}
}

void etk::object::add_subject(subject* baby)
{
	_subjects.insert(baby);
}

void etk::object::del_subject(subject* baby)
{
	_subjects.erase(baby);
}

void etk::object::print_set()
{
	BUGGER_BUFFER(msg) << "OBJECT "<<pointer_as_string(this)<<":{";
	std::set<subject*>::iterator i = _subjects.begin();
	if (i!=_subjects.end())  {
		BUGGER_BUFFER(msg)<<pointer_as_string(*i);
		for (i++; i!=_subjects.end(); i++) {
			BUGGER_BUFFER(msg)<<","<<pointer_as_string(*i);
		}
	}
	BUGGER(msg)<<"}" ;
}





etk::subject::subject(object* parent)
: _object (parent)
{ 
	if (_object) {
		//MONITOR(_object->msg) << "Creating subject:"<<pointer_as_string(this)<<" of object:"<<pointer_as_string(_object) ;
		_object->add_subject(this);
		msg = (_object->msg);
	}
}

etk::subject::subject(const subject& sibling)
: _object (sibling._object)
{ 
	if (_object) {
		//MONITOR(_object->msg) << "Creating subject:"<<pointer_as_string(this)<<" of object:"<<pointer_as_string(_object) ;
		_object->add_subject(this);
	}
}

subject& etk::subject::operator=(const subject& sibling)
{ 
	if (_object != sibling._object) {
		_object->del_subject(this);
		_object = sibling._object;
	}
	if (_object) {
		//MONITOR(_object->msg) << "Creating subject:"<<pointer_as_string(this)<<" of object:"<<pointer_as_string(_object) ;
		_object->add_subject(this);
	}
	return *this;
}

void etk::subject::reparent(object* parent)
{
	if (_object) {
		_object->del_subject(this);
	}
	_object = parent;
	if (_object) {
		_object->add_subject(this);
	}
}

etk::subject::~subject()
{
	if (_object) {
		//MONITOR(_object->msg) << "Destroying subject:"<<pointer_as_string(this)<<" of object:"<<pointer_as_string(_object) ;
		_object->del_subject(this);
	}
}
