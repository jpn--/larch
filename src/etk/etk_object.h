/*
 *  toolbox_object.h
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


#ifndef _TOOLBOX_OBJECT_SUBJECT_H_
#define _TOOLBOX_OBJECT_SUBJECT_H_

#include <set>
#include "etk_messenger.h"

namespace etk {
	

	class object;
	class subject;

	class object
	{
		std::set<subject*> _subjects;	
	public:
        logging_service msg;
		object(logging_service* m=NULL);
		object(logging_service& ms);
		virtual ~object();	
		void add_subject(subject* baby);
		void del_subject(subject* baby);
		void print_set();
		friend class subject;
	};

	class subject
	{
		object* _object;	
	public:
		logging_service msg;
		subject(object* parent=0);
		subject(const subject& sibling);
		subject& operator=(const subject& sibling);
		virtual ~subject();
		void reparent(object* parent);
		const object* obj() const { return _object; }
	};

}
#endif // _TOOLBOX_OBJECT_SUBJECT_H_
