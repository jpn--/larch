/*
 *  etk_refcount.h
 *
 *  Copyright 2007-2017 Jeffrey Newman
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

#ifndef __Hangman__etk_refcount__
#define __Hangman__etk_refcount__



#ifdef SWIG

%feature("ref")   etk::refcounted "$this->incref();"
%feature("unref") etk::refcounted "$this->decref();"

#endif // def SWIG


#ifndef SWIG

#define REF_INC(x)   if (x) {x->incref();}
#define REF_CLEAR(x) if (x) {x->decref(); x=nullptr;}

#endif // ndef swig


namespace etk {
	
	
	class refcounted  {
		// implement the ref counting mechanism
		int _refcount;
		
		int increase_ref();
		int decrease_ref();
				
	public:
		refcounted();
		virtual ~refcounted(){};
		int incref() const;
		int decref() const;
		int ref_count() const;
		void lifeboat() const;
	};

	
};
#endif /* defined(__Hangman__etk_refcount__) */
