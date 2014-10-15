//
//  etk_refcount.h
//  Hangman
//
//  Created by Jeffrey Newman on 4/15/14.
//  Copyright (c) 2014 Jeffrey Newman. All rights reserved.
//

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
