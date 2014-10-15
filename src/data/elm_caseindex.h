//
//  elm_caseindex.h
//  Hangman
//
//  Created by Jeffrey Newman on 4/15/14.
//  Copyright (c) 2014 Jeffrey Newman. All rights reserved.
//

#ifndef __Hangman__elm_caseindex__
#define __Hangman__elm_caseindex__

#include "etk_refcount.h"
#include "etk_thread.h"

#ifdef SWIG
%{
// In SWIG, these headers are available to the c++ wrapper,
// but are not themselves wrapped
#include <map>
#include <vector>
%}
#endif // SWIG

#ifndef SWIG
#include <map>
#include <vector>
#endif // ndef SWIG


namespace elm {
	
	#ifndef SWIG
	class caseindex_t;
	typedef boosted::shared_ptr<caseindex_t> caseindex;
	typedef boosted::weak_ptr<caseindex_t>   caseindex_;
	#endif // ndef SWIG

	
	class caseindex_t
	{

		#ifndef SWIG
		caseindex_ myself;
		#endif // SWIG
		
		// caseids are unique 64 bit integers in arbitrary not necessarily sequential order naming the cases
		// casenums are sequential unsigned integers for the cases in order
		
		std::map<long long,size_t>	_casenums;
		std::vector<long long>      _caseids;
		
	public:

#ifndef SWIG
		const size_t&    operator[]      (const long long& caseid) const ;
#endif // SWIG
		const size_t&    casenum_from_id (const long long& caseid) const { return operator[](caseid); }
		const long long& caseid_from_num (const size_t& index)     const ;
		
		std::size_t size() const	{ return _caseids.size(); }
		const std::vector<long long>& caseids() const { return _caseids; }
		bool contains(const long long& caseid) const;

		void   clear();
		size_t add_caseid(const long long& caseid);
		void add_caseids(const std::vector<long long>& caseid);
		
		#ifndef SWIG
	public:
		caseindex_t();
		#endif // SWIG

	public:
		static caseindex create();
		virtual ~caseindex_t();


	};
	



};




#endif /* defined(__Hangman__elm_caseindex__) */
