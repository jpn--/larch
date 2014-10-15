//
//  elm_caseindex.cpp
//  Hangman
//
//  Created by Jeffrey Newman on 4/15/14.
//  Copyright (c) 2014 Jeffrey Newman. All rights reserved.
//

#include "etk.h"
#include "elm_caseindex.h"



const size_t& elm::caseindex_t::operator[] (const long long& caseid) const
{
	auto i =_casenums.find(caseid);
	if (i==_casenums.end()) {
		OOPS("caseid ",caseid," not found");
	}
	return i->second;
}


const long long& elm::caseindex_t::caseid_from_num (const size_t& index) const
{
	if (index >= _caseids.size()) {
		OOPS("casenum ",index," out of range, only ",_caseids.size()," cases are known");
	}
	return _caseids[index];
}

bool elm::caseindex_t::contains(const long long& caseid) const
{
	auto i =_casenums.find(caseid);
	if (i==_casenums.end()) {
		return false;
	}
	return true;
}

void elm::caseindex_t::clear()
{
	
}

size_t elm::caseindex_t::add_caseid(const long long& caseid)
{
	_caseids.push_back(caseid);
	size_t s = _caseids.size();
	_casenums[caseid] = s;
	return s;
}

void elm::caseindex_t::add_caseids(const std::vector<long long>& caseid)
{
	for (auto i=caseid.begin(); i!=caseid.end(); i++) {
		add_caseid(*i);
	}
}

elm::caseindex_t::caseindex_t()
{
}


elm::caseindex_t::~caseindex_t()
{
	
}

elm::caseindex elm::caseindex_t::create()
{
	elm::caseindex p = boosted::make_shared<elm::caseindex_t>();
	p->myself = p;
	return p;
}


