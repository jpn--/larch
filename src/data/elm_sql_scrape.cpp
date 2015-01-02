/*
 *  elm_sql_scrape.cpp
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

#include <iostream>

#include <cstring>
#include "etk.h"
#include "elm_sql_facet.h"
#include "elm_sql_scrape.h"
#include "elm_cellcode.h"
#include "elm_vascular.h"

using namespace std;
using namespace etk;


elm::ScrapePtr elm::Scrape::pointer()
{
	return myself.lock();
}

elm::ScrapePtr elm::Scrape::create(elm::Facet* parent, int style)
{
	elm::ScrapePtr p = boosted::make_shared<elm::Scrape>(elm::Scrape(parent, style));
	p->myself = elm::ScrapePtr_weak (p);
	return p;
}

elm::ScrapePtr elm::Scrape::copy() const
{
	elm::ScrapePtr p = boosted::make_shared<elm::Scrape>(elm::Scrape(parent, _style));
	p->myself = elm::ScrapePtr_weak (p);

	p->_nVars_        =_nVars_;
	p->_firstcasenum  =0;
	p->_numberofcases =0;
	p->_VarNames      =_VarNames;
	p->_style         =_style;

	return p;
}

elm::Scrape elm::Scrape::create_idco(elm::Facet* parent)
{
	return elm::Scrape(parent, IDCO);
}

elm::Scrape elm::Scrape::create_idca(elm::Facet* parent)
{
	return elm::Scrape(parent, IDCA);
}

elm::Scrape elm::Scrape::create_choo(elm::Facet* parent)
{
	elm::Scrape x (parent, CHOO);
	x._nVars_ = 1;
	return x;
}

elm::Scrape elm::Scrape::create_wght(elm::Facet* parent)
{
	elm::Scrape x (parent, WGHT);
	x._nVars_ = 1;
	return x;
}

elm::Scrape elm::Scrape::create_aval(elm::Facet* parent)
{
	elm::Scrape x (parent, AVAL);
	x._nVars_ = 1;
	return x;
}



elm::Scrape::Scrape(elm::Facet* parent, int style)
: parent         (parent)
, _nVars_        (0)
, _repository    ()
, _bools         ()
, _firstcasenum  (0)
, _numberofcases (0)
, _VarNames      ()
, _style         (style)
, _repo_lock     (etk::reading_lockout::open())
, _bool_lock     (etk::reading_lockout::open())
, _stmt          (parent->sql_statement_readonly(""))
{
	if (_style&(CHOO|AVAL|WGHT)) _nVars_ = 1;
}

elm::Scrape::Scrape(const elm::Scrape& x)
: parent         (x.parent)
, _nVars_        (x._nVars_)
, _repository    ()
, _bools         ()
, _firstcasenum  (0)
, _numberofcases (0)
, _VarNames      (x._VarNames)
, _style         (x._style)
, _repo_lock     (etk::reading_lockout::open())
, _bool_lock     (etk::reading_lockout::open())
, _stmt          (x.parent->sql_statement_readonly(""))
{
 
}

elm::Scrape::Scrape()
: parent         (NULL)
, _nVars_        (0)
, _repository    ()
, _bools         ()
, _firstcasenum  (0)
, _numberofcases (0)
, _VarNames      ()
, _style         (0)
, _repo_lock     ()
, _bool_lock     ()
, _stmt          ()
{
	OOPS("do not create scrapes directly, use a create_* method");
}


elm::Scrape::~Scrape()
{ 
	tearDown(true);
}




void elm::Scrape::tearDown(bool force)
{
	if (!force) {
		if (_repo_lock->check_use_count()) OOPS("There is a repository read lock active, it is not safe to tearDown");
		if (_bool_lock->check_use_count()) OOPS("There is a bool read lock active, it is not safe to tearDown");
	}
	_repository.destroy();
	_bools.destroy();
	_firstcasenum=0;
	_numberofcases=0;
}



void elm::Scrape::use_variables( const std::vector<std::string>& varnames )
{
	if (_style&(CHOO|AVAL|WGHT)) OOPS("Cannot set variable names for this style directly; use facet queries instead.");

	_VarNames.clear();
	_nVars_ = 0;
	if (_style&(CHOO|AVAL|WGHT)) _nVars_ = 1;
	std::vector<std::string>::const_iterator x;
	for (x = varnames.begin(); x != varnames.end(); x++) {
		_VarNames.push_back(*x);
		if (_style&IDCA) parent->check_ca(*x);
		if (_style&IDCO) parent->check_co(*x);
		_nVars_++;
	}
}

size_t elm::Scrape::add_var( const std::string& varname )
{	
	if (_style&(CHOO|AVAL|WGHT)) OOPS("Cannot set variable names for this style directly; use facet queries instead.");

	if (_style&IDCA) parent->check_ca(varname);
	if (_style&IDCO) parent->check_co(varname);
	for (unsigned i=0;i<_VarNames.size();i++) {
		if (varname==_VarNames.at(i)) {
			return i;
		}
	}
	_VarNames.push_back(varname);
	_nVars_++;
	return _VarNames.size()-1;
}

std::vector<std::string> elm::Scrape::using_variables() const
{
	return _VarNames;
}

size_t elm::Scrape::nVars() const
{
	return _nVars_;
}

const unsigned& elm::Scrape::nAlts(const long long& c) const
{
	if (!parent) {
		OOPS(("no data source known"));
	}
	return parent->nAlts();
}

const unsigned& elm::Scrape::nCases() const
{
	if (!parent) {
		OOPS(("no data source known"));
	}
	return parent->nCases();
}



etk::ptr_lockout<const double> elm::Scrape::values(const unsigned& firstcasenum, const size_t& numberofcases)
{
	if (firstcasenum < _firstcasenum) load_values(firstcasenum, numberofcases);
	if (firstcasenum+numberofcases > _firstcasenum+_numberofcases) load_values(firstcasenum, numberofcases);
	return ptr_lockout<const double>(_repository.ptr(firstcasenum-_firstcasenum), _repo_lock);
}

etk::ptr_lockout<const bool> elm::Scrape::boolvalues(const unsigned& firstcasenum, const size_t& numberofcases)
{
	if (firstcasenum < _firstcasenum) load_values(firstcasenum, numberofcases);
	if (firstcasenum+numberofcases > _firstcasenum+_numberofcases) load_values(firstcasenum, numberofcases);
	return ptr_lockout<const bool>(_bools.ptr(firstcasenum-_firstcasenum), _bool_lock);
}

etk::ptr_lockout<const double> elm::Scrape::values(const unsigned& firstcasenum, const size_t& numberofcases) const
{
	if (firstcasenum < _firstcasenum) {
		OOPS("const error on reading values");
	}
	if (firstcasenum+numberofcases > _firstcasenum+_numberofcases) {
		OOPS("const error on reading values");
	}
	if (_repository.size()==0) {
		OOPS("float values not loaded");
	}
	;
	return ptr_lockout<const double>(_repository.ptr(firstcasenum-_firstcasenum), const_cast<elm::Scrape*>(this)->_repo_lock);
}

etk::ptr_lockout<const bool> elm::Scrape::boolvalues(const unsigned& firstcasenum, const size_t& numberofcases) const
{
	if (firstcasenum < _firstcasenum) {
		OOPS("const error on reading values");
	}
	if (firstcasenum+numberofcases > _firstcasenum+_numberofcases) {
		OOPS("const error on reading values");
	}
	if (_bools.size()==0) {
		OOPS("bool values not loaded");
	}
	return ptr_lockout<const bool>(_bools.ptr(firstcasenum-_firstcasenum), const_cast<elm::Scrape*>(this)->_bool_lock);
}


double elm::Scrape::value(const unsigned& c, const unsigned& a, const unsigned& v) const
{
	return *(values(c,1)+(a*nVars())+v);
}

double elm::Scrape::value(const unsigned& c, const unsigned& v) const
{
	return *(values(c,1)+v);
}

bool elm::Scrape::boolvalue(const unsigned& c, const unsigned& a, const unsigned& v) const
{
	return *(boolvalues(c,1)+(a*nVars())+v);
}
bool elm::Scrape::boolvalue(const unsigned& c, const unsigned& v) const
{
	return *(boolvalues(c,1)+v);
}


bool elm::Scrape::_as_bool() const
{
	if (_style&AVAL) return true;
	return false;
}


void elm::Scrape::load_values(const size_t& firstcasenum, const size_t& numberofcases)
{
	if (firstcasenum==0 && numberofcases==0 && fully_loaded()) {
		return;
	}
	
	if (firstcasenum!=0 || numberofcases!=0) {
		if (is_loaded_in_range(firstcasenum,numberofcases)) {
			return;
		}
	}
	
	size_t cs = numberofcases;
	if (numberofcases==0) cs = parent->nCases();
	if (firstcasenum+numberofcases>parent->nCases()) cs = parent->nCases() - firstcasenum;

	if (_style&(CHOO|AVAL|WGHT)) _nVars_ = 1;

	if (_style&IDCA) {
		if (cs==0 || parent->nAlts()==0 || nVars()==0) return;
	} else if (_style&IDCO) {
		if (cs==0 || nVars()==0) return;
	}

	if (_repo_lock->check_use_count()) {
		OOPS("There is a repository read lock active, cannot load new data now\n", describe_loaded_range(),
			 "\nAsking for case ",firstcasenum, " to case ", firstcasenum+numberofcases);
	}
	if (_bool_lock->check_use_count()) {
		OOPS("There is a bool read lock active, cannot load new data now\n", describe_loaded_range(),
			 "\nAsking for case ",firstcasenum, " to case ", firstcasenum+numberofcases);
	}
		

	if (_as_bool()) {
		if ((firstcasenum>=_firstcasenum)&&(firstcasenum+cs<=_firstcasenum+_numberofcases)&&(_bools.size()>=cs)) return;
		_repository.destroy();
	} else {
		if ((firstcasenum>=_firstcasenum)&&(firstcasenum+cs<=_firstcasenum+_numberofcases)&&(_repository.size()>=cs)) return;
		_bools.destroy();
	}
	
	cellcodeset bad_codes;
		
	if (_style&IDCA) {
		if (cs==0 || parent->nAlts()==0 || nVars()==0) return;
		_repository.resize(cs,parent->nAlts(),nVars());
	} else if (_style&IDCO) {
		if (cs==0 || nVars()==0) return;
		_repository.resize(cs,nVars());
	} else if (_style&CHOO) {
		_repository.resize(cs,parent->nAlts(),1);
	} else if (_style&AVAL) {
		_bools.resize(cs,parent->nAlts(),1);
	} else if (_style&WGHT) {
		_repository.resize(cs,1);
	} else {
		OOPS("Unknown scrape style ",_style);
	}
	if (_as_bool()) {
		_bools.initialize(false);
	} else {
		_repository.initialize(0.0);
	}
	
	if (nVars()==0 || cs<=0) return;


	const VAS_Cell* alt;
	cellcode alt_code;
	long long current_caseid;
	size_t c = 0;

	OOPS("This function is broken, sorry.");
	
	if (_style&IDCA) { 
//		_stmt->prepare( parent->query_idca(_VarNames,firstcasenum,cs) );
	} else if (_style&IDCO) {
//		_stmt->prepare( parent->query_idco(_VarNames,firstcasenum,cs) );
	} else if (_style&CHOO) {
//		_stmt->prepare( parent->query_choice(firstcasenum,cs) );
	} else if (_style&AVAL) {
//		_stmt->prepare( parent->query_avail(firstcasenum,cs) );
	} else if (_style&WGHT) {
//		_stmt->prepare( parent->query_weight(firstcasenum,cs) );
	}
		
	MONITOR(parent->msg) << "SQL: "<< _stmt->sql() ;
	std::string the_stmt = _stmt->sql();

	clock_t prevmsgtime = clock();
	clock_t timenow;
	
	_stmt->execute();
	if (_stmt->status()==SQLITE_ROW) {
		current_caseid = _stmt->getInt(0);
	}
	while (_stmt->status()==SQLITE_ROW) {
		try { 
			if (current_caseid != _stmt->getInt64(0)) {
				c++;
				current_caseid = _stmt->getInt64(0);
			}
			timenow = clock();
			if (timenow > prevmsgtime + (CLOCKS_PER_SEC * 3)) {
				MONITOR(parent->msg) << "reading case "<< current_caseid << ", " << 100.0*double(c)/double(cs) << "% ..." ;
				prevmsgtime = clock();
			}
		}
		SPOO {
			_stmt->execute();
			continue;
		}
		if (_style&(IDCA|AVAL|CHOO)) {
			alt_code = _stmt->getInt64(1);
			if (alt_code==0) {
				bad_codes.insert(alt_code);
				_stmt->execute();
				continue;
			}
			try { 
				alt = parent->DataDNA(current_caseid)->cell_from_code(alt_code);
			} SPOO {
				bad_codes.insert(alt_code);
				_stmt->execute();
				continue;
			}
			if (_as_bool()) {
				_stmt->getBools (2, 2+nVars(),_bools.ptr(c,alt->slot()));
			} else {
				_stmt->getDoubles (2, 2+nVars(),_repository.ptr(c,alt->slot()));
			}
		} else if (_style&(IDCO|WGHT)) {
			if (_as_bool()) {
				_stmt->getBools (1, 1+nVars(),_bools.ptr(c));
			} else {
				_stmt->getDoubles (1, 1+nVars(),_repository.ptr(c));
			}
		}
		_stmt->execute();
	}
	MONITOR(parent->msg) << "table read end" ;
	if (bad_codes.size()) {
		ostringstream badness;
		badness << "while reading data ( " << _stmt->sql() << " ) there are " << bad_codes.size() << " unidentified cell codes:\n";
		for (std::set<cellcode>::iterator b=bad_codes.begin(); b!=bad_codes.end(); b++)
			badness << *b << "\n"; 
		OOPS(badness.str());
	}
	_firstcasenum = firstcasenum;
	_numberofcases = cs;
}




bool elm::Scrape::fully_loaded(bool boolean) const
{
	if (boolean && _repository.size()>0) return false;
	if (!boolean && _bools.size()>0) return false;
	if (_firstcasenum==0 && _numberofcases==parent->nCases()) return true;
	if (_nVars_==0) return true;
	return false;
}

bool elm::Scrape::is_loaded_in_range(const size_t& firstcasenum, const size_t& numberofcases) const
{
	if (_as_bool() && _repository.size()>0) return false;
	if (!_as_bool() && _bools.size()>0) return false;
	if (_firstcasenum<=firstcasenum && _firstcasenum+_numberofcases>=firstcasenum+numberofcases) return true;
	return false;
}

std::string elm::Scrape::describe_loaded_range() const
{
	return cat("Loaded from case ",_firstcasenum," to case ",_firstcasenum+_numberofcases);
}


std::string elm::Scrape::printcase(const unsigned& r) const
{
	std::ostringstream ret;
	unsigned x2, x3;
	char depMarker, colMarker, rowMarker;
	if (_style&(IDCO|WGHT)) {
		depMarker = ' ';
		colMarker = '\t';
		rowMarker = '\n';
		for ( x2=0; x2<nVars(); x2++ ) { 
			ret << value(r,x2) << colMarker;
		}
		ret << rowMarker;
	} else if (_style&(IDCA|CHOO)) {
		depMarker = '\t';
		colMarker = '\n';
		rowMarker = '\n';
		for ( x2=0; x2<nAlts(); x2++ ) { 
			for ( x3=0; x3<nVars(); x3++ ) { 
				ret << value(r,x2,x3) << depMarker;
			}
			ret << colMarker;
		}
		ret << rowMarker;
	}
	return ret.str();
}

std::string elm::Scrape::printcases(unsigned rstart, const unsigned& rfinish) const
{
	std::ostringstream ret;
	for (; rstart < rfinish; rstart++) {
		ret << printcase(rstart);
	}
	return ret.str();
}


std::string elm::Scrape::printboolcase(const unsigned& r) const
{
	std::ostringstream ret;
	unsigned x2, x3;
	char depMarker, colMarker, rowMarker;
	if (_style&(IDCO|WGHT)) {
		depMarker = ' ';
		colMarker = '\t';
		rowMarker = '\n';
		for ( x2=0; x2<nVars(); x2++ ) { 
			ret << boolvalue(r,x2) << colMarker;
		}
		ret << rowMarker;
	} else if (_style&(AVAL|CHOO|IDCA)) {
		depMarker = '\t';
		colMarker = '\n';
		rowMarker = '\n';
		for ( x2=0; x2<nAlts(); x2++ ) { 
			for ( x3=0; x3<nVars(); x3++ ) { 
				ret << boolvalue(r,x2,x3) << depMarker;
			}
			ret << colMarker;
		}
		ret << rowMarker;
	}
	return ret.str();
}

std::string elm::Scrape::printboolcases(unsigned rstart, const unsigned& rfinish) const
{
	std::ostringstream ret;
	for (; rstart < rfinish; rstart++) {
		ret << printboolcase(rstart);
	}
	return ret.str();
}





void elm::Scrape::ExportData (double* ExportTo, const unsigned& c, const unsigned& a, const unsigned& numberOfAlts) const
{
	if (_style&(CHOO|IDCA)) {
		cblas_dcopy(nVars(),values(c,1)+(a*nVars()),1,ExportTo,1);
		ExportTo+=nVars();
	} else if (_style&(WGHT|IDCO)) {
		memset(ExportTo,0,nVars()*numberOfAlts*sizeof(double));
		cblas_dcopy(nVars(),values(c,1),1,ExportTo+a,numberOfAlts);
	}
}
void elm::Scrape::ExportData	(double* ExportTo, const unsigned& c, const unsigned& a, const double& scale, const unsigned& numberOfAlts) const
{
	if (_style&(CHOO|IDCA)) {
		cblas_dcopy(nVars(),values(c,1)+(a*nVars()),1,ExportTo,1);
		cblas_dscal(nVars(), scale, ExportTo,1);
		ExportTo+=nVars();
	} else if (_style&(WGHT|IDCO)) {
		memset(ExportTo,0,(nVars()*numberOfAlts)*sizeof(double));
		cblas_dcopy(nVars(),values(c,1),1,ExportTo+a,numberOfAlts);
		cblas_dscal(nVars(), scale, ExportTo+a,numberOfAlts);
		ExportTo+=(nVars()*numberOfAlts);
	}
}

void elm::Scrape::OverlayData(double* ExportTo, const unsigned& c, const unsigned& a, const double& scale, const unsigned& numberOfAlts) const
{
	if (_style&(CHOO|IDCA)) {
		cblas_daxpy(nVars(),scale,values(c,1)+(a*nVars()),1,ExportTo,1);
		ExportTo+=nVars();
	} else if (_style&(WGHT|IDCO)) {
		cblas_daxpy(nVars(),scale,values(c,1),1,ExportTo+a,numberOfAlts);
		ExportTo+=(nVars()*numberOfAlts);
	}		
}

PyObject* elm::Scrape::getArray()
{
	PyObject* x = _repository.get_object();
	if (!x) {
		load_values();
		x = _repository.get_object();
	}
	return x;
}

PyObject* elm::Scrape::getBoolArray()
{
	PyObject* x = _bools.get_object();
	if (!x) {
		load_values();
		x = _bools.get_object();
	}
	return x;
}

