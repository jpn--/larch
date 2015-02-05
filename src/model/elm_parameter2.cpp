/*
 *  elm_parameter.cpp
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


#include "elm_parameter2.h"
#include "elm_parameterlist.h"
#include "elm_inputstorage.h"
//#include "elm_sql_array.h"
#include "elm_names.h"

using namespace etk;
using namespace elm;
using namespace std;

elm::parametex::parametex(const string& f, elm::ParameterList* mod) // constructor
: freedom (f)
, mdl (mod)
{ }

double elm::parametex::pullvalue(const double* pullSource) const 
{
	return 0;
}
double elm::parametex::pullfield(const std::string& field) const 
{
	return 0;
}

void elm::parametex::pushvalue(double* pushDest, const double& q) const
{
	
}
void elm::parametex::pushfield(const std::string& field, const double& q) const
{

}

#include <iomanip>

std::string elm::parametex::print() const
{
	ostringstream ret;
	ret << "Freedom: " << std::setw(20) << std::left << freedom ;
	ret << "Parameter: "<< std::setw(12) << std::left << "Default";
	if (mdl) {
		ret << "Freedom Slot: "<<mdl->FNames[freedom] << "\n";
	} else {
		ret << "Freedom Slot: n/a\n";
	}
	return ret.str();
}

std::string elm::parametex::smallprint() const
{
	ostringstream ret;
	ret << "<Default>";
	return ret.str();
}


///////// CONSTANT ////////////

elm::parametex_constant::parametex_constant(const double& val) // constructor
: parametex ("Constant",NULL)
, _value (val)
{ }

double elm::parametex_constant::pullvalue(const double* pullSource) const
{
	return _value;
}
double elm::parametex_constant::pullfield(const std::string& field) const
{
	if (field=="value") return _value;
	OOPS("not implemented");
}

void elm::parametex_constant::pushvalue(double* pushDest, const double& q) const
{

}
void elm::parametex_constant::pushfield(const std::string& field, const double& q) const
{

}

std::string elm::parametex_constant::print() const
{
	ostringstream ret;
	ret << "Freedom: " << std::setw(20) << std::left << freedom ;
	ret << "Parameter: "<< std::setw(12) << std::left << "Constant="<<_value<<"\n";
	return ret.str();
}
/*{
	ostringstream ret;
	ret << "Freedom: " << freedom << "\n";
	ret << "Parameter: Constant = " << _value << "\n";
	return ret.str();
}*/


std::string elm::parametex_constant::smallprint() const
{
	ostringstream ret;
	ret << "Constant=" << _value;
	return ret.str();
}


///////// EQUAL ////////////

elm::parametex_equal::parametex_equal(const string& f, elm::ParameterList* mod) // constructor
: parametex (f,mod)
{ }

double elm::parametex_equal::pullvalue(const double* pullSource) const
{
	if (mdl) return pullSource[mdl->FNames[freedom]];
	return elm::parametex::pullvalue(pullSource);
}
double elm::parametex_equal::pullfield(const std::string& field) const
{
	if (field=="value") {
		if (mdl) return mdl->FInfo[freedom].value;
		return elm::parametex::pullfield(field);
	}
	OOPS("not implemented");
}
void elm::parametex_equal::pushvalue(double* pushDest, const double& q) const
{
	if (mdl)  pushDest[mdl->FNames[freedom]] += q;
}
void elm::parametex_equal::pushfield(const std::string& field, const double& q) const
{
	if (mdl)  mdl->FInfo[freedom].value += q;
}

std::string elm::parametex_equal::print() const
{
	ostringstream ret;
	ret << "Freedom: " << std::setw(20) << std::left << freedom ;
	ret << "Parameter: "<< std::setw(12) << std::left << "Equal";
	if (mdl) {
		ret << "Freedom Slot: "<<mdl->FNames[freedom] << "\n";
	} else {
		ret << "Freedom Slot: n/a\n";
	}
	return ret.str();
}



std::string elm::parametex_equal::smallprint() const
{
	ostringstream ret;
	ret << freedom;
	return ret.str();
}




///////// SCALE //////////


elm::parametex_scale::parametex_scale(const string& f, elm::ParameterList* mod, const double& m) // constructor
: parametex (f,mod)
, _multiplier (m)
{ }

double elm::parametex_scale::pullvalue(const double* pullSource) const
{
	if (mdl) return (_multiplier * pullSource[mdl->FNames[freedom]]);
	return elm::parametex::pullvalue(pullSource);
}
double elm::parametex_scale::pullfield(const std::string& field) const
{
	if (field=="value") {
		if (mdl) return _multiplier * mdl->FInfo[freedom].value;
		return elm::parametex::pullfield(field);
	}
	OOPS("not implemented");
}

void elm::parametex_scale::pushvalue(double* pushDest, const double& q) const
{
	if (mdl)  pushDest[mdl->FNames[freedom]] += (q / _multiplier);
}
void elm::parametex_scale::pushfield(const std::string& field, const double& q) const
{
	if (mdl)  mdl->FInfo[freedom].value += (q / _multiplier);
}

std::string elm::parametex_scale::print() const
{
	ostringstream ret;
	ret << "Freedom: " << std::setw(20) << std::left << freedom ;
	ret << "Parameter: "<< std::setw(12) << std::left << "Scale "<< _multiplier;
	if (mdl) {
		ret << "Freedom Slot: "<<mdl->FNames[freedom] << "\n";
	} else {
		ret << "Freedom Slot: n/a\n";
	}
	return ret.str();
}
/*{
	ostringstream ret;
	ret << "Freedom: " << freedom << "\n";
	ret << "Parameter: Scale "<<_multiplier<<"\n";
	if (mdl) {
		ret << "Freedom Slot: "<<mdl->FNames[freedom] << "\n";
	} else {
		ret << "Freedom Slot: n/a\n";
	}
	return ret.str();
}*/


std::string elm::parametex_scale::smallprint() const
{
	ostringstream ret;
	ret << freedom << "*" << _multiplier;
	return ret.str();
}














elm::paramArray::paramArray(const unsigned& r, const unsigned& c, const unsigned& d)
: three_dim(r,c,d)
, z()
{ }

elm::paramArray::paramArray()
: three_dim(0,0,0)
, z()
{ }

elm::paramArray::~paramArray()
{ 
	delete_and_clear();
}


parametexr& elm::paramArray::operator()(const unsigned& r, const unsigned& c, const unsigned& d)
{
	if (r>=rows) {
		OOPS("paramArray row access out of range, asking ",r," but having only ",rows);
	}
	if (c>=cols) {
		OOPS("paramArray col access out of range, asking ",c," but having only ",cols);
	} 
	if (d>=deps) {
		OOPS("paramArray dep access out of range, asking ",d," but having only ",deps);
	}
	return z[r*cols*deps+c*deps+d];
}
	
const parametexr elm::paramArray::operator()(const unsigned& r, const unsigned& c, const unsigned& d) const
{
	if (r>=rows) {
		OOPS("const paramArray row access out of range, asking ",r," but having only ",rows);
	}
	if (c>=cols) {
		OOPS("const paramArray col access out of range, asking ",c," but having only ",cols);
	} 
	if (d>=deps) {
		OOPS("const paramArray dep access out of range, asking ",d," but having only ",deps);
	}
	return z[r*cols*deps+c*deps+d];
}
	
parametexr& elm::paramArray::operator[](const unsigned& r)
{
	if (r>=rows*cols*deps) {
		OOPS("paramArray single-index access out of range, asking ",r," but having only ",rows*cols*deps);
	}
	return z[r];
}
	
const parametexr elm::paramArray::operator[](const unsigned& r) const
{
	if (r>=rows*cols*deps) {
		OOPS("const paramArray single-index access out of range, asking ",r," but having only ",rows*cols*deps);
	}
	return z[r];
}


void elm::paramArray::resize(const unsigned& r, const unsigned& c, const unsigned& d)
{
	if (r*c*d<rows*cols*deps) {
		for (size_t i=r*c*d; i<rows*cols*deps; i++) {
			if (z[i]) {
				z[i].reset();
				z[i] = parametexr();
			}
		}
	}
	z.resize(r*c*d, parametexr());
	rows=r;
	cols=c;
	deps=d;
}

void elm::paramArray::clear()
{
	resize(0);
}

void elm::paramArray::delete_and_clear()
{
	resize(0);
}

std::string elm::paramArray::__str__() const
{	
	ostringstream ret;
	ret << "<class larch.ParameterLinkArray size=("<<rows<<","<<cols<<","<<deps<<")>\n";
	for (size_t r=0; r<rows; r++) for (size_t c=0; c<cols; c++) for (size_t d=0; d<deps; d++) {
		if (z[r*cols*deps+c*deps+d]) ret << "["<<r<<","<<c<<","<<d<<"]" << z[r*cols*deps+c*deps+d]->print();
	}
	return ret.str();
}

std::string elm::paramArray::__repr__() const
{	
	return "<class larch.ParameterLinkArray>";
}


void elm::paramArray::pull(const etk::ndarray* listorder, etk::ndarray* apporder)
{
	pull_from_freedoms2(*this, apporder->ptr(), listorder->ptr());
}


void elm::paramArray::push(etk::ndarray* listorder, const etk::ndarray* apporder)
{
	push_to_freedoms2(*this, apporder->ptr(), listorder->ptr());
}

void elm::paramArray::pull_field(const std::string& field, etk::ndarray* apporder)
{
	for (unsigned i=0; i<length(); i++) {
		if (z[i]){
			apporder->operator[](i) = z[i]->pullfield(field);
		}
	}
}
void elm::paramArray::push_field(const std::string& field, const etk::ndarray* apporder)
{
	for (unsigned i=0; i<length(); i++) {
		if (z[i]){
			 z[i]->pushfield(field, apporder->operator[](i));
		}
	}
}


void elm::pull_from_freedoms2(const paramArray& par,       double* ops, const double* fr)
{
	for (unsigned i=0; i<par.length(); i++) {
		if (par[i]){
			ops[i] = par[i]->pullvalue(fr);
		}
	}
}
void elm::push_to_freedoms2  (const paramArray& par, const double* ops,       double* fr)
{
	for (unsigned i=0; i<par.length(); i++) {
		if (par[i]){
			par[i]->pushvalue(fr,ops[i]);
		}
	}
}

std::string elm::push_to_freedoms2_  (const paramArray& par, const double* ops,       double* fr)
{
	std::ostringstream x;
	for (unsigned i=0; i<par.length(); i++) {
		if (par[i]){
			par[i]->pushvalue(fr,ops[i]);
			x << "pushed "<<ops[i]<<" using par["<<i<<"]="<< par[i]->print();
		}
	}
	return x.str();
}










