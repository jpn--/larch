/*
 *  elm_sql_queryset.cpp
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

#include <Python.h>
#include "elm_queryset.h"

#include "etk_exception.h"

elm::QuerySet::~QuerySet()
{
	Py_CLEAR(this->py_validator);
}


elm::QuerySet::QuerySet(elm::Facet* validator, PyObject* validator2)
: validator (validator)
, py_validator(nullptr)
{
	Py_XINCREF(validator2);
	this->py_validator = validator2;
}


void elm::QuerySet::set_validator_(elm::Facet* validator, PyObject* validator2)
{
	Py_CLEAR(this->py_validator);

	this->validator = validator;
	Py_XINCREF(validator2);
	this->py_validator = validator2;
}

PyObject* elm::QuerySet::get_validator()
{
	if (this->py_validator) {
		Py_INCREF(this->py_validator);
		return this->py_validator;
	}
	Py_RETURN_NONE;
}


std::string elm::QuerySet::__repr__() const
{
	return "<larch.core.QuerySet>";
}

std::string elm::QuerySet::actual_type() const
{
	return "QuerySet";
}



std::string elm::QuerySet::qry_idco   (const bool& corrected) const
{
	return "QuerySet is an abstract base class, use a derived class";
}
std::string elm::QuerySet::qry_idca   (const bool& corrected) const
{
	return "QuerySet is an abstract base class, use a derived class";
}
std::string elm::QuerySet::qry_idco_  () const
{
	return "QuerySet is an abstract base class, use a derived class";
}
std::string elm::QuerySet::qry_idca_  () const
{
	return "QuerySet is an abstract base class, use a derived class";
}
std::string elm::QuerySet::qry_alts   () const
{
	return "QuerySet is an abstract base class, use a derived class";
}
std::string elm::QuerySet::qry_caseids() const
{
	return "QuerySet is an abstract base class, use a derived class";
}
std::string elm::QuerySet::qry_choice () const
{
	return "QuerySet is an abstract base class, use a derived class";
}
std::string elm::QuerySet::qry_weight () const
{
	return "QuerySet is an abstract base class, use a derived class";
}
std::string elm::QuerySet::qry_avail  () const
{
	return "QuerySet is an abstract base class, use a derived class";
}

bool elm::QuerySet::unweighted() const
{
	return false;
}
bool elm::QuerySet::all_alts_always_available() const
{
	return false;
}



PyObject* elm::QuerySet::pickled  () const
{
	Py_RETURN_NONE;
}






std::string elm::QuerySet::tbl_idco   (const bool& corrected) const
{
	return "("+qry_idco(corrected)+") AS larch_idco";
}

std::string elm::QuerySet::tbl_idca   (const bool& corrected) const
{
	return "("+qry_idca(corrected)+") AS larch_idca";
}

std::string elm::QuerySet::tbl_alts   () const
{
	return "("+qry_alts()+") AS larch_alternatives";
}

std::string elm::QuerySet::tbl_caseids() const
{
	return "("+qry_caseids()+") AS larch_caseids";
}

std::string elm::QuerySet::tbl_choice () const
{
	return "("+qry_choice()+") AS larch_choice";
}

std::string elm::QuerySet::tbl_weight () const
{
	return "("+qry_weight()+") AS larch_weight";
}

std::string elm::QuerySet::tbl_avail  () const
{
	if (qry_avail()=="") {
		return "";
		//OOPS("empty avail query");
	}

	return "("+qry_avail()+") AS larch_avail";
}



