/*
 *  versioning.cpp
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


#include "elm_version.h"
#include <iostream>
#include <sstream>
#include <cstdlib>

#ifdef __GNUC__
#define __COMPILER__ "GCC"
#else 
#define __COMPILER__ "unidentified compiler"
#endif

#ifdef __CMAKE__
#define _SVNVERS_ "CMAKE"
#else
#define _SVNVERS_ "XXX"
#endif

#ifndef __BUILDCONFIG__
#define __BUILDCONFIG__ "Unknown Configuration"
#endif

#ifndef _SVNVERS_
#define _SVNVERS_ "GIT"
#endif

#ifndef __VERSION__
#define __VERSION__ ""
#endif

std::string elm::svn_version()
{
	char * pPath;
	pPath = getenv ("SVN_VERSION");
	std::string vers = _SVNVERS_;
	if (pPath!=NULL)
		vers = pPath;
	return vers;
}

std::string elm::build_configuration()
{
	std::string bldcfgvalue (__BUILDCONFIG__);
	return bldcfgvalue;
}

const int elm::major_version = 2;
const int elm::minor_version = 3;

std::string elm::version()
{
	std::ostringstream ret;
	ret << "ELM Core version "<<elm::major_version<<"."<<elm::minor_version;
	ret << " (Hangman " << build_configuration() << ") built using "<< __COMPILER__ <<" "<< __VERSION__;
	ret << " on " << __DATE__ << " at " << __TIME__;
	return ret.str();
}
	
