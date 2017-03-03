/*
 *  toolbox_logger.cpp
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


#include "etk_logger.h"
#include "etk_exception.h"
#include <string>

#include <iomanip>
#include <iostream>

using namespace etk;

std::string __TrimFinalLineFeeds(std::string s)
{
	std::string::reverse_iterator rit;
	while (*(s.rbegin()) == '\n' || *(s.rbegin()) == '\r') {
		s = s.substr(0,s.length()-1);
	}
	return s;
}




loggerToPy::loggerToPy(const std::string& loggerName, const std::string& loggerSubName)
: logObj ( NULL )
{
	std::string ln (loggerName);
	if ((ln!="") && (loggerSubName!="")) {
		ln += ".";
		ln += loggerSubName;
	}
	if (ln!="") {
		PyObject* sysLoggingModule = PyImport_ImportModule("logging");
		PyObject* sysGetLogger = PyObject_GetAttrString(sysLoggingModule, "getLogger");	
		logObj = PyObject_CallFunction(sysGetLogger, "(s)", ln.c_str());
		Py_CLEAR(sysGetLogger);
		Py_CLEAR(sysLoggingModule);
	}
}

loggerToPy::~loggerToPy()
{
	Py_CLEAR(logObj);
}

/*
void loggerToPy::FATALpy(const std::string& msg)
{
	if (logObj) {
		std::string s = __TrimFinalLineFeeds(msg);
		PyObject* o = PyObject_CallMethod(logObj, "critical", "(s)", s.c_str());
		Py_CLEAR(o);
	}
}

void loggerToPy::ERRORpy(const std::string& msg)
{
	if (logObj) {
		std::string s = __TrimFinalLineFeeds(msg);
		PyObject* o = PyObject_CallMethod(logObj, "error", "(s)", s.c_str());
		Py_CLEAR(o);
	}
}

void loggerToPy::WARNpy(const std::string& msg)
{
	if (logObj) {
		std::string s = __TrimFinalLineFeeds(msg);
		PyObject* o = PyObject_CallMethod(logObj, "warning", "(s)", s.c_str());
		Py_CLEAR(o);
	}
}

void loggerToPy::INFOpy(const std::string& msg)
{
	if (logObj) {
		std::string s = __TrimFinalLineFeeds(msg);
		PyObject* o = PyObject_CallMethod(logObj, "info", "(s)", s.c_str());
		Py_CLEAR(o);
	}
}

void loggerToPy::DEBUGpy(const std::string& msg)
{
	if (logObj) {
		std::string s = __TrimFinalLineFeeds(msg);
		PyObject* o = PyObject_CallMethod(logObj, "debug", "(s)", s.c_str());
		Py_CLEAR(o);
	}
}

 */
 
void loggerToPy::LOGpy(const unsigned& lvl, const std::string& msg)
{
	if (logObj) {
		std::string s = __TrimFinalLineFeeds(msg);
		PyObject* o = PyObject_CallMethod(logObj, "log", "Is", lvl, s.c_str());
		Py_CLEAR(o);
	}
}

void loggerToPyTest()
{
	loggerToPy eL ("etk.core");
//	eL.FATALpy("this message is fatal");
//	eL.WARNpy("this message is warn");
//	eL.ERRORpy("this message is error");
//	eL.INFOpy("this message is info");
//	eL.DEBUGpy("this message is debug");
	eL.LOGpy(25,"this message is 25");	
	eL.LOGpy(55,"this message is 55");	
}



/* LOGGING SERVICE */

void etk::logging_service::_get_python_object()
{ 
	Py_CLEAR(logObj);
	if (py_logger_name!="") {
		PyObject* sysLoggingModule = PyImport_ImportModule("logging");
		PyObject* sysGetLogger = PyObject_GetAttrString(sysLoggingModule, "getLogger");	
		logObj = PyObject_CallFunction(sysGetLogger, "(s)", py_logger_name.c_str());
		Py_CLEAR(sysGetLogger);
		Py_CLEAR(sysLoggingModule);
	}
}

PyObject* etk::logging_service::get_logger()
{
	if (!logObj) {
		Py_RETURN_NONE;
	} else {
		Py_INCREF( logObj );
		return logObj;
	}
}

PyObject* etk::logging_service::set_logger(PyObject* z)
{
	if (z == Py_None) {
    	Py_CLEAR(logObj);
		py_logger_name = "";
		Py_RETURN_NONE;
	}

	PyObject* name = PyObject_GetAttrString(z, "name");
	py_logger_name = PyString_ExtractCppString(name);
	Py_CLEAR(name);
	
	Py_CLEAR(logObj);
	Py_XINCREF( z );
	logObj = z;

	if (!logObj) {
		Py_RETURN_NONE;
	} else {
		Py_INCREF( logObj );
		return logObj;
	}
}


void etk::logging_service::change_logger_name(const std::string& new_name)
{
	py_logger_name = new_name;
	Py_CLEAR(logObj);
	if (new_name!="") {
		_get_python_object();
	}
}

std::string etk::logging_service::get_logger_name() const
{
	return py_logger_name;
}


etk::logging_service::logging_service(const std::string& loggerName, const std::string& loggerSubName)
: logObj (NULL)
, py_logger_name (loggerName)
{ 
	std::string ln (loggerName);
	if ((py_logger_name!="") && (loggerSubName!="")) {
		py_logger_name += ".";
		py_logger_name += loggerSubName;
	}
	_get_python_object();
}

etk::logging_service::logging_service(const logging_service& s)
: logObj (NULL)
, py_logger_name (s.py_logger_name)
{ 
	_get_python_object();
}

etk::logging_service::logging_service(logging_service* s)
: logObj (NULL)
, py_logger_name ()
{ 
	if (s) {
		py_logger_name = s->py_logger_name;
	}
	_get_python_object();
}

etk::logging_service& etk::logging_service::operator=(const etk::logging_service& s)
{
	Py_CLEAR(logObj);
	this->py_logger_name = s.py_logger_name;
	_get_python_object();
	return *this;
}

etk::logging_service::~logging_service()
{ 
	Py_CLEAR(logObj);
}

void etk::logging_service::mute()
{
	Py_CLEAR(logObj);
	if (outputfile) {
		outputfile.close();
	}
}



void etk::logging_service::output_message(const unsigned& level, const std::string& message)
{
	if (logObj) {
		
		boosted::lock_guard<boosted::mutex> LOCK(etk::python_global_mutex);
		
		int interrupt ( PyErr_CheckSignals() );
	
		PyObject* ptype (NULL);
		PyObject* pvalue (NULL); 
		PyObject* ptraceback (NULL);
	//	if (PyErr_Occurred()) {
			PyErr_Fetch(&ptype, &pvalue, &ptraceback);
	//	}
	
		try {
			std::string s = __TrimFinalLineFeeds(message);
			PyObject* o = PyObject_CallMethod(logObj, "log", "Is", level, s.c_str());
			Py_CLEAR(o);
		} catch (...) {
			std::cerr << "ERROR LOGGING\n";
		}
		
	//	if (ptype) {
			PyErr_Restore( ptype,  pvalue,  ptraceback);
	//	}
	
		if (interrupt) PyErr_SetInterrupt();
	}
	
}

etk::log_instance::log_instance(etk::logging_service& svc, const unsigned& lvl)
: logObj (svc.logObj)
, level   (lvl)
, svc    (&svc)
{

}

etk::log_instance::log_instance(etk::logging_service& svc, const unsigned& lvl, const std::string& buffer)
: logObj (svc.logObj)
, level   (lvl)
, svc    (&svc)
{ 

	(*this) << buffer;
}

etk::log_instance& etk::log_instance::operator<<(const std::string& msg)
{
	msgs << msg;
	return *(this);
}

etk::log_instance& etk::log_instance::operator<<(const char* msg)
{
	msgs << msg;
	return *(this);
}

etk::log_instance& etk::log_instance::operator<<(const char& msg)
{
	msgs << msg;
	return *(this);
}

etk::log_instance& etk::log_instance::operator<<(const double& msg)
{
	msgs << msg;
	return *(this);
}

etk::log_instance& etk::log_instance::operator<<(const unsigned& msg)
{
	msgs << msg;
	return *(this);
}

etk::log_instance& etk::log_instance::operator<<(const unsigned long& msg)
{
	msgs << msg;
	return *(this);
}

etk::log_instance& etk::log_instance::operator<<(const unsigned long long& msg)
{
	msgs << msg;
	return *(this);
}

etk::log_instance& etk::log_instance::operator<<(const long long& msg)
{
	msgs << msg;
	return *(this);
}

etk::log_instance& etk::log_instance::operator<<(const int& msg)
{
	msgs << msg;
	return *(this);
}

etk::log_instance& etk::log_instance::operator<<(const void* msg)
{
	msgs << msg;
	return *(this);
}

etk::log_instance& etk::log_instance::operator<<(std::ostream& ( *pf )(std::ostream&))
{
	msgs << pf;
	return *(this);
}

etk::log_instance& etk::log_instance::operator<<(std::ios& ( *pf )(std::ios&))
{
	msgs << pf;
	return *(this);
}
etk::log_instance& etk::log_instance::operator<<(std::ios_base& ( *pf )(std::ios_base&))
{
	msgs << pf;
	return *(this);
}


etk::log_instance::~log_instance()
{
	svc->output_message(level, msgs.str());

/*
	if (logObj) {
		try {
			std::string s = __TrimFinalLineFeeds(msgs.str());
			PyObject* o = PyObject_CallMethod(logObj, "log", "Is", level, s.c_str());
			Py_CLEAR(o);
		} catch (...) {
			std::cerr << "ERROR LOGGING\n";
		}
	}
	*/
}


etk::periodic::periodic(const unsigned& ticks)
: prev_update_time (clock())
, interval (ticks*CLOCKS_PER_SEC)
{ }

std::string etk::full_precision_hex(const double& x)
{
	std::ostringstream s;
	s << std::hexfloat << x;
	std::string ss = s.str();
	
	if (ss.size()>20)
		return ss.substr(0,9)+" "+ss.substr(9,5)+" "+ss.substr(14,5)+" "+ss.substr(19);
	if (ss.size()>16)
		return ss.substr(0,9)+" "+ss.substr(9,5)+" "+ss.substr(14);
	if (ss.size()>11)
		return ss.substr(0,9)+" "+ss.substr(9);
	return ss;
}


bool etk::periodic::ping() 
{
	clock_t timenow = clock();
	if (timenow - prev_update_time > interval) {
		prev_update_time = timenow;
		return true;
	}
	return false;
}


