/*
 *  etk_messenger.cpp
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



#include "etk_messenger.h"
#include "etk_exception.h"

#include <iostream>


std::string etk::timestamp()
{
	char t [25];
	time_t now;
	time(&now);
	strftime(t,25,"%b %d %I:%M:%S %p",localtime(&now));
	return t;
}

std::string etk::apply_timestamp(const std::string& logmsg)
{
	unsigned i (0);
	size_t j (0);
	size_t k (0);
    std::ostringstream ret;
	std::string t (timestamp());
	while (i<logmsg.length()-1) {
		ret << t << " ";
		j = logmsg.find_first_of("\n",i);
		k = j-i+1;
		ret << logmsg.substr(i,k);
		i += k;
	}
	return ret.str();
}

/*
void etk::messenger_service::connect_file(const std::string& filename, std::ios::openmode mode)
{
	msgr->connect_file( filename, mode );
}

void etk::messenger_service::disconnect_file()
{
	msgr->disconnect_file( );
}
*/

void etk::messenger::connect_file(const std::string& filename, std::ios::openmode mode)
{
	if (outputfilename != filename) {
		outputfilename = filename;
		if (outputfile) outputfile.close( );
		outputfile.open( filename.c_str(), mode );
		if( !outputfile ) {
			std::cerr << "Error opening output stream" << std::endl;
			OOPS("error in opening output file '",filename,"'");
		}   
	}
}

void etk::messenger::disconnect_file()
{
	ping(true);
	outputfilename.clear();
	outputfile.close( );
}

etk::messenger::messenger(const std::string& loggerName)
: mute_stdout (false)
#ifndef NO_PYTHON_LOGGING
//, logger ("etk.core",loggerName)
#endif
, current_level (10)
{
	if (outputfile) outputfile.close( );
}

etk::messenger::~messenger()
{
	if (outputfile) outputfile.close( );
}


void etk::messenger::ping(const bool& suppress_timestamp)
{
#ifdef NO_PYTHON_LOGGING
	if (outputfile){
		if (!suppress_timestamp) {
			outputfile << apply_timestamp(msgs.str());
		} else {
			outputfile << msgs.str();
		}
	} 
	if (!mute_stdout){
		cout << (msgs.str());		
	}
#else
	logger.LOGpy(current_level, msgs.str());
#endif
	msgs.str("");
	msgs.clear();
}

void etk::messenger::message(const std::string& msg)
{
	msgs << msg;
//	ping();
}

etk::messenger_service& etk::messenger::operator<<(const std::string& msg)
{
	msgs << msg;
//	ping();
	return *(dynamic_cast<etk::messenger_service*>(this));
}

etk::messenger_service& etk::messenger::operator<<(const char* msg)
{
	msgs << msg;
//	ping();
	return *(dynamic_cast<etk::messenger_service*>(this));
}

etk::messenger_service& etk::messenger::operator<<(const char& msg)
{
	msgs << msg;
//	ping();
	return *(dynamic_cast<etk::messenger_service*>(this));
}

etk::messenger_service& etk::messenger::operator<<(const double& msg)
{
	msgs << msg;
//	ping();
	return *(dynamic_cast<etk::messenger_service*>(this));
}

etk::messenger_service& etk::messenger::operator<<(const unsigned& msg)
{
	msgs << msg;
//	ping();
	return *(dynamic_cast<etk::messenger_service*>(this));
}

etk::messenger_service& etk::messenger::operator<<(const unsigned long& msg)
{
	msgs << msg;
//	ping();
	return *(dynamic_cast<etk::messenger_service*>(this));
}

etk::messenger_service& etk::messenger::operator<<(const int& msg)
{
	msgs << msg;
//	ping();
	return *(dynamic_cast<etk::messenger_service*>(this));
}

etk::messenger_service& etk::messenger::operator<<(const etk::messengerWidth& msg)
{
	msgs.width(msg.w);
	return *(dynamic_cast<etk::messenger_service*>(this));
}

etk::messenger_service& etk::messenger::operator<<(const etk::messengerPrecision& msg)
{
	msgs.precision(msg.w);
	return *(dynamic_cast<etk::messenger_service*>(this));
}

etk::messenger_service& etk::messenger::operator<<(const messengerLevel& msg)
{
	current_level = msg.lvl;
	return *(dynamic_cast<etk::messenger_service*>(this));
}

etk::messenger_service& etk::messenger::operator<<(const messengerPing& msg)
{
	ping(msg.suppress_timestamp);
	if (msg.flusher) {
		if (outputfile){
			outputfile.flush();
		} 
		if (!mute_stdout){
            std::cout.flush();		
		}		
	}	
	return *(dynamic_cast<etk::messenger_service*>(this));
}


etk::messenger_service::messenger_service(messenger_service* m)
: msgr (m) 
{ }

etk::messenger_service::messenger_service(messenger_service& m)
: msgr (&m) 
{ }

void etk::messenger_service::assign_messenger (messenger_service* m) 
{ 
	msgr = m; 
}
void etk::messenger_service::assign_messenger (messenger_service& m     ) 
{ 
	msgr = &m; 
}

etk::messenger_service& etk::messenger_service::operator=(const messenger_service& m) 
{ 
	msgr = m.msgr; return *this; 
}



etk::messenger_service& etk::messenger_service::operator<<(const std::string& msg)
{
	if (msgr) (*msgr) << msg;
	return *this;
}
etk::messenger_service& etk::messenger_service::operator<<(const char* msg)
{
	if (msgr) (*msgr) << msg;
	return *this;
}
etk::messenger_service& etk::messenger_service::operator<<(const char& msg)
{
	if (msgr) (*msgr) << msg;
	return *this;
}
etk::messenger_service& etk::messenger_service::operator<<(const double& msg)
{
	if (msgr) (*msgr) << msg;
	return *this;
}
etk::messenger_service& etk::messenger_service::operator<<(const unsigned& msg)
{
	if (msgr) (*msgr) << msg;
	return *this;
}
etk::messenger_service& etk::messenger_service::operator<<(const unsigned long& msg)
{
	if (msgr) (*msgr) << msg;
	return *this;
}
etk::messenger_service& etk::messenger_service::operator<<(const int& msg)
{
	if (msgr) (*msgr) << msg;
	return *this;
}

etk::messenger_service& etk::messenger_service::operator<<(const messengerWidth& msg)
{
	if (msgr) (*msgr) << msg;
	return *this;
}

etk::messenger_service& etk::messenger_service::operator<<(const messengerPrecision& msg)
{
	if (msgr) (*msgr) << msg;
	return *this;
}

etk::messenger_service& etk::messenger_service::operator<<(const messengerLevel& msg)
{
	if (msgr) (*msgr) << msg;
	return *this;
}

etk::messenger_service& etk::messenger_service::operator<<(const messengerPing& msg)
{
	if (msgr) (*msgr) << msg;
	return *this;
}

	/*
etk::status_update::status_update(messenger_service* m, const unsigned& ticks)
: prev_update_time (clock())
, interval (ticks*CLOCKS_PER_SEC)
, msgr (m)
{ }

etk::status_update::status_update(messenger_service& m, const unsigned& ticks)
: prev_update_time (clock())
, interval (ticks*CLOCKS_PER_SEC)
, msgr (&m)
{ }


void etk::status_update::tell(const std::string& msg)
{
	clock_t timenow = clock();
	if (timenow - prev_update_time > interval) {
		MONITOR(*msgr) << msg ;
		prev_update_time = timenow;
	}
}
*/




