/*
 *  toolbox_logger.h
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


#ifndef __TOOLBOX_LOGGER__
#define __TOOLBOX_LOGGER__

#include "etk_python.h"
#include <string>
#include <fstream>
#include <sstream>

namespace etk {

	class loggerToPy {		
		PyObject* logObj;
	public:
		loggerToPy(const std::string& loggerName="", const std::string& loggerSubName="");
		~loggerToPy();
//		void FATALpy(const std::string& msg);
//		void ERRORpy(const std::string& msg);
//		void WARNpy(const std::string& msg);
//		void INFOpy(const std::string& msg);
//		void DEBUGpy(const std::string& msg);
		void LOGpy(const unsigned& lvl, const std::string& msg);
	};
	
	
	class logging_service {

	  private:
		PyObject* logObj;
		std::ofstream outputfile;
		void _get_python_object();

		// This contains the name of the pylogger and/or output file in use. Only the
		//  names should be passed to copies of this logging_service, which then can 
		//  get their own python objects to operate with.
		std::string py_logger_name;

	  public:
		std::ostringstream buffered_message;

	  public:
		// Constructors and Destructor
		logging_service(const std::string& loggerName="", const std::string& loggerSubName="");
		logging_service(const logging_service& s);
		logging_service(logging_service* s);
		logging_service& operator=(const logging_service& s);
		~logging_service();
		
		void change_logger_name(const std::string& new_name);
		std::string get_logger_name() const;
				
		void connect_file(const std::string& filename,
					      std::ios::openmode mode=(std::ios::app|std::ios::out));
		void disconnect_file();
		
		void mute();
		
		void output_message(const unsigned& level, const std::string& message);
		
		PyObject* get_logger();
		PyObject* set_logger(PyObject* z);
		
		friend class log_instance;
	};
	
	
	class log_instance {
		PyObject* logObj;
		logging_service* svc;
		std::ostringstream msgs;	
		unsigned level;
	public:
		
		// Streaming
		// Stream output messages into the log
		log_instance& operator<<(const std::string& msg);
		log_instance& operator<<(const char* msg);
		log_instance& operator<<(const char& msg);
		log_instance& operator<<(const double& msg);
		log_instance& operator<<(const unsigned& msg);
		log_instance& operator<<(const unsigned long& msg);
		log_instance& operator<<(const unsigned long long& msg);
		log_instance& operator<<(const long long& msg);
		log_instance& operator<<(const int& msg);
		log_instance& operator<<(std::ostream& ( *pf )(std::ostream&));
		log_instance& operator<<(std::ios& ( *pf )(std::ios&));
		log_instance& operator<<(std::ios_base& ( *pf )(std::ios_base&));
		
		// Constructor
		// Generally, create log_instance as an anonymous class object.
		log_instance(logging_service& svc, const unsigned& lvl);
		log_instance(logging_service& svc, const unsigned& lvl, const std::string& buffer);
		//log_instance(log_instance& i);
		
		// Destructor
		// ~log_instance will be destroyed at the end of the expression.
		// The destructor sends the msgs to the output stream[s].
		~log_instance();
	};
	
	class quiet_instance {
	public:
		quiet_instance() { }
		void operator&(log_instance&) { }   // This has to be an operator with a precedence lower than << but higher than ?:
	};
	
	
	
	
	class periodic {
		clock_t prev_update_time;
		clock_t interval;
	public:
		periodic(const unsigned& ticks);
		bool ping();
	};
	
	
}



#define BUGGER(msgr)  etk::log_instance((msgr),05)
#define MONITOR(msgr) etk::log_instance((msgr),10)
#define INFO(msgr)    etk::log_instance((msgr),20)
#define WARN(msgr)    etk::log_instance((msgr),30)
#define ERRO(msgr)    etk::log_instance((msgr),40)
#define FATAL(msgr)   etk::log_instance((msgr),50)

#define BUGGER_BUFFER(msgr)  (msgr).buffered_message
#define MONITOR_BUFFER(msgr) (msgr).buffered_message
#define INFO_BUFFER(msgr)    (msgr).buffered_message
#define WARN_BUFFER(msgr)    (msgr).buffered_message
#define ERRO_BUFFER(msgr)    (msgr).buffered_message
#define FATAL_BUFFER(msgr)   (msgr).buffered_message

#define QUIET(msgr)  (void) 0

#define BUGGER_(msgr_ptr, message) {if (msgr_ptr) { etk::log_instance(*(msgr_ptr),05) << message; } else { }}
#define MONITOR_(msgr_ptr, message) {if (msgr_ptr) { etk::log_instance(*(msgr_ptr),10) << message; } else { }}
#define INFO_(msgr_ptr, message) {if (msgr_ptr) { etk::log_instance(*(msgr_ptr),20) << message; } else { }}
#define WARN_(msgr_ptr, message) {if (msgr_ptr) { etk::log_instance(*(msgr_ptr),30) << message; } else { }}
#define ERROR_(msgr_ptr, message) {if (msgr_ptr) { etk::log_instance(*(msgr_ptr),40) << message; } else { }}
#define FATAL_(msgr_ptr, message) {if (msgr_ptr) { etk::log_instance(*(msgr_ptr),50) << message; } else { }}

#define PERIODICALLY(ticker, streamer) !((ticker).ping()) ? (void) 0 : etk::quiet_instance() & streamer


#endif // __TOOLBOX_LOGGER__

