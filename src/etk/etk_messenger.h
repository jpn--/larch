/*
 *  toolbox_messenger.h
 *
 *  Copyright 2007-2013 Jeffrey Newman
 *
 *  This file is part of ELM.
 *  
 *  ELM is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  ELM is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with ELM.  If not, see <http://www.gnu.org/licenses/>.
 *  
 */


#ifndef __TOOLBOX_MESSENGER__
#define __TOOLBOX_MESSENGER__

#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>

#include "etk_logger.h"
/*
#define LOGR(locale, substance, content) if (substance) { (locale) << content << "\n" << etk::messengerPing(); }
#define LOGR_CONDITIONAL(locale, content, level) { (locale) << content << "\n" << etk::messengerLevel(level) << etk::messengerPing(); }

#define BUGGER(locale, content) LOGR_CONDITIONAL(locale, content, 10)
#define INFO(locale, content)      LOGR_CONDITIONAL(locale, content, 20)
#define WARNINGS(locale, content)  LOGR_CONDITIONAL(locale, content, 30)
#define FATAL(locale, content)     LOGR_CONDITIONAL(locale, content, 50)

#define MONITOR(x) ((x) << etk::messengerLevel(20))
#define   << "\n" << etk::messengerPing()

#define WARN(x)   ((x) << etk::messengerLevel(30))
#define    << "\n" << etk::messengerPing()

#define ERRO(x)   ((x) << etk::messengerLevel(40))
#define    << "\n" << etk::messengerPing()
*/
// OUTPUT CONTROLLER //
namespace etk {

	std::string timestamp();
	std::string apply_timestamp(const std::string& logmsg);

	class messenger;

	
	class messengerWidth { public:
		unsigned w;
		messengerWidth(const unsigned& ww):w(ww) { }
	};
	
	class messengerPrecision { public:
		unsigned w;
		messengerPrecision(const unsigned& ww):w(ww) { }
	};

	class messengerLevel { public:
		unsigned lvl;
		messengerLevel(const unsigned& v):lvl(v) { }
	};
	
	class messengerPing { public:
		bool suppress_timestamp;
		bool flusher;
		messengerPing(const bool& st=false, const bool& fl=false):suppress_timestamp(st),flusher(fl) { }
	};
	
	// MESSENGER SERVICE ///////////////////////////////////////////////////////
	// This service acts as a pseudo-pointer to a messenger object.
	// By default, the pointer is NULL.
	// Messages can be sent using the << operator to the messenger service.
	// If no pointer has been set, the messeges are ignored, otherwise they
	//  are logged by the designated messenger.
	class messenger_service {
		messenger_service* msgr;
	public:	
		messenger_service(messenger_service* m=NULL);
		messenger_service(messenger_service& m);
		void assign_messenger (messenger_service* m=NULL) ;
		void assign_messenger (messenger_service& m     ) ;
		
		messenger_service& operator=(const messenger_service& m) ;
		
		virtual messenger_service& operator<<(const std::string& msg);
		virtual messenger_service& operator<<(const char* msg);
		virtual messenger_service& operator<<(const char& msg);
		virtual messenger_service& operator<<(const double& msg);
		virtual messenger_service& operator<<(const unsigned& msg);
		virtual messenger_service& operator<<(const unsigned long& msg);
		virtual messenger_service& operator<<(const int& msg);
		virtual messenger_service& operator<<(const messengerWidth& msg);
		virtual messenger_service& operator<<(const messengerPrecision& msg);
		virtual messenger_service& operator<<(const messengerLevel& msg);
		virtual messenger_service& operator<<(const messengerPing& msg);

//		virtual void connect_file(const std::string& filename, std::ios::openmode mode=(std::ios::app|std::ios::out));
//		virtual void disconnect_file();

		virtual ~messenger_service() { }
	};
	
	
	// MESSENGER ///////////////////////////////////////////////////////////////
	// A messenger object receives messages, and logs them to some output place.
	// 
	class messenger: public messenger_service {
		std::ostringstream msgs;	
		std::ofstream outputfile;
		std::string outputfilename;
		void ping(const bool& suppress_timestamp);
//#ifndef NO_PYTHON_LOGGING
		loggerToPy logger;
//#endif
		unsigned current_level;
	public:
		messenger(const std::string& loggerName="");
		virtual ~messenger();

		void message(const std::string& msg);
		
		virtual messenger_service& operator<<(const std::string& msg);
		virtual messenger_service& operator<<(const char* msg);
		virtual messenger_service& operator<<(const char& msg);
		virtual messenger_service& operator<<(const double& msg);
		virtual messenger_service& operator<<(const unsigned& msg);
		virtual messenger_service& operator<<(const unsigned long& msg);
		virtual messenger_service& operator<<(const int& msg);
		virtual messenger_service& operator<<(const messengerWidth& msg);
		virtual messenger_service& operator<<(const messengerPrecision& msg);		
		virtual messenger_service& operator<<(const messengerLevel& msg);		
		virtual messenger_service& operator<<(const messengerPing& msg);
		
		virtual void connect_file(const std::string& filename, std::ios::openmode mode=(std::ios::app|std::ios::out));
		virtual void disconnect_file();
		bool mute_stdout;
	};
	
	
	/*
	class status_update {
		clock_t prev_update_time;
		clock_t interval;
		messenger_service* msgr;
	public:
		status_update(messenger_service* m, const unsigned& ticks);
		status_update(messenger_service& m, const unsigned& ticks);
		void tell(const std::string& msg);
	};
*/
	
	
	
	
	
	template<class T>
	std::string pointer_as_string(T* t)
	{
		std::ostringstream ret;
		ret << t;
		return ret.str();
	}

	template<class T>
	std::string thing_as_string(const T& t)
	{
		std::ostringstream ret;
		ret << t;
		return ret.str();
	}
	


	
} // end namespace etk

#endif

