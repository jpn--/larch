/*
 *  elm_runstats.h
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


#ifndef __ELM_RUNSTATS_H__
#define __ELM_RUNSTATS_H__

#include <ctime>
#include <string>
#include <sstream>
#ifdef __APPLE__
#include <sys/time.h>
#else
#ifndef SWIG
#include <time.h>
struct timeval
{
	long	tv_sec;		/* seconds */
	long	tv_usec;	/* and microseconds */
	
	timeval& operator=(const timeval& other){tv_sec=other.tv_sec; tv_usec=other.tv_usec; return *this;}
};
#endif // ndef SWIG
#endif // __APPLE__

#include <chrono>


#include "etk.h"

namespace elm {

	class runstats {
	public:
		unsigned iteration;
		std::string results;		
		std::string timestamp;
		int number_threads;
		int number_cpu_cores;

		std::string processor;
		
		std::vector<std::string> process_label;
		std::vector< std::chrono::time_point<std::chrono::high_resolution_clock> > process_starttime;
		std::vector< std::chrono::time_point<std::chrono::high_resolution_clock> > process_endtime;

		std::vector<std::string> _notes;
		
		double elapsed_time() const;
		double runtime_seconds() const;
		std::string runtime() const;
		
		runstats(long startTimeSec=0, long startTimeUSec=0,
				 long endTimeSec=0, long endTimeUSec=0,
				 unsigned iteration=0,
				 std::string results="",
				 std::string notes="",
				 std::string timestamp="",
				 std::string processor="?",
				 int number_threads=-9,
				 int number_cpu=-9);
		runstats(const runstats& other);
		runstats(PyObject* dictionary);
		~runstats();

		PyObject* other();
		void set_other(PyObject* other);

		void prepend_timing(const runstats& previously);
		void append_timing(const runstats& subsequently);
		
	private:
		void restart();
		void iter();

		PyObject* _other_attr;
		
	private:
//		std::string _notes;
		friend class Model2;
	public:
		std::string notes() const;
		
		void write(std::string note);
		void write(char* note);
		void flush();
		void write_result(const std::string& a, const std::string& b="", const std::string& c="");
		
		std::string __repr__() const;
		PyObject* dictionary() const;
		std::string pickled_dictionary() const;
		void read_from_dictionary(PyObject* dictionary);
		
		
		void start_process(const std::string& name);
		void end_process();
		
		double process_duration(const std::string& name) const;
		double process_duration(const size_t& number) const;
		double total_duration() const;
		std::string process_duration_fancy(const std::string& name) const;
		std::string process_duration_fancy(const size_t& number) const;
		std::string total_duration_fancy() const;

		PyObject* __getstate__() const;
		
		#ifdef SWIG
		%pythoncode %{
		def __setstate__(self, state):
			self.__init__()
			self.read_from_dictionary(state)
		%}
		#endif // SWIG

		
	};


}

#endif

