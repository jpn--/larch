/*
 *  elm_runstats.cpp
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


#include "elm_runstats.h"
#include <string>
#include <sstream>
#include <cmath>
#include "etk.h"
#include <iostream>

#ifdef __APPLE__
double timeval_subtract (const timeval& y, const timeval& x)
{
	double t (0);
	t += double(y.tv_sec - x.tv_sec);
	t += (double(y.tv_usec - x.tv_usec) / 1000000);
	return t;
}
#endif
	
	
double elm::runstats::elapsed_time() const
{
//#ifdef __APPLE__
//	timeval t;
//	gettimeofday( &t, NULL);
//	return timeval_subtract(t,startTime);
//#else
//	return 0;
//#endif
	return total_duration();
}

double elm::runstats::runtime_seconds() const
{
//#ifdef __APPLE__
//	return timeval_subtract(endTime, startTime);
//#else
//	return 0;
//#endif
	return total_duration();
}

std::string elm::runstats::runtime() const
{
	return etk::hours_minutes_seconds(runtime_seconds(),true);
}

elm::runstats::runstats(long startTimeSec, long startTimeUSec,
				 long endTimeSec, long endTimeUSec,
				 unsigned iteration,
				 std::string results,
				 std::string notes,
				 std::string timestamp,
				 std::string processor,
				 int number_threads,
				 int number_cpu)
: startTime ()
, endTime ()
, iteration (iteration)
, _notes (notes)
, results (results)
, timestamp (timestamp)
, processor(processor)
, number_threads(number_threads)
, number_cpu_cores(number_cpu)
{
	//#ifdef __APPLE__
	//	if (startTimeSec || startTimeUSec) {
	//		startTime.tv_sec = startTimeSec;
	//		startTime.tv_usec = startTimeUSec;
	//		endTime.tv_sec = endTimeSec;
	//		endTime.tv_usec = endTimeUSec;
	//	} else {
	//		gettimeofday( &startTime, NULL);
	//		endTime.tv_sec = 0;
	//		endTime.tv_usec = 0;
	//	}
	//#endif
	if (this->processor=="?") this->processor=etk::discovered_platform_description;
	if (this->number_threads==-9) {
		this->number_threads = etk::number_of_cpu;
	}
	if (this->number_cpu_cores==-9) {
		this->number_cpu_cores = etk::number_of_cpu;
	}
}

elm::runstats::runstats(const runstats& other)
: iteration (other.iteration)
, _notes (other._notes)
, results (other.results)
, timestamp (other.timestamp)
, processor(other.processor)
, process_label(other.process_label)
, process_starttime(other.process_starttime)
, process_endtime(other.process_endtime)
, number_threads(other.number_threads)
{
	if (this->processor=="?") this->processor=etk::discovered_platform_description;
}

elm::runstats::runstats(PyObject* dictionary)
: startTime ()
, endTime ()
, iteration (0)
, _notes ()
, results ()
, timestamp ()
, processor ()
, number_threads(0)
{
	read_from_dictionary(dictionary);
}

void elm::runstats::restart()
{
	iteration = 0;
	_notes.clear();
	results.clear();
	process_endtime.clear();
	process_starttime.clear();
	process_label.clear();
}


void elm::runstats::iter()
{
	iteration++;
}

std::string elm::runstats::notes() const
{
	return _notes;
}

std::string elm::runstats::__repr__() const
{
	std::ostringstream x;
	x << "<larch.core.runstats, ";
	if (results.empty()) x << "No result";
	x << results << " in " << runtime()<<">";
	return x.str();
}

void elm::runstats::write(std::string note)
{
//	if (_notes=="") {
//		_notes = note;
//		return;
//	}
//	_notes += "\n";
	_notes += note;
}

void elm::runstats::write(char* note)
{
//	if (_notes=="") {
//		_notes = note;
//		return;
//	}
//	_notes += "\n";
	_notes += note;
}

void elm::runstats::flush()
{
	
}

void elm::runstats::write_result(const std::string& a, const std::string& b, const std::string& c)
{
	if (results!="") {
		results += "/n";
	}
	results += a;
	results += b;
	results += c;
}

PyObject* elm::runstats::dictionary() const
{
	PyObject* P = PyDict_New();
//	#ifdef __APPLE__
//	etk::py_add_to_dict(P, "startTimeSec", startTime.tv_sec);
//	etk::py_add_to_dict(P, "startTimeUSec", startTime.tv_usec);
//	etk::py_add_to_dict(P, "endTimeSec", endTime.tv_sec);
//	etk::py_add_to_dict(P, "endTimeUSec", endTime.tv_usec);
//	#endif
	etk::py_add_to_dict(P, "iteration", iteration);
	etk::py_add_to_dict(P, "results", results);
	etk::py_add_to_dict(P, "notes", _notes);
	etk::py_add_to_dict(P, "timestamp", timestamp);
	etk::py_add_to_dict(P, "processor", processor);
	etk::py_add_to_dict(P, "number_threads", number_threads);
	etk::py_add_to_dict(P, "number_cpu_cores", number_cpu_cores);

	etk::py_add_to_dict(P, "process_label", process_label);
	etk::py_add_to_dict(P, "process_starttime", process_starttime);
	etk::py_add_to_dict(P, "process_endtime", process_endtime);

	// These are not used in reading in the data, but are for convenience in printing out.
	std::vector< std::string > fancy_durations;
	for (size_t i=0; i<process_endtime.size(); i++) {
		fancy_durations.push_back(process_duration_fancy(i));
	}

	etk::py_add_to_dict(P, "process_durations", fancy_durations);
	etk::py_add_to_dict(P, "total_duration", total_duration_fancy());
	etk::py_add_to_dict(P, "total_duration_seconds", total_duration());
	
	return (P);
}

std::string elm::runstats::pickled_dictionary() const
{


	PyObject* D = dictionary();
	PyObject* pickle = PyObject_CallMethod(etk::pickle_module, "dumps", "O", D);
	Py_CLEAR(D);
	
	PyObject* b85pickle = PyObject_CallMethod(etk::base64_module, "b85encode", "O", pickle);
	
	std::string v = PyBytes_AsString(b85pickle);
	Py_CLEAR(pickle);
	Py_CLEAR(b85pickle);
	return v;
}

void elm::runstats::read_from_dictionary(PyObject* P)
{
	PyObject* pickle = nullptr;
	PyObject* b85pickle = nullptr;
	
//	auto q1 = PyList_CheckExact(P);
//	auto q2 = PyUnicode_Check(P);
	
	
	if (PyList_CheckExact(P)) {
		if (PyList_Size(P)==1) {
			P = PyList_GetItem(P, 0);
		} else {
			OOPS("need a dict or a list of ony one dict");
		}
	}
	
	if (PyUnicode_Check(P)) {
		b85pickle = PyObject_CallMethod(etk::base64_module, "b85decode", "O", P);
		pickle = PyObject_CallMethod(etk::pickle_module, "loads", "O", b85pickle);
		if (pickle != NULL) {
			P = pickle;
		}
	}


	int x;
//	#ifdef __APPLE__
//	x=etk::py_read_from_dict(P, "startTimeSec", startTime.tv_sec);
//	if (x!=0) OOPS("error in reading run_stats startTimeSec, code ",x);
//	x=etk::py_read_from_dict(P, "startTimeUSec", startTime.tv_usec);
//	if (x!=0) OOPS("error in reading run_stats startTimeUSec");
//	x=etk::py_read_from_dict(P, "endTimeSec", endTime.tv_sec);
//	if (x!=0) OOPS("error in reading run_stats endTimeSec");
//	x=etk::py_read_from_dict(P, "endTimeUSec", endTime.tv_usec);
//	if (x!=0) OOPS("error in reading run_stats endTimeUSec");
//	#endif
	x=etk::py_read_from_dict(P, "iteration", iteration);
	if (x!=0) OOPS("error in reading run_stats iteration");
	x=etk::py_read_from_dict(P, "results", results);
	if (x!=0) OOPS("error in reading run_stats results");
	x=etk::py_read_from_dict(P, "timestamp", timestamp);
	if (x!=0) OOPS("error in reading run_stats timestamp");
	x=etk::py_read_from_dict(P, "processor", processor);
	if (x!=0) OOPS("error in reading run_stats processor");
	x=etk::py_read_from_dict(P, "notes", _notes);
	if (x!=0) OOPS("error in reading run_stats notes");
	x=etk::py_read_from_dict(P, "number_threads", number_threads);
	if (x!=0) OOPS("error in reading run_stats number_threads");
	x=etk::py_read_from_dict(P, "number_cpu_cores", number_cpu_cores);
	if (x!=0) OOPS("error in reading run_stats number_cpu_cores");

	x=etk::py_read_from_dict(P, "process_label", process_label);
	if (x!=0) OOPS("error in reading run_stats process_label");
	x=etk::py_read_from_dict(P, "process_starttime", process_starttime);
	if (x!=0) OOPS("error in reading run_stats process_starttime");
	x=etk::py_read_from_dict(P, "process_endtime", process_endtime);
	if (x!=0) OOPS("error in reading run_stats process_endtime");


	Py_XDECREF(b85pickle);
	Py_XDECREF(pickle);
}


//		std::vector<std::string> process_label;
//		std::vector< std::chrono::time_point<std::chrono::high_resolution_clock> > process_starttime;
//		std::vector< std::chrono::time_point<std::chrono::high_resolution_clock> > process_endtime;

void elm::runstats::start_process(const std::string& name)
{
	if (process_starttime.size() > process_endtime.size()) {
		end_process();
	}
	process_label.push_back(name);
	process_starttime.push_back(std::chrono::high_resolution_clock::now());
	
//	std::cerr<<"start_process "<<name<<"\n";
}

void elm::runstats::end_process(){
	while (process_starttime.size() > process_endtime.size())
		process_endtime.push_back(std::chrono::high_resolution_clock::now());

//	std::cerr<<"end_process "<<"\n";
}

double elm::runstats::process_duration(const std::string& name) const
{
	std::string upper_name = etk::to_uppercase(name);
	for (size_t i=0; i<process_label.size(); i++) {
		if (upper_name == etk::to_uppercase(process_label[i])) return process_duration(i);
	}
	return 0;
}
double elm::runstats::process_duration(const size_t& number) const
{
	if (number>=process_starttime.size()) OOPS("start time missing for process number ",number);
	if (number>=process_endtime.size()) OOPS("end time missing for process number ",number);
	
	double x = std::chrono::duration_cast<std::chrono::milliseconds>(process_endtime[number] - process_starttime[number]).count();
	x /= 1000.0;
	return x;
}

double elm::runstats::total_duration() const
{
	double x;
	if (process_starttime.size()==0) return 0;
	if (process_endtime.size()==0) return 0;
	if (process_starttime.size() > process_endtime.size()) {
		x = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - process_starttime[0]).count();
	} else {
		x = std::chrono::duration_cast<std::chrono::milliseconds>(process_endtime.back() - process_starttime[0]).count();
	}
	x /= 1000.0;
	return x;
}



std::string elm::runstats::process_duration_fancy(const std::string& name) const
{
	return etk::hours_minutes_seconds(process_duration(name),true);
}

std::string elm::runstats::process_duration_fancy(const size_t& number) const
{
	return etk::hours_minutes_seconds(process_duration(number),true);
}

std::string elm::runstats::total_duration_fancy() const
{
	return etk::hours_minutes_seconds(total_duration(),true);
}


PyObject* elm::runstats::__getstate__() const
{
	PyObject* P = PyDict_New();

	etk::py_add_to_dict(P, "iteration", iteration);
	etk::py_add_to_dict(P, "results", results);
	etk::py_add_to_dict(P, "notes", _notes);
	etk::py_add_to_dict(P, "timestamp", timestamp);
	etk::py_add_to_dict(P, "processor", processor);

	etk::py_add_to_dict(P, "process_label", process_label);
	etk::py_add_to_dict(P, "process_starttime", process_starttime);
	etk::py_add_to_dict(P, "process_endtime", process_endtime);
	
	return P;
}

