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
#ifdef __APPLE__
	timeval t;
	gettimeofday( &t, NULL);
	return timeval_subtract(t,startTime);
#else
	return 0;
#endif
}

double elm::runstats::runtime_seconds() const
{
#ifdef __APPLE__
	return timeval_subtract(endTime, startTime);
#else
	return 0;
#endif
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
				 std::string processor)
: startTime ()
, endTime ()
, iteration (iteration)
, _notes (notes)
, results (results)
, timestamp (timestamp)
, processor(processor)
{
#ifdef __APPLE__
	if (startTimeSec || startTimeUSec) {
		startTime.tv_sec = startTimeSec;
		startTime.tv_usec = startTimeUSec;
		endTime.tv_sec = endTimeSec;
		endTime.tv_usec = endTimeUSec;
	} else {
		gettimeofday( &startTime, NULL);
		endTime.tv_sec = 0;
		endTime.tv_usec = 0;
	}
#endif
}

elm::runstats::runstats(const runstats& other)
: startTime (other.startTime)
, endTime (other.endTime)
, iteration (other.iteration)
, _notes (other._notes)
, results (other.results)
, timestamp (other.timestamp)
, processor(other.processor)
{
}

elm::runstats::runstats(PyObject* dictionary)
: startTime ()
, endTime ()
, iteration (0)
, _notes ()
, results ()
, timestamp ()
, processor ()
{
	read_from_dictionary(dictionary);
}

void elm::runstats::restart()
{
#ifdef __APPLE__
	gettimeofday( &startTime, NULL);
#endif
	iteration = 0;
	_notes.clear();
	results.clear();
}

void elm::runstats::finish()
{
#ifdef __APPLE__
	gettimeofday( &endTime, NULL);
#endif
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
	if (results.empty()) x << "No result";
	x << results << " in " << runtime();
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
	#ifdef __APPLE__
	etk::py_add_to_dict(P, "startTimeSec", startTime.tv_sec);
	etk::py_add_to_dict(P, "startTimeUSec", startTime.tv_usec);
	etk::py_add_to_dict(P, "endTimeSec", endTime.tv_sec);
	etk::py_add_to_dict(P, "endTimeUSec", endTime.tv_usec);
	#endif
	etk::py_add_to_dict(P, "iteration", iteration);
	etk::py_add_to_dict(P, "results", results);
	etk::py_add_to_dict(P, "notes", _notes);
	etk::py_add_to_dict(P, "timestamp", timestamp);
	etk::py_add_to_dict(P, "processor", processor);
	
	return etk::py_one_item_list(P);
}

void elm::runstats::read_from_dictionary(PyObject* P)
{
	int x;
	#ifdef __APPLE__
	x=etk::py_read_from_dict(P, "startTimeSec", startTime.tv_sec);
	if (x!=0) OOPS("error in reading run_stats startTimeSec, code ",x);
	x=etk::py_read_from_dict(P, "startTimeUSec", startTime.tv_usec);
	if (x!=0) OOPS("error in reading run_stats startTimeUSec");
	x=etk::py_read_from_dict(P, "endTimeSec", endTime.tv_sec);
	if (x!=0) OOPS("error in reading run_stats endTimeSec");
	x=etk::py_read_from_dict(P, "endTimeUSec", endTime.tv_usec);
	if (x!=0) OOPS("error in reading run_stats endTimeUSec");
	#endif
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
}
