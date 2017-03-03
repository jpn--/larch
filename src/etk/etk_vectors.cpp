/*
 *  etk_vectors.cpp
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


#include "etk_vectors.h"
#include <sstream>
#include <iostream>

etk::strvec::strvec(const std::vector<std::string>& x)
: std::vector<std::string> (x)
{ }

etk::strvec::strvec()
: std::vector<std::string> ()
{ }


bool etk::strvec::contains (const std::string& j) const
{
	std::vector<std::string>::const_iterator k = begin();
	while (k!=end()) {
		if (*k == j) return true;
		k++;
	}
	return false;
}

std::string etk::strvec::join(const std::string& j) const
{
	std::ostringstream x;
	if (this->size() > 0) x << this->at(0);
	for (size_t i=1; i<this->size(); i++) {
		x << j << this->at(i);
	}
	return x.str();
}




size_t etk::strvec::push_back_if_unique( const std::string& j )
{
	for (size_t i=0;i<size();i++) {
		if (j==at(i)) {
			return i;
		}
	}
	push_back(j);
	return size()-1;
}

size_t etk::push_back_if_unique( std::vector<std::string>& vec, const std::string& j )
{
	for (size_t i=0;i<vec.size();i++) {
		if (j==vec.at(i)) {
			return i;
		}
	}
	vec.push_back(j);
	return vec.size()-1;
}


std::vector<std::string> etk::strvec::as_std_string() const
{
	std::vector<std::string> x (*this);
	return x;
}






etk::strvec split_string(const std::string& s, char delim) {
	etk::strvec x;
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim)) {
        x.push_back(item);
    }
    return x;
}


double etk::dblvec::average() const {
	double ret (0);
	double div (0);
	for (const_iterator i= begin(); i!=end(); i++) {
		ret += *i;
		div++;
	}
	if (!div) return 0.;
	return ret/div;
}

double etk::dblvec::moving_average(unsigned long s) const {
	if (s >= size()) return average();
	s = size() - s;	
	double ret (0);
	double div (0);
	for (const_iterator i= begin()+s; i!=end(); i++) {
		ret += *i;
		div++;
	}
	if (!div) return 0.;
	return ret/div;
}

unsigned long etk::dblvec::moving_average_size(unsigned long s) const {
	if (s >= size()) return s;
	return size();
}

double etk::dblvec::moving_average_improvement(unsigned long s) const {
	if (size()<2) return 0.;
	if (s >= size()) return (back()-front())/double(size()-1);
	return (back()-at(size() - s - 1))/double(s);
}

unsigned long etk::dblvec::moving_average_improvement_size(unsigned long s) const {
	if (size()<2) return 0;
	if (s >= size()) return size()-1;
	return s;
}


/*

etk::jobber::jobber(size_t length, int n)
: std::deque<chunk>()
, terminate(false)
{
	if (length) populate(length, n);
}

void etk::jobber::populate(size_t length, int n)
{
	ask.lock();
	size_t chunksize = length/n;
	size_t chunkleft = length%n;
	
	size_t begin = 0;
	
	for (int i=0; i<n; i++) {
		if (chunkleft) {
			push_back(etk::chunk(begin, chunksize+1));
			chunkleft--;
			begin += chunksize+1;
		} else {
			push_back(etk::chunk(begin, chunksize));
			begin += chunksize;
		}
	}
	ask.unlock();
	has_some.notify_all();
}


etk::chunk etk::jobber::next_job()
{
	etk::chunk ret(SIZE_T_MAX,SIZE_T_MAX);
	
	boosted::unique_lock<boosted::mutex> lock(ask);
	
	while (size()==0 && !terminate) {
		has_some.wait(lock);
	}
	
	if (terminate) {
		return ret;
	}
	
	if (size()) {
		ret.first = front().first;
		ret.length = front().length;
		pop_front();
		jobs_out.insert(ret.first);
	}
	
	return ret;
}

void etk::jobber::finished_job(const size_t& job_id)
{
	boosted::unique_lock<boosted::mutex> lock(ask);
	jobs_out.erase(job_id);
	if (jobs_out.size()==0) {
		// release threads waiting on job completion
		lock.unlock();
	}
}


void etk::jobber::end(bool really)
{
	ask.lock();
	terminate = really;
	ask.unlock();
	has_some.notify_all();
}

bool etk::jobber::work_remains() const
{
	return (size()+jobs_out.size() > 0);
}
*/

