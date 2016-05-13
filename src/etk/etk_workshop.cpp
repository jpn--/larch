/*
 *  etk_thread.cpp
 *
 *  Copyright 2007-2016 Jeffrey Newman
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
 
 
#include "etk_thread.h"
#include "etk_workshop.h"
#include "etk_exception.h"
 
void etk::workshop::work(size_t firstcase, size_t numberofcases, boosted::mutex* result_mutex)
{
	double x = 0.0;
	for (unsigned i=0; i<1000000*numberofcases; i++) {
			x += double((rand() % 10))/double(1000);
			x -= double((rand() % 10))/double(1000);
	}
}

void etk::workshop::startwork(etk::dispatcher* dispatcher, boosted::mutex* result_mutex)
{
	while (!release_workshop) {
		etk::job thisjob = dispatcher->next_job();
		boosted::unique_lock<boosted::mutex> LOCK(timecard);
		if (thisjob.is_null() || release_workshop) {
			break;
		}
		if (thisjob.is_skip()) {
			LOCK.unlock();
			continue;
		}
		try {
			work(thisjob.first,thisjob.length, result_mutex);
		} catch(const etk::exception_t &err) {
			dispatcher->etk_exception_on_job(thisjob.first, err);
			LOCK.unlock();
			continue;
		} catch(const std::exception &err) {
			dispatcher->std_exception_on_job(thisjob.first, err);
			LOCK.unlock();
			continue;
		}
		dispatcher->finished_job(thisjob.first);
		LOCK.unlock();
	}
}









#include <iostream>



etk::dispatcher::dispatcher(int nThreads, size_t nJobs, workshop_builder_t workshop_builder)
: nThreads(nThreads)
, nJobs(nJobs)
, result_mutex()
, workshops ()
, threads ()
, schedule_size(10)
, workshop_builder(workshop_builder)
, terminate(false)
, exception_message()
, exception_count(0)
, zeroprob_exception_count(0)
, exception_mutex()
{
	for (int i=0; i<nThreads; i++) {
		add_thread();
	}
}

void etk::dispatcher::add_thread()
{
		workshops.push_back(workshop_builder());
		boosted::shared_ptr<boosted::thread> thrd = boosted::make_shared<boosted::thread>(&workshop::startwork, workshops.back(), this, &result_mutex);
		threads.push_back( thrd );
}

etk::dispatcher::~dispatcher()
{
	release();
}

void etk::dispatcher::dispatch(int nThreads, workshop_updater_t* updater)
{
	exception_count = 0;
	zeroprob_exception_count = 0;
	if (nThreads != -9 && nThreads > threads.size()) {
		for (int i=threads.size(); i<nThreads; i++) {
			add_thread();
		}
	}
	else if (nThreads != -9 && nThreads < threads.size()) {
		release();
		for (int i=0; i<nThreads; i++) {
			add_thread();
		}
	}
	else if (nThreads != -9 && updater) {
		// update threads
		for (int i=0; i<workshops.size(); i++) {
			(*updater)(workshops[i]);
		}
	}
	request_work();
	boosted::unique_lock<boosted::mutex> LOCK(workdone_mutex);
	while (work_remains()) {
		jobs_done.wait_for(LOCK, boosted::chrono::milliseconds(20));
	}
	if (exception_count) {
		OOPS(exception_message);
	}
	if (zeroprob_exception_count) {
		OOPS_ZEROPROB(exception_message);
	}
}

void etk::dispatcher::release()
{
	for (unsigned i=0; i<workshops.size(); i++) {
		if (workshops[i]) {
			workshops[i]->release_workshop = true;
		}
	}
	terminate = true;
	has_jobs.notify_all();
	for (unsigned i=0; i<threads.size(); i++) {
		if (threads[i]->joinable()) threads[i]->join();
	}
	workshops.clear();
	threads.clear();
	terminate = false;
}








void etk::dispatcher::request_work()
{
	// Divide up all the discrete tasks into sets of work to complete.
	queue_mutex.lock();
	int n = nThreads*schedule_size;
	if (n > nJobs) n = nJobs;
	if (n>0) {
		size_t chunksize = nJobs/n;
		size_t chunkleft = nJobs%n;
		
		size_t begin = 0;
		
		for (int i=0; i<n; i++) {
			if (chunkleft) {
				jobs_waiting.push_back(etk::job(begin, chunksize+1));
				chunkleft--;
				begin += chunksize+1;
			} else {
				jobs_waiting.push_back(etk::job(begin, chunksize));
				begin += chunksize;
			}
		}
	}
	queue_mutex.unlock();
	if (jobs_waiting.size()) has_jobs.notify_all();
}


etk::job etk::dispatcher::next_job()
{
	etk::job ret(SIZE_T_MAX,SIZE_T_MAX);
	
	boosted::unique_lock<boosted::mutex> lock(queue_mutex);
	
	while (jobs_waiting.size()==0 && !terminate) {
		has_jobs.wait(lock);
	}
	
	if (terminate) {
		return ret;
	}
	
	if (jobs_waiting.size()) {
		ret.first = jobs_waiting.front().first;
		ret.length = jobs_waiting.front().length;
		jobs_waiting.pop_front();
		if (!ret.is_skip()) jobs_out.insert(ret.first);
	}
	
	return ret;
}

void etk::dispatcher::finished_job(const size_t& job_id)
{
	boosted::unique_lock<boosted::mutex> lock(queue_mutex);
	jobs_out.erase(job_id);
	if (jobs_out.size()==0) {
		// release threads waiting on job completion
		lock.unlock();
		jobs_done.notify_all();
	}
}

void etk::dispatcher::etk_exception_on_job(const size_t& job_id, const etk::exception_t& err)
{
	boosted::unique_lock<boosted::mutex> lock(queue_mutex);
	boosted::unique_lock<boosted::mutex> elock(exception_mutex);
	jobs_out.erase(job_id);
	exception_message += std::string(err.what()) + "\n";
	
	if (err.code()==OOPSCODE_ZEROPROB) {
		zeroprob_exception_count++;
	} else {
		exception_count++;
	}
	elock.unlock();
	if (jobs_out.size()==0) {
		// release threads waiting on job completion
		lock.unlock();
		jobs_done.notify_all();
	}
	
	
}


void etk::dispatcher::std_exception_on_job(const size_t& job_id, const std::exception& err)
{
	boosted::unique_lock<boosted::mutex> lock(queue_mutex);
	boosted::unique_lock<boosted::mutex> elock(exception_mutex);
	jobs_out.erase(job_id);
	exception_message += std::string(err.what()) + "\n";
	exception_count++;
	elock.unlock();
	if (jobs_out.size()==0) {
		// release threads waiting on job completion
		lock.unlock();
		jobs_done.notify_all();
	}
	
	
}



bool etk::dispatcher::work_remains() 
{
	boosted::unique_lock<boosted::mutex> local_lock(queue_mutex);
	return (jobs_waiting.size()+jobs_out.size() > 0);
}


