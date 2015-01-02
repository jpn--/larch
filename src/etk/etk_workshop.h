/*
 *  etk_thread.h
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


#ifndef __TOOLBOX_WORKSHOPS__
#define __TOOLBOX_WORKSHOPS__

#include <limits.h>
#include "etk_thread.h"
#include "etk_vectors.h"

#ifndef SIZE_T_MAX
#define	SIZE_T_MAX	ULONG_MAX	/* max value for a size_t */
#endif // ndef SIZE_T_MAX

namespace etk {



	class dispatcher;

	class workshop {
	
	public:
		virtual void work(size_t firstcase, size_t numberofcases, boosted::mutex* result_mutex);
		void startwork(etk::dispatcher* dispatcher, boosted::mutex* result_mutex);
	
		boosted::mutex timecard;
		bool release_workshop;
		
		workshop(): release_workshop(false) {}
	};




	struct job {
		size_t first;
		size_t length;
		
		job(size_t first, size_t length)
		: first(first)
		, length(length)
		{}
		
		inline bool is_null() {return first==SIZE_T_MAX && length==SIZE_T_MAX;}
		inline bool is_skip() {return first==SIZE_T_MAX && length==SIZE_T_MAX-1;}
		inline void make_null() { first=SIZE_T_MAX; length=SIZE_T_MAX;}
		inline void make_skip() { first=SIZE_T_MAX; length=SIZE_T_MAX-1;}
		
		static job skip() {return job(SIZE_T_MAX,SIZE_T_MAX-1);}
	};



	class dispatcher {
	
		friend class workshop;
	
		int nThreads;
		size_t nJobs;
		boosted::mutex result_mutex;
		std::vector< boosted::shared_ptr<workshop> > workshops;
		std::vector< boosted::shared_ptr<boosted::thread> > threads;
		boosted::function< boosted::shared_ptr<workshop>() > workshop_builder;

		void add_thread();

		bool terminate;

		boosted::mutex queue_mutex;
		std::deque<job> jobs_waiting;
		std::set<size_t> jobs_out;
		job next_job();
		void finished_job(const size_t& job_id);
		void exception_on_job(const size_t& job_id, const std::exception& err);
		void request_work();
		bool work_remains() const;
		boosted::condition_variable has_jobs;
		boosted::condition_variable jobs_done;

		boosted::mutex workdone_mutex;
		
	  public:
		int schedule_size;
		
		dispatcher(int nThreads, size_t nJobs, boosted::function< boosted::shared_ptr<workshop>() > workshop_builder);
		~dispatcher();
		void dispatch(int nThreads=-9);
		void release();
		
		boosted::mutex exception_mutex;
		int exception_count;
		std::string exception_message;
	};


//	void dispatch(int nThreads, size_t nJobs, boosted::function< boosted::shared_ptr<workshop>() > workshop_builder);



}

#define USE_DISPATCH(x,threads,...) if (!(x)) {(x)=boosted::make_shared<etk::dispatcher>(threads,__VA_ARGS__);} else {} (x)->dispatch(threads)


#endif // __TOOLBOX_WORKSHOPS__
