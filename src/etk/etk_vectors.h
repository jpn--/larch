/*
 *  etk_vectors.h
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


#ifndef __TOOLBOX_VECTORS_H__
#define __TOOLBOX_VECTORS_H__

#include <vector>
#include <string>
#include <deque>
#include <set>
//#include <mutex>
#include "etk_thread.h"

#ifndef SIZE_T_MAX
#define	SIZE_T_MAX	ULONG_MAX	/* max value for a size_t */
#endif // ndef SIZE_T_MAX

namespace etk {
	
	class strvec: public std::vector<std::string> {
		
	public:
		bool contains (const std::string& j) const;
		strvec(const std::vector<std::string>& x);
		strvec();
		
		std::string join(const std::string& j) const;
		
		size_t push_back_if_unique(const std::string& j);
		
		std::vector<std::string> as_std_string() const;
	};

	#ifndef SWIG

	size_t push_back_if_unique( std::vector<std::string>& vec, const std::string& j );

	
	strvec split_string(const std::string& s, char delim);
	
	class dblvec: public std::vector<double> {
	public:
		double average() const;
		double moving_average(unsigned long s) const;
		unsigned long moving_average_size(unsigned long s) const;
		double moving_average_improvement(unsigned long s) const;
		unsigned long moving_average_improvement_size(unsigned long s) const;
	};

	template <class T>
	size_t find_first(const T& prototype, std::vector<T> vec)
	{
		size_t k = 0;
		while (k<vec.size()) {
			if (vec[k] == prototype) return k;
			k++;
		}
		return SIZE_T_MAX;
	}


	#endif // ndef SWIG
/*
	struct chunk {
		size_t first;
		size_t length;
		
		chunk(size_t first, size_t length)
		: first(first)
		, length(length)
		{}
		
		bool is_null() {return first==SIZE_T_MAX && length==SIZE_T_MAX;}
	};





	class jobber: public std::deque<chunk> {
	public:
		boosted::mutex ask;
		boosted::condition_variable has_some;
		boosted::condition_variable jobs_done;
		bool terminate;
		
		jobber(size_t length=0, int n=1);
		chunk next_job();
		std::set<size_t> jobs_out;
		
		void finished_job(const size_t& job_id);
		
		void populate(size_t length, int n=1);
		void end(bool really=true);
		
		bool work_remains() const;
	};
*/

}

#endif
