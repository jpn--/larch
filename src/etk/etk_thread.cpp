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
 
#ifndef NOTHREAD
 
#include <iostream>
#include <string>
#include "etk_thread.h"

 
 
 

 
/* TODO;

#include <string>
#include <ctime>


void threadpause( double seconds )
{
	std::clock_t endwait;
	endwait = clock () + seconds * CLOCKS_PER_SEC ;
	while (clock() < endwait) {}
}

etk::threadString::threadString(const std::string& s)
: _s (s)
{
	
}

std::string etk::threadString::read()
{
	boost::mutex::scoped_lock lock(_mutex); 
	std::string ret = _s;
	return ret;
}

void etk::threadString::write(const std::string& s)
{
	boost::mutex::scoped_lock lock(_mutex); 
	_s = s;
	return;
}

class thread_ending_message {
	std::string endMessage;
public:
	void operator()() const { std::cerr << endMessage << "\n"; };
	thread_ending_message(const std::string& em="thread ending"): endMessage (em) { boost::this_thread::at_thread_exit(*this); }
}; 




etk::threadCounter::threadCounter(const unsigned& start, const unsigned& stop, const unsigned& step)
: _dex (start)
, _start (start)
, _stop (stop)
, _step (step)
{
	
}

bool etk::threadCounter::get(unsigned& cntr)
{
	boost::mutex::scoped_lock lock(_mutex); 
	if (_dex < _stop) {
		cntr = _dex;
		_dex += _step;
		return true;
	} else {
		return false;
	}
}

bool etk::threadCounter::incomplete() const
{
	if (_dex < _stop) {
		return true;
	} else {
		return false;
	}
}






#include "etk_random.h"



void etk::spew(etk::threadString* tS)
{
	thread_ending_message("spew thread ending");
	etk::randomizer R;
	int i =0;
	double rr;
	try {
		while (1) {
			//threadpause(0.5);
			for (unsigned j=0; j<10000000; j++) {
				rr += R.next() / 1000000.0;
			}
			std::cout << i++ << "\t" << tS->read() << "\n";
			boost::this_thread::interruption_point();
		}
	} 
	catch (boost::thread_interrupted) {
		std::cout << "end\n";
		throw;
	}
}

void etk::replacer(etk::threadString* tS)
{
	thread_ending_message("repl thread ending");
	etk::randomizer R;
	std::string repl;
	while (repl != "q") {
		std::cin >> repl;
		tS->write(repl);
	}
}


void etk::threadtester()
{

	std::cout << "Hello Thread Test!\n";
	etk::threadString tSo ("empty");
	etk::threadString* tS = &tSo;
	
	boost::thread spawnAsk (replacer,tS);
	boost::thread spawnTell (spew,tS);
	boost::thread spawnTell2 (spew,tS);
	
	spawnAsk.join();
	spawnTell.interrupt();
	spawnTell2.interrupt();
	spawnTell.join();
	spawnTell2.join();
	return;
}
	*/


#endif // NOTHREAD
 



////////////////////////////////////////////////////////////////////////////////
#include <thread>
#include <vector>
#include <cmath>
////////////////////////////////////////////////////////////////////////////////
 
class ThreadPool {
 
public:
 
	template<typename Index, typename Callable>
	static void ParallelFor(Index start, Index end, Callable func) {
		// Estimate number of threads in the pool
		const static unsigned nb_threads_hint = std::thread::hardware_concurrency();
		const static unsigned nb_threads = (nb_threads_hint == 0u ? 8u : nb_threads_hint);
 
		// Size of a slice for the range functions
		Index n = end - start + 1;
		Index slice = (Index) std::round(n / static_cast<double> (nb_threads));
		slice = std::max(slice, Index(1));
 
		// [Helper] Inner loop
		auto launchRange = [&func] (int k1, int k2) {
			for (Index k = k1; k < k2; k++) {
				func(k);
			}
		};
 
		// Create pool and launch jobs
		std::vector<std::thread> pool;
		pool.reserve(nb_threads);
		Index i1 = start;
		Index i2 = std::min(start + slice, end);
		for (unsigned i = 0; i + 1 < nb_threads && i1 < end; ++i) {
			pool.emplace_back(launchRange, i1, i2);
			i1 = i2;
			i2 = std::min(i2 + slice, end);
		}
		if (i1 < end) {
			pool.emplace_back(launchRange, i1, end);
		}
 
		// Wait for jobs to finish
		for (std::thread &t : pool) {
			if (t.joinable()) {
				t.join();
			}
		}
	}
 
	// Serial version for easy comparison
	template<typename Index, typename Callable>
	static void SequentialFor(Index start, Index end, Callable func) {
		for (Index i = start; i < end; i++) {
			func(i);
		}
	}
 
};
 
////////////////////////////////////////////////////////////////////////////////
//#include <iostream>
//#include <mutex>
////////////////////////////////////////////////////////////////////////////////
 
//int main() {
//	std::mutex critical;
//	ThreadPool::ParallelFor(0, 16, [&] (int i) {
//		std::lock_guard<std::mutex> lock(critical);
//		std::cout << i << std::endl;
//	});
//	return 0;
//}
//




