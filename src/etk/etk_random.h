/*
 *  etk_random.h
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


#ifndef __TOOLBOX_RANDOM_H__
#define __TOOLBOX_RANDOM_H__

#include <vector>
#include <map>
#include <stdlib.h>

#ifndef __APPLE__
#include <time.h>
#endif

namespace etk {
	
	class recallable {
	public:
		virtual double next()=0;
		void burn(const unsigned& n);
		virtual ~recallable() { }
	};

	class randomizer: public recallable { 
	public:
		double uniform();
		virtual double next() { return uniform(); }
		
		double WeightLow(const unsigned& w=1);  // Output random float in [0,1] weighted low
		double WeightHigh(const unsigned& w=1); // Output random float in [0,1] weighted high
		unsigned dexHigh(const unsigned& top, const unsigned& w=1);
		unsigned dexLow(const unsigned& top, const unsigned& w=1);
		unsigned dex(const unsigned& top);
		
		randomizer() {
            srand((unsigned int)time((time_t *)NULL));
		};
		virtual ~randomizer() { }
	};
	
	class halton: public recallable {
		const unsigned Prime;
		unsigned NextPower;
		unsigned ticker;
		double PreviousReturn;
		const double Divisor;
		double CurrentDivisor;
	public:
		virtual double next();
		halton(const unsigned& seed);
		virtual ~halton() { }
	};
	
	const unsigned& prime (const unsigned& n);
	
	template <class T>
	void shuffle (size_t N, T* X, unsigned incX, recallable* R) {
				
		std::map<double, unsigned> reordering;
		std::pair<double, unsigned> P;
		
		// Construct Shuffler
		unsigned i;
		for (i=0; i<N; i++) {
			P.first = R->next();
			P.second = i;
			reordering.insert(P);
		}
		
		// Record Originals
		std::vector<T> originals (N);
		for (i=0; i<N; i++) {
			originals[i] = X[i*incX];
		}	
		
		// Reordering
		std::map<double, unsigned>::iterator j;
		for (j=reordering.begin(), i=0; j!=reordering.end(); j++, i++) {
			X[j->second*incX] = originals[i];
		}	
	}
	
	template <class T>
	void shuffle (std::vector<T>& Xvec, recallable* R) {
		shuffle (Xvec.size(), &Xvec[0], 1, R);
	}
}

#endif

