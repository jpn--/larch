/*
 *  etk_random.cpp
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


#include "etk_random.h"
#include "etk_exception.h"

void etk::recallable::burn(const unsigned& n)
{
	for (unsigned i=0; i<n; i++) {
		next();
	}
}

double etk::randomizer::uniform() {
	
	union {double f; unsigned i[2];} convert;
	
	// Get 32 random bits
	unsigned r = rand() << 16;               
	r ^= rand();
	
#if defined(_M_IX86) || defined(_M_X64) || defined(__LITTLE_ENDIAN__)
	convert.i[0] =  r << 20;
	convert.i[1] = (r >> 12) | 0x3FF00000;
	return convert.f - 1.0;
	
#elif defined(__BIG_ENDIAN__)
	convert.i[1] =  r << 20;
	convert.i[0] = (r >> 12) | 0x3FF00000;
	return convert.f - 1.0;
	
#else
	return (double)r * (1./((double)(unsigned)(-1L)+1.));
	
#endif
	
}

double etk::randomizer::WeightLow(const unsigned& w) {
	double ret = uniform();
	for (unsigned i=0; i<w; i++) ret *= uniform();
	return ret;
}

double etk::randomizer::WeightHigh(const unsigned& w) {
	return (1-WeightLow(w));
}

unsigned etk::randomizer::dexHigh(const unsigned& top, const unsigned& w)
{
	unsigned r = unsigned(top  * WeightHigh(w)); 
	if (r > top) r = top;
	return r;
}

unsigned etk::randomizer::dexLow(const unsigned& top, const unsigned& w)
{
	unsigned r = unsigned(top  * WeightLow(w)); 
	if (r > top) r = top;
	return r;
}

unsigned etk::randomizer::dex(const unsigned& top)
{
	unsigned r = unsigned(top  * uniform()); 
	if (r > top) r = top;
	return r;
}


etk::halton::halton(const unsigned& seed)
: Prime (etk::prime(seed))
, ticker (0)
, PreviousReturn (0)
, Divisor (1 / (double)etk::prime(seed))
, CurrentDivisor (1)
{
	
}

#include <iostream>
double etk::halton::next()
{
	if (ticker % Prime) {
		ticker++;
		PreviousReturn += Divisor;
		return PreviousReturn;
	}
	PreviousReturn = 0;
	double d (Divisor);
	unsigned t = ticker;
	unsigned left;
	while (t) {
		left = t % Prime;
		PreviousReturn += (double)left * d;		
		t /= Prime;
		d *= Divisor;
	}
	ticker++;
	return PreviousReturn;
}

const unsigned& etk::prime (const unsigned& n)
{				 
	static const unsigned 
	Primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 
		31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 
		73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 
		127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 
		179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 
		233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 
		283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 
		353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 
		419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 
	467, 479, 487, 491, 499, 503, 509, 521, 523, 541};	
	if (n > 99) OOPS("Only the first 100 primes are available");
	return Primes[n];
}

