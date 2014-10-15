/*
 *  etk_time.cpp
 *
 *  Copyright 2007-2013 Jeffrey Newman
 *
 *  This file is part of ELM.
 *  
 *  ELM is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  ELM is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with ELM.  If not, see <http://www.gnu.org/licenses/>.
 *  
 */


#include "etk_time.h"
#include <ios>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <ctime>

std::string etk::hours_minutes_seconds (double seconds, bool always_show_minutes)
{
	double minutes = floor(seconds / 60);
	seconds = fmod(seconds, 60);
	double hours = floor(minutes / 60);
	minutes = fmod(minutes, 60);
	double partial_secs = modf(seconds, &seconds);
	partial_secs *= 100;
	int psecs = int(partial_secs+0.5);
	std::ostringstream ret;
	ret << std::setfill('0');
	if (hours>0) ret << int(hours) << ":"<< std::setw( 2 );	
	if (minutes>0 || hours>0 || always_show_minutes) ret  << int(minutes) << ":" << std::setw( 2 );
	unsigned prec = 1;
	if (seconds >= 10.0) prec = 2;
	ret << std::setprecision(prec) << seconds;
	
	div_t temp;
	temp = div( psecs, 10 );
	
	if (psecs) ret << "." << temp.quot;
	if (temp.rem) ret << temp.rem;
	return ret.str();
}


void etk::pause( double seconds )
{
	std::clock_t endwait;
	endwait = clock () + seconds * CLOCKS_PER_SEC ;
	while (clock() < endwait) {}
}

std::string etk::current_datetime()
{
  time_t rawtime;
  time ( &rawtime );
  return ctime (&rawtime);

}


