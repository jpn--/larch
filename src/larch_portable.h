/*
 *  larch_portable.h
 *
 *  Copyright 2007-2015 Jeffrey Newman
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

#ifndef Larch_larch_portable_h
#define Larch_larch_portable_h

/*
 header file to make basic math functions portable.
 isinf, isnan and isfinite are in C99 standard
 all platforms should use them.
 */

#include <math.h>
#include <cmath>

// cblas and clapack libraries vary by platform
#ifdef __APPLE__
#	include <Accelerate/Accelerate.h>
#	define DEBUG_APPLE
#	define _larch_init_
#else
#	include <cblas.h>
#	define _larch_init_ openblas_set_num_threads(1)
#endif

// platform specific isInf
#if defined(__inline_isinf)
#	define isInf(_a) (__inline_isinf(_a))	/* MacOSX/Darwin definition (old) */
#elif defined(isinf)
#	define isInf(_a) (isinf(_a))
#elif defined(_MSC_VER)
#include	<float.h>
#	define isInf(_a) (!_isnan(_a) && !_finite(_a))   /* Microsoft */
#else
#	define isInf(_a) (std::isinf(_a))
#endif

// platform specific isNan
#if defined(__inline_isnan)
#	define isNan(_a) (__inline_isnan(_a))	/* MacOSX/Darwin definition (old) */
#elif defined(isnan)
#	define isNan(_a) (isnan(_a))
#elif defined(_MSC_VER)
#   include <float.h>
#	define isNan(_a) (_isnan(_a)) 	/* microsoft */
#else
#	define isNan(_a) (std::isnan(_a))
#endif

// platform specific isFinite
#if defined(__inline_isfinite)
#	define isFinite(_a) (__inline_isfinite(_a))	/* MacOSX/Darwin definition */
#elif defined(isfinite)
#	define isFinite(_a) (isfinite(_a))
#elif defined(_MSC_VER)
#	include <float.h>
#	define isFinite(_a) (_finite(_a))  /* microsoft */
#else
#	define isFinite(_a) (std::isfinite(_a))
#endif


#ifdef _MSC_VER
#define  DIRSEP  '\\'
#else
#define  DIRSEP  '/'
#endif

inline double ELM_NAN_( )
{
	unsigned long _elm__nan__[2]={0xffffffff, 0x7fffffff};
	return *( double* )_elm__nan__;
}

#ifndef NAN
# ifdef __builtin_nan
#  define NAN __builtin_nan("")
# elif defined __nanf
#  define NAN (*(__const float *)(__const void *)__nanf)
# else
#  define NAN ELM_NAN_()
# endif
#endif

#endif /* Larch_larch_portable_h */



