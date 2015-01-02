/*
 *  etk_math.h
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







#ifndef __TOOLBOX_MATH__
#define __TOOLBOX_MATH__

namespace etk {

//void dppinv (double* A, int N);

// X * A * alpha + Y * beta -> Y
void element_multiply(const int& N, const double& alpha, const double * A, const int& incA, 
					  const double * X, const int& incX, 
					  const double& beta, double * Y, const int& incY) ;

void simple_inplace_element_multiply(const int& N, const double * A, const int& incA, 
									 double * Y, const int& incY) ;

void simple_element_multiply(const int& N, const double * A, const int& incA, 
							 const double * X, const int& incX, 
							 double * Y, const int& incY) ;

};
#endif

