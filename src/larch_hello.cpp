//
//  larch_hello.cpp
//  Larch
//
//  Created by Jeffrey Newman on 9/18/14.
//  Copyright (c) 2014 Jeffrey Newman. All rights reserved.
//

#include "larch_portable.h"
#include "larch_hello.h"
#include <iostream>
#include <sstream>



double ddot_trial()
{
	double x[] = {1.0,2.0,10.0};
	double y[] = {0.1,0.01,10.0};
	
	double zz= cblas_ddot(3, x, 1, y, 1);
//	double zz= 992.992;
	return zz;
}



std::string greeting()
{
	std::ostringstream z;
	z << "Greetings, Earthling.  ";
	z << "You math problem answer is "<<ddot_trial();

	return z.str();
}

