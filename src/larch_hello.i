//
//  larch_hello.h
//  Larch
//
//  Created by Jeffrey Newman on 9/18/14.
//  Copyright (c) 2014 Jeffrey Newman. All rights reserved.
//

#ifndef __Larch__larch_hello__
#define __Larch__larch_hello__

%module larch

%{
#define PY_ARRAY_UNIQUE_SYMBOL _LARCH_HELLO_PY_ARRAY_UNIQUE_SYMBOL_
#include <numpy/arrayobject.h>
%}


%include "std_string.i"

%{
#include "larch_portable.h"
#include "etk/etk_python.h"

#include "larch_hello.h"
%}

std::string greeting();
double ddot_trial();

#endif /* defined(__Larch__larch_hello__) */
