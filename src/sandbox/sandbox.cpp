/*
 *  pysandbox.cpp
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



#include "Python.h"
#include <string>
#include <iostream>

int main(int argc, char **argv) {
	Py_Initialize();
	PyRun_SimpleString("print ('.'*40, 'larch sandbox', '.'*40)");
	PyRun_SimpleString("import os, sys");
	PyRun_SimpleString("print( 'CWD:', os.getcwd() )");
//	PyRun_SimpleString("sys.path.insert(0,os.getcwd())");
	PyRun_SimpleString("print( 'Version:', sys.version )");
	PyRun_SimpleString("print( 'Executable:', sys.executable )");
	PyRun_SimpleString("print( 'Prefix:', sys.prefix )");
	PyRun_SimpleString("exec(open('/Users/jpn/Dropbox/Larch/src/sandbox/sandbox.py').read())");
	PyRun_SimpleString("print('=====> Testing Sandbox Complete. <=====')");
	Py_Finalize();
	return 0;
}