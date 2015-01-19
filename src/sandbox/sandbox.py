#
#  Copyright 2007-2015 Jeffrey Newman
#
#  This file is part of Larch.
#
#  Larch is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Larch is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Larch.  If not, see <http://www.gnu.org/licenses/>.
#


print ("="*80)
print ("="*32," sandbox module ","="*32,sep="")
print ("="*80)

import larch

def execfile(filename):
	import __main__
	with open(filename) as f:
		code = compile(f.read(), filename, 'exec')
		exec(code, globals(), __main__.__dict__)


print ("-"*80)
print ("larch.__file__")
print(larch.__file__)
print ("-"*80)

#d = larch.DB('/Users/jpn/Dropbox/Larch/py/data_warehouse/swissmetro.sqlite')
#
#print(d.list_queries())
#q = d.store["queries:{}".format('default')]
#print(q)
#print("%%%%%%%")
#
#d.load_queries()
#d.save_queries('default')
#
#d.load_queries('weighted')
#d.save_queries('weighted')
#
#d = larch.DB('/Users/jpn/Dropbox/Larch/py/data_warehouse/MTCwork.sqlite')
#d.load_queries()
#d.save_queries('default')



import scipy.optimize

import logging, os, sys, pprint, math
logging.basicConfig(format='%(levelname)s:%(name)s:%(message)s', level=5)

bitstring = "%x" % sys.maxsize, sys.maxsize > 2**32
if bitstring[1]:
	print("Python is running in 64 bit mode")
else:
	print("Python is running in 32 bit mode")
print ("="*80)

print( 'CWD:', os.getcwd() )
pprint.pprint( "sys.PATH" )
pprint.pprint( sys.path )
import larch
from larch.utilities import flux

print(larch.versions)







def assertAlmostEqual(x,y,delta=0.00001):
	if math.isinf(x) ^ math.isinf(y):
		print ("%s, %s [%s]"%(str(x),str(y),str(flux(x,y))))
		return -2
	if flux(x,y) > 5e-3:
		print ("%s, %s [%s]"%(str(x),str(y),str(flux(x,y))))
		return -1
	else:
		print ("assert ok")
		return 0


import larch.core
from larch.core import DB, Model


def test_swissmetro_1():
	swissmetro_alts = {
	1:('Train','TRAIN_AV*(SP!=0)'),
	2:('SM','SM_AV'),
	3:('Car','CAR_AV*(SP!=0)'),
	}
	d = DB.CSV_idco(filename="/Users/jpn/Dropbox/Hangman/data_warehouse/swissmetro.csv",
				   choice="CHOICE", weight="(1.0*(GROUPid==2)+1.2*(GROUPid==3))*0.8890991",
			   tablename="data", savename=None, alts=swissmetro_alts, safety=True)
	d.queries.set_idco_query(d.queries.get_idco_query()+" WHERE CHOICE!=0 AND (PURPOSE==1 OR PURPOSE==3)")
	reweight_factor = 0.8890991
	#d.refresh()
	d.sql()
	print("nCases={0}    nAlts={1}".format(d.nCases(),d.nAlts()))
	m = Model(d);
	m.logger(True)
	m.parameter("ASC_TRAIN",0)
	m.parameter("B_TIME"   ,0)
	m.parameter("B_COST"   ,0)
	m.parameter("ASC_CAR"  ,0)
	m.utility.co("1",1,"ASC_TRAIN") 
	m.utility.co("1",3,"ASC_CAR") 
	m.utility.co("TRAIN_TT",1,"B_TIME")
	m.utility.co("SM_TT",2,"B_TIME") 
	m.utility.co("CAR_TT",3,"B_TIME") 
	m.utility.co("TRAIN_CO*(GA==0)",1,"B_COST")
	m.utility.co("SM_CO*(GA==0)"   ,2,"B_COST") 
	m.utility.co("CAR_CO",        3,"B_COST") 
	m.option.gradient_diagnostic=3
	m.option.calculate_std_err=1
	m.setUp()
	d.sql()
	#raise RuntimeError()
	m.option.threads = 2
	m.estimate()
	print(d.nCases(), d.nAlts())
	print(m)
	d.display("SELECT sum(TRAIN_AV*(SP!=0)), sum(CAR_AV*(SP!=0)), sum(SM_AV) FROM "+d.tbl_idco())
	d.display("SELECT sum(CHOICE==1), sum(CHOICE==2), sum(CHOICE==3) FROM "+d.tbl_idco())
	d.display("SELECT altid, sum(CHOICE) FROM larch_choice GROUP BY altid")
	d.display("SELECT altid, sum(avail) FROM larch_avail GROUP BY altid")
	assertAlmostEqual( -5273.743, m.LL(), 3 )
	assertAlmostEqual( -0.7564616727277875,   m.parameter("ASC_TRAIN").value,9 )
	assertAlmostEqual( -0.0132,  m.parameter("B_TIME").value   ,4 )
	assertAlmostEqual( -0.0112,  m.parameter("B_COST").value   ,4 )
	assertAlmostEqual( -0.114,   m.parameter("ASC_CAR").value  ,3 )
	assertAlmostEqual( 0.0560,   m.parameter("ASC_TRAIN").std_err,4 )
	assertAlmostEqual( 0.000569, m.parameter("B_TIME").std_err   ,6 )
	assertAlmostEqual( 0.000520, m.parameter("B_COST").std_err   ,6 )
	assertAlmostEqual( 0.0432,   m.parameter("ASC_CAR").std_err  ,4 )
	assertAlmostEqual( 0.08233420341563234,   m.parameter("ASC_TRAIN").robust_std_err,9 )
	assertAlmostEqual( 0.0010327853000539932, m.parameter("B_TIME").robust_std_err   ,9 )
	assertAlmostEqual( 0.0006668017481417176, m.parameter("B_COST").robust_std_err   ,9 )
	assertAlmostEqual( 0.05856745364734663,   m.parameter("ASC_CAR").robust_std_err  ,9 )



def test_nl2_single_cycle():
	print("test_nl2_single_cycle")
	d = DB.Example('MTC');
	d.logger(True)
	m = Model(d)
	m.logger(True)
	m.parameter ("cost",0)
	m.parameter("time",0)
	m.parameter("con2",0)
	m.parameter("con3",0)
	m.parameter("con4",0)
	m.parameter("con5",0)
	m.parameter("con6",0)
	m.parameter("inc2",0)
	m.parameter("inc3",0)
	m.parameter("inc4",0)
	m.parameter("inc5",0)
	m.parameter("inc6",0)
	m.parameter("autoNest", 1.0, 1.0)
	m.utility.ca("tottime","time")
	m.utility.ca("totcost","cost")
	m.utility.co("HHINC","SR2","inc2")
	m.utility.co("HHINC","SR3+","inc3")
	m.utility.co("HHINC","Tran","inc4")
	m.utility.co("HHINC","Bike","inc5")
	m.utility.co("HHINC","Walk","inc6")
	m.utility.co("1","SR2","con2")
	m.utility.co("1","SR3+","con3")
	m.utility.co("1","Tran","con4")
	m.utility.co("1","Bike","con5")
	m.utility.co("1","Walk","con6")
	m.nest("auto", 8, "autoNest")
	m.link(8, 1)
	m.link(8, 2)
	m.link(8, 3)
	m.link(8, 4)
	m.option.gradient_diagnostic = 2
	print("test_nl2_single_cycle:model setUp")
	prv = m.db.provision(m.needs())
	for k,v in prv.items():
		print("  ---",k)
	m.provision()
	m.setUp()
	print("test_nl2_single_cycle:model loglike")
	assertAlmostEqual( -7309.600971749863, m.loglike(), delta=0.00000001 )
	g = m.d_loglike()
	assertAlmostEqual( -127397.53666666638, g[ 0], delta=0.000001 )
	assertAlmostEqual( 42104.2, g[ 1], delta=0.01 )
	assertAlmostEqual( 687.7  , g[ 2], delta=0.01 )
	assertAlmostEqual( 1043.7 , g[ 3], delta=0.01 )
	assertAlmostEqual( 380.78333333332864, g[ 4], delta=0.00000001 )
	assertAlmostEqual( 279.8  , g[ 5], delta=0.01 )
	assertAlmostEqual( 113.65 , g[ 6], delta=0.001 )
	assertAlmostEqual( 41179.541666666715, g[ 7], delta=0.0000001 )
	assertAlmostEqual( 60691.541666666686, g[ 8], delta=0.0000001 )
	assertAlmostEqual( 24028.374999999985, g[ 9], delta=0.0000001 )
	assertAlmostEqual( 17374.8, g[10], delta=0.01 )
	assertAlmostEqual( 7739.108333333339, g[11], delta=0.0000001 )
	assertAlmostEqual(-537.598179908304, g[12], delta=0.00000001 )
	v =(-0.0053047 ,
		-0.0746418 ,
		-2.07691   ,
		-9.2709    ,
		-0.185096  ,
		-1.65575   ,
		1.86265    ,
		-0.00427335,
		0.0126937  ,
		-0.00710494,
		-0.0160185 ,
		-0.00716963,
		2.32074    ,
		)
	assertAlmostEqual( -4213.0122967116695, m.loglike(v), delta=0.00000001 )
	g = m.d_loglike()
	print(m.parameter_names()[0])
	assertAlmostEqual( -4140.972283757319, g[ 0], delta=0.0000001 )
	print(m.parameter_names()[1])
	assertAlmostEqual(  12836.151998017447, g[ 1], delta=0.0000001 )
	print(m.parameter_names()[2])
	assertAlmostEqual(  263.627, g[ 2], delta=0.001 )
	print(m.parameter_names()[3])
	assertAlmostEqual(  -32.735, g[ 3], delta=0.001 )
	print(m.parameter_names()[4])
	assertAlmostEqual(  105.441, g[ 4], delta=0.0001 )
	print(m.parameter_names()[5])
	assertAlmostEqual( -24.52301491338605, g[ 5], delta=0.00000001 )
	print(m.parameter_names()[6])
	assertAlmostEqual(  84.57760963703537, g[ 6], delta=0.00000001 )
	print(m.parameter_names()[7])
	assertAlmostEqual(  15532.15438906018, g[ 7], delta=0.00000001 )
	print(m.parameter_names()[8])
	assertAlmostEqual( -1748.1507300171443, g[ 8], delta=0.00000001 )
	print(m.parameter_names()[9])
	assertAlmostEqual(  6338.61, g[ 9], delta=0.01 )
	print(m.parameter_names()[10])
	assertAlmostEqual( -1252.58, g[10], delta=0.01 )
	print(m.parameter_names()[11])
	assertAlmostEqual(  4531.231730902182, g[11], delta=0.00000001 )
	print(m.parameter_names()[12])
	assertAlmostEqual(  294.4276572113675, g[12], delta=0.00000001 )
	m.tearDown()
	m.option.gradient_diagnostic = 10
	m.estimate()

def test_swissmetro_09nested():
	d = DB.Example('swissmetro')
	m = Model(d)
	m.logger(True)
	m.parameter("ASC_TRAIN",0)
	m.parameter("B_TIME"   ,0)
	m.parameter("B_COST"   ,0)
	m.parameter("ASC_CAR"  ,0)
	m.parameter("existing" ,1,1)
	m.utility.co("1",1,"ASC_TRAIN")
	m.utility.co("1",3,"ASC_CAR")
	m.utility.co("TRAIN_TT",1,"B_TIME")
	m.utility.co("SM_TT",2,"B_TIME")
	m.utility.co("CAR_TT",3,"B_TIME")
	m.utility.co("TRAIN_CO*(GA==0)",1,"B_COST")
	m.utility.co("SM_CO*(GA==0)"   ,2,"B_COST")
	m.utility.co("CAR_CO",        3,"B_COST")
	m.option.gradient_diagnostic = 10
	m.option.calculate_std_err = 1
	m.option.threads = 4
	m.nest("existing", 4, "existing")
	m.link(4, 1)
	m.link(4, 3)
	m.estimate()
	#del m
	#del d
	

#ms = larch.Model.Example()
#ms.estimate_scipy()


#test_swissmetro_1()
#test_nl2_single_cycle()

#import larch.examples
#larch.examples.load_example(109)
#m = larch.Model.Example()
#m.logger(True)
#print(m.save_buffer())
#m.estimate()
#print(m)

#larch.logging.setLevel(30)
#test_swissmetro_09nested()
#
#import larch.test
#larch.test.run()

#import gc
#gc.collect()#

larch.logging.setLevel(1)
sys.path.append("/Users/jpn/Dropbox/CamSys/Memphis/")
os.chdir("/Users/jpn/Dropbox/CamSys/Memphis/")
execfile("/Users/jpn/Dropbox/CamSys/Memphis/dest_main.py")

larch.logging.setLevel(1)
dcm.loglike()

