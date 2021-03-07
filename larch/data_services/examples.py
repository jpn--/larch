import os
import tables as tb
import pandas as pd

def MTC():
	from larch.dataframes import DataFrames
	from larch.data_warehouse import example_file
	ca = pd.read_csv(example_file('MTCwork.csv.gz'), index_col=('casenum', 'altnum'))
	ca['altnum'] = ca.index.get_level_values('altnum')
	dt = DataFrames(
		ca,
		ch="chose",
		crack=True,
		alt_codes=[1, 2, 3, 4, 5, 6],
		alt_names=['DA', 'SR2', 'SR3', 'TRANSIT', 'BIKE', 'WALK']
	)
	dt.data_ce_as_ca("_avail_")
	return dt
	# from .service import DataService
	# from .h5 import H5PodCA, H5PodCO
	# warehouse_file = os.path.join( os.path.dirname(__file__), '..', 'data_warehouse', 'MTCwork.h5d')
	# f = tb.open_file(warehouse_file, mode='r')
	# idca = H5PodCA(f.root.larch.idca)
	# idco = H5PodCO(f.root.larch.idco)
	# return DataService(pods=[idca,idco], altids=[1,2,3,4,5,6], altnames=['DA','SR2','SR3','TRANSIT','BIKE','WALK'])


def EXAMPVILLE(model=None):
	from ..util import Dict
	evil = Dict()
	from .service import DataService
	import numpy
	from .h5 import H5PodCA, H5PodCO, H5PodRC, H5PodCS
	from ..omx import OMX
	warehouse_dir = os.path.join( os.path.dirname(__file__), '..', 'data_warehouse', )
	evil.skims = OMX(os.path.join(warehouse_dir,'exampville.omx'), mode='r')
	evil.tours = H5PodCO(os.path.join(warehouse_dir,'exampville_tours.h5'), mode='r', ident='tours')
	# hhs = H5PodCO(os.path.join(warehouse_dir,'exampville_hh.h5'))
	# persons = H5PodCO(os.path.join(warehouse_dir,'exampville_person.h5'))
	# tours.merge_external_data(hhs, 'HHID', )
	# tours.merge_external_data(persons, 'PERSONID', )
	# tours.add_expression("HOMETAZi", "HOMETAZ-1", dtype=int)
	# tours.add_expression("DTAZi", "DTAZ-1", dtype=int)
	evil.skims_rc = H5PodRC(evil.tours.HOMETAZi[:], evil.tours.DTAZi[:], groupnode=evil.skims.data, ident='skims_rc')
	evil.tours_stack = H5PodCS([evil.tours, evil.skims_rc], storage=evil.tours, ident='tours_stack_by_mode').set_alts([1,2,3,4,5])
	DA = 1
	SR = 2
	Walk = 3
	Bike = 4
	Transit = 5
	# tours_stack.set_bunch('choices', {
	# 	DA: 'TOURMODE==1',
	# 	SR: 'TOURMODE==2',
	# 	Walk: 'TOURMODE==3',
	# 	Bike: 'TOURMODE==4',
	# 	Transit: 'TOURMODE==5',
	# })
	#
	# tours_stack.set_bunch('availability', {
	# 	DA: '(AGE>=16)',
	# 	SR: '1',
	# 	Walk: 'DIST<=3',
	# 	Bike: 'DIST<=15',
	# 	Transit: 'RAIL_TIME>0',
	# })
	evil.mode_ids = [DA, SR, Walk, Bike, Transit]
	evil.mode_names = ['DA', 'SR', 'Walk', 'Bike', 'Transit']
	nZones = 15
	evil.dest_ids = numpy.arange(1,nZones+1)
	evil.logsums = H5PodCA(os.path.join(warehouse_dir,'exampville_mc_logsums.h5'), mode='r', ident='logsums')
	return evil

def SWISSMETRO():
	from ..util.temporaryfile import TemporaryGzipInflation
	warehouse_dir = os.path.join( os.path.dirname(__file__), '..', 'data_warehouse', )
	from .service import DataService
	from .h5 import H5PodCO, H5PodCS
	warehouse_file = TemporaryGzipInflation(os.path.join(warehouse_dir, "swissmetro.h5.gz"))
	f = tb.open_file(warehouse_file, mode='r')
	idco = H5PodCO(f.root.larch.idco)
	stack = H5PodCS(
		[idco], ident='stack_by_mode', alts=[1,2,3],
		traveltime={1: "TRAIN_TT", 2: "SM_TT", 3: "CAR_TT"},
		cost={1: "TRAIN_CO*(GA==0)", 2: "SM_CO*(GA==0)", 3: "CAR_CO"},
		avail={1:'TRAIN_AV*(SP!=0)', 2:'SM_AV', 3:'CAR_AV*(SP!=0)'},
		choice={1: "CHOICE==1", 2: "CHOICE==2", 3: "CHOICE==3"},
	)
	return DataService(pods=[idco, stack], altids=[1,2,3], altnames=['Train', 'SM', 'Car'])


def ITINERARY_RAW():
	warehouse_file = os.path.join( os.path.dirname(__file__), '..', 'data_warehouse', 'itinerary_data.csv.gz')
	import pandas
	return pandas.read_csv(warehouse_file)


def example_file(filename):
	warehouse_file = os.path.normpath( os.path.join( os.path.dirname(__file__), '..', 'data_warehouse', filename) )
	if os.path.exists(warehouse_file):
		return warehouse_file
	raise FileNotFoundError(f"there is no example data file '{warehouse_file}' in data_warehouse")
