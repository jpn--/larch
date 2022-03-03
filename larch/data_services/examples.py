import os
import tables as tb
import pandas as pd

def MTC(format='dataframes'):
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
	if format in ('dataset', 'datapool'):
		from ..dataset import Dataset, DataArray
		dataset = Dataset.from_dataframe(dt.data_co.rename_axis(index='caseid'))
		dataset = dataset.merge(Dataset.from_dataframe(dt.data_ce_as_ca().rename_axis(index=('caseid', 'altid'))))
		dataset['avail'] = DataArray(dt.data_av.values, dims=['caseid', 'altid'], coords=dataset.coords)
		dataset.coords['alt_names'] = DataArray(
			['DA', 'SR2', 'SR3+', 'Transit', 'Bike', 'Walk'],
			dims=['altid'],
		)
		dataset.flow.CASEID = 'caseid'
		dataset.flow.ALTID = 'altid'
		return dataset
	elif format == 'dataframes':
		dt.data_ce_as_ca("_avail_")
		return dt
	else:
		raise ValueError(f"undefined format {format}")


def EXAMPVILLE(format='dataframes', model='mode', cache_dir=None):
	if format == 'datatree' and model == 'mode':
		from ..examples import example
		from ..dataset import Dataset, DataArray, DataTree
		_hh, _pp, _tour, _skims = example(200, ['hh', 'pp', 'tour', 'skims'])
		tours = Dataset.construct(
			_tour.set_index('TOURID'), caseid='TOURID',
		)
		tours.coords['altid'] = DataArray(
			[1, 2, 3, 4, 5], dims="_altid_",
		)
		tours.coords['altname'] = DataArray(
			['DA', 'SR', 'Walk', 'Bike', 'Transit'], dims="_altid_",
		)
		od_skims = Dataset.construct.from_omx(_skims)
		hh = Dataset(_hh.set_index('HHID'))
		pp = Dataset(_pp.set_index('PERSONID'))
		tree = DataTree(
			tours=tours,
			hh=hh,
			pp=pp,
			od=od_skims,
			do=od_skims,
			root_node_name='tours',
			relationships=(
				"tours.HHID @ hh.HHID",
				"tours.PERSONID @ pp.PERSONID",
				"hh.HOMETAZ @ od.otaz",
				"tours.DTAZ @ od.dtaz",
				"hh.HOMETAZ @ do.dtaz",
				"tours.DTAZ @ do.otaz",
			),
		)
		return tree
	elif format == 'dataframes' and model == 'mode':
		from ..examples import example
		hh, pp, tour, skims = example(200, ['hh', 'pp', 'tour', 'skims'])
		raw = tour.merge(hh, on='HHID').merge(pp, on=('HHID', 'PERSONID'))
		raw["HOMETAZi"] = raw["HOMETAZ"] - 1
		raw["DTAZi"] = raw["DTAZ"] - 1
		raw = raw.join(skims.get_rc_dataframe(raw.HOMETAZi, raw.DTAZi))
		from ..dataframes import DataFrames
		return DataFrames(
			co=raw,
			alt_codes=[1, 2, 3, 4, 5],
			alt_names=['DA', 'SR', 'Walk', 'Bike', 'Transit'],
			ch_name='TOURMODE',
		)
	else:
		raise ValueError(f"undefined format {format} for Exampville {model} choice model")

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
