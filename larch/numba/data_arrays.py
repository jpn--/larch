import numpy as np
import pandas as pd
import xarray as xr
import logging
from typing import NamedTuple

from ..dim_names import CASEID, ALTID

class _case_slice:

    def __get__(self, obj, objtype=None):
        self.parent = obj
        return self

    def __getitem__(self, idx):
        return type(self.parent)(
            **{
                k: getattr(self.parent, k)[idx] if len(k)==2 else getattr(self.parent, k)
                for k in self.parent._fields
            }
        )


class DataArrays(NamedTuple):
    ch: np.ndarray
    av: np.ndarray
    wt: np.ndarray
    co: np.ndarray
    ca: np.ndarray
    alt_codes: np.ndarray = None
    alt_names: np.ndarray = None

    cs = _case_slice()

    @property
    def alternatives(self):
        if self.alt_codes is not None:
            if self.alt_names is not None:
                return dict(zip(self.alt_codes, self.alt_names))
            else:
                return {i:str(i) for i in self.alt_codes}
        else:
            raise ValueError("alt_codes not defined")


def to_dataset(dataframes):
    caseindex_name = CASEID
    altindex_name = ALTID
    from ..dataset import Dataset
    from xarray import DataArray
    coords = {
        CASEID: dataframes.caseindex.values,
        ALTID: dataframes.alternative_codes(),
    }
    ds = Dataset(coords=coords)
    if dataframes.data_co is not None:
        caseindex_name = dataframes.data_co.index.name
        ds.update(Dataset.from_dataframe(dataframes.data_co))
    if dataframes.data_ca is not None:
        caseindex_name = dataframes.data_ca.index.names[0]
        altindex_name = dataframes.data_ca.index.names[1]
        ds.update(Dataset.from_dataframe(dataframes.data_ca))
    alt_names = dataframes.alternative_names()
    if alt_names:
        ds.coords['alt_names'] = DataArray(alt_names, dims=(altindex_name,))
    return ds


def prepare_data(
        datashare,
        request,
        float_dtype=None,
        cache_dir=None,
        flows=None,
):
    log = logging.getLogger("Larch")
    from ..dataset import Dataset, DataArray, DataTree
    if float_dtype is None:
        float_dtype = np.float64
    log.debug(f"building dataset from datashare coords: {datashare.coords}")
    model_dataset = Dataset(
        coords=datashare.coords,
    )

    from .model import NumbaModel # avoid circular import
    if isinstance(request, NumbaModel):
        alts = request.graph.elemental_names()
        if ALTID not in model_dataset.coords:
            model_dataset.coords[ALTID] = list(alts.keys())
        if 'alt_names' not in model_dataset.coords:
            model_dataset.coords['alt_names'] = DataArray(list(alts.values()), dims=(ALTID,))
        request = request.required_data()

    if flows is None:
        flows = {}

    if isinstance(datashare, (DataTree, )):
        log.debug(f"adopting existing DataTree")
        if not datashare.relationships_are_digitized:
            datashare.digitize_relationships(inplace=True)
        shared_data_ca = datashare
        shared_data_co = datashare.drop_dims(datashare.ALTID, ignore_missing_dims=True)
    else:
        log.debug(f"initializing new DataTree")
        shared_data_ca = DataTree(main=datashare)
        shared_data_ca.digitize_relationships(inplace=True)
        shared_data_co = shared_data_ca.drop_dims(shared_data_ca.ALTID, ignore_missing_dims=True)

    if 'co' in request:
        log.debug(f"requested co data: {request['co']}")
        model_dataset, flows['co'] = _prep_co(
            model_dataset,
            shared_data_co,
            request['co'],
            tag='co',
            dtype=float_dtype,
            cache_dir=cache_dir,
            flow=flows.get('co'),
        )
    if 'ca' in request:
        log.debug(f"requested ca data: {request['ca']}")
        model_dataset, flows['ca'] = _prep_ca(
            model_dataset,
            shared_data_ca,
            request['ca'],
            tag='ca',
            dtype=float_dtype,
            cache_dir=cache_dir,
            flow=flows.get('ca'),
        )

    if 'choice_ca' in request:
        log.debug(f"requested choice_ca data: {request['choice_ca']}")
        model_dataset, flows['choice_ca'] = _prep_ca(
            model_dataset,
            shared_data_ca,
            [request['choice_ca']],
            tag='ch',
            preserve_vars=False,
            dtype=float_dtype,
            cache_dir=cache_dir,
            flow=flows.get('choice_ca'),
        )
    if 'choice_co_code' in request:
        log.debug(f"requested choice_co_code data: {request['choice_co_code']}")
        choicecodes = datashare[request['choice_co_code']]
        da_ch = DataArray(
            float_dtype(0),
            dims=[CASEID, ALTID],
            coords={
                CASEID: model_dataset.coords[CASEID],
                ALTID: model_dataset.coords[ALTID],
            },
            name='ch',
        )
        for i,a in enumerate(model_dataset.coords[ALTID]):
            da_ch[:, i] = (choicecodes == a)
        model_dataset = model_dataset.merge(da_ch)
    if 'choice_co_vars' in request:
        log.debug(f"requested choice_co_vars data: {request['choice_co_vars']}")
        raise NotImplementedError('choice_co_vars')
    if 'choice_any' in request:
        log.debug(f"requested choice_any data: {request['choice_any']}")
        raise NotImplementedError('choice_any')

    if 'weight_co' in request:
        log.debug(f"requested weight_co data: {request['weight_co']}")
        model_dataset, flows['weight_co'] = _prep_co(
            model_dataset,
            shared_data_co,
            [request['weight_co']],
            tag='wt',
            preserve_vars=False,
            dtype=float_dtype,
            cache_dir=cache_dir,
            flow=flows.get('weight_co'),
        )

    if 'avail_ca' in request:
        log.debug(f"requested avail_ca data: {request['avail_ca']}")
        model_dataset, flows['avail_ca'] = _prep_ca(
            model_dataset,
            shared_data_ca,
            [request['avail_ca']],
            tag='av',
            preserve_vars=False,
            dtype=np.int8,
            cache_dir=cache_dir,
            flow=flows.get('avail_ca'),
        )
    if 'avail_co' in request:
        log.debug(f"requested avail_co data: {request['avail_co']}")
        av_co_expressions = {
            a: request['avail_co'].get(a, '0')
            for a in model_dataset.coords[ALTID].values
        }
        model_dataset, flows['avail_co'] = _prep_co(
            model_dataset,
            shared_data_co,
            av_co_expressions,
            tag='av',
            preserve_vars=False,
            dtype=np.int8,
            dim_name=ALTID,
            cache_dir=cache_dir,
            flow=flows.get('avail_co'),
        )
    if 'avail_any' in request:
        log.debug(f"requested avail_any data: {request['avail_any']}")
        raise NotImplementedError('avail_any')

    return model_dataset, flows

def flownamer(tag, definition_spec, extra_hash_features=()):
    import hashlib, base64
    defs_hash = hashlib.md5()
    defs_hash.update(str(tag).encode("utf8"))
    for k, v in definition_spec.items():
        defs_hash.update(str(k).encode("utf8"))
        defs_hash.update(str(v).encode("utf8"))
    for k in extra_hash_features:
        defs_hash.update(str(k).encode("utf8"))
    return "pipeline_"+(base64.b32encode(defs_hash.digest())).decode().replace("=","")


def _prep_ca(
        model_dataset,
        shared_data_ca,
        vars_ca,
        tag='ca',
        preserve_vars=True,
        dtype=None,
        cache_dir=None,
        flow=None,
):
    from ..dataset import Dataset, DataArray, DataTree
    assert isinstance(shared_data_ca, DataTree)
    if not isinstance(vars_ca, dict):
        vars_ca = {i:i for i in vars_ca}
    flowname = flownamer(tag, vars_ca, shared_data_ca._hash_features())
    if flow is None or flowname != flow.name:
        flow = shared_data_ca.setup_flow(vars_ca, cache_dir=cache_dir, name=flowname)
    arr = flow.load(
        shared_data_ca,
        dtype=dtype,
    )
    caseid_dim = shared_data_ca.CASEID
    if preserve_vars or len(vars_ca)>1:
        arr = arr.reshape(
            model_dataset.dims.get(caseid_dim),
            model_dataset.dims.get(ALTID),
            -1,
        )
        da = DataArray(
            arr,
            dims=[caseid_dim, ALTID, f"var_{tag}"],
            coords={
                caseid_dim: model_dataset.coords[caseid_dim],
                ALTID: model_dataset.coords[ALTID],
                f"var_{tag}": list(vars_ca.keys()),
            },
            name=tag,
        )
    else:
        arr = arr.reshape(
            model_dataset.dims.get(caseid_dim),
            model_dataset.dims.get(ALTID),
        )
        da = DataArray(
            arr,
            dims=[caseid_dim, ALTID],
            coords={
                caseid_dim: model_dataset.coords[caseid_dim],
                ALTID: model_dataset.coords[ALTID],
            },
            name=tag,
        )
    return model_dataset.merge(da), flow


def _prep_co(
        model_dataset,
        shared_data_co,
        vars_co,
        tag='co',
        preserve_vars=True,
        dtype=None,
        dim_name=None,
        cache_dir=None,
        flow=None,
):
    from ..dataset import DataArray, DataTree
    assert isinstance(shared_data_co, DataTree)
    if not isinstance(vars_co, dict):
        vars_co = {i: i for i in vars_co}
    flowname = flownamer(tag, vars_co, shared_data_co._hash_features())
    if flow is None or flowname != flow.name:
        flow = shared_data_co.setup_flow(vars_co, cache_dir=cache_dir, name=flowname)
    arr = flow.load(
        shared_data_co,
        dtype=dtype,
    )
    caseid_dim = shared_data_co.CASEID
    if preserve_vars or len(vars_co)>1:
        if dim_name is None:
            dim_name = f"var_{tag}"
        arr = arr.reshape(
            model_dataset.dims.get(caseid_dim),
            -1,
        )
        da = DataArray(
            arr,
            dims=[caseid_dim, dim_name],
            coords={
                caseid_dim: model_dataset.coords[caseid_dim],
                dim_name: list(vars_co.keys()),
            },
            name=tag,
        )
    else:
        arr = arr.reshape(
            model_dataset.dims.get(caseid_dim),
        )
        da = DataArray(
            arr,
            dims=[caseid_dim],
            coords={
                caseid_dim: model_dataset.coords[caseid_dim],
            },
            name=tag,
        )
    return model_dataset.merge(da), flow

