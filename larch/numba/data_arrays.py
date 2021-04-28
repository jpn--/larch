import numpy as np
from typing import NamedTuple


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
    caseindex_name = '_caseid_'
    altindex_name = '_altid_'
    from sharrow import Dataset
    from xarray import DataArray
    ds = Dataset()
    if dataframes.data_co is not None:
        caseindex_name = dataframes.data_co.index.name
        ds.update(Dataset.from_dataframe(dataframes.data_co))
    if dataframes.data_ca is not None:
        caseindex_name = dataframes.data_ca.index.names[0]
        altindex_name = dataframes.data_ca.index.names[1]
        ds.update(Dataset.from_dataframe(dataframes.data_ca))
    altnames = dataframes.alternative_names()
    if altnames:
        ds.coords['alt_names'] = DataArray(altnames, dims=(altindex_name,))
    return ds

def prepare_data(datashare, request):
    pass