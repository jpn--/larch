import warnings
import numpy as np
try:
    from sharrow import Dataset as _sharrow_Dataset, SharedData as _sharrow_SharedData, DataArray
except ImportError:
    raise RuntimeError("larch.dataset requires the sharrow library")


class DataPool(_sharrow_SharedData):

    @property
    def n_cases(self):
        return self.main.dims['_caseid_']


class Dataset(_sharrow_Dataset):

    __slots__ = ()

    def __new__(cls, *args, caseid=None, **kwargs):
        import logging
        logging.getLogger("SHARROW").debug(f"NEW INSTANCE {cls}")
        obj = super().__new__(cls)
        super(cls, obj).__init__(*args, **kwargs)
        if caseid is not None:
            if caseid not in obj.dims:
                raise ValueError(f"no dim named '{caseid}' to make into _caseid_")
            obj = obj.rename_dims({caseid: '_caseid_'})
        return obj

    def __init__(self, *args, **kwargs):
        pass # init for superclass happens inside __new__

    @property
    def n_cases(self):
        return self.dims['_caseid_']

    @property
    def n_alts(self):
        if '_altid_' in self.dims:
            return self.dims['_altid_']
        if 'n_alts' in self.attrs:
            return self.attrs['n_alts']
        raise ValueError('no n_alts set')

    def _repr_html_(self):
        html = super()._repr_html_()
        html = html.replace("sharrow.Dataset", "larch.Dataset")
        return html

    def __repr__(self):
        r = super().__repr__()
        r = r.replace("sharrow.Dataset", "larch.Dataset")
        return r

    def to_arrays(self, graph, float_dtype=np.float64):
        from .numba.data_arrays import DataArrays
        from .numba.cascading import array_av_cascade, array_ch_cascade

        if 'co' in self:
            co = self['co'].values.astype(float_dtype)
        else:
            co = np.empty( (self.n_cases, 0), dtype=float_dtype)

        if 'ca' in self:
            ca = self['ca'].values.astype(float_dtype)
        else:
            ca = np.empty( (self.n_cases, self.n_alts, 0), dtype=float_dtype)

        if 'wt' in self:
            wt = self['wt'].astype(float_dtype)
        else:
            wt = np.ones(self.n_cases, dtype=float_dtype)

        if 'ch' in self:
            ch = array_ch_cascade(self['ch'].values, graph, dtype=float_dtype)
        else:
            ch = np.zeros([self.n_cases, len(graph)], dtype=float_dtype)

        if 'av' in self:
            av = array_av_cascade(self['av'].values, graph)
        else:
            av = np.ones([self.n_cases, len(graph)], dtype=np.int8)

        return DataArrays(
            ch, av, wt, co, ca
        )

    def validate_format(self):
        error_msgs = []
        warn_msgs = []
        if '_caseid_' not in self.dims:
            error_msgs.append(
                "- There is no dimensions named `_caseid_`. "
            )
        if '_altid_' not in self.dims:
            warn_msgs.append(
                "- There is no dimensions named `_altid_`. "
            )
        msgs = []
        if error_msgs:
            msgs.append("ERRORS:")
            msgs.extend(error_msgs)
        if warn_msgs:
            msgs.append("WARNINGS:")
            msgs.extend(warn_msgs)
        return msgs

    def query_cases(self, query):
        return self.query({'_caseid_': query})

