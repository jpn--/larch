import logging
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from collections.abc import Mapping
from xarray.core import dtypes

try:
    from sharrow import Dataset as _sharrow_Dataset, DataArray
    from sharrow import DataTree as _sharrow_DataTree
except ImportError:
    warnings.warn("larch.dataset requires the sharrow library")
    class _noclass:
        pass
    _sharrow_Dataset = _noclass
    DataArray = xr.DataArray
    _sharrow_DataTree = _noclass

from .dim_names import CASEID as _CASEID, ALTID as _ALTID

class Dataset(_sharrow_Dataset):
    """
    A xarray.Dataset extended interface for use with Larch.

    A Dataset consists of variables, coordinates and attributes which
    together form a self describing dataset.

    Dataset implements the mapping interface with keys given by variable
    names and values given by DataArray objects for each variable name.

    One dimensional variables with name equal to their dimension are
    index coordinates used for label based indexing.

    For Larch, one dimension of each Dataset must typiccally be named '_caseid_',
    and this dimension is used to identify the individual discrete choice
    observations or simulations in the data. The `caseid` argument can be
    used to set an existing dimension as '_caseid_' on Dataset construction.

    Parameters
    ----------
    data_vars : dict-like, optional
        A mapping from variable names to :py:class:`~xarray.DataArray`
        objects, :py:class:`~xarray.Variable` objects or to tuples of
        the form ``(dims, data[, attrs])`` which can be used as
        arguments to create a new ``Variable``. Each dimension must
        have the same length in all variables in which it appears.

        The following notations are accepted:

        - mapping {var name: DataArray}
        - mapping {var name: Variable}
        - mapping {var name: (dimension name, array-like)}
        - mapping {var name: (tuple of dimension names, array-like)}
        - mapping {dimension name: array-like}
          (it will be automatically moved to coords, see below)

        Each dimension must have the same length in all variables in
        which it appears.

    coords : dict-like, optional
        Another mapping in similar form as the `data_vars` argument,
        except the each item is saved on the dataset as a "coordinate".
        These variables have an associated meaning: they describe
        constant/fixed/independent quantities, unlike the
        varying/measured/dependent quantities that belong in
        `variables`. Coordinates values may be given by 1-dimensional
        arrays or scalars, in which case `dims` do not need to be
        supplied: 1D arrays will be assumed to give index values along
        the dimension with the same name.

        The following notations are accepted:

        - mapping {coord name: DataArray}
        - mapping {coord name: Variable}
        - mapping {coord name: (dimension name, array-like)}
        - mapping {coord name: (tuple of dimension names, array-like)}
        - mapping {dimension name: array-like}
          (the dimension name is implicitly set to be the same as the
          coord name)

        The last notation implies that the coord name is the same as
        the dimension name.

    attrs : dict-like, optional
        Global attributes to save on this dataset.

    caseid : str, optional, keyword only
        This named dimension will be marked as the '_caseid_' dimension.

    alts : str or Mapping or array-like, keyword only
        If given as a str, this named dimension will be marked as the
        '_altid_' dimension.  Otherwise, give a Mapping that defines
        alternative names and (integer) codes or an array of codes.
    """

    __slots__ = ()

    def __new__(cls, *args, caseid=None, alts=None, **kwargs):
        import logging
        logging.getLogger("sharrow").debug(f"NEW INSTANCE {cls}")
        obj = super().__new__(cls)
        super(cls, obj).__init__(*args, **kwargs)
        return cls.__initialize_for_larch(obj, caseid, alts)

    @classmethod
    def __initialize_for_larch(cls, obj, caseid=None, alts=None):
        if caseid is not None:
            if caseid not in obj.dims:
                raise ValueError(f"no dim named '{caseid}' to make into {_CASEID}")
            obj.attrs[_CASEID] = caseid
        if isinstance(alts, pd.Series):
            alts_dim_name = alts.name or alts.index.name or '_altid_'
            alts_k = DataArray(
                alts.index, dims=alts_dim_name,
            )
            alts_v = DataArray(
                alts.values, dims=alts_dim_name,
            )
        elif isinstance(alts, Mapping):
            alts_dim_name = '_altid_'
            alts_k = DataArray(
                list(alts.keys()), dims=alts_dim_name,
            )
            alts_v = DataArray(
                list(alts.values()), dims=alts_dim_name,
            )
        elif isinstance(alts, str):
            alts_dim_name = alts
            alts_k = alts_v = None
        elif alts is None:
            alts_dim_name = None
            alts_k = alts_v = None
        else:
            alts_dim_name = getattr(alts, 'name', '_altid_')
            alts_v = np.asarray(alts).reshape(-1)
            alts_k = None
        if alts_dim_name:
            obj.attrs[_ALTID] = alts_dim_name
        if alts_k is not None:
            if np.issubdtype(alts_v, np.integer) and not np.issubdtype(alts_k, np.integer):
                obj.coords[alts_dim_name] = alts_v
                obj.coords['alt_names'] = alts_k
            else:
                obj.coords[alts_dim_name] = alts_k
                obj.coords['alt_names'] = alts_v
        elif alts_v is not None:
            obj.coords[alts_dim_name] = alts_v
        return obj

    @classmethod
    def construct(cls, source, caseid=None, alts=None):
        if isinstance(source, pd.DataFrame):
            source = cls.from_dataframe(source)
        else:
            source = super().construct(source)
        return cls.__initialize_for_larch(source, caseid, alts)

    def __init__(self, *args, **kwargs):
        pass # init for superclass happens inside __new__

    @property
    def CASEID(self):
        result = self.attrs.get(_CASEID, None)
        if result is None:
            warnings.warn("no defined CASEID")
            return _CASEID
        return result

    @CASEID.setter
    def CASEID(self, dim_name):
        if dim_name not in self.dims:
            raise ValueError(f"{dim_name} not in dims")
        self.attrs[_CASEID] = dim_name

    @property
    def ALTID(self):
        result = self.attrs.get(_ALTID, None)
        if result is None:
            warnings.warn("no defined ALTID")
            return _ALTID
        return result

    @ALTID.setter
    def ALTID(self, dim_name):
        self.attrs[_ALTID] = dim_name

    @property
    def n_cases(self):
        try:
            return self.dims[self.CASEID]
        except KeyError:
            logging.getLogger().error(f"missing {self.CASEID!r} among dims {self.dims}")
            raise

    @property
    def n_alts(self):
        if self.ALTID in self.dims:
            return self.dims[self.ALTID]
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
        if self.CASEID not in self.dims:
            error_msgs.append(
                f"- There is no dimensions named `{self.CASEID}`. "
            )
        if self.ALTID not in self.dims:
            warn_msgs.append(
                f"- There is no dimensions named `_altid_`. "
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
        return self.query({self.CASEID: query})

    def dissolve_coords(self, dim, others=None):
        d = self.reset_index(dim)
        a = d[f"{dim}_"]
        mapper = dict((j, i) for (i, j) in enumerate(a.to_series()))
        mapper_f = np.vectorize(mapper.get)
        if others is None:
            others = []
        if isinstance(others, str):
            others = [others]
        for other in others:
            d[other] = xr.apply_ufunc(mapper_f, d[other])
        return d

    def dissolve_zero_variance(self, dim, inplace=False):
        if inplace:
            obj = self
        else:
            obj = self.copy()
        for k in obj.variables:
            if dim in obj[k].dims:
                if obj[k].std(dim=dim).max() < 1e-10:
                    obj[k] = obj[k].min(dim=dim)
        return obj

    def set_dtypes(self, dtypes, inplace=False, on_error='warn'):
        if isinstance(dtypes, pd.DataFrame):
            dtypes = dtypes.dtypes
        if inplace:
            obj = self
        else:
            obj = self.copy()
        for k in obj:
            try:
                obj[k] = obj[k].astype(dtypes[k])
            except Exception as err:
                if on_error == 'warn':
                    warnings.warn(f"{err!s} on converting {k}")
                elif on_error == 'raise':
                    raise
        return obj

    @classmethod
    def from_idca(cls, df, crack=True, alts=None):
        if df.index.nlevels != 2:
            raise ValueError("source idca dataframe must have a two "
                             "level MultiIndex giving case and alt id's")
        caseidname, altidname = df.index.names
        ds = cls.construct(df, alts=alts)
        if crack:
            ds = ds.dissolve_zero_variance(altidname)
        ds = ds.set_dtypes(df)
        renames = {}
        if caseidname is not None:
            renames[caseidname] = _CASEID
        if altidname is not None:
            renames[altidname] = _ALTID
        ds = ds.rename(renames)
        return ds

    @classmethod
    def from_idco(cls, df, alts=None):
        if df.index.nlevels != 1:
            raise ValueError("source idco dataframe must have a one "
                             "level Index giving case id's")
        caseidname = df.index.name
        ds = cls.construct(df, alts=alts)
        ds = ds.set_dtypes(df)
        if caseidname is not None:
            ds = ds.rename({caseidname: _CASEID})
        return ds


    def setup_flow(self, *args, **kwargs):
        return DataTree(main=self).setup_flow(*args, **kwargs)


class DataTree(_sharrow_DataTree):

    DatasetType = Dataset

    def __init__(
        self,
        graph=None,
        root_node_name=None,
        extra_funcs=(),
        extra_vars=None,
        cache_dir=None,
        relationships=(),
        force_digitization=False,
        **kwargs,
    ):
        super().__init__(
            graph=graph,
            root_node_name=root_node_name,
            extra_funcs=extra_funcs,
            extra_vars=extra_vars,
            cache_dir=cache_dir,
            relationships=relationships,
            force_digitization=force_digitization,
            **kwargs,
        )
        self.dim_order = (self.CASEID, self.ALTID)

    @property
    def CASEID(self):
        result = self.root_dataset.attrs.get(_CASEID, None)
        if result is None:
            warnings.warn("no defined CASEID")
            return _CASEID
        return result

    @property
    def ALTID(self):
        result = self.root_dataset.attrs.get(_ALTID, None)
        if result is None:
            warnings.warn("no defined ALTID")
            return _ALTID
        return result

    @property
    def n_cases(self):
        return self.root_dataset.dims[self.CASEID]

    @property
    def n_alts(self):
        return self.root_dataset.dims[self.ALTID]

    def query_cases(self, query):
        obj = self.copy()
        obj.root_dataset = obj.root_dataset.query_cases(query)
        return obj


def merge(
        objects,
        compat= "no_conflicts",
        join= "outer",
        fill_value = dtypes.NA,
        combine_attrs = "override",
        *,
        caseid=None,
        alts=None,
):
    """
    Merge any number of xarray objects into a single larch.Dataset as variables.

    Parameters
    ----------
    objects : iterable of Dataset or iterable of DataArray or iterable of dict-like
        Merge together all variables from these objects. If any of them are
        DataArray objects, they must have a name.

    compat : {"identical", "equals", "broadcast_equals", "no_conflicts", "override"}, optional
        String indicating how to compare variables of the same name for
        potential conflicts:
        - "broadcast_equals": all values must be equal when variables are
          broadcast against each other to ensure common dimensions.
        - "equals": all values and dimensions must be the same.
        - "identical": all values, dimensions and attributes must be the
          same.
        - "no_conflicts": only values which are not null in both datasets
          must be equal. The returned dataset then contains the combination
          of all non-null values.
        - "override": skip comparing and pick variable from first dataset

    join : {"outer", "inner", "left", "right", "exact"}, optional
        String indicating how to combine differing indexes in objects.
        - "outer": use the union of object indexes
        - "inner": use the intersection of object indexes
        - "left": use indexes from the first object with each dimension
        - "right": use indexes from the last object with each dimension
        - "exact": instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - "override": if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.

    fill_value : scalar or dict-like, optional
        Value to use for newly missing values. If a dict-like, maps
        variable names to fill values. Use a data array's name to
        refer to its values.

    combine_attrs : {"drop", "identical", "no_conflicts", "drop_conflicts", \
                    "override"} or callable, default: "override"
        A callable or a string indicating how to combine attrs of the objects being
        merged:
        - "drop": empty attrs on returned Dataset.
        - "identical": all attrs must be the same on every object.
        - "no_conflicts": attrs from all objects are combined, any that have
          the same name must also have the same value.
        - "drop_conflicts": attrs from all objects are combined, any that have
          the same name but different values are dropped.
        - "override": skip comparing and copy attrs from the first dataset to
          the result.
        If a callable, it must expect a sequence of ``attrs`` dicts and a context object
        as its only parameters.

    caseid : str, optional, keyword only
        This named dimension will be marked as the '_caseid_' dimension.

    alts : str or Mapping or array-like, keyword only
        If given as a str, this named dimension will be marked as the
        '_altid_' dimension.  Otherwise, give a Mapping that defines
        alternative names and (integer) codes or an array of codes.

    Returns
    -------
    Dataset
        Dataset with combined variables from each object.
    """
    return Dataset(
        xr.merge(
            objects,
            compat="no_conflicts",
            join="outer",
            fill_value=dtypes.NA,
            combine_attrs="override",
        ),
        caseid=caseid,
        alts=alts,
    )
