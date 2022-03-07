import numpy as np
import pandas as pd
import xarray as xr
import sharrow.dataset as sd
from sharrow.dataset import construct
from .patch import register_dataset_classmethod

from typing import Mapping

from .dim_names import CASEID, ALTID, CASEALT, ALTIDX, CASEPTR, GROUPID, INGROUP

def _steal(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper.__doc__ = func.__doc__
    wrapper.__name__ = func.__name__
    return staticmethod(wrapper)

def _initialize_for_larch(obj, caseid=None, alts=None):
    """

    Parameters
    ----------
    obj : Dataset
        The dataaset being initialized.
    caseid : str, optional
        The name of a dimension referencing cases.
    alts : Mapping or str or array-like, optional
        If given as a mapping, links alternative codes to names.
        A string names a dimension that defines the alternatives.
        An array or list of integers gives codes for the alternatives,
        which are otherwise unnamed.

    Returns
    -------
    Dataset
    """
    if caseid is not None:
        if caseid not in obj.dims:
            raise ValueError(f"no dim named '{caseid}' to make into {CASEID}")
        obj.attrs[CASEID] = caseid
    if isinstance(alts, pd.Series):
        alts_dim_name = alts.name or alts.index.name or '_altid_'
        alts_k = xr.DataArray(
            alts.index, dims=alts_dim_name,
        )
        alts_v = xr.DataArray(
            alts.values, dims=alts_dim_name,
        )
    elif isinstance(alts, Mapping):
        alts_dim_name = '_altid_'
        alts_k = xr.DataArray(
            list(alts.keys()), dims=alts_dim_name,
        )
        alts_v = xr.DataArray(
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
        obj.attrs[ALTID] = alts_dim_name
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


@xr.register_dataset_accessor("construct")
class _DatasetConstruct:

    _parent_class = xr.Dataset

    def __new__(cls, *args, **kwargs):
        if kwargs:
            source = construct(*args)
            return _initialize_for_larch(source, **kwargs)
        else:
            return super().__new__(cls, *args, **kwargs)

    def __init__(self, x=None, **kwargs):
        self._obj = x

    @classmethod
    def __call__(cls, source, caseid=None, alts=None):
        """
        A generic constructor for creating Datasets from various similar objects.

        Parameters
        ----------
        source : pandas.DataFrame, pyarrow.Table, xarray.Dataset, or Sequence[str]
            The source from which to create a Dataset.  DataFrames and Tables
            are converted to Datasets that have one dimension (the rows) and
            seperate variables for each of the columns.  A list of strings
            creates a dataset with those named empty variables.
        caseid : str, optional
            The name of a dimension referencing cases.
        alts : Mapping or str or array-like, optional
            If given as a mapping, links alternative codes to names.
            A string names a dimension that defines the alternatives.
            An array or list of integers gives codes for the alternatives,
            which are otherwise unnamed.

        Returns
        -------
        Dataset
        """
        source = construct(source)
        return _initialize_for_larch(source, caseid, alts)

    from_omx = _steal(sd.from_omx)
    from_omx_3d = _steal(sd.from_omx_3d)

    @classmethod
    def from_idco(cls, df, alts=None):
        """
        Construct a Dataset from an idco-format DataFrame.

        Parameters
        ----------
        df : DataFrame
            The input data should be an idco-format DataFrame, with
            the caseid's in a single-level index,
        alts : Mapping or array-like, optional
            If given as a mapping, links alternative codes to names.
            An array or list of integers gives codes for the alternatives,
            which are otherwise unnamed.

        Returns
        -------
        Dataset
        """
        if df.index.nlevels != 1:
            raise ValueError("source idco dataframe must have a one "
                             "level Index giving case id's")
        caseidname = df.index.name or 'index'
        ds = cls()(df, caseid=caseidname, alts=alts)
        ds = ds.set_dtypes(df)
        return ds

    @classmethod
    def from_idca(cls, df, crack=True, altnames=None, avail='_avail_', fill_missing=None):
        """
        Construct a Dataset from an idca-format DataFrame.

        This method loads the data as dense arrays.

        Parameters
        ----------
        df : DataFrame
            The input data should be an idca-format or idce-format DataFrame,
            with the caseid's and altid's in a two-level pandas MultiIndex.
        crack : bool, default True
            If True, the `dissolve_zero_variance` method is applied before
            repairing dtypes, to ensure that missing value are handled
            properly.
        altnames : Mapping, optional
            If given as a mapping, links alternative codes to names.
            An array or list of strings gives names for the alternatives,
            sorted in the same order as the codes.
        avail : str, default '_avail_'
            When the imported data is in idce format (i.e. sparse) then
            an availability indicator is computed and given this name.
        fill_missing : scalar or Mapping, optional
            Fill values to use for missing values when imported data is
            in idce format (i.e. sparse).  Give a single value to use
            globally, or a mapping of {variable: value} or {dtype: value}.

        Returns
        -------
        Dataset

        See Also
        --------
        Dataset.from_idce : Construct a Dataset from a sparse idca-format DataFrame.

        """
        if df.index.nlevels != 2:
            raise ValueError("source idca dataframe must have a two "
                             "level MultiIndex giving case and alt id's")
        caseidname, altidname = df.index.names

        # check altids are integers, if they are not then fix it
        if df.index.levels[1].dtype.kind != 'i':
            if altnames is None:
                altnames = df.index.levels[1]
                df.index = df.index.set_levels(np.arange(1, len(altnames) + 1), level=1)
            else:
                new_index = df.index.get_level_values(1).astype(pd.CategoricalDtype(altnames))
                df.index = df.index.set_codes(new_index.codes, level=1).set_levels(np.arange(1, len(altnames) + 1), level=1)

        ds = cls()(df, caseid=caseidname, alts=altidname)
        if crack:
            ds = ds.dc.dissolve_zero_variance()
        ds = ds.set_dtypes(df)
        if altnames is not None:
            ds = ds.dc.set_altnames(altnames)
        if avail not in ds and len(df) < ds.dc.n_cases * ds.dc.n_alts:
            av = xr.DataArray.from_series(pd.Series(1, index=df.index)).fillna(0).astype(np.int8)
            ds[avail] = av
            if fill_missing is not None:
                if isinstance(fill_missing, Mapping):
                    for k, i in ds.items():
                        if ds.dc.ALTID not in i.dims:
                            continue
                        if k not in fill_missing and i.dtype not in fill_missing:
                            continue
                        filler = fill_missing.get(k, fill_missing[i.dtype])
                        ds[k] = i.where(ds['_avail_']!=0, filler)
                else:
                    for k, i in ds.items():
                        if ds.dc.ALTID not in i.dims:
                            continue
                        ds[k] = i.where(ds['_avail_']!=0, fill_missing)
        return ds

    @classmethod
    def from_idce(cls, df, crack=True, altnames=None, dim_name=None, alt_index='alt_idx', case_index=None, case_pointer=None):
        """
        Construct a Dataset from a sparse idca-format DataFrame.

        Parameters
        ----------
        df : DataFrame
            The input data should be an idca-format or idce-format DataFrame,
            with the caseid's and altid's in a two-level pandas MultiIndex.
        crack : bool, default False
            If True, the `dissolve_zero_variance` method is applied before
            repairing dtypes, to ensure that missing value are handled
            properly.
        altnames : Mapping, optional
            If given as a mapping, links alternative codes to names.
            An array or list of strings gives names for the alternatives,
            sorted in the same order as the codes.
        dim_name : str, optional
            Name to apply to the sparse index dimension.
        alt_index : str, default 'alt_idx'
            Add the alt index (position) for each sparse data row as a
            coords array with this name.
        case_index : str, optional
            Add the case index (position) for each sparse data row as a
            coords array with this name. If not given, this array is not
            stored but it can still be reconstructed later from the case
            pointers.
        case_pointer : str, optional
            Use this name for the case_ptr dimension, overriding the
            default.

        Returns
        -------
        Dataset

        See Also
        --------
        Dataset.from_idca : Construct a dense Dataset from a idca-format DataFrame.
        """
        if df.index.nlevels != 2:
            raise ValueError("source idce dataframe must have a two "
                             "level MultiIndex giving case and alt id's")
        caseidname, altidname = df.index.names
        caseidname = caseidname or CASEID
        altidname = altidname or ALTID
        dim_name = dim_name or CASEALT
        case_pointer = case_pointer or CASEPTR
        ds = cls()(df.reset_index(drop=True).rename_axis(index=dim_name))
        ds.coords[caseidname] = xr.DataArray(df.index.levels[0], dims=caseidname)
        ds.coords[altidname] = xr.DataArray(df.index.levels[1], dims=altidname)
        if case_index is not None:
            ds.coords[case_index] = xr.DataArray(df.index.codes[0], dims=dim_name)
        if alt_index is None:
            raise ValueError('alt_index cannot be None')
        ds.coords[alt_index] = xr.DataArray(df.index.codes[1], dims=dim_name)
        ds.coords[case_pointer] = xr.DataArray(
            np.where(np.diff(df.index.codes[0], prepend=np.nan, append=np.nan))[0],
            dims=case_pointer,
        )
        ds.attrs['_exclude_dims_'] = (caseidname, altidname, case_pointer)
        ds.attrs[CASEID] = caseidname
        ds.attrs[ALTID] = altidname
        ds.attrs[CASEALT] = dim_name
        ds.attrs[ALTIDX] = alt_index
        ds.attrs[CASEPTR] = case_pointer
        ds = ds.drop_vars(dim_name)
        if crack:
            ds = ds.dc.dissolve_zero_variance()
        ds = ds.set_dtypes(df)
        if altnames is not None:
            ds = ds.dc.set_altnames(altnames)
        return ds
