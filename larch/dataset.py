import logging
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import pathlib
from collections.abc import Mapping
from xarray.core import dtypes

try:
    from sharrow import Dataset as _sharrow_Dataset
    from sharrow import DataArray as _sharrow_DataArray
    from sharrow import DataTree as _sharrow_DataTree
except ImportError:
    warnings.warn("larch.dataset requires the sharrow library")
    class _noclass:
        pass
    _sharrow_Dataset = xr.Dataset
    _sharrow_DataArray = xr.DataArray
    _sharrow_DataTree = _noclass

from .dim_names import CASEID as _CASEID, ALTID as _ALTID


class DataArray(_sharrow_DataArray):

    __slots__ = ()

    @classmethod
    def zeros(cls, *coords, dtype=np.float64, name=None, attrs=None):
        """
        Construct a dataset filled with zeros.

        Parameters
        ----------
        coords : Tuple[array-like]
            A sequence of coordinate vectors.  Ideally each should have a
            `name` attribute that names a dimension, otherwise placeholder
            names are used.
        dtype : dtype, default np.float64
            dtype of the new array. If omitted, it defaults to np.float64.
        name : str or None, optional
            Name of this array.
        attrs : dict_like or None, optional
            Attributes to assign to the new instance. By default, an empty
            attribute dictionary is initialized.

        Returns
        -------
        DataArray
        """
        dims = []
        shape = []
        coo = {}
        for n, c in enumerate(coords):
            i = getattr(c, "name", f"dim_{n}")
            dims.append(i)
            shape.append(len(c))
            coo[i] = c
        return cls(
            data=np.zeros(shape, dtype=dtype),
            coords=coo,
            dims=dims,
            name=name,
            attrs=attrs,
        )

    def to_zarr(self, store=None, *args, **kwargs):
        """
        Write DataArray contents to a zarr store.

        All parameters are passed directly to :py:meth:`xarray.Dataset.to_zarr`.

        Notes
        -----
        Only xarray.Dataset objects can be written to netCDF files, so
        the xarray.DataArray is converted to a xarray.Dataset object
        containing a single variable. If the DataArray has no name, or if the
        name is the same as a coordinate name, then it is given the name
        ``"__xarray_dataarray_variable__"``.

        See Also
        --------
        Dataset.to_zarr
        """
        from xarray.backends.api import DATAARRAY_VARIABLE

        if self.name is None:
            # If no name is set then use a generic xarray name
            dataset = self.to_dataset(name=DATAARRAY_VARIABLE)
        else:
            # No problems with the name - so we're fine!
            dataset = self.to_dataset()

        if isinstance(store, (str, pathlib.Path)) and len(args)==0:
            if str(store).endswith('.zip'):
                import zarr
                with zarr.ZipStore(store, mode='w') as zstore:
                    dataset.to_zarr(zstore, **kwargs)
                return

        return dataset.to_zarr(*args, **kwargs)

    @classmethod
    def from_zarr(cls, *args, name=None, **kwargs):
        dataset = xr.open_zarr(*args, **kwargs)
        if name is None:
            names = set(dataset.variables) - set(dataset.coords)
            if len(names) == 1:
                name = names.pop()
            else:
                raise ValueError("cannot infer name to load")
        return cls(dataset[name])


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
            wt = self['wt'].values.astype(float_dtype)
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

    def query_cases(self, query, parser="pandas", engine=None):
        """
        Return a new dataset with each array indexed along the CASEID dimension.

        The indexers are given as strings containing Python expressions to be
        evaluated against the data variables in the dataset.

        Parameters
        ----------
        query : str
            Python expressions to be evaluated against the data variables
            in the dataset. The expressions will be evaluated using the pandas
            eval() function, and can contain any valid Python expressions but cannot
            contain any Python statements.
        parser : {"pandas", "python"}, default: "pandas"
            The parser to use to construct the syntax tree from the expression.
            The default of 'pandas' parses code slightly different than standard
            Python. Alternatively, you can parse an expression using the 'python'
            parser to retain strict Python semantics.
        engine : {"python", "numexpr", None}, default: None
            The engine used to evaluate the expression. Supported engines are:

            - None: tries to use numexpr, falls back to python
            - "numexpr": evaluates expressions using numexpr
            - "python": performs operations as if you had eval’d in top level python

        Returns
        -------
        obj : Dataset
            A new Dataset with the same contents as this dataset, except each
            array is indexed by the results of the query on the CASEID dimension.

        See Also
        --------
        Dataset.isel
        pandas.eval
        """
        return self.query({self.CASEID: query}, parser=parser, engine=engine)

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

    def dissolve_zero_variance(self, dim='<ALTID>', inplace=False):
        if dim == '<ALTID>':
            dim = self.ALTID
        if inplace:
            obj = self
        else:
            obj = self.copy()
        for k in obj.variables:
            if obj[k].dtype.kind in {'U', 'S'}:
                continue
            if dim in obj[k].dims:
                if obj[k].std(dim=dim).max() < 1e-10:
                    obj[k] = obj[k].min(dim=dim)
        return obj

    def set_dtypes(self, dtypes, inplace=False, on_error='warn'):
        """
        Set the dtypes for the variables in this Dataset.
        Parameters
        ----------
        dtypes : Mapping or DataFrame
            Mapping of names to dtypes, or a DataFrame to infer such a
            mapping.
        inplace : bool, default False
            Whether to convert dtypes inplace.
        on_error : {'warn', 'raise', 'ignore'}
            What to do when a type conversion triggers an error.

        Returns
        -------
        Dataset
        """
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

    def set_altnames(self, altnames, inplace=False):
        """
        Set the alternative names for this Dataset.

        Parameters
        ----------
        altnames : Mapping or array-like
            A mapping of (integer) codes to names, or an array or names
            of the same length and order as the alternatives already
            defined in this Dataset.

        Returns
        -------
        Dataset
        """
        if inplace:
            obj = self
        else:
            obj = self.copy()
        if isinstance(altnames, Mapping):
            names = xr.DataArray(
                [altnames.get(i, None) for i in obj[obj.ALTID].values],
                dims=obj.ALTID,
            )
        elif isinstance(altnames, DataArray):
            names = altnames
        else:
            names = xr.DataArray(
                np.asarray(altnames),
                dims=obj.ALTID,
            )
        obj.coords['altnames'] = names
        return obj

    @classmethod
    def from_idca(cls, df, crack=True, altnames=None, avail='_avail_', fill_unavail=None):
        """
        Construct a Dataset from an idco-format DataFrame.

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
        fill_unavail : scalar or Mapping, optional
            Fill values to use for missing values when imported data is
            in idce format (i.e. sparse).  Give a single value to use
            globally, or a mapping of {variable: value} or {dtype: value}.

        Returns
        -------
        Dataset
        """
        if df.index.nlevels != 2:
            raise ValueError("source idca dataframe must have a two "
                             "level MultiIndex giving case and alt id's")
        caseidname, altidname = df.index.names
        ds = cls.construct(df, caseid=caseidname, alts=altidname)
        if crack:
            ds = ds.dissolve_zero_variance()
        ds = ds.set_dtypes(df)
        if altnames is not None:
            ds = ds.set_altnames(altnames)
        if avail not in ds and len(df) < ds.n_cases * ds.n_alts:
            av = DataArray.from_series(pd.Series(1, index=df.index)).fillna(0).astype(np.int8)
            ds[avail] = av
            if fill_unavail is not None:
                if isinstance(fill_unavail, Mapping):
                    for k, i in ds.items():
                        if ds.ALTID not in i.dims:
                            continue
                        if k not in fill_unavail and i.dtype not in fill_unavail:
                            continue
                        filler = fill_unavail.get(k, fill_unavail[i.dtype])
                        ds[k] = i.where(ds['_avail_']!=0, filler)
                else:
                    for k, i in ds.items():
                        if ds.ALTID not in i.dims:
                            continue
                        ds[k] = i.where(ds['_avail_']!=0, fill_unavail)
        return ds

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
        caseidname = df.index.name
        ds = cls.construct(df, caseid=caseidname, alts=alts)
        ds = ds.set_dtypes(df)
        return ds

    def setup_flow(self, *args, **kwargs):
        return DataTree(main=self).setup_flow(*args, **kwargs)

    def caseids(self):
        """
        Access the caseids coordinates as an index.

        Returns
        -------
        pd.Index
        """
        return self.indexes[self.CASEID]


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

    def caseids(self):
        """
        Access the caseids coordinates as an index.

        Returns
        -------
        pd.Index
        """
        return self.root_dataset.indexes[self.CASEID]


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


def choice_avail_summary(dataset, graph=None, availability_co_vars=None):
    """
    Generate a summary of choice and availability statistics.

    Parameters
    ----------
    dataset : Dataset
        The loaded dataset to summarize, which should have
        `ch` and `av` variables.
    graph : NestingTree, optional
        The nesting graph.
    availability_co_vars : dict, optional
        Also attach the definition of the availability conditions.

    Returns
    -------
    pandas.DataFrame
    """
    if graph is None:
        if 'ch' in dataset:
            ch_ = dataset['ch'].copy()
        else:
            ch_ = None
        av_ = dataset.get('av')
    else:
        from .numba.cascading import array_av_cascade, array_ch_cascade

        ch_ = array_ch_cascade(dataset.get('ch'), graph)
        av_ = array_av_cascade(dataset.get('av'), graph)

    if ch_ is not None:
        ch = ch_.sum(0)
    else:
        ch = None

    if av_ is not None:
        av = av_.sum(0)
    else:
        av = None

    arr_wt = dataset.get('wt')
    if arr_wt is not None:
        if ch_ is not None:
            ch_w = pd.Series((ch_ * arr_wt.values).sum(0))
        else:
            ch_w = None
        if av_ is not None:
            av_w = pd.Series((av_ * arr_wt.values).sum(0))
        else:
            av_w = None
        show_wt = np.any(ch != ch_w)
    else:
        ch_w = ch
        av_w = av
        show_wt = False

    if av_ is not None:
        ch_[av_ > 0] = 0
    if ch_ is not None and ch_.sum() > 0:
        ch_but_not_av = ch_.sum(0)
        if arr_wt is not None:
            ch_but_not_av_w = pd.Series((ch_ * arr_wt.values).sum(0), index=ch_.columns)
        else:
            ch_but_not_av_w = ch_but_not_av
    else:
        ch_but_not_av = None
        ch_but_not_av_w = None

    from collections import OrderedDict
    od = OrderedDict()

    if graph is not None:
        od['name'] = pd.Series(graph.standard_sort_names, index=graph.standard_sort)
    else:
        od['name'] = dataset.get('alt_names')

    if show_wt:
        od['chosen weighted'] = ch_w
        od['chosen unweighted'] = ch
        od['available weighted'] = av_w
        od['available unweighted'] = av
    else:
        od['chosen'] = ch
        od['available'] = av
    if ch_but_not_av is not None:
        if show_wt:
            od['chosen but not available weighted'] = ch_but_not_av_w
            od['chosen but not available unweighted'] = ch_but_not_av
        else:
            od['chosen but not available'] = ch_but_not_av

    if availability_co_vars is not None:
        od['availability condition'] = pd.Series(
            availability_co_vars.values(),
            index=availability_co_vars.keys(),
        )

    result = pd.DataFrame.from_dict(od)
    if graph is not None:
        totals = result.loc[graph.root_id, :]
        result.drop(index=graph.root_id, inplace=True)
    else:
        totals = result.sum()

    for tot in (
            'chosen',
            'chosen weighted',
            'chosen unweighted',
            'chosen but not available',
            'chosen but not available weighted',
            'chosen but not available unweighted',
            'chosen thus available',
            'not available so not chosen'
    ):
        if tot in totals:
            result.loc['< Total All Alternatives >', tot] = totals[tot]

    result.loc['< Total All Alternatives >', pd.isnull(result.loc['< Total All Alternatives >', :])] = ""
    result.drop('_root_', errors='ignore', inplace=True)

    if 'availability condition' in result:
        result['availability condition'] = result['availability condition'].fillna('')

    for i in (
            'chosen',
            'chosen but not available',
            'chosen thus available',
            'not available so not chosen',
    ):
        if i in result.columns and all(result[i] == result[i].astype(int)):
            result[i] = result[i].astype(int)

    for i in (
            'available',
    ):
        if i in result.columns:
            j = result.columns.get_loc(i)
            if all(result.iloc[:-1,j] == result.iloc[:-1,j].astype(int)):
                result.iloc[:-1,j] = result.iloc[:-1,j].astype(int)

    return result
