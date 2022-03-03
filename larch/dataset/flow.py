import numpy as np
import pandas as pd
import xarray as xr
import numba as nb
from typing import Mapping
import logging
import warnings

from .dim_names import CASEID, ALTID, CASEALT, ALTIDX, CASEPTR, GROUPID, INGROUP


@nb.njit
def case_ptr_to_indexes(n_casealts, case_ptrs):
    case_index = np.zeros(n_casealts, dtype=np.int64)
    for c in range(case_ptrs.shape[0]-1):
        case_index[case_ptrs[c]:case_ptrs[c + 1]] = c
    return case_index


@nb.njit
def ce_dissolve_zero_variance(ce_data, ce_caseptr):
    """

    Parameters
    ----------
    ce_data : array-like, shape [n_casealts] one-dim only
    ce_altidx
    ce_caseptr
    n_alts

    Returns
    -------
    out : ndarray
    flag : int
        1 if variance was detected, 0 if no variance was found and
        the `out` array is valid.
    """
    failed = 0
    if ce_caseptr.ndim == 2:
        ce_caseptr1 = ce_caseptr[:,-1]
    else:
        ce_caseptr1 = ce_caseptr[1:]
    shape = (ce_caseptr1.shape[0], )
    out = np.zeros(shape, dtype=ce_data.dtype)
    c = 0
    out[0] = ce_data[0]
    for row in range(ce_data.shape[0]):
        if row == ce_caseptr1[c]:
            c += 1
            out[c] = ce_data[row]
        else:
            if out[c] != ce_data[row]:
                failed = 1
                break
    return out, failed


class _GenericFlow:

    def __init__(self, x=None):
        self._obj = x
        self._flow_library = {}

    @property
    def CASEID(self):
        """str : The _caseid_ dimension of this Dataset, if defined."""
        result = self._obj.attrs.get(CASEID, None)
        return result

    def __set_attr(self, attr_name, target_value, check_dim=True):
        if target_value is None:
            if attr_name in self._obj.attrs:
                del self._obj.attrs[attr_name]
        else:
            if check_dim and target_value not in self._obj.dims:
                raise ValueError(f"cannot set {attr_name}, {target_value} not in dims")
            self._obj.attrs[attr_name] = target_value

    @CASEID.setter
    def CASEID(self, dim_name):
        self.__set_attr(CASEID, dim_name, check_dim=self.GROUPID is not None)

    @property
    def ALTID(self):
        """str : The _altid_ dimension of this Dataset, if defined."""
        result = self._obj.attrs.get(ALTID, None)
        return result

    @ALTID.setter
    def ALTID(self, dim_name):
        self.__set_attr(ALTID, dim_name)

    @property
    def CASEALT(self):
        """str : The _casealt_ dimension of this Dataset, if defined."""
        result = self._obj.attrs.get(CASEALT, None)
        return result

    @CASEALT.setter
    def CASEALT(self, dim_name):
        self.__set_attr(CASEALT, dim_name)

    @property
    def ALTIDX(self):
        """str : The _alt_idx_ dimension of this Dataset, if defined."""
        result = self._obj.attrs.get(ALTIDX, None)
        return result

    @ALTIDX.setter
    def ALTIDX(self, dim_name):
        self.__set_attr(ALTIDX, dim_name, False)

    @property
    def CASEPTR(self):
        """str : The _caseptr_ dimension of this Dataset, if defined."""
        result = self._obj.attrs.get(CASEPTR, None)
        return result

    @CASEPTR.setter
    def CASEPTR(self, dim_name):
        self.__set_attr(CASEPTR, dim_name, False)

    @property
    def GROUPID(self):
        """str : The _groupid_ dimension of this Dataset, if defined."""
        result = self._obj.attrs.get(GROUPID, None)
        return result

    @GROUPID.setter
    def GROUPID(self, dim_name):
        self.__set_attr(GROUPID, dim_name)

    @property
    def INGROUP(self):
        """str : The _ingroup_ dimension of this Dataset, if defined."""
        result = self._obj.attrs.get(INGROUP, None)
        return result

    @INGROUP.setter
    def INGROUP(self, dim_name):
        self.__set_attr(INGROUP, dim_name)

    def set_altids(self, altids, dim_name=None, inplace=False):
        """
        Set the alternative ids for this Dataset.

        Parameters
        ----------
        altids : array-like of int
            Integer id codes.
        dim_name : str, optional
            Use this dimension name for alternatives.
        inplace : bool, default False
            When true, apply the transformation in-place on the Dataset,
            otherwise return a modified copy.

        Returns
        -------
        Dataset
        """
        if inplace:
            obj = self
        else:
            obj = self.copy()
        dim_name = dim_name or getattr(altids, 'name', None) or obj.attrs.get(ALTID, ALTID)
        if not isinstance(altids, xr.DataArray):
            altids = xr.DataArray(
                np.asarray(altids),
                dims=(dim_name),
            )
        obj.coords[dim_name] = altids
        obj.ALTID = dim_name
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
        inplace : bool, default False
            When true, apply the transformation in-place on the Dataset,
            otherwise return a modified copy.

        Returns
        -------
        Dataset
        """
        if inplace:
            obj = self._obj
        else:
            obj = self._obj.copy()
        if isinstance(altnames, Mapping):
            a = obj.flow.ALTID
            names = xr.DataArray(
                [altnames.get(i, None) for i in obj[a].values],
                dims=a,
            )
        elif isinstance(altnames, xr.DataArray):
            names = altnames
        else:
            names = xr.DataArray(
                np.asarray(altnames),
                dims=obj.flow.ALTID,
            )
        obj.coords['altnames'] = names
        return obj

    def as_tree(self, label='main', exclude_dims=()):
        """
        Convert this Dataset to a DataTree.

        For |idco| and |idca| datasets, the result will generally be a
        single-node tree.  For |idce| data, there will be

        Parameters
        ----------
        label : str, default 'main'
            Name to use for the root node in the tree.
        exclude_dims : Tuple[str], optional
            Exclude these dimensions, in addition to any
            dimensions listed in the `_exclude_dims_` attribute.

        Returns
        -------
        DataTree
        """
        from ..dataset import DataTree
        if self.CASEPTR is not None:
            case_index = case_ptr_to_indexes(self._obj.dims[self.CASEALT], self[self.CASEPTR].values)
            obj = self._obj.assign({'_case_index_': xr.DataArray(case_index, dims=(self.CASEALT))})
            tree = DataTree(**{label: obj.drop_dims(self.CASEID)})
            ds = obj.keep_dims(self.CASEID)
            ds.attrs.pop('_exclude_dims_', None)
            ds.attrs.pop('_caseptr_', None)
            ds.attrs.pop('_casealt_', None)
            ds.attrs.pop('_alt_idx_', None)
            tree.add_dataset(
                'idcoVars',
                ds,
                relationships=(
                    f"{label}._case_index_ -> idcoVars.{self.CASEID}"
                )
            )
        else:
            tree = DataTree(**{label: self._obj})
        return tree

    def setup_flow(self, *args, **kwargs):
        """
        Set up a new Flow for analysis using the structure of this DataTree.

        This method creates a new `DataTree` with only this Dataset as
        the root Dataset labeled `main`.  All other arguments are passed
        through to `DataTree.setup_flow`.

        Returns
        -------
        Flow
        """
        return self.as_tree().setup_flow(*args, **kwargs)

    def caseids(self):
        """
        Access the caseids coordinates as an index.

        Returns
        -------
        pd.Index
        """
        return self.indexes[self.CASEID]

    def altids(self):
        """
        Access the altids coordinates as an index.

        Returns
        -------
        pd.Index
        """
        return self.indexes[self.ALTID]

    def groupids(self):
        """
        Access the groupids coordinates as an index.

        Returns
        -------
        pd.Index
        """
        return self.indexes[self.GROUPID]

    @property
    def alts_mapping(self):
        """Dict[int,str] : Mapping of alternative codes to names"""
        a = self._obj.coords[self.ALTID]
        if 'alt_names' in a.coords:
            return dict(zip(a.values, a.coords['alt_names'].values))
        else:
            return dict(zip(a.values, a.values))

    @property
    def n_cases(self):
        try:
            return self.dims[self.CASEID]
        except KeyError:
            try:
                return self.dims[self.GROUPID] * self.dims[self.INGROUP]
            except KeyError:
                pass
            logging.getLogger().error(f"missing {self.CASEID!r} among dims {self.dims}")
            raise

    def transfer_dimension_attrs(self, target):
        if not isinstance(target, (xr.DataArray, xr.Dataset)):
            return target
        updates = {}
        for i in [CASEID, ALTID, GROUPID, INGROUP]:
            j = self._obj.attrs.get(i, None)
            if j is not None:
                updates[i] = j
        return target.assign_attrs(updates)

    def get_expr(self, expression):
        """
        Access or evaluate an expression.

        Parameters
        ----------
        expression : str

        Returns
        -------
        DataArray
        """
        try:
            result = self._obj[expression]
        except (KeyError, IndexError):
            if expression in self._flow_library:
                flow = self._flow_library[expression]
            else:
                flow = self.setup_flow({expression: expression})
                self._flow_library[expression] = flow
            if not flow.tree.root_dataset is self:
                flow.tree = self.as_tree()
            result = flow.load_dataarray().isel(expressions=0)
        result = self.transfer_dimension_attrs(result)
        return result


    def dissolve_zero_variance(self, dim='<ALTID>', inplace=False):
        """
        Dissolve dimension on variables where it has no variance.

        This method is convenient to convert variables that have
        been loaded as |idca| or |idce| format into |idco| format where
        appropriate.

        Parameters
        ----------
        dim : str, optional
            The name of the dimension to potentially dissolve.
        inplace : bool, default False
            Whether to dissolve variables in-place.

        Returns
        -------
        Dataset
        """
        if dim == '<ALTID>':
            dim = self.ALTID
        if inplace:
            obj = self
        else:
            obj = self.copy()
        for k in obj.variables:
            if obj[k].dtype.kind in {'U', 'S', 'O'}:
                continue
            if dim in obj[k].dims:
                try:
                    dissolve = (obj[k].std(dim=dim).max() < 1e-10)
                except TypeError:
                    pass
                else:
                    if dissolve:
                        obj[k] = obj[k].min(dim=dim)
            elif obj[k].dims == (self.CASEALT,):
                proposal, flag = ce_dissolve_zero_variance(obj[k].values, obj[self.CASEPTR].values)
                if flag == 0:
                    obj = obj.assign({k: xr.DataArray(proposal, dims=(self.CASEID))})
        return obj

    def dissolve_coords(self, dim, others=None):
        d = self._obj.reset_index(dim)
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


@xr.register_dataarray_accessor("flow")
class _DaFlow(_GenericFlow):

    _parent_class = xr.DataArray

    @property
    def n_alts(self):
        if self.ALTID in self._obj.dims:
            return self._obj.shape[self.dims.index(self.ALTID)]
        if 'n_alts' in self._obj.attrs:
            return self._obj.attrs['n_alts']
        raise ValueError('no n_alts set')


@xr.register_dataset_accessor("flow")
class _DatasetFlow(_GenericFlow):

    _parent_class = xr.Dataset

    @property
    def n_alts(self):
        if self.ALTID in self._obj.dims:
            return self._obj.dims[self.ALTID]
        if 'n_alts' in self._obj.attrs:
            return self._obj.attrs['n_alts']
        raise ValueError('no n_alts set')

    def __getitem__(self, name):
        # pass dimension attrs to DataArray
        result = self._obj[name]
        result = self.transfer_dimension_attrs(result)
        return result

    def __getattr__(self, name):
        # pass dimension attrs to DataArray
        result = getattr(self._obj, name)
        result = self.transfer_dimension_attrs(result)
        return result

    def __contains__(self, item):
        return self._obj.__contains__(item)

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
        result = self._obj.query({self.CASEID: query}, parser=parser, engine=engine)
        result = self.transfer_dimension_attrs(result)
        return result

    def to_arrays(self, graph, float_dtype=np.float64):
        from ..numba.data_arrays import DataArrays
        from ..numba.cascading import array_av_cascade, array_ch_cascade

        if 'co' in self:
            co = self['co'].values.astype(float_dtype)
        else:
            co = np.empty( (self.n_cases, 0), dtype=float_dtype)

        if 'ca' in self:
            ca = self['ca'].values.astype(float_dtype)
        else:
            ca = np.empty( (self.n_cases, self.n_alts, 0), dtype=float_dtype)

        if 'ce_data' in self:
            ce_data = self['ce_data'].values.astype(float_dtype)
        else:
            ce_data = np.empty( (0, 0), dtype=float_dtype)

        if self.ALTIDX is not None:
            ce_altidx = self[self.ALTIDX].values
        else:
            ce_altidx = np.empty( (0), dtype=np.int16)

        if self.CASEPTR is not None:
            ce_caseptr = np.lib.stride_tricks.sliding_window_view(
                self[self.CASEPTR].values, 2
            )
        else:
            ce_caseptr = np.empty( (self.n_cases, 0), dtype=np.int16)

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
            ch, av, wt, co, ca, ce_data, ce_altidx, ce_caseptr
        )
