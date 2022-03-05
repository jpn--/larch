import numpy as np
import pandas as pd
import xarray as xr
from typing import Mapping
import warnings

from sharrow.accessors import register_dataset_method, register_dataarray_method


class NameConflictWarning(Warning):
    """Warning for conflicts in name registration."""


def _register_classmethod(name, cls):
    def decorator(classmeth):
        if hasattr(cls, name):
            warnings.warn(
                f"registration of classmethod {classmeth!r} under name {name!r} "
                f"for type {cls!r} is overriding a preexisting attribute with "
                f"the same name.",
                NameConflictWarning,
                stacklevel=2,
            )
        setattr(cls, name, classmethod(classmeth))
        return classmeth

    return decorator


def register_dataarray_classmethod(func):
    """
    Register a custom classmethod on xarray.DataArray objects.

    Use this as a decorator to add class methods.

    Parameters
    ----------
    func : Callable
        Class method to add.  The name is inferred from the
        original name of this function.
    """
    return _register_classmethod(func.__name__, xr.DataArray)(func)


def register_dataset_classmethod(func):
    """
    Register a custom classmethod on xarray.Dataset objects.

    Use this as a decorator to add class methods.

    Parameters
    ----------
    func : Callable
        Class method to add.  The name is inferred from the
        original name of this function.
    """
    return _register_classmethod(func.__name__, xr.Dataset)(func)


@register_dataset_method
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
        if k not in dtypes:
            continue
        try:
            obj[k] = obj[k].astype(dtypes[k])
        except Exception as err:
            if on_error == 'warn':
                warnings.warn(f"{err!r} on converting {k}")
            elif on_error == 'raise':
                raise
    return obj
