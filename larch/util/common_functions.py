
from ..roles import P,X, LinearFunction
import numpy
import pandas
import re, ast
from numpy import log, exp, log1p, absolute, fabs, sqrt, isnan, isfinite, logaddexp, \
    fmin, fmax, nan_to_num, sin, cos, pi


# duplicate these here for legacy compatibility
from .data_expansion import piece, piecewise_linear, parse_piece



def polynomial( x, p, powers=None, funs=None ):

    if powers is None:
        powers = {
            1: "",
            2: "_Squared",
            3: "_Cubed",
        }
    z = LinearFunction()
    for pwr, label in powers.items():
        if pwr==1:
            z = z + X(x) * P(f"{p}{label}")
        else:
            z = z + X(f"({x})**{pwr}") * P(f"{p}{label}")

    if funs is not None:
        for fun, label in funs.items():
            z = z + X(f"{fun}({x})") * P(f"{p}{label}")

    return z


def fourier_series( x, p=None, length=4):
    z = LinearFunction()
    if p is None:
        p = x
    for i in range(length):
        func = 'cos' if i % 2 else 'sin'
        mult = ((i // 2) + 1) * 2
        z = z + X(f'{func}({x}*{mult}*pi)') * P(f'{func}_{mult}π{p}')
    return z


def fourier_expansion_names(basename, length=4):
    """
    Get the names of items for a fourier series expansion.

    Parameters
    ----------
    basename : str
        The input data name
    length : int
        Length of expansion series

    Returns
    -------
    list of str
    """
    columns = []
    for i in range(length):
        func = 'cos' if i % 2 else 'sin'
        mult = ((i // 2) + 1) * 2
        columns.append(f'{func}({basename}*{mult}*pi)')
    return columns


def fourier_expansion(s, length=4, column=None, inplace=False):
    """
    Expand a pandas Series into a DataFrame containing a fourier series.

    Parameters
    ----------
    s : pandas.Series or pandas.DataFrame
        The input data
    length : int
        Length of expansion series
    column : str, optional
        If `s` is given as a DataFrame, use this column

    Returns
    -------
    pandas.DataFrame
    """
    if isinstance(s, pandas.DataFrame):
        input = s
        if len(s.columns) == 1:
            s = s.iloc[:, 0]
        elif column is not None:
            s = s.loc[:, column]
    else:
        input = None
    columns = fourier_expansion_names(s.name, length=length)
    df = pandas.DataFrame(
        data=0,
        index=s.index,
        columns=columns,
    )
    for i, col in enumerate(columns):
        func = numpy.cos if i % 2 else numpy.sin
        mult = ((i // 2) + 1) * 2
        df.iloc[:, i] = func(s * mult * numpy.pi)
    if inplace and input is not None:
        input[columns] = df
    else:
        return df


def normalize(x, std_div=1):
    """
    Normalize an array by subtracting the mean and dividing by the standard deviation.

    Parameters
    ----------
    x : ndarray
    std_div : numeric, optional
        Divide by this many standard deviations.  Defaults to 1, but you may consider using 2
        per [1]_.

    Returns
    -------
    ndarray

    References
    ----------
    .. [1] A. Gelman, Scaling regression inputs by dividing by two standard deviations,
       Statistics in medicine 27 (15) (2008) 2865–2873.
    """
    m = numpy.nanmean(x)
    s = numpy.nanstd(x)
    s *= std_div
    return (x-m)/s

