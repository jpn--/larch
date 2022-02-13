import numpy as np
import pandas as pd
import seaborn as sns
import xmle
from larch.util.bounded_kde import bounded_gaussian_kde
import altair as alt
from scipy.stats import gaussian_kde

from .dataset import Dataset, DataArray


def fair_range(x, buffering=0.1):
    lo, hi = np.percentile(x, [1,99])
    buffer = (hi-lo) * buffering
    return lo - buffer, hi + buffer


def scotts_factor(self, n, d=1):
    """
    Compute Scott's factor.

    Parameters
    ----------
    n : numeric
        Effective number of data points, either the cardinality
        of the array or the total weight.
    d : int, default 1
        Number of dimensions.

    Returns
    -------
    float
    """
    return np.power(n, -1./(d+4))


def silverman_factor(self, n, d=1):
    """
    Compute the Silverman factor.

    Parameters
    ----------
    n : numeric
        Effective number of data points, either the cardinality
        of the array or the total weight.
    d : int, default 1
        Number of dimensions.

    Returns
    -------
    float
    """
    return np.power(n*(d+2.0)/4.0, -1./(d+4))

import xarray as xr

def idca_statistics(var, bw_method='scott', choice=None, avail=None, wgt=None, return_plot_data=False, scale_ch=True):
    name = var.name or "Variable"

    if avail is None:
        avail = DataArray(xr.ones_like(var, dtype=np.int8))
    if wgt is None:
        wgt = np.ones(var.n_cases, dtype=np.float32)
    n_alts = var.n_alts
    n = np.prod(var.shape)
    if bw_method == 'scott':
        bandwidth = scotts_factor(n, 1)
    elif bw_method == 'silverman':
        bandwidth = scotts_factor(n, 1)
    elif np.isscalar(bw_method) and not isinstance(bw_method, str):
        bandwidth = bw_method
    else:
        raise ValueError(
            "`bw_method` should be 'scott', 'silverman', or a scalar"
        )
    stats = Dataset({
        'mean': var.mean(var.CASEID),
        'std': var.std(var.CASEID),
    })
    data = {}
    data_ch = {}
    data_av = {}
    for k in range(n_alts):
        if stats['std'][k] > 0:
            if 'alt_names' in var.coords:
                key = str(var.coords['alt_names'][k].data)
            elif var.ALTID in var.coords:
                key = str(var.coords[var.ALTID][k].data)
            else:
                key = str(k)
            data[key] = var.isel({var.ALTID: k}).values
            if choice is not None:
                data_ch[key] = choice.isel({choice.ALTID: k}).values
            data_av[key] = avail.isel({avail.ALTID: k}).values
    x = np.linspace(*fair_range(var), 100)
    kde = {
        k: bounded_gaussian_kde(v, bw_method=bandwidth, weights=wgt)(x)
        for k, v in data.items()
    }
    if choice is not None:
        kde_ch = {}
        for k, v in data.items():
            y = bounded_gaussian_kde(v, bw_method=bandwidth, weights=data_ch[k]*wgt)(x)
            if scale_ch:
                y = y * (data_ch[k]*wgt).sum() / (data_av[k]*wgt).sum()
            kde_ch[k] = y
        plot_data = pd.concat([pd.DataFrame(kde), pd.DataFrame(kde_ch)], axis=1, keys=['Density', 'Chosen'])
        plot_data = plot_data.set_index(x).stack().rename_axis(index=(name,'Alternative')).reset_index()
    else:
        kde_ch = None
        plot_data = pd.DataFrame(kde, index=x).stack().rename("Density").rename_axis(index=(name,'Alternative')).reset_index()
    if return_plot_data:
        return plot_data
    selection = alt.selection_single(fields=['Alternative'], bind='legend', empty='all')
    selection2 = alt.selection_single(fields=['Alternative'], bind='legend', empty='none')
    ax = alt.Chart(
        plot_data
    )
    lines = ax.mark_line().encode(
        x=name,
        y='Density',
        color='Alternative',
        strokeDash='Alternative',
        tooltip="Alternative",
        opacity=alt.condition(selection, alt.value(1), alt.value(0.0)),
    ).add_selection(
        selection
    )
    if choice is not None:
        areas = ax.mark_area().encode(
            x=name,
            y='Chosen',
            color=alt.Color('Alternative', legend=None),
            strokeDash='Alternative',
            tooltip="Alternative",
            opacity=alt.condition(selection2, alt.value(1 if scale_ch else 0.75), alt.value(0.0)),
        ).add_selection(
            selection2
        )
        lines = lines + areas

    return lines, stats
