import numpy as np
import pandas as pd
from numba import njit, prange

@njit(parallel=True, nogil=True)
def cascade_or(arr, dn_slots, up_slots):
    for i in prange(arr.shape[0]):
        for j in range(dn_slots.size):
            arr[i,up_slots[j]] |= arr[i,dn_slots[j]]

@njit(parallel=True, nogil=True)
def cascade_sum(arr, dn_slots, up_slots):
    for i in prange(arr.shape[0]):
        for j in range(dn_slots.size):
            arr[i,up_slots[j]] += arr[i,dn_slots[j]]

def data_av_cascade(dataframes, graph):
    """
    Create an extra wide dataframe with availability rolled up to nests.

    This has all the same elemental alternatives as the original
    `data_av` dataframe, plus columns for all nests.  For each
    nest, if any component is available, then the nest is also
    indicated as available.

    Parameters
    ----------
    dataframes : DataFrames
    graph : NestingTree

    Returns
    -------
    array
    """
    result = np.zeros((len(dataframes.data_av.index), len(graph)), dtype=np.int8)
    result[: ,:graph.n_elementals()] = dataframes.data_av
    ups, dns, _1, _2 = graph.edge_slot_arrays()
    cascade_or(result, dns, ups)
    return result

def data_ch_cascade(dataframes, graph, dtype=None):
    """
    Create an extra wide dataframe with choices rolled up to nests.

    This has all the same elemental alternatives as the original
    `data_ch` dataframe, plus columns for all nests.  For each
    nest, if any component is chosen, then the nest is also
    indicated as chosen, with a magnitude equal to the sum of its
    parts.

    Parameters
    ----------
    dataframes : DataFrames
    graph : NestingTree

    Returns
    -------
    array
    """
    result = np.zeros(
        (len(dataframes.data_ch.index), len(graph)),
        dtype=dtype or dataframes.data_ch.dtype,
    )
    result[: ,:graph.n_elementals()] = dataframes.data_ch
    ups, dns, _1, _2 = graph.edge_slot_arrays()
    cascade_sum(result, dns, ups)
    return result

def array_av_cascade(arr_av, graph):
    """
    Create an extra wide array with availability rolled up to nests.

    This has all the same elemental alternatives as the original
    `arr_av` array, plus columns for all nests.  For each
    nest, if any component is available, then the nest is also
    indicated as available.

    Parameters
    ----------
    arr_av : ndarray
    graph : NestingTree

    Returns
    -------
    array
    """
    if arr_av is None:
        return None
    result = np.zeros((arr_av.shape[0], len(graph)), dtype=np.int8)
    result[: ,:graph.n_elementals()] = arr_av
    ups, dns, _1, _2 = graph.edge_slot_arrays()
    cascade_or(result, dns, ups)
    return result

def array_ch_cascade(arr_ch, graph, dtype=None):
    """
    Create an extra wide array with choices rolled up to nests.

    This has all the same elemental alternatives as the original
    `arr_ch` array, plus columns for all nests.  For each
    nest, if any component is chosen, then the nest is also
    indicated as chosen, with a magnitude equal to the sum of its
    parts.

    Parameters
    ----------
    arr_ch : ndarray
    graph : NestingTree

    Returns
    -------
    array
    """
    if arr_ch is None:
        return None
    result = np.zeros(
        (arr_ch.shape[0], len(graph)),
        dtype=dtype or arr_ch.dtype,
    )
    result[: ,:graph.n_elementals()] = arr_ch
    ups, dns, _1, _2 = graph.edge_slot_arrays()
    cascade_sum(result, dns, ups)
    return result
