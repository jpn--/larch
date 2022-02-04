import logging

import numpy as np
import pandas as pd
from numba import guvectorize, njit
from numba import int8 as i8
from numba import int32 as i32
from numba import float32 as f32
from numba import float64 as f64
from numba import boolean
from ..model import Model as _BaseModel
from ..exceptions import MissingDataError
from ..util import dictx
from .cascading import data_av_cascade, data_ch_cascade
from ..dataframes import DataFrames
from ..dataset import Dataset, DataTree
from collections import namedtuple
from .data_arrays import DataArrays


import warnings
warnings.warn( ### EXPERIMENTAL ### )
    "\n\n"
    "### larch.numba is experimental, and not feature-complete ###\n"
    " the first time you import on a new system, this package will\n"
    " compile optimized binaries for your machine, which may take \n"
    " a little while, please be patient \n"
)

@njit(cache=True)
def minmax(x):
    maximum = x[0]
    minimum = x[0]
    for i in x[1:]:
        if i > maximum:
            maximum = i
        elif i < minimum:
            minimum = i
    return (minimum, maximum)

@njit(cache=True)
def outside_range(x, bottom, top):
    for i in x[:]:
        if i == -np.inf:
            continue
        if i > top:
            return True
        elif i < bottom:
            return True
    return False


@njit(error_model='numpy', fastmath=True, cache=True)
def utility_from_data_co(
        model_utility_co_alt,          # int input shape=[n_co_features]
        model_utility_co_param_scale,  # float input shape=[n_co_features]
        model_utility_co_param,        # int input shape=[n_co_features]
        model_utility_co_data,         # int input shape=[n_co_features]
        parameter_arr,                 # float input shape=[n_params]
        holdfast_arr,                  # float input shape=[n_params]
        array_av,                      # int8 input shape=[n_alts]
        data_co,                       # float input shape=[n_co_vars]
        utility_elem,                  # float output shape=[n_alts]
        dutility_elem,                 # float output shape=[n_alts, n_params]
):
    for i in range(model_utility_co_alt.shape[0]):
        altindex = model_utility_co_alt[i]
        param_value = parameter_arr[model_utility_co_param[i]]
        param_holdfast = holdfast_arr[model_utility_co_param[i]]
        if array_av[altindex]:
            if model_utility_co_data[i] == -1:
                utility_elem[altindex] += param_value * model_utility_co_param_scale[i]
                if not param_holdfast:
                    dutility_elem[altindex, model_utility_co_param[i]] += model_utility_co_param_scale[i]
            else:
                _temp = data_co[model_utility_co_data[i]] * model_utility_co_param_scale[i]
                utility_elem[altindex] += _temp * param_value
                if not param_holdfast:
                    dutility_elem[altindex, model_utility_co_param[i]] += _temp



@njit(error_model='numpy', fastmath=True, cache=True)
def quantity_from_data_ca(
        model_q_ca_param_scale,         # float input shape=[n_q_ca_features]
        model_q_ca_param,               # int input shape=[n_q_ca_features]
        model_q_ca_data,                # int input shape=[n_q_ca_features]
        model_q_scale_param,            # int input scalar
        parameter_arr,                  # float input shape=[n_params]
        holdfast_arr,                   # float input shape=[n_params]
        array_av,                       # int8 input shape=[n_alts]
        array_ca,                       # float input shape=[n_alts, n_ca_vars]
        utility_elem,                   # float output shape=[n_alts]
        dutility_elem,                  # float output shape=[n_alts, n_params]
):
    n_alts = array_ca.shape[0]

    if model_q_scale_param[0] >= 0:
        scale_param_value = parameter_arr[model_q_scale_param[0]]
        scale_param_holdfast = holdfast_arr[model_q_scale_param[0]]
    else:
        scale_param_value = 1.0
        scale_param_holdfast = 1

    for j in range(n_alts):

        # if self._array_ce_reversemap is not None:
        #     if c >= self._array_ce_reversemap.shape[0] or j >= self._array_ce_reversemap.shape[1]:
        #         row = -1
        #     else:
        #         row = self._array_ce_reversemap[c, j]
        row = -1

        if array_av[j]: # and row != -1:

            if model_q_ca_param.shape[0]:
                for i in range(model_q_ca_param.shape[0]):
                    # if row >= 0:
                    #     _temp = self._array_ce[row, self.model_quantity_ca_data[i]]
                    # else:
                    _temp = (
                        array_ca[j, model_q_ca_data[i]]
                        * model_q_ca_param_scale[i]
                        * np.exp(parameter_arr[model_q_ca_param[i]])
                    )
                    utility_elem[j] += _temp
                    if not holdfast_arr[model_q_ca_param[i]]:
                        dutility_elem[j, model_q_ca_param[i]] += _temp * scale_param_value

                for i in range(model_q_ca_param.shape[0]):
                    if not holdfast_arr[model_q_ca_param[i]]:
                        dutility_elem[j, model_q_ca_param[i]] /= utility_elem[j]

                _tempsize = np.log(utility_elem[j])
                utility_elem[j] = _tempsize * scale_param_value
                if (model_q_scale_param[0] >= 0) and not scale_param_holdfast:
                    dutility_elem[j, model_q_scale_param[0]] += _tempsize

        else:
            utility_elem[j] = -np.inf


@njit(error_model='numpy', fastmath=True, cache=True)
def utility_from_data_ca(
        model_utility_ca_param_scale,   # int input shape=[n_u_ca_features]
        model_utility_ca_param,         # int input shape=[n_u_ca_features]
        model_utility_ca_data,          # int input shape=[n_u_ca_features]
        parameter_arr,                  # float input shape=[n_params]
        holdfast_arr,                   # float input shape=[n_params]
        array_av,                       # int8 input shape=[n_alts]
        array_ca,                       # float input shape=[n_alts, n_ca_vars]
        utility_elem,                   # float output shape=[n_alts]
        dutility_elem,                  # float output shape=[n_alts, n_params]
):
    n_alts = array_ca.shape[0]
    for j in range(n_alts):

        # if self._array_ce_reversemap is not None:
        #     if c >= self._array_ce_reversemap.shape[0] or j >= self._array_ce_reversemap.shape[1]:
        #         row = -1
        #     else:
        #         row = self._array_ce_reversemap[c, j]
        row = -1

        if array_av[j]: # and row != -1:

            #     if self.model_quantity_ca_param.shape[0]:
            #         for i in range(self.model_quantity_ca_param.shape[0]):
            #             if row >= 0:
            #                 _temp = self._array_ce[row, self.model_quantity_ca_data[i]]
            #             else:
            #                 _temp = self._array_ca[c, j, self.model_quantity_ca_data[i]]
            #             _temp *= self.model_quantity_ca_param_value[i] * self.model_quantity_ca_param_scale[i]
            #             U[j] += _temp
            #             if not self.model_quantity_ca_param_holdfast[i]:
            #                 dU[j, self.model_quantity_ca_param[i]] += _temp * self.model_quantity_scale_param_value
            #
            #         for i in range(self.model_quantity_ca_param.shape[0]):
            #             if not self.model_quantity_ca_param_holdfast[i]:
            #                 dU[j, self.model_quantity_ca_param[i]] /= U[j]
            #
            #         if Q is not None:
            #             Q[j] = U[j]
            #         IF
            #         DOUBLE_PRECISION:
            #         _temp = log(U[j])
            #     ELSE:
            #     _temp = logf(U[j])
            # U[j] = _temp * self.model_quantity_scale_param_value
            # if (self.model_quantity_scale_param >= 0) and not self.model_quantity_scale_param_holdfast:
            #     dU[j, self.model_quantity_scale_param] += _temp

            for i in range(model_utility_ca_param.shape[0]):
                if row >= 0:
                    _temp = 0.0 #_temp = self._array_ce[row, self.model_utility_ca_data[i]]
                else:
                    _temp = array_ca[j, model_utility_ca_data[i]]
                _temp *= model_utility_ca_param_scale[i]
                utility_elem[j] += _temp * parameter_arr[model_utility_ca_param[i]]
                if not holdfast_arr[model_utility_ca_param[i]]:
                    dutility_elem[j, model_utility_ca_param[i]] += _temp
        else:
            utility_elem[j] = -np.inf


def _type_signature(sig, precision=32):
    result = ()
    for s in sig:
        if s == "f":
            result += (f32[:],) if precision==32 else (f64[:],)
        elif s == "F":
            result += (f32[:,:],) if precision == 32 else (f64[:,:],)
        elif s == "r":
            result += (f32,) if precision == 32 else (f64,)
        elif s == "i":
            result += (i32[:],)
        elif s == "I":
            result += (i32[:,:],)
        elif s == "b":
            result += (i8[:],)
        elif s == "S":
            result += (i8,)
        elif s == 'B':
            result += (boolean, )
    return result


def _type_signatures(sig):
    return [
        _type_signature(sig, precision=32),
        _type_signature(sig, precision=64),
    ]

@njit(error_model='numpy', fastmath=True, cache=True)
def _numba_utility_to_loglike(
        n_alts,
        edgeslots,     # int input shape=[edges, 4]
        mu_slots,      # int input shape=[nests]
        start_slots,   # int input shape=[nests]
        len_slots,     # int input shape=[nests]
        holdfast_arr,  # int8 input shape=[n_params]
        parameter_arr, # float input shape=[n_params]
        array_ch,      # float input shape=[nodes]
        array_av,      # int8 input shape=[nodes]
        array_wt,      # float input shape=[]
        return_flags,  #
        dutility,      #
        utility,       # float output shape=[nodes]
        logprob,       # float output shape=[nodes]
        probability,   # float output shape=[nodes]
        bhhh,          # float output shape=[n_params, n_params]
        d_loglike,     # float output shape=[n_params]
        loglike,       # float output shape=[]
):

    assert edgeslots.shape[1] == 4
    upslots   = edgeslots[:,0]  # int input shape=[edges]
    dnslots   = edgeslots[:,1]  # int input shape=[edges]
    visit1    = edgeslots[:,2]  # int input shape=[edges]
    allocslot = edgeslots[:,3]  # int input shape=[edges]

    assert return_flags.size == 4
    only_utility = return_flags[0]          # [19] int8 input
    return_probability = return_flags[1]    # [20] bool input
    return_grad = return_flags[2]           # [21] bool input
    return_bhhh = return_flags[3]           # [22] bool input

    #util_nx = np.zeros_like(utility)
    #mu_extra = np.zeros_like(util_nx)
    loglike[0] = 0.0

    if True: # outside_range(utility[:n_alts], -0.0, 0.0):
        for up in range(n_alts, utility.size):
            up_nest = up - n_alts
            n_children_for_parent = len_slots[up_nest]
            shifter = -np.inf
            shifter_position = -1
            if mu_slots[up_nest] < 0:
                mu_up = 1.0
            else:
                mu_up = parameter_arr[mu_slots[up_nest]]
            if mu_up:
                for n in range(n_children_for_parent):
                    edge = start_slots[up_nest] + n
                    dn = dnslots[edge]
                    if utility[dn] > -np.inf:
                        z = utility[dn] / mu_up
                        if z > shifter:
                            shifter = z
                            shifter_position = dn
                        # TODO alphas
                        # if alpha[edge] > 0:
                        #     z = (logalpha[edge] + utility[child]) / mu[parent]
                        #     if z > shifter:
                        #         shifter = z
                        #         shifter_position = child
                for n in range(n_children_for_parent):
                    edge = start_slots[up_nest] + n
                    dn = dnslots[edge]
                    if utility[dn] > -np.inf:
                        if shifter_position == dn:
                            utility[up] += 1
                        else:
                            utility[up] += np.exp((utility[dn] / mu_up) - shifter)
                        # if alpha[edge] > 0:
                        #     if shifter_position == child:
                        #         utility[parent] += 1
                        #     else:
                        #         z = ((logalpha[edge] + utility[child]) / mu[parent]) - shifter
                        #         utility[parent] += exp(z)
                utility[up] = (np.log(utility[up]) + shifter) * mu_up
            else: # mu_up is zero
                for n in range(n_children_for_parent):
                    edge = start_slots[up_nest] + n
                    dn = dnslots[edge]
                    if utility[dn] > utility[up]:
                        utility[up] = utility[dn]
    else:
        for s in range(upslots.size):
            dn = dnslots[s]
            up = upslots[s]
            up_nest = up - n_alts
            dn_nest = dn - n_alts
            if mu_slots[up_nest] < 0:
                mu_up = 1.0
            else:
                mu_up = parameter_arr[mu_slots[up_nest]]
            if visit1[s]>0 and dn>=n_alts:
                log_dn = np.log(utility[dn])
                #mu_extra[dn] += log_dn + util_nx[dn]/utility[dn]
                if mu_slots[dn_nest] < 0:
                    mu_dn = 1.0
                else:
                    mu_dn = parameter_arr[mu_slots[dn_nest]]
                utility[dn] = log_dn * mu_dn
            util_dn = utility[dn]
            exp_util_dn_mu_up = np.exp(util_dn / mu_up)
            utility[up] += exp_util_dn_mu_up
            #util_nx[up] -= util_dn * exp_util_dn_mu_up / mu_up

        #mu_extra[mu_extra.size-1] += np.log(utility[utility.size-1]) + util_nx[-1]/utility[utility.size-1]
        utility[utility.size-1] = np.log(utility[utility.size-1])

    if only_utility == 2:
        return

    for s in range(upslots.size):
        dn = dnslots[s]
        up = upslots[s]
        if mu_slots[up - n_alts] < 0:
            mu_up = 1.0
        else:
            mu_up = parameter_arr[mu_slots[up - n_alts]]
        if np.isinf(utility[up]) and utility[up] < 0:
            logprob[dn] = -np.inf
        else:
            logprob[dn] = (utility[dn] - utility[up]) / mu_up
        if array_ch[dn]:
            loglike[0] += logprob[dn] * array_ch[dn] * array_wt[0]

    if return_probability or return_grad or return_bhhh:

        # logprob becomes conditional_probability
        conditional_probability = logprob
        for i in range(logprob.size):
            if array_av[i]:
                conditional_probability[i] = np.exp(logprob[i])

        # probability
        probability[-1] = 1.0
        for s in range(upslots.size-1, -1, -1):
            dn = dnslots[s]
            if array_av[dn]:
                up = upslots[s]
                probability[dn] = probability[up] * conditional_probability[dn]
            else:
                probability[dn] = 0.0

        if return_grad or return_bhhh:

            d_loglike[:] = 0.0

            # d utility
            for s in range(upslots.size):
                dn = dnslots[s]
                up = upslots[s]
                if array_av[dn]:
                    cond_prob = conditional_probability[dn]
                    if dn >= n_alts:
                        dn_mu_slot = mu_slots[dn-n_alts]
                        if dn_mu_slot >= 0:
                            dutility[dn, dn_mu_slot] += utility[dn]
                            dutility[dn, dn_mu_slot] /= parameter_arr[dn_mu_slot]
                    up_mu_slot = mu_slots[up - n_alts]
                    if up_mu_slot >= 0:
                        # FIXME: alpha slots to appear here if cross-nesting is activated
                        dutility[up, up_mu_slot] -= cond_prob * (utility[dn])
                    dutility[up, :] += cond_prob * dutility[dn, :]

            # d probability
            # scratch = np.zeros_like(parameter_arr)
            # d_probability = np.zeros_like(dutility)
            # for s in range(upslots.size-1, -1, -1):
            #     dn = dnslots[s]
            #     if array_ch[dn]:
            #         up = upslots[s]
            #         scratch[:] = dutility[dn] - dutility[up]
            #         up_mu_slot = mu_slots[up-n_alts]
            #         if up_mu_slot < 0:
            #             mu_up = 1.0
            #         else:
            #             mu_up = parameter_arr[up_mu_slot]
            #         if mu_up:
            #             if up_mu_slot >= 0:
            #                 scratch[up_mu_slot] += (utility[up] - utility[dn]) / mu_up
            #                 # FIXME: alpha slots to appear here if cross-nesting is activated
            #             multiplier = probability[up] / mu_up
            #         else:
            #             multiplier = 0
            #
            #         scratch[:] *= multiplier
            #         scratch[:] += d_probability[up, :]
            #         d_probability[dn, :] += scratch[:] * conditional_probability[dn] # FIXME: for CNL, use edge not dn

            # d probability alternate path slightly lower memory usage and some faster
            d_probability = np.zeros_like(dutility)
            for s in range(upslots.size-1, -1, -1):
                dn = dnslots[s]
                if array_ch[dn]:
                    up = upslots[s]
                    up_mu_slot = mu_slots[up - n_alts]
                    if up_mu_slot < 0:
                        mu_up = 1.0
                    else:
                        mu_up = parameter_arr[up_mu_slot]
                    for p in range(parameter_arr.size):
                        if mu_up:
                            scratch_ = dutility[dn, p] - dutility[up, p]
                            if p == up_mu_slot:
                                scratch_ += (utility[up] - utility[dn]) / mu_up
                                # FIXME: alpha slots to appear here if cross-nesting is activated
                            scratch_ *= probability[up] / mu_up
                        else:
                            scratch_ = 0
                        scratch_ += d_probability[up, p]
                        d_probability[dn, p] += scratch_ * conditional_probability[dn] # FIXME: for CNL, use edge not dn

            if return_bhhh:
                bhhh[:] = 0.0

            # d loglike
            for a in range(n_alts):
                this_ch = array_ch[a]
                if this_ch == 0:
                    continue
                total_probability_a = probability[a]
                # if total_probability_a > 0:
                #     tempvalue = d_probability[a, :] / total_probability_a
                #     if return_bhhh:
                #         bhhh += np.outer(tempvalue,tempvalue) * this_ch * array_wt[0]
                #     d_loglike += tempvalue * array_wt[0]
                #
                if total_probability_a > 0:
                    if total_probability_a < 1e-250:
                        total_probability_a = 1e-250
                    tempvalue = d_probability[a, :] * (this_ch / total_probability_a)
                    dLL_temp = tempvalue / this_ch
                    d_loglike += tempvalue * array_wt[0]
                    if return_bhhh:
                        bhhh += np.outer(dLL_temp,dLL_temp) * this_ch * array_wt[0]


_master_shape_signature = (
    '(qca),(qca),(qca),(), '
    '(uca),(uca),(uca), '
    '(uco),(uco),(uco),(uco), '
    '(edges,four), '
    '(nests),(nests),(nests), '
    '(params),(params), '
    '(nodes),(nodes),(),(vco),(alts,vca), '
    '(four)->'
    '(nodes),(nodes),(nodes),(params,params),(params),()'
)


def _numba_master(
        model_q_ca_param_scale,  # float input shape=[n_q_ca_features]
        model_q_ca_param,        # int input shape=[n_q_ca_features]
        model_q_ca_data,         # int input shape=[n_q_ca_features]
        model_q_scale_param,     # int input scalar

        model_utility_ca_param_scale,  # [0] float input shape=[n_u_ca_features]
        model_utility_ca_param,        # [1] int input shape=[n_u_ca_features]
        model_utility_ca_data,         # [2] int input shape=[n_u_ca_features]

        model_utility_co_alt,          # [3] int input shape=[n_co_features]
        model_utility_co_param_scale,  # [4] float input shape=[n_co_features]
        model_utility_co_param,        # [5] int input shape=[n_co_features]
        model_utility_co_data,         # [6] int input shape=[n_co_features]

        edgeslots,     # int input shape=[edges, 4]
        # upslots,       # [7] int input shape=[edges]
        # dnslots,       # [8] int input shape=[edges]
        # visit1,        # [9] int input shape=[edges]
        # allocslot,     # [10] int input shape=[edges]

        mu_slots,      # [11] int input shape=[nests]
        start_slots,   # [12] int input shape=[nests]
        len_slots,     # [13] int input shape=[nests]

        holdfast_arr,  # [12] int8 input shape=[n_params]
        parameter_arr, # [13] float input shape=[n_params]

        array_ch,      # [14] float input shape=[nodes]
        array_av,      # [15] int8 input shape=[nodes]
        array_wt,      # [16] float input shape=[]
        array_co,      # [17] float input shape=[n_co_vars]
        array_ca,      # [18] float input shape=[n_alts, n_ca_vars]

        return_flags,
        # only_utility,        # [19] int8 input
        # return_probability,  # [20] bool input
        # return_grad,         # [21] bool input
        # return_bhhh,         # [22] bool input

        utility,       # [23] float output shape=[nodes]
        logprob,       # [24] float output shape=[nodes]
        probability,   # [25] float output shape=[nodes]
        bhhh,          # [26] float output shape=[n_params, n_params]
        d_loglike,     # [27] float output shape=[n_params]
        loglike,       # [28] float output shape=[]
):
    n_alts = array_ca.shape[0]

    # assert edgeslots.shape[1] == 4
    # upslots   = edgeslots[:,0]  # int input shape=[edges]
    # dnslots   = edgeslots[:,1]  # int input shape=[edges]
    # visit1    = edgeslots[:,2]  # int input shape=[edges]
    # allocslot = edgeslots[:,3]  # int input shape=[edges]

    assert return_flags.size == 4
    only_utility = return_flags[0]            # int8 input
    # return_probability = return_flags[1]    # bool input
    # return_grad = return_flags[2]           # bool input
    # return_bhhh = return_flags[3]           # bool input

    utility[:] = 0.0
    dutility = np.zeros((utility.size, parameter_arr.size), dtype=utility.dtype)

    quantity_from_data_ca(
        model_q_ca_param_scale,  # float input shape=[n_q_ca_features]
        model_q_ca_param,        # int input shape=[n_q_ca_features]
        model_q_ca_data,         # int input shape=[n_q_ca_features]
        model_q_scale_param,     # int input scalar
        parameter_arr,           # float input shape=[n_params]
        holdfast_arr,            # float input shape=[n_params]
        array_av,                # int8 input shape=[n_nodes]
        array_ca,                # float input shape=[n_alts, n_ca_vars]
        utility[:n_alts],        # float output shape=[n_alts]
        dutility[:n_alts],
    )

    if only_utility == 3:
        if model_q_scale_param[0] >= 0:
            scale_param_value = parameter_arr[model_q_scale_param[0]]
        else:
            scale_param_value = 1.0
        utility[:n_alts] = np.exp(utility[:n_alts] / scale_param_value)
        return

    utility_from_data_ca(
        model_utility_ca_param_scale,  # int input shape=[n_u_ca_features]
        model_utility_ca_param,        # int input shape=[n_u_ca_features]
        model_utility_ca_data,         # int input shape=[n_u_ca_features]
        parameter_arr,                 # float input shape=[n_params]
        holdfast_arr,                  # float input shape=[n_params]
        array_av,                      # int8 input shape=[n_nodes]
        array_ca,                      # float input shape=[n_alts, n_ca_vars]
        utility[:n_alts],              # float output shape=[n_alts]
        dutility[:n_alts],
    )

    utility_from_data_co(
        model_utility_co_alt,          # int input shape=[n_co_features]
        model_utility_co_param_scale,  # float input shape=[n_co_features]
        model_utility_co_param,        # int input shape=[n_co_features]
        model_utility_co_data,         # int input shape=[n_co_features]
        parameter_arr,                 # float input shape=[n_params]
        holdfast_arr,                  # float input shape=[n_params]
        array_av,                      # int8 input shape=[n_nodes]
        array_co,                      # float input shape=[n_co_vars]
        utility[:n_alts],              # float output shape=[n_alts]
        dutility[:n_alts],
    )

    if only_utility == 1: return

    _numba_utility_to_loglike(
        n_alts,
        edgeslots,      # int input shape=[edges, 4]
        mu_slots,       # int input shape=[nests]
        start_slots,    # int input shape=[nests]
        len_slots,      # int input shape=[nests]
        holdfast_arr,   # int8 input shape=[n_params]
        parameter_arr,  # float input shape=[n_params]
        array_ch,       # float input shape=[nodes]
        array_av,       # int8 input shape=[nodes]
        array_wt,       # float input shape=[]
        return_flags,
        dutility,
        utility,        # float output shape=[nodes]
        logprob,        # float output shape=[nodes]
        probability,    # float output shape=[nodes]
        bhhh,           # float output shape=[n_params, n_params]
        d_loglike,      # float output shape=[n_params]
        loglike,        # float output shape=[]
    )



_numba_master_vectorized = guvectorize(
    _type_signatures("fiii fii ifii I iii bf fbffF b fffFff"),
    _master_shape_signature,
    nopython=True,
    fastmath=True,
    target='parallel',
    cache=True,
)(
    _numba_master,
)


@njit(cache=True)
def softplus(i, sharpness=10):
    cut = 10 / sharpness
    if i > cut:
        return i
    # elif i < -cut:
    #     return 0.0
    else:
        return np.log1p(np.exp(i * sharpness)) / sharpness


@njit(cache=True)
def d_softplus(i, sharpness=10):
    cut = 1000 / sharpness
    if i > cut:
        return 1.0
    # elif i < -cut:
    #     return 0.0
    else:
        return 1 / (1 + np.exp(-i * sharpness))


@guvectorize(
    [
        _type_signature("ffffffff", precision=32),
        _type_signature("ffffffff", precision=64),
    ],
    (
        '(params),(params),(params),(),()->(),(params),(params)'
    ),
    cache=True,
    nopython=True,
)
def bounds_penalty(
        param_array,           # [] float input shape=[n_params]
        lower_bounds,          # [] float input shape=[n_params]
        upper_bounds,          # [] float input shape=[n_params]
        constraint_intensity,  # [] float input shape=[]
        constraint_sharpness,  # [] float input shape=[]
        penalty,               # [] float output shape=[]
        d_penalty,             # [] float output shape=[n_params]
        d_penalty_binding,     # [] float output shape=[n_params]
):
    # penalty = 0.0
    # d_penalty = np.zeros_like(param_array)
    # d_penalty_binding = np.zeros_like(param_array)
    penalty[0] = 0.0
    d_penalty[:] = 0.0
    d_penalty_binding[:] = 0.0
    for i in range(param_array.size):
        diff_threshold = np.minimum(upper_bounds[i] - lower_bounds[i], 0.0001)
        low_diff = lower_bounds[i] - param_array[i]
        low_penalty = -softplus(low_diff, constraint_sharpness[0])
        high_diff = param_array[i] - upper_bounds[i]
        high_penalty = -softplus(high_diff, constraint_sharpness[0])
        penalty[0] += (low_penalty + high_penalty) * constraint_intensity[0]
        if low_penalty:
            d_penalty[i] += d_softplus(lower_bounds[i] - param_array[i], constraint_sharpness[0]) * constraint_intensity[0]
        if high_penalty:
            d_penalty[i] -= d_softplus(param_array[i] - upper_bounds[i], constraint_sharpness[0]) * constraint_intensity[0]
        if np.absolute(high_diff) < diff_threshold:
            d_penalty_binding[i] -= 0.5 * constraint_intensity[0]
        elif np.absolute(low_diff) < diff_threshold:
            d_penalty_binding[i] += 0.5 * constraint_intensity[0]
    #return penalty, d_penalty, d_penalty_binding


def _numba_penalty(
        param_array,           # [] float input shape=[n_params]
        lower_bounds,          # [] float input shape=[n_params]
        upper_bounds,          # [] float input shape=[n_params]
        constraint_intensity,  # [] float input shape=[]
        constraint_sharpness,  # [] float input shape=[]
        bhhh,                  # [] float output shape=[n_params, n_params]
        d_loglike,             # [] float output shape=[n_params]
        loglike,               # [] float output shape=[]
):
    penalty = 0.0
    d_penalty = np.zeros_like(d_loglike)
    for i in range(param_array.size):
        low_penalty = -softplus(lower_bounds[i] - param_array[i], constraint_sharpness[0])
        high_penalty = -softplus(param_array[i] - upper_bounds[i], constraint_sharpness[0])
        penalty += (low_penalty + high_penalty) * constraint_intensity[0]
        if low_penalty:
            d_penalty[i] += d_softplus(lower_bounds[i] - param_array[i], constraint_sharpness[0]) * constraint_intensity[0]
        if high_penalty:
            d_penalty[i] -= d_softplus(param_array[i] - upper_bounds[i], constraint_sharpness[0]) * constraint_intensity[0]
    loglike[0] += penalty
    d_loglike[:] += d_penalty
    bhhh[:] += np.outer(d_penalty, d_penalty)


_numba_penalty_vectorized = guvectorize(
    _type_signatures("fffffFff"),
    (
        '(params),(params),(params),(),()->(params,params),(params),()'
    ),
    nopython=True,
    fastmath=True,
    target='parallel',
    cache=True,
)(
    _numba_penalty,
)



def model_co_slots(data_provider, model, dtype=np.float64):
    len_co = sum(len(_) for _ in model.utility_co.values())
    model_utility_co_alt = np.zeros([len_co], dtype=np.int32)
    model_utility_co_param_scale = np.ones([len_co], dtype=dtype)
    model_utility_co_param = np.zeros([len_co], dtype=np.int32)
    model_utility_co_data = np.zeros([len_co], dtype=np.int32)

    j = 0

    param_loc = {}
    for _n, _pname in enumerate(model._frame.index):
        param_loc[_pname] = _n
    data_loc = {}
    if isinstance(data_provider, DataFrames):
        if data_provider.data_co is not None:
            for _n, _dname in enumerate(data_provider.data_co.columns):
                data_loc[_dname] = _n
        alternative_codes = data_provider.alternative_codes()
    elif isinstance(data_provider, Dataset):
        if 'var_co' in data_provider.indexes:
            for _n, _dname in enumerate(data_provider.indexes['var_co']):
                data_loc[_dname] = _n
        alternative_codes = data_provider.indexes[data_provider.ALTID]
    else:
        raise TypeError(f"data_provider must be DataFrames or Dataset not {type(data_provider)}")

    for alt, func in model.utility_co.items():
        altindex = alternative_codes.get_loc(alt)
        for i in func:
            model_utility_co_alt[j] = altindex
            model_utility_co_param[j] = param_loc[str(i.param)]  # model._frame.index.get_loc(str(i.param))
            model_utility_co_param_scale[j] = i.scale
            if i.data == '1':
                model_utility_co_data[j] = -1
            else:
                model_utility_co_data[j] = data_loc[str(i.data)]  # self._data_co.columns.get_loc(str(i.data))
            j += 1

    return (
        model_utility_co_alt,
        model_utility_co_param_scale,
        model_utility_co_param,
        model_utility_co_data,
    )


def model_u_ca_slots(data_provider, model, dtype=np.float64):
    if isinstance(data_provider, DataFrames):
        looker = lambda tag: data_provider.data_ca_or_ce.columns.get_loc(str(tag))
    elif isinstance(data_provider, Dataset):
        looker = lambda tag: data_provider.indexes['var_ca'].get_loc(str(tag))
    else:
        raise TypeError(f"data_provider must be DataFrames or Dataset not {type(data_provider)}")
    len_model_utility_ca = len(model.utility_ca)
    model_utility_ca_param_scale = np.ones([len_model_utility_ca], dtype=dtype)
    model_utility_ca_param = np.zeros([len_model_utility_ca], dtype=np.int32)
    model_utility_ca_data = np.zeros([len_model_utility_ca], dtype=np.int32)
    for n, i in enumerate(model.utility_ca):
        model_utility_ca_param[n] = model._frame.index.get_loc(str(i.param))
        model_utility_ca_data[n] = looker(i.data)
        model_utility_ca_param_scale[n] = i.scale
    return (
        model_utility_ca_param_scale,
        model_utility_ca_param,
        model_utility_ca_data,
    )

def model_q_ca_slots(data_provider, model, dtype=np.float64):
    if isinstance(data_provider, DataFrames):
        looker = lambda tag: data_provider.data_ca_or_ce.columns.get_loc(str(tag))
    elif isinstance(data_provider, Dataset):
        looker = lambda tag: data_provider.indexes['var_ca'].get_loc(str(tag))
    else:
        raise TypeError(f"data_provider must be DataFrames or Dataset not {type(data_provider)}")
    len_model_q_ca = len(model.quantity_ca)
    model_q_ca_param_scale = np.ones([len_model_q_ca], dtype=dtype)
    model_q_ca_param = np.zeros([len_model_q_ca], dtype=np.int32)
    model_q_ca_data = np.zeros([len_model_q_ca], dtype=np.int32)
    if model.quantity_scale:
        model_q_scale_param = model._frame.index.get_loc(str(model.quantity_scale))
    else:
        model_q_scale_param = np.zeros([1], dtype=np.int32)-1
    for n, i in enumerate(model.quantity_ca):
        model_q_ca_param[n] = model._frame.index.get_loc(str(i.param))
        model_q_ca_data[n] = looker(i.data)
        model_q_ca_param_scale[n] = i.scale
    return (
        model_q_ca_param_scale,
        model_q_ca_param,
        model_q_ca_data,
        model_q_scale_param,
    )


class _case_slice:
    def __get__(self, obj, objtype=None):
        self.parent = obj
        return self
    def __getitem__(self, idx):
        return type(self.parent)(**{k: getattr(self.parent, k)[idx] for k in self.parent._fields})


WorkArrays = namedtuple(
    'WorkArrays',
    ['utility', 'logprob', 'probability', 'bhhh', 'd_loglike', 'loglike'],
)
WorkArrays.cs = _case_slice()


FixedArrays = namedtuple(
    'FixedArrays',
    [
        'qca_scale', 'qca_param_slot', 'qca_data_slot',
        'qscale_param_slot',
        'uca_scale', 'uca_param_slot', 'uca_data_slot',
        'uco_alt_slot', 'uco_scale', 'uco_param_slot', 'uco_data_slot',
        'edge_slots',
        'mu_slot', 'start_edges', 'len_edges',
    ]
)


class NumbaModel(_BaseModel):

    _null_slice = (None, None, None)

    def __init__(self, *args, float_dtype=np.float64, datatree=None, **kwargs):
        for a in args:
            if datatree is None and isinstance(a, (DataTree, Dataset)):
                datatree = a
        super().__init__(*args, **kwargs)
        self._dataset = None
        self._fixed_arrays = None
        self._data_arrays = None
        self.work_arrays = None
        self.float_dtype = float_dtype
        self.constraint_intensity = 0.0
        self.constraint_sharpness = 0.0
        self._constraint_funcs = None
        self.datatree = datatree

    def mangle(self, *args, **kwargs):
        super().mangle(*args, **kwargs)
        self._dataset = None
        self._fixed_arrays = None
        self._data_arrays = None
        self.work_arrays = None
        self._array_ch_cascade = None
        self._array_av_cascade = None
        self._constraint_funcs = None

    def initialize_graph(self, dataframes=None, alternative_codes=None, alternative_names=None, root_id=0):
        """
        Write a nesting tree graph for a MNL model.

        Parameters
        ----------
        dataframes : DataFrames, optional
            Use this to determine the included alternatives.
        alternative_codes : array-like, optional
            Explicitly give alternative codes. Ignored if `dataframes` is given
            or if the model has dataframes or a dataservice already set.
        alternative_names : array-like, optional
            Explicitly give alternative names. Ignored if `dataframes` is given
            or if the model has dataframes or a dataservice already set.
        root_id : int, default 0
            The id code of the root node.

        Raises
        ------
        ValueError
            The model is unable to infer the alternative codes to use.  This can
            be avoided by giving alternative codes explicitly or having previously
            set dataframes or a dataservice that will give the alternative codes.
        """
        if self.datatree is not None:

            from ..dataset import DataTree

            def get_coords_array(*names):
                for name in names:
                    if name in self.datatree.root_dataset.coords:
                        return self.datatree.root_dataset.coords[name].values

            if alternative_codes is None:
                alternative_codes = get_coords_array(
                    self.datatree.ALTID,
                    '_altid_', 'altid', 'alt_id', 'alt_ids',
                    'alternative_id', 'alternative_ids',
                )
            if alternative_names is None:
                alternative_names = get_coords_array(
                    'altname', 'altnames', 'alt_name', 'alt_names',
                    'alternative_name', 'alternative_names',
                )

        super().initialize_graph(
            dataframes=dataframes,
            alternative_codes=alternative_codes,
            alternative_names=alternative_names,
            root_id=root_id,
        )

    def reflow_data_arrays(self):
        """
        Reload the internal data_arrays so they are consistent with the datatree.
        """
        if self.graph is None:
            self._data_arrays = None
            return

        datatree = self.datatree
        if datatree is not None:
            from .data_arrays import prepare_data
            self.dataset, self.dataflows = prepare_data(
                datashare=datatree,
                request=self,
                float_dtype=self.float_dtype,
                cache_dir=datatree.cache_dir,
                flows=getattr(self, 'dataflows', None),
            )
            self._data_arrays = self.dataset.to_arrays(
                self.graph,
                float_dtype=self.float_dtype,
            )
            if self.work_arrays is not None:
                self._rebuild_work_arrays()

        elif self.dataframes is not None: # work from old DataFrames

            n_nodes = len(self.graph)
            if self.dataframes.data_ch is not None:
                _array_ch_cascade = data_ch_cascade(self.dataframes, self.graph, dtype=self.float_dtype)
            else:
                _array_ch_cascade = np.zeros([self.n_cases, 0], dtype=self.float_dtype)
            if self.dataframes.data_av is not None:
                _array_av_cascade = data_av_cascade(self.dataframes, self.graph)
            else:
                _array_av_cascade = np.ones([self.n_cases, n_nodes], dtype=np.int8)
            if self.dataframes.data_wt is not None:
                _array_wt = self.dataframes.array_wt().astype(self.float_dtype)
            else:
                _array_wt = np.ones(self.n_cases, dtype=self.float_dtype)

            _array_co = self.dataframes.array_co(force=True)
            if _array_co.dtype != self.float_dtype:
                _array_co = _array_co.astype(self.float_dtype)

            _array_ca = self.dataframes.array_ca(force=True)
            if _array_ca.dtype != self.float_dtype:
                _array_ca = _array_ca.astype(self.float_dtype)

            self._data_arrays = DataArrays(
                (_array_ch_cascade),
                (_array_av_cascade),
                (_array_wt.reshape(-1)),
                (_array_co),
                (_array_ca),
            )

    def _rebuild_work_arrays(self, n_cases=None, n_nodes=None, n_params=None, on_missing_data='silent'):
        log = logging.getLogger("Larch")
        if n_cases is None:
            try:
                n_cases = self.n_cases
            except MissingDataError as err:
                if on_missing_data != 'silent':
                    log.error("MissingDataError, cannot rebuild work arrays")
                self.work_arrays = None
                if on_missing_data == 'raise':
                    raise
                return
        if n_nodes is None:
            n_nodes = len(self.graph)
        if n_params is None:
            n_params = len(self._frame)
        _need_to_rebuild_work_arrays = True
        if self.work_arrays is not None:
            if (
                    (self.work_arrays.utility.shape[0] == n_cases)
                    and (self.work_arrays.utility.shape[1] == n_nodes)
                    and (self.work_arrays.d_loglike.shape[1] == n_params)
                    and (self.work_arrays.utility.dtype == self.float_dtype)
            ):
                _need_to_rebuild_work_arrays = False
        if _need_to_rebuild_work_arrays:
            log.debug("rebuilding work arrays")
            self.work_arrays = WorkArrays(
                utility=np.zeros([n_cases, n_nodes], dtype=self.float_dtype),
                logprob=np.zeros([n_cases, n_nodes], dtype=self.float_dtype),
                probability=np.zeros([n_cases, n_nodes], dtype=self.float_dtype),
                bhhh=np.zeros([n_cases, n_params, n_params], dtype=self.float_dtype),
                d_loglike=np.zeros([n_cases, n_params], dtype=self.float_dtype),
                loglike=np.zeros([n_cases], dtype=self.float_dtype),
            )

    def _rebuild_fixed_arrays(self):
        data_provider = self.data_as_loaded
        if data_provider is not None:
            (
                model_utility_ca_param_scale,
                model_utility_ca_param,
                model_utility_ca_data,
            ) = model_u_ca_slots(data_provider, self, dtype=self.float_dtype)
            (
                model_utility_co_alt,
                model_utility_co_param_scale,
                model_utility_co_param,
                model_utility_co_data,
            ) = model_co_slots(data_provider, self, dtype=self.float_dtype)
            (
                model_q_ca_param_scale,
                model_q_ca_param,
                model_q_ca_data,
                model_q_scale_param,
            ) = model_q_ca_slots(data_provider, self, dtype=self.float_dtype)
            node_slot_arrays = self.graph.node_slot_arrays(self)
            n_alts = self.graph.n_elementals()
            self._fixed_arrays = FixedArrays(
                model_q_ca_param_scale,
                model_q_ca_param,
                model_q_ca_data,
                model_q_scale_param,

                model_utility_ca_param_scale,
                model_utility_ca_param,
                model_utility_ca_data,

                model_utility_co_alt,
                model_utility_co_param_scale,
                model_utility_co_param,
                model_utility_co_data,

                np.stack(self.graph.edge_slot_arrays()).T,
                node_slot_arrays[0][n_alts:],
                node_slot_arrays[1][n_alts:],
                node_slot_arrays[2][n_alts:],
            )
        else:
            self._fixed_arrays = None


    def unmangle(self, force=False):
        super().unmangle(force=force)
        if self._dataset is None or force:
            self.reflow_data_arrays()
        if self._fixed_arrays is None or force:
            self._rebuild_fixed_arrays()
            self._rebuild_work_arrays()
        if self._constraint_funcs is None:
            self._constraint_funcs = [c.as_soft_penalty() for c in self.constraints]

    def __prepare_for_compute(
            self,
            x=None,
            allow_missing_ch=False,
            allow_missing_av=False,
            caseslice=None,
    ):
        if caseslice is None:
            caseslice = slice(caseslice)
        missing_ch, missing_av = False, False
        if self._dataframes is None and self.datatree is None:
            raise MissingDataError('dataframes and datatree are both not set, maybe you need to call `load_data` first?')
        if self._dataframes is not None and not self._dataframes.is_computational_ready(activate=True):
            raise ValueError('DataFrames is not computational-ready')
        self.unmangle()
        if x is not None:
            self.set_values(x)
        if self._dataframes is not None:
            if self.dataframes.data_ch is None:
                if allow_missing_ch:
                    missing_ch = True
                    self._data_arrays = DataArrays(
                        np.zeros(self._data_arrays.av.shape, dtype=self._data_arrays.wt.dtype),
                        self._data_arrays.av,
                        self._data_arrays.wt,
                        self._data_arrays.co,
                        self._data_arrays.ca,
                    )
                else:
                    raise MissingDataError('model.dataframes does not define data_ch')
            if self.dataframes.data_av is None:
                if allow_missing_av:
                    missing_av = True
                else:
                    raise MissingDataError('model.dataframes does not define data_av')
        if self.work_arrays is None:
            self._rebuild_work_arrays(on_missing_data='raise')
        return (
            *self._fixed_arrays,
            self._frame.holdfast.to_numpy(),
            self.pvals.astype(self.float_dtype), # float input shape=[n_params]
            *self._data_arrays.cs[caseslice][:5], # TODO fix when not using named tuple
        )

    def constraint_violation(
            self,
            on_violation='raise',
            intensity_check=False,
    ):
        """
        Check if constraints are currently violated.

        Parameters
        ----------
        on_violation : {'raise', 'return'}
            If set to 'raise', an exception is raised if any model constraint,
            including any bound constraint, is violated.  Otherwise, this method
            returns a message describing the first constraint violation found, or
            an empty string if no violation are found.
        intensity_check : bool, default False
            If True, when the model's `constraint_intensity` attribute is set
            to zero, this function always returns OK (empty string).

        Returns
        -------
        str
            If no exception is raised, this method returns a message describing
            the first constraint violation found, or an empty string if no
            violation are found.
        """
        OK = ""
        if intensity_check and self.constraint_intensity == 0:
            return OK
        over_max = self.pf.value > self.pf.maximum
        if np.any(over_max):
            failure = np.where(over_max)[0][0]
            failure_message = (
                f"{self.pf.index[failure]} over maximum "
                f"({self.pf.value.iloc[failure]} > {self.pf.maximum.iloc[failure]})"
            )
            if on_violation != 'raise':
                return failure_message
            raise ValueError(failure_message)
        under_min = self.pf.value < self.pf.minimum
        if np.any(under_min):
            failure = np.where(under_min)[0][0]
            failure_message = (
                f"{self.pf.index[failure]} under minimum "
                f"({self.pf.value.iloc[failure]} < {self.pf.minimum.iloc[failure]})"
            )
            if on_violation != 'raise':
                return failure_message
            raise ValueError(failure_message)
        for c in self.constraints:
            if c.fun(self.pvals) < 0:
                failure_message = str(c)
                if on_violation != 'raise':
                    return failure_message
                raise ValueError(failure_message)
        return OK

    def fit_bhhh(self, *args, **kwargs):
        from .optimization import fit_bhhh
        return fit_bhhh(self, *args, **kwargs)

    def constraint_penalty(self, x=None):
        if x is not None:
            self.set_values(x)
        import time
        start = time.time()
        penalty, dpenalty, dpenalty_binding = bounds_penalty(
            self.pvals.astype(self.float_dtype),
            self.pf.minimum.to_numpy().astype(self.float_dtype),
            self.pf.maximum.to_numpy().astype(self.float_dtype),
            self.float_dtype(self.constraint_intensity),
            self.float_dtype(self.constraint_sharpness),
        )
        for (cf, dcf, dcf_bind) in self._constraint_funcs:
            penalty += cf(
                self.pvals,
                self.constraint_intensity,
                self.constraint_sharpness,
            )
            dpenalty += dcf(
                self.pvals,
                self.constraint_intensity,
                self.constraint_sharpness,
            )
            dpenalty_binding += dcf_bind(
                self.pvals,
                self.constraint_intensity,
            )
        return penalty, dpenalty, dpenalty_binding

    def constraint_converge_tolerance(self, x=None):
        args = self.__prepare_for_compute(
            x,
            allow_missing_ch=False,
        )
        args_flags = args + (np.asarray([
            0,     # only_utility
            False, # return_probability
            True,  # return_gradient
            True,  # return_bhhh
        ], dtype=np.int8),)
        with np.errstate(divide='ignore', over='ignore', ):
            _numba_master_vectorized(
                *args_flags,
                out=tuple(self.work_arrays),
            )
            if self.constraint_intensity:
                penalty, dpenalty, dpenalty_binding = self.constraint_penalty()
                self.work_arrays.loglike[:] += penalty
                self.work_arrays.d_loglike[:] += np.expand_dims(dpenalty_binding, 0)
                self.work_arrays.bhhh[:] = np.einsum('ij,ik->ijk', self.work_arrays.d_loglike, self.work_arrays.d_loglike)
        bhhh = self.work_arrays.bhhh.sum(0)
        dloglike = self.work_arrays.d_loglike.sum(0)
        freedoms = (self.pf.holdfast == 0).to_numpy()
        from .optimization import propose_direction
        direction = propose_direction(bhhh, dloglike, freedoms)
        tolerance = np.dot(direction, dloglike) - self.n_cases
        return tolerance

    def _loglike_runner(
            self,
            x=None,
            only_utility=0,
            return_gradient=False,
            return_probability=False,
            return_bhhh=False,
            start_case=None,
            stop_case=None,
            step_case=None,
    ):
        caseslice = slice(start_case, stop_case, step_case)
        args = self.__prepare_for_compute(
            x,
            allow_missing_ch=return_probability or (only_utility>0),
            caseslice=caseslice,
        )
        args_flags = args + (np.asarray([
            only_utility,
            return_probability,
            return_gradient,
            return_bhhh,
        ], dtype=np.int8),)
        try:
            with np.errstate(divide='ignore', over='ignore', ):
                try:
                    result_arrays = WorkArrays(*_numba_master_vectorized(
                        *args_flags,
                        out=tuple(self.work_arrays.cs[caseslice]),
                    ))
                except ValueError:
                    result_arrays = WorkArrays(*_numba_master_vectorized(
                        *args_flags,
                        #out=tuple(self.work_arrays.cs[caseslice]),
                    ))

                if self.constraint_intensity:
                    penalty, dpenalty, dpenalty_binding = self.constraint_penalty()
                    self.work_arrays.loglike[caseslice] += penalty
                    self.work_arrays.d_loglike[caseslice] += np.expand_dims(dpenalty, 0)
                    self.work_arrays.bhhh[caseslice] = np.einsum(
                        'ij,ik->ijk',
                        self.work_arrays.d_loglike[caseslice],
                        self.work_arrays.d_loglike[caseslice],
                    )
                else:
                    penalty = 0.0

        except:
            shp = lambda y: getattr(y, 'shape', 'scalar')
            dtp = lambda y: getattr(y, 'dtype', f'{type(y)} ')
            import inspect
            arg_names = list(inspect.signature(_numba_master).parameters)
            arg_name_width = max(len(j) for j in arg_names)

            in_sig, out_sig = _master_shape_signature.split("->")
            in_sig_shapes = in_sig.split("(")[1:]
            out_sig_shapes = out_sig.split("(")[1:]
            print(in_sig_shapes)
            print(out_sig_shapes)
            print("# Input Arrays")
            for n, (a, s) in enumerate(zip(args_flags, in_sig_shapes)):
                s = s.rstrip(" ),")
                print(f" {arg_names[n]:{arg_name_width}} [{n:2}] {s.strip():9}: {dtp(a)}{shp(a)}")
            print("# Output Arrays")
            for n, (a, s) in enumerate(zip(self.work_arrays, out_sig_shapes), start=n+1):
                s = s.rstrip(" ),")
                print(f" {arg_names[n]:{arg_name_width}} [{n:2}] {s.strip():9}: {dtp(a)}{shp(a)}")
            raise
        return result_arrays, penalty

    @property
    def weight_normalization(self):
        try:
            return self.dataframes.weight_normalization
        except AttributeError:
            return 1.0

    def loglike(
            self,
            x=None,
            *,
            start_case=None, stop_case=None, step_case=None,
            **kwargs
    ):
        """
        Compute the log likelihood of the model.

        Parameters
        ----------
        x : array-like or dict, optional
            New values to set for the parameters before evaluating
            the log likelihood.  If given as array-like, the array must
            be a vector with length equal to the length of the
            parameter frame, and the given vector will replace
            the current values.  If given as a dictionary,
            the dictionary is used to update the parameters.
        start_case : int, default 0
            The first case to include in the log likelihood computation.
            To include all cases, start from 0 (the default).
        stop_case : int, default -1
            One past the last case to include in the log likelihood
            computation.  This is processed as usual for Python slicing
            and iterating, and negative values count backward from the
            end.  To include all cases, end at -1 (the default).
        step_case : int, default 1
            The step size of the case iterator to use in likelihood
            calculation.  This is processed as usual for Python slicing
            and iterating.  To include all cases, step by 1 (the default).

        Returns
        -------
        float
        """
        result_arrays, penalty = self._loglike_runner(
            x, start_case=start_case, stop_case=stop_case, step_case=step_case
        )
        result = result_arrays.loglike.sum() * self.weight_normalization
        if start_case is None and stop_case is None and step_case is None:
            self._check_if_best(result)
        return result

    def d_loglike(
            self,
            x=None,
            *,
            start_case=None, stop_case=None, step_case=None,
            return_series=False,
            **kwargs,
    ):
        result_arrays, penalty = self._loglike_runner(
            x,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
            return_gradient=True,
        )
        result = result_arrays.d_loglike.sum(0) * self.weight_normalization
        if return_series:
            result = pd.Series(result, index=self._frame.index)
        return result

    def loglike_casewise(
            self,
            x=None,
            *,
            start_case=None, stop_case=None, step_case=None,
            **kwargs,
    ):
        result_arrays, penalty = self._loglike_runner(
            x,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
        )
        return result_arrays.loglike * self.weight_normalization

    def d_loglike_casewise(
            self,
            x=None,
            *,
            start_case=None, stop_case=None, step_case=None,
            **kwargs,
    ):
        result_arrays, penalty = self._loglike_runner(
            x,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
            return_gradient=True,
        )
        return result_arrays.d_loglike * self.weight_normalization

    def bhhh(
            self,
            x=None,
            *,
            start_case=None, stop_case=None, step_case=None,
            return_dataframe=False,
            **kwargs,
    ):
        result_arrays, penalty = self._loglike_runner(
            x,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
            return_bhhh=True,
        )
        result = result_arrays.bhhh.sum(0) * self.weight_normalization
        if return_dataframe:
            result = pd.DataFrame(
                result, columns=self._frame.index, index=self._frame.index
            )
        return result

    def _wrap_as_dataframe(
            self,
            arr,
            return_dataframe,
            start_case=None,
            stop_case=None,
            step_case=None,
    ):
        if return_dataframe:
            idx = self.datatree.caseids()
            if idx is not None:
                idx = idx[start_case:stop_case:step_case]
            if return_dataframe == 'names':
                return pd.DataFrame(
                    data=arr,
                    columns=self.graph.standard_sort_names[:arr.shape[1]],
                    index=idx,
                )
            result = pd.DataFrame(
                data=arr,
                columns=self.graph.standard_sort[:arr.shape[1]],
                index=idx,
            )
            if return_dataframe == 'idce':
                return result.stack()[self._dataframes._data_av.stack().astype(bool).values]
            elif return_dataframe == 'idca':
                return result.stack()
            else:
                return result
        return arr

    def probability(
            self,
            x=None,
            *,
            start_case=None,
            stop_case=None,
            step_case=None,
            return_dataframe=False,
            include_nests=False,
    ):
        result_arrays, penalty = self._loglike_runner(
            x,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
            return_probability=True,
        )
        pr = result_arrays.probability
        if not include_nests:
            pr = pr[:, :self.graph.n_elementals()]
        return self._wrap_as_dataframe(
            pr,
            return_dataframe,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
        )

    def utility(
            self,
            x=None,
            *,
            start_case=None,
            stop_case=None,
            step_case=None,
            return_dataframe=None,
    ):
        result_arrays, penalty = self._loglike_runner(
            x,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
            only_utility=2,
        )
        return self._wrap_as_dataframe(
            result_arrays.utility,
            return_dataframe,
        )

    def quantity(
            self,
            x=None,
            *,
            start_case=None,
            stop_case=None,
            step_case=None,
            return_dataframe=None,
    ):
        result_arrays, penalty = self._loglike_runner(
            x,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
            only_utility=3,
        )
        return self._wrap_as_dataframe(
            result_arrays.utility,
            return_dataframe,
        )

    def logsums(
            self,
            x=None,
            *,
            start_case=None,
            stop_case=None,
            step_case=None,
            arr=None,
    ):
        result_arrays, penalty = self._loglike_runner(
            x,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
            only_utility=2,
        )
        if arr is not None:
            arr[start_case:stop_case:step_case] = result_arrays.utility[:,-1]
            return arr
        return result_arrays.utility[:,-1]

    def loglike2(
            self,
            x=None,
            *,
            start_case=None,
            stop_case=None,
            step_case=None,
            persist=0,
            leave_out=-1,
            keep_only=-1,
            subsample=-1,
            return_series=True,
            probability_only=False,
    ):
        result_arrays, penalty = self._loglike_runner(
            x,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
            return_gradient=True,
        )
        result = dictx(
            ll=result_arrays.loglike.sum() * self.weight_normalization,
            dll=result_arrays.d_loglike.sum(0) * self.weight_normalization,
        )
        if start_case is None and stop_case is None and step_case is None:
            self._check_if_best(result.ll)
        if return_series:
            result['dll'] = pd.Series(result['dll'], index=self._frame.index, )
        return result

    def loglike2_bhhh(
            self,
            x=None,
            *,
            return_series=False,
            start_case=None,
            stop_case=None,
            step_case=None,
            persist=0,
            leave_out=-1, keep_only=-1, subsample=-1,
    ):
        result_arrays, penalty = self._loglike_runner(
            x,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
            return_gradient=True,
            return_bhhh=True,
        )
        result = dictx(
            ll=result_arrays.loglike.sum() * self.weight_normalization,
            dll=result_arrays.d_loglike.sum(0) * self.weight_normalization,
            bhhh=result_arrays.bhhh.sum(0) * self.weight_normalization,
        )
        from ..model.persist_flags import PERSIST_LOGLIKE_CASEWISE, PERSIST_D_LOGLIKE_CASEWISE
        if persist & PERSIST_LOGLIKE_CASEWISE:
            result['ll_casewise'] = result_arrays.loglike * self.weight_normalization
        if persist & PERSIST_D_LOGLIKE_CASEWISE:
            result['dll_casewise'] = result_arrays.d_loglike * self.weight_normalization
        if start_case is None and stop_case is None and step_case is None:
            self._check_if_best(result.ll)
        if return_series:
            result['dll'] = pd.Series(result['dll'], index=self._frame.index, )
            result['bhhh'] = pd.DataFrame(
                result['bhhh'], index=self._frame.index, columns=self._frame.index
            )
        result['penalty'] = penalty
        return result

    def d2_loglike(
            self,
            x=None,
            *,
            start_case=None,
            stop_case=None,
            step_case=None,
            leave_out=-1,
            keep_only=-1,
            subsample=-1,
    ):
        return super().d2_loglike(
            x=x,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
            leave_out=leave_out,
            keep_only=keep_only,
            subsample=subsample,
        )

    def neg_loglike(
            self,
            x=None,
            start_case=None,
            stop_case=None,
            step_case=None,
            leave_out=-1,
            keep_only=-1,
            subsample=-1,
    ):
        return super().neg_loglike(
            x=x,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
            leave_out=leave_out,
            keep_only=keep_only,
            subsample=subsample,
        )

    def neg_loglike2(
            self,
            x=None,
            start_case=None,
            stop_case=None,
            step_case=None,
            leave_out=-1,
            keep_only=-1,
            subsample=-1,
    ):
        return super().neg_loglike2(
            x=x,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
            leave_out=leave_out,
            keep_only=keep_only,
            subsample=subsample,
        )

    def jumpstart_bhhh(
            self,
            steplen=0.5,
            jumpstart=0,
            jumpstart_split=5,
            leave_out=-1,
            keep_only=-1,
            subsample=-1,
            logger=None,
    ):
        """
        Jump start optimization

        Parameters
        ----------
        steplen
        jumpstart
        jumpstart_split

        """
        if logger is None:
            class NoLogger:
                debug = lambda *x: None
                info = lambda *x: None
            logger = NoLogger()

        for jump in range(jumpstart):
            j_pvals = self.pvals.copy()
            #
            # jump_breaks = list(
            #     range(0, n_cases, n_cases // jumpstart_split + (1 if n_cases % jumpstart_split else 0))
            # ) + [n_cases]
            #
            # for j0, j1 in zip(jump_breaks[:-1], jump_breaks[1:]):
            for j0 in range(jumpstart_split):
                result = self.loglike2_bhhh(
                    start_case=j0,
                    step_case=jumpstart_split,
                    leave_out=leave_out,
                    keep_only=keep_only,
                    subsample=subsample,
                )
                current_dll = result.dll
                current_bhhh = result.bhhh
                bhhh_inv = self._free_slots_inverse_matrix(current_bhhh)
                direction = np.dot(current_dll, bhhh_inv)
                j_pvals += direction * steplen
                logger.debug(f"jump to {j_pvals}")
                self.set_values(j_pvals)


    def set_dataframes(
            self,
            x,
            check_sufficiency=True,
            *,
            raw=False,
    ):
        """

        Parameters
        ----------
        x : larch.DataFrames
        check_sufficiency : bool, default True
            Run a check

        Returns
        -------

        """
        if raw:
            return super().set_dataframes(x, check_sufficiency, raw=raw)

        x.computational = True
        self.clear_best_loglike()

        # self.unmangle() # don't do a full unmangle here, it will fail if the old data is incomplete
        if self._is_mangled():
            self._scan_all_ensure_names()
        if check_sufficiency:
            x.check_data_is_sufficient_for_model(self)
        super().set_dataframes(x, check_sufficiency=False, raw=True)

        self.reflow_data_arrays()
        n_cases = x.n_cases
        if self.graph is None:
            self.initialize_graph(
                alternative_codes=x.alternative_codes(),
                alternative_names=x.alternative_names(),
            )
        n_nodes = len(self.graph)
        n_params = len(self._frame)
        self._rebuild_work_arrays(n_cases, n_nodes, n_params)


    def _frame_values_have_changed(self):
        pass # nothing to do for numba model

    def __getstate__(self):
        state = dict(
            float_dtype=self.float_dtype,
            constraint_intensity=self.constraint_intensity,
            constraint_sharpness=self.constraint_sharpness,
            _constraint_funcs=self._constraint_funcs,
        )
        return super().__getstate__(), state

    def __setstate__(self, state):
        self.float_dtype = state[1]['float_dtype']
        self.constraint_intensity = state[1]['constraint_intensity']
        self.constraint_sharpness = state[1]['constraint_sharpness']
        self._constraint_funcs = state[1]['_constraint_funcs']
        super().__setstate__(state[0])

    @property
    def n_cases(self):
        """int : The number of cases in the attached data."""
        data_as_possible = self.data_as_possible
        if data_as_possible is None:
            raise MissingDataError("no data are set")
        return data_as_possible.n_cases

    def total_weight(self):
        """
        The total weight of cases in the loaded data.

        Returns
        -------
        float
        """
        if self._data_arrays is not None:
            return self._data_arrays.wt.sum()
        raise MissingDataError("no data_arrays are set")

    @property
    def datatree(self):
        """DataTree : A source for data for the model"""
        try:
            return self._datatree
        except AttributeError:
            return None

    @datatree.setter
    def datatree(self, tree):
        from ..dataset import DataTree
        if tree is self.datatree:
            return
        if isinstance(tree, DataTree) or tree is None:
            self._datatree = tree
            self.mangle()
        elif isinstance(tree, Dataset):
            self._datatree = DataTree(main=tree)
            self.mangle()
        else:
            try:
                self._datatree = DataTree(main=Dataset.construct(tree))
            except Exception as err:
                raise TypeError(f"datatree must be DataTree not {type(tree)}") from err
            else:
                self.mangle()


    @property
    def dataset(self):
        """larch.Dataset : A source for data for the model"""
        try:
            return self._dataset
        except AttributeError:
            return None

    @dataset.setter
    def dataset(self, dataset):
        if dataset is self.dataset:
            return
        from xarray import Dataset as _Dataset
        if isinstance(dataset, Dataset):
            self._dataset = dataset
            self._data_arrays = None
        elif isinstance(dataset, _Dataset):
            self._dataset = Dataset(dataset)
            self._data_arrays = None
        else:
            raise TypeError(f"dataset must be Dataset not {type(dataset)}")

    @dataset.deleter
    def dataset(self):
        self._dataset = None
        self._data_arrays = None

    @property
    def data_as_loaded(self):
        if self.dataset is not None:
            return self.dataset
        if self.dataframes is not None:
            return self.dataframes
        return None

    @property
    def data_as_possible(self):
        if self.dataset is not None:
            return self.dataset
        if self.dataframes is not None:
            return self.dataframes
        if self.datatree is not None:
            return self.datatree
        if self.dataservice is not None:
            return self.dataservice
        return None

    @property
    def float_dtype(self):
        try:
            return self._float_dtype
        except AttributeError:
            return None

    @float_dtype.setter
    def float_dtype(self, float_dtype):
        if self.float_dtype != float_dtype:
            self.mangle()
        self._float_dtype = float_dtype

    def choice_avail_summary(self):
        """
        Generate a summary of choice and availability statistics.

        Returns
        -------
        pandas.DataFrame
        """
        from ..dataset import choice_avail_summary
        self.unmangle()
        graph = None if self.is_mnl() else self.graph
        return choice_avail_summary(
            self.dataset,
            graph,
            self.availability_co_vars,
        )
