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

import warnings
warnings.warn( ### EXPERIMENTAL ### )
    "\n\n"
    "### larch.numba is experimental, and not feature-complete ###\n"
    " the first time you import on a new system, this package will\n"
    " compile optimized binaries for your machine, which may take \n"
    " a little while, please be patient \n"
)

@njit
def minmax(x):
    maximum = x[0]
    minimum = x[0]
    for i in x[1:]:
        if i > maximum:
            maximum = i
        elif i < minimum:
            minimum = i
    return (minimum, maximum)

@njit
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



def model_co_slots(dataframes, model, dtype=np.float64):
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
    if dataframes.data_co is not None:
        for _n, _dname in enumerate(dataframes.data_co.columns):
            data_loc[_dname] = _n

    alternative_codes = dataframes.alternative_codes()

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


def model_u_ca_slots(dataframes, model, dtype=np.float64):
    len_model_utility_ca = len(model.utility_ca)
    model_utility_ca_param_scale = np.ones([len_model_utility_ca], dtype=dtype)
    model_utility_ca_param = np.zeros([len_model_utility_ca], dtype=np.int32)
    model_utility_ca_data = np.zeros([len_model_utility_ca], dtype=np.int32)
    for n, i in enumerate(model.utility_ca):
        model_utility_ca_param[n] = model._frame.index.get_loc(str(i.param))
        model_utility_ca_data[n] = dataframes.data_ca_or_ce.columns.get_loc(str(i.data))
        model_utility_ca_param_scale[n] = i.scale
    return (
        model_utility_ca_param_scale,
        model_utility_ca_param,
        model_utility_ca_data,
    )

def model_q_ca_slots(dataframes, model, dtype=np.float64):
    len_model_q_ca = len(model.quantity_ca)
    model_q_ca_param_scale = np.ones([len_model_q_ca], dtype=dtype)
    model_q_ca_param = np.zeros([len_model_q_ca], dtype=np.int32)
    model_q_ca_data = np.zeros([len_model_q_ca], dtype=np.int32)
    if model.quantity_scale:
        model_q_scale_param = model._frame.index.get_loc(str(model.quantity_scale))
    else:
        model_q_scale_param = -1
    for n, i in enumerate(model.quantity_ca):
        model_q_ca_param[n] = model._frame.index.get_loc(str(i.param))
        model_q_ca_data[n] = dataframes.data_ca_or_ce.columns.get_loc(str(i.data))
        model_q_ca_param_scale[n] = i.scale
    return (
        model_q_ca_param_scale,
        model_q_ca_param,
        model_q_ca_data,
        model_q_scale_param,
    )



from collections import namedtuple
WorkArrays = namedtuple(
    'WorkArrays',
    ['utility', 'logprob', 'probability', 'bhhh', 'd_loglike', 'loglike'],
)

DataArrays = namedtuple(
    'DataArrays',
    ['ch', 'av', 'wt', 'co', 'ca'],
)

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

    def __init__(self, *args, float_dtype = np.float64, **kwargs):
        super().__init__(*args, **kwargs)
        self._fixed_arrays = None
        self.work_arrays = None
        self.float_dtype = float_dtype

    def mangle(self, *args, **kwargs):
        super().mangle(*args, **kwargs)
        self._fixed_arrays = None
        self.work_arrays = None
        self._array_ch_cascade = None
        self._array_av_cascade = None

    def unmangle(self, force=False):
        super().unmangle(force=force)
        if self._fixed_arrays is None or force:
            n_nodes = len(self.graph)
            n_alts = self.graph.n_elementals()
            n_nests = n_nodes - n_alts
            n_params = len(self._frame)
            if self.dataframes is not None:
                (
                    model_utility_ca_param_scale,
                    model_utility_ca_param,
                    model_utility_ca_data,
                ) = model_u_ca_slots(self.dataframes, self, dtype=self.float_dtype)
                (
                    model_utility_co_alt,
                    model_utility_co_param_scale,
                    model_utility_co_param,
                    model_utility_co_data,
                ) = model_co_slots(self.dataframes, self, dtype=self.float_dtype)
                (
                    model_q_ca_param_scale,
                    model_q_ca_param,
                    model_q_ca_data,
                    model_q_scale_param,
                ) = model_q_ca_slots(self.dataframes, self, dtype=self.float_dtype)

                node_slot_arrays = self.graph.node_slot_arrays(self)

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

                if self.dataframes.data_ch is not None:
                    _array_ch_cascade = self.dataframes.data_ch_cascade(self.graph).to_numpy()
                else:
                    _array_ch_cascade = np.empty([self.n_cases, 0], dtype=self.float_dtype)
                if self.dataframes.data_av is not None:
                    _array_av_cascade = self.dataframes.data_av_cascade(self.graph).to_numpy()
                else:
                    _array_av_cascade = np.ones([self.n_cases, n_nodes], dtype=np.int8)
                if self.dataframes.data_wt is not None:
                    _array_wt = self.dataframes.array_wt().astype(self.float_dtype)
                else:
                    _array_wt = self.float_dtype(1.0)

                self._data_arrays = DataArrays(
                    (_array_ch_cascade.astype(self.float_dtype)),
                    (_array_av_cascade),
                    (_array_wt.reshape(-1)),
                    (self.dataframes.array_co(force=True).astype(self.float_dtype)),
                    (self.dataframes.array_ca(force=True).astype(self.float_dtype)),
                )
            else:
                self._fixed_arrays = None
                self._data_arrays = None

            try:
                n_cases = self.n_cases
            except MissingDataError:
                self.work_arrays = None
            else:
                self.work_arrays = WorkArrays(
                    utility=np.zeros([self.n_cases, n_nodes], dtype=self.float_dtype),
                    logprob=np.zeros([self.n_cases, n_nodes], dtype=self.float_dtype),
                    probability=np.zeros([self.n_cases, n_nodes], dtype=self.float_dtype),
                    bhhh=np.zeros([self.n_cases, n_params, n_params], dtype=self.float_dtype),
                    d_loglike=np.zeros([self.n_cases, n_params], dtype=self.float_dtype),
                    loglike=np.zeros([self.n_cases], dtype=self.float_dtype),
                )

    def __prepare_for_compute(
            self,
            x=None,
            allow_missing_ch=False,
            allow_missing_av=False,
    ):
        missing_ch, missing_av = False, False
        if self.dataframes is None:
            raise MissingDataError('dataframes is not set, maybe you need to call `load_data` first?')
        if not self.dataframes.is_computational_ready(activate=True):
            raise ValueError('DataFrames is not computational-ready')
        if x is not None:
            self.set_values(x)
        self.unmangle()
        if self.dataframes.data_ch is None:
            if allow_missing_ch:
                missing_ch = True
            else:
                raise MissingDataError('model.dataframes does not define data_ch')
        if self.dataframes.data_av is None:
            if allow_missing_av:
                missing_av = True
            else:
                raise MissingDataError('model.dataframes does not define data_av')
        return (
            *self._fixed_arrays,
            self._frame.holdfast.to_numpy(),
            self.pvals.astype(self.float_dtype), # float input shape=[n_params]
            *self._data_arrays,
        )

    def _loglike_runner(
            self,
            x=None,
            only_utility=0,
            return_gradient=False,
            return_probability=False,
            return_bhhh=False,
    ):
        args = self.__prepare_for_compute(x)
        args_flags = args + (np.asarray([
            only_utility,
            return_probability,
            return_gradient,
            return_bhhh,
        ], dtype=np.int8),)
        try:
            result_arrays = WorkArrays(*_numba_master_vectorized(
                *args_flags,
                out=tuple(self.work_arrays),
            ))
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
        return result_arrays

    def loglike(
            self,
            x=None,
            *,
            start_case=0, stop_case=-1, step_case=1,
            persist=0,
            leave_out=-1, keep_only=-1, subsample=-1,
            probability_only=False,
    ):
        result_arrays = self._loglike_runner(x)
        result = result_arrays.loglike.sum() * self.dataframes.weight_normalization
        if start_case == 0 and stop_case == -1 and step_case == 1:
            self._check_if_best(result)
        return result

    def d_loglike(self, x=None, *args, return_series=False, **kwargs):
        result_arrays = self._loglike_runner(x, return_gradient=True)
        result = result_arrays.d_loglike.sum(0) * self.dataframes.weight_normalization
        if return_series:
            result = pd.Series(result, index=self._frame.index)
        return result

    def loglike_casewise(self, x=None, *args, **kwargs):
        result_arrays = self._loglike_runner(x)
        return result_arrays.loglike * self.dataframes.weight_normalization

    def d_loglike_casewise(self, x=None, *args, **kwargs):
        result_arrays = self._loglike_runner(x, return_gradient=True)
        return result_arrays.d_loglike * self.dataframes.weight_normalization

    def bhhh(self, x=None, *args, return_dataframe=False, **kwargs):
        result_arrays = self._loglike_runner(x, return_bhhh=True)
        result = result_arrays.bhhh.sum(0) * self.dataframes.weight_normalization
        if return_dataframe:
            result = pd.DataFrame(
                result, columns=self._frame.index, index=self._frame.index
            )
        return result

    def _wrap_as_dataframe(
            self,
            arr,
            return_dataframe,
            start_case=0,
            stop_case=-1,
            step_case=1,
    ):
        if return_dataframe:
            idx = self.dataframes.caseindex
            if idx is not None:
                if stop_case == -1:
                    stop_case = len(idx)
                idx = idx[start_case:stop_case:step_case]
            if return_dataframe == 'names':
                return pd.DataFrame(
                    data=arr,
                    columns=self.graph.standard_sort_names[:pr.shape[1]],
                    index=idx,
                )
            result = pd.DataFrame(
                data=arr,
                columns=self.graph.standard_sort[:pr.shape[1]],
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
            start_case=0,
            stop_case=-1,
            step_case=1,
            return_dataframe=False,
            include_nests=False,
    ):
        result_arrays = self._loglike_runner(x, return_probability=True)
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

    def utility(self, x=None, return_dataframe=None):
        result_arrays = self._loglike_runner(x, only_utility=2)
        return self._wrap_as_dataframe(
            result_arrays.utility,
            return_dataframe,
        )

    def quantity(self, x=None, return_dataframe=None):
        result_arrays = self._loglike_runner(x, only_utility=3)
        return self._wrap_as_dataframe(
            result_arrays.utility,
            return_dataframe,
        )

    def logsums(self, x=None, arr=None):
        result_arrays = self._loglike_runner(x, only_utility=2)
        if arr is not None:
            arr[:] = result_arrays.utility[:,-1]
            return arr
        return result_arrays.utility[:,-1]

    def loglike2(
            self,
            x=None,
            *,
            start_case=0,
            stop_case=-1,
            step_case=1,
            persist=0,
            leave_out=-1,
            keep_only=-1,
            subsample=-1,
            return_series=True,
            probability_only=False,
    ):
        result_arrays = self._loglike_runner(x, return_gradient=True)
        result = dictx(
            ll=result_arrays.loglike.sum() * self.dataframes.weight_normalization,
            dll=result_arrays.d_loglike.sum(0) * self.dataframes.weight_normalization,
        )
        if start_case == 0 and stop_case == -1 and step_case == 1:
            self._check_if_best(result.ll)
        if return_series:
            result['dll'] = pd.Series(result['dll'], index=self._frame.index, )
        return result

    def loglike2_bhhh(
            self,
            x=None,
            *,
            return_series=False,
            start_case=0, stop_case=-1, step_case=1,
            persist=0,
            leave_out=-1, keep_only=-1, subsample=-1,
    ):
        result_arrays = self._loglike_runner(x, return_gradient=True, return_bhhh=True)
        result = dictx(
            ll=result_arrays.loglike.sum() * self.dataframes.weight_normalization,
            dll=result_arrays.d_loglike.sum(0) * self.dataframes.weight_normalization,
            bhhh=result_arrays.bhhh.sum(0) * self.dataframes.weight_normalization,
        )
        from ..model.persist_flags import PERSIST_LOGLIKE_CASEWISE, PERSIST_D_LOGLIKE_CASEWISE
        if persist & PERSIST_LOGLIKE_CASEWISE:
            result['ll_casewise'] = result_arrays.loglike * self.dataframes.weight_normalization
        if persist & PERSIST_D_LOGLIKE_CASEWISE:
            result['dll_casewise'] = result_arrays.d_loglike * self.dataframes.weight_normalization
        if start_case == 0 and stop_case == -1 and step_case == 1:
            self._check_if_best(result.ll)
        if return_series:
            result['dll'] = pd.Series(result['dll'], index=self._frame.index, )
            result['bhhh'] = pd.DataFrame(
                result['bhhh'], index=self._frame.index, columns=self._frame.index
            )
        return result
